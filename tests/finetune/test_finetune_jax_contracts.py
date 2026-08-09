"""Invariants the training loop must satisfy, checked without a GPU.

Deliberately NOT an end-to-end run against a real DINOv3 backbone or the real
ImageNet-1k dataset: that needs an A100 and is covered by the Raven smoke in
the plan's exit criteria. What IS exercised here, with fakes standing in for
equimo/datasets, is the actual control flow of `main`, `stream_micro_batches`
and `evaluate_split` -- because every failure mode this module cares about
is silent: a run that violates one of these invariants produces dump files
that load fine and figures that are wrong.

The IBProbit arm's `__call__` (used for its logits/loss) runs a TF32 matmul
preset that JAX's CPU backend refuses outright (`ValueError: The precision
'TF32_TF32_F32' is not supported by dot_general on CPU` -- this reproduces on
this checkout via `bllarse.losses.bayesian.IBProbit.__call__` even outside
this module, e.g. tests/losses/test_bayesian.py fails to collect on CPU for
the same reason). That is a pre-existing, repo-wide property of the loss, not
something introduced here. It means the IBProbit forward/eval kernels
(`jax_steps.ibprobit_micro_forward[_nograd]`, `eval_logits_ibprobit`) cannot
be exercised for real on CPU; the IBProbit test below replaces exactly those
three call sites with stubs that skip the real forward. `jax_steps.apply_grads`
is also wrapped, but only to count calls -- it still delegates to and runs the
real implementation. `IBProbit.update` (the CAVI head update) is the real
function throughout, since it uses a CPU-safe matmul preset. That leaves the
surrounding control flow -- in particular "no optimizer step at step == 1" --
checked against the real `main` loop.
"""

import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dinov3"))

import finetune_jax as fj  # noqa: E402

from bllarse import evaldump, metrics as bllarse_metrics  # noqa: E402
from bllarse.finetune import jax_steps  # noqa: E402
from bllarse.losses import IBProbit  # noqa: E402


# --- Shared fakes -----------------------------------------------------------
#
# A real DINOv3 backbone needs equimo and a GPU. These fakes stand in for it
# and for the HuggingFace `datasets` split, so `main` can be driven end to end
# through its actual control flow on CPU.


class DummyBackbone(eqx.Module):
    """Tiny stand-in for the equimo ViT: mean-pool then a linear projection.

    Accepts a (3, H, W) channels-first image and a `key` kwarg, exactly the
    calling convention `jax_steps.forward_features` uses.
    """

    proj: jax.Array

    def __call__(self, x, key=None):
        return jnp.mean(x, axis=(1, 2)) @ self.proj


class FakeDatasetSplit:
    """Minimal stand-in for a HuggingFace split: supports both `split[i]`
    (a single row, for imagenet.load_images_uint8) and `split["label"]`
    (the whole column, for stream_micro_batches / main)."""

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return list(self._labels)
            if key == "image":
                return list(self._images)
            raise KeyError(key)
        return {"image": self._images[key], "label": self._labels[key]}


def _fake_images(n, size):
    return [
        Image.fromarray(np.full((size, size, 3), fill_value=(i * 17) % 256, dtype=np.uint8))
        for i in range(n)
    ]


def _fake_prefetch_iterator(iterator, n_prefetch=2):
    """Stand-in for vit_classification.prefetch_iterator: the identical
    queue/thread wrapping technique, duplicated so the fake module doesn't
    need to import the real (heavier) one just for this one function."""
    from queue import Queue
    from threading import Thread

    q = Queue(maxsize=n_prefetch)
    sentinel = object()

    def fill():
        for item in iterator:
            q.put(item)
        q.put(sentinel)

    t = Thread(target=fill, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


def _install_fake_heavy_modules(monkeypatch, *, embed_dim, num_classes, img_size,
                                train_n=8, eval_n=10):
    """Stand in for vit_classification and datasets so `main` never touches
    equimo or the real HuggingFace `datasets` package. Returns the args
    common to every caller; individual tests still build their own argv.
    """
    fake_vit = types.ModuleType("vit_classification")
    fake_vit.DATASET_CONFIGS = {
        "fake": {
            "hf_path": "fake/fake",
            "num_classes": num_classes,
            "label_key": "label",
            "img_key": "image",
            "test_split": "validation",
        }
    }
    fake_vit.EQUIMO_MODELS = {
        "fake_model": {"img_size": img_size, "embed_dim": embed_dim, "channels_first": True}
    }
    fake_vit.IMAGENET_MEAN = np.zeros(3)
    fake_vit.IMAGENET_STD = np.ones(3)

    def get_pretrained_backbone(model_name, key):
        return DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))

    fake_vit.get_pretrained_backbone = get_pretrained_backbone
    # stream_micro_batches (I5) imports prefetch_iterator from
    # vit_classification -- duplicated here (same queue/thread technique as
    # the real one in scripts/dinov3/vit_classification.py) so the fake
    # module doesn't need the real, heavier one.
    fake_vit.prefetch_iterator = _fake_prefetch_iterator

    train_split = FakeDatasetSplit(
        _fake_images(train_n, img_size), [i % num_classes for i in range(train_n)]
    )
    eval_split = FakeDatasetSplit(
        _fake_images(eval_n, img_size), [i % num_classes for i in range(eval_n)]
    )

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda hf_path, name=None: {
        "train": train_split, "validation": eval_split,
    }

    monkeypatch.setitem(sys.modules, "vit_classification", fake_vit)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    # resolve_plan derives `steps` from imagenet.IMAGENET_N_TRAIN, not from
    # len(train_split) -- the real constant (1.28M) would make even a 1-epoch
    # test run for a million steps, so it is patched down to the fake
    # dataset's actual size.
    monkeypatch.setattr(fj.imagenet, "IMAGENET_N_TRAIN", train_n)


def _base_argv(*, arm, dump_root, split_cache_dir, img_size, epochs, batch_size,
               micro_batch_size, **extra):
    argv = [
        "--arm", arm,
        "--model", "fake_model",
        "--dataset", "fake",
        "--img-size", str(img_size),
        "--allow-non-224",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--micro-batch-size", str(micro_batch_size),
        # Evaluate every optimizer step and dump every evaluation: these runs
        # are a handful of steps long, so the production quantum (16,384
        # samples) would schedule nothing at all.
        "--eval-quantum", str(batch_size),
        "--dump-every-k-evals", "1",
        "--dump-root", str(dump_root),
        "--group-id", "g",
        "--uid", "u",
        "--split-cache-dir", str(split_cache_dir),
        "--seed", "137",
    ]
    for key, value in extra.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


# --- Specification tests (pre-existing resolve_plan / evaldump contracts) --


def test_samples_seen_tracks_the_effective_batch(tmp_path):
    """samples_seen is Fig 5's x-axis; it must count the effective batch."""
    evaldump.write_dump(
        tmp_path, "g", "u", 7,
        np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64),
        arm="ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=7 * 1024,
    )
    record = next(iter(evaldump.iter_dumps(tmp_path, "g", "u")))
    assert record.meta["samples_seen"] == 7168


def test_evaluate_split_returns_logits_and_scalars_without_writing(tmp_path):
    """evaluate_split computes; it never writes.

    Evaluation now happens every eval_quantum samples -- 122 times in a tuning
    run -- while at most a stride of those points is dumped, so a function that
    wrote on every call would produce 4.9 GB per run. The caller buffers the
    argmin's logits and writes once, from _write_best.
    """
    key = jax.random.PRNGKey(0)
    embed_dim, num_classes, n = 5, 4, 7
    backbone = DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))
    head = eqx.nn.Linear(embed_dim, num_classes, key=key)

    images_uint8 = np.random.default_rng(0).integers(0, 256, size=(n, 2, 2, 3), dtype=np.uint8)
    labels = np.random.default_rng(1).integers(0, num_classes, size=(n,)).astype(np.int64)
    mean, std = np.zeros(3), np.ones(3)

    args = fj.build_argparser().parse_args([
        "--arm", "ce", "--batch-size", "4", "--micro-batch-size", "4",
        "--eval-chunk", "3",
        "--dump-root", str(tmp_path), "--group-id", "g", "--uid", "u",
    ])

    logits, scalars = fj.evaluate_split(
        "ce", backbone, head, images_uint8, labels,
        args=args, mean=mean, std=std, key=key,
    )

    # eval_chunk=3 over n=7 forces 3 chunks (3, 3, 1); every image must be
    # scored exactly once, not dropped or duplicated by the chunk loop.
    assert logits.shape == (n, num_classes)
    assert "nll" in scalars
    assert not list(evaldump.iter_dumps(tmp_path, "g", "u"))


def test_evaluate_split_subset_masks_logits_outside_the_subset(tmp_path):
    """`subset` must actually mask logits, not just be accepted and ignored."""
    key = jax.random.PRNGKey(1)
    embed_dim, num_classes, n = 6, 5, 6
    backbone = DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))
    head = eqx.nn.Linear(embed_dim, num_classes, key=key)
    images_uint8 = np.random.default_rng(2).integers(0, 256, size=(n, 2, 2, 3), dtype=np.uint8)
    labels = np.random.default_rng(3).integers(0, num_classes, size=(n,)).astype(np.int64)
    mean, std = np.zeros(3), np.ones(3)
    args = fj.build_argparser().parse_args([
        "--arm", "ce", "--batch-size", "4", "--micro-batch-size", "4",
        "--dump-root", str(tmp_path), "--group-id", "g", "--uid", "u",
    ])

    full_logits, _ = fj.evaluate_split(
        "ce", backbone, head, images_uint8, labels,
        args=args, mean=mean, std=std, key=key,
    )
    full_logits = np.asarray(full_logits, dtype=np.float32)
    # Pick the subset to deliberately exclude row 0's top class, so masking
    # is guaranteed to move that row's prediction -- a no-op mask (e.g. one
    # that always includes the argmax) would let this pass by accident.
    excluded = int(np.argmax(full_logits[0]))
    subset = [c for c in range(num_classes) if c != excluded]

    masked_logits, out = fj.evaluate_split(
        "ce", backbone, head, images_uint8, labels,
        args=args, mean=mean, std=std, key=key, subset=subset,
    )
    masked_logits = np.asarray(masked_logits, dtype=np.float32)

    # The excluded class is driven to a large negative value everywhere...
    assert np.all(masked_logits[:, excluded] < -1000)
    # ...every kept class is left alone...
    for c in subset:
        np.testing.assert_allclose(masked_logits[:, c], full_logits[:, c], atol=1e-5)
    # ...and row 0's prediction actually changed as a result.
    assert int(np.argmax(masked_logits[0])) != excluded
    # The returned metrics are computed on the MASKED array, not the raw one.
    assert out["acc"] == pytest.approx(
        float(np.mean(np.argmax(masked_logits, axis=-1) == labels))
    )


def test_masked_logits_survive_the_fp16_dump_round_trip(tmp_path):
    """MASK_NEG must not overflow float16 on the way to disk.

    evaldump.write_dump stores logits as float16 (max finite magnitude
    65504). The original choice of -1e9 OVERFLOWS to -inf there: softmax's
    probability for that class then underflows to exactly 0.0, and
    0.0 * (-inf) = NaN, so entropy_ood_score and fit_temperature go NaN on a
    masked row while msp_ood_score (which never multiplies by the raw logit)
    still looks healthy -- a downstream figure would show a working MSP
    column next to a silently-NaN entropy column. This test must FAIL if
    MASK_NEG reverts to -1e9 (verified by mutation in the fix report).
    """
    num_classes, n = 5, 8
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(n, num_classes)).astype(np.float32)
    keep = np.zeros(num_classes, dtype=bool)
    keep[[0, 2]] = True
    masked = np.where(keep, logits, fj.MASK_NEG)
    labels = rng.integers(0, num_classes, size=(n,)).astype(np.int64)

    path = evaldump.write_dump(
        tmp_path, "g", "u", 1, masked, labels,
        arm="ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=4,
    )
    loaded = evaldump.load_dump(path).logits.astype(np.float32)

    assert np.all(np.isfinite(loaded)), "the masked logits overflowed the fp16 dump"
    entropy = np.asarray(bllarse_metrics.entropy_ood_score(loaded))
    assert np.all(np.isfinite(entropy)), "entropy went NaN on a masked row"
    temperature = bllarse_metrics.fit_temperature(loaded, labels)
    assert np.isfinite(temperature), "fit_temperature went NaN on masked rows"


# --- stream_micro_batches ---------------------------------------------------


def test_stream_micro_batches_drops_the_short_trailing_batch_and_reshuffles():
    n, micro, img_size = 5, 2, 4
    ds_config = {"img_key": "image", "label_key": "label"}
    split = FakeDatasetSplit(_fake_images(n, img_size), list(range(n)))
    mean, std = np.zeros(3), np.ones(3)
    seed = 42

    # n=5, micro=2: only floor(5/2)=2 full batches per epoch (positions
    # 0:2, 2:4); position 4 is the short trailing batch and must be skipped,
    # so the 3rd requested micro-batch has to come from a FRESH reshuffle
    # (epoch 1), not from index 4 of epoch 0.
    batches = list(fj.stream_micro_batches(
        split, ds_config, micro, img_size, mean, std,
        total_micro=3, seed=seed, num_workers=2,
    ))
    assert len(batches) == 3

    from bllarse.finetune import protocol
    order0 = protocol.shuffled_indices(n, seed, 0)
    order1 = protocol.shuffled_indices(n, seed, 1)
    expected_label_batches = [order0[0:2], order0[2:4], order1[0:2]]

    for (images, labels), expected in zip(batches, expected_label_batches):
        assert images.shape == (micro, 3, img_size, img_size)
        np.testing.assert_array_equal(np.asarray(labels), expected)


def test_stream_micro_batches_uses_the_prefetch_buffer(monkeypatch):
    """I5: stream_micro_batches must actually route through
    vit_classification.prefetch_iterator(n_prefetch=10). Order and count are
    already covered by the sibling test above and are UNCHANGED by adding
    prefetch (it's a concurrency change, not a correctness one) -- so a
    version that dropped the wrap entirely would still pass that test. Only
    spying on the call itself catches a revert.
    """
    import vit_classification

    calls = []
    real_prefetch_iterator = vit_classification.prefetch_iterator

    def spy_prefetch_iterator(iterator, n_prefetch=2):
        calls.append(n_prefetch)
        return real_prefetch_iterator(iterator, n_prefetch=n_prefetch)

    monkeypatch.setattr(vit_classification, "prefetch_iterator", spy_prefetch_iterator)

    n, micro, img_size = 5, 2, 4
    ds_config = {"img_key": "image", "label_key": "label"}
    split = FakeDatasetSplit(_fake_images(n, img_size), list(range(n)))
    mean, std = np.zeros(3), np.ones(3)

    batches = list(fj.stream_micro_batches(
        split, ds_config, micro, img_size, mean, std,
        total_micro=3, seed=42, num_workers=2,
    ))

    assert len(batches) == 3, "the prefetch wrap must not change the total count"
    assert calls, "stream_micro_batches never called prefetch_iterator"
    assert all(n == 10 for n in calls), f"expected n_prefetch=10 every call, got {calls}"


# --- main(): the real control flow, driven end to end with fakes -----------


def test_main_ce_arm_dumps_both_splits_at_every_scheduled_step(tmp_path, monkeypatch):
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        # imagenet.IMAGENET_N_TRAIN is patched to 8; epochs=1.0 * 8 // effective(4) = 2 steps.
        # accum_plan(4, 2) -> micro=2, num_accum=2, effective=4. Request and
        # achieved now always agree, because design A8 pins the micro batch to
        # a divisor of the target and accum_plan raises otherwise -- the case
        # this test used to distinguish (requested 5, achieved 4) is no longer
        # reachable, and test_protocol asserts it raises instead.
        # OOD is off here: this test is about the val/test schedule, not OOD,
        # and leaving --ood-every-k-dumps at its default would make main()
        # call load_ood_sets, which reaches out to precompute_ood_features
        # (network) -- see test_finetune_jax_ood.py for the OOD-specific
        # coverage.
        epochs=1.0, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)

    dump_calls = []
    fj.main(args, on_dump=lambda i, s, n, sc: dump_calls.append((i, s, n, dict(sc))))

    records = list(evaldump.iter_dumps(dump_root, "g", "u"))
    by_split = {}
    samples_seen = {}
    for r in records:
        by_split.setdefault(r.meta["split"], set()).add(r.meta["step"])
        samples_seen[(r.meta["split"], r.meta["step"])] = r.meta["samples_seen"]
        assert r.meta["logit_kind"] == "softmax_logits"

    # Both splits, every scheduled step (1 and 2) -- fig1_curves.score_run's
    # invariant, now checked against the real loop instead of a hand call.
    assert by_split["val"] == by_split["test"] == {1, 2}
    # effective_batch is 4 (2 micro x 2 accum), so samples_seen = step * 4.
    assert samples_seen[("val", 1)] == 4
    assert samples_seen[("test", 2)] == 8

    # eval_index = samples_seen // eval_quantum, and the quantum is set to the
    # batch here, so index i means i * 4 samples -- the same amount of training
    # for any rung of the ladder, which is what makes it a valid pruner key.
    assert [c[0] for c in dump_calls] == [1, 2]
    assert [c[1] for c in dump_calls] == [1, 2]
    assert [c[2] for c in dump_calls] == [4, 8]
    for _, _, _, scalars in dump_calls:
        assert set(scalars) == {
            "val_acc", "val_ece", "val_nll", "val_brier",
            "test_acc", "test_ece", "test_nll", "test_brier",
        }


def test_main_ce_arm_actually_trains_not_a_no_op(tmp_path, monkeypatch):
    """Closes two mutations that survived the original suite:

    (c) apply_grads never called at all -- CE training is a total no-op, and
    (d) grads_acc divided by num_accum an extra time before apply_grads,
        which already divides by num_accum internally (a silent 2x-too-small
        update).

    A simple "did the backbone/head change at all" check does NOT catch (d):
    verified empirically that AdamW's per-parameter normalization
    (m_hat / sqrt(v_hat)) is EXACTLY invariant to a constant rescale of the
    gradient applied consistently from step 1 -- run twice, once correct and
    once with the extra division, the final backbone params came out
    bit-identical to 8 significant figures. So in addition to the "changed at
    all" check, this test captures the exact grads pytree `main()` passes
    into apply_grads at every step and compares it against an independently
    summed reference built from the real (unmocked) per-micro-batch grads --
    that is what actually distinguishes "summed once" from "summed then
    halved again".
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )

    grad_calls = []
    apply_calls = []
    dump_snapshots = []

    real_ce_micro_grad = jax_steps.ce_micro_grad
    real_apply_grads = jax_steps.apply_grads
    real_evaluate_split = fj.evaluate_split

    def capturing_ce_micro_grad(model, loss_fn, images, labels, key, bf16=False):
        grads, loss = real_ce_micro_grad(model, loss_fn, images, labels, key, bf16=bf16)
        grad_calls.append(jax.tree.map(lambda a: np.array(a), grads))
        return grads, loss

    def capturing_apply_grads(model, grads, opt_state, optimizer, num_accum):
        # np.array(), not a bare identity tree.map: apply_grads donates
        # `grads` (donate="all-except-first"), so a shallow snapshot would
        # alias a buffer XLA frees/reuses inside this very call.
        apply_calls.append((jax.tree.map(lambda a: np.array(a), grads), num_accum))
        return real_apply_grads(model, grads, opt_state, optimizer, num_accum)

    def capturing_evaluate_split(arm, backbone, head_or_loss, *rest, **kw):
        dump_snapshots.append((backbone, head_or_loss))
        return real_evaluate_split(arm, backbone, head_or_loss, *rest, **kw)

    monkeypatch.setattr(jax_steps, "ce_micro_grad", capturing_ce_micro_grad)
    monkeypatch.setattr(jax_steps, "apply_grads", capturing_apply_grads)
    monkeypatch.setattr(fj, "evaluate_split", capturing_evaluate_split)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        # OOD off -- this test is about the grads/apply_grads control flow,
        # not OOD; see the note in the sibling test above.
        epochs=1.0, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)
    fj.main(args, on_dump=None)

    num_accum, steps = 2, 2  # accum_plan(5, 2) -> micro=2, num_accum=2; 2 steps

    # (c): apply_grads must actually run once per step, not be skipped.
    assert len(apply_calls) == steps
    assert len(grad_calls) == steps * num_accum

    # (d): the grads apply_grads receives must be the raw, undivided sum of
    # that step's micro-batch grads -- apply_grads (jax_steps.py, untouched)
    # already divides by num_accum internally.
    for step_idx, (received_grads, received_num_accum) in enumerate(apply_calls):
        assert received_num_accum == num_accum
        raw = grad_calls[step_idx * num_accum:(step_idx + 1) * num_accum]
        expected = raw[0]
        for other in raw[1:]:
            expected = jax.tree.map(lambda a, b: a + b, expected, other)
        for got, exp in zip(jax.tree.leaves(received_grads), jax.tree.leaves(expected)):
            np.testing.assert_allclose(
                got, exp, rtol=1e-6, atol=1e-8,
                err_msg=(
                    f"step {step_idx}: apply_grads did not receive the raw sum "
                    f"of this step's micro-batch grads (it already divides by "
                    f"num_accum internally -- dividing again before the call "
                    f"halves every update)"
                ),
            )

    # Documentation-level sanity check on top of the above: the trained
    # params must actually differ from their known initial values. This
    # alone would NOT have caught (d) (see docstring), but it is still the
    # simplest and most direct way to catch a total no-op like (c) and any
    # other bug that leaves the model completely unchanged.
    key0 = jax.random.PRNGKey(int(args.seed))
    key0, model_key0, init_key0, fwd_key0 = jax.random.split(key0, 4)
    expected_initial_proj = jax.random.normal(model_key0, (3, embed_dim))
    expected_initial_head = eqx.nn.Linear(embed_dim, num_classes, key=init_key0)

    final_backbone, final_head = dump_snapshots[-1]
    assert not np.allclose(
        np.asarray(final_backbone.proj), np.asarray(expected_initial_proj)
    ), "backbone params never changed -- training is a no-op"
    assert not np.allclose(
        np.asarray(final_head.weight), np.asarray(expected_initial_head.weight)
    ), "head params never changed -- training is a no-op"


def test_main_ibprobit_arm_skips_the_optimizer_step_at_step_one(tmp_path, monkeypatch):
    """Design invariant: the seed posterior gives a zero trunk gradient, so
    IBProbit must take no optimizer step at step == 1, but must still call
    loss_fn.update at every step (including step 1) and dump both splits
    everywhere. jax_steps.ibprobit_micro_forward[_nograd] and
    eval_logits_ibprobit are the only three things faked here (see module
    docstring for why: TF32 is not supported on the CPU backend);
    apply_grads and IBProbit.update are the REAL functions.
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )

    call_log = []

    def fake_forward(backbone, loss_fn, images, labels, key, bf16=False):
        call_log.append("forward")
        grads = jax.tree.map(lambda a: jnp.ones_like(a) * 1e-3,
                             eqx.filter(backbone, eqx.is_array))
        return grads, jnp.zeros((images.shape[0], embed_dim)), jnp.array(0.0)

    def fake_forward_nograd(backbone, loss_fn, images, labels, key, bf16=False):
        call_log.append("forward_nograd")
        return jnp.zeros((images.shape[0], embed_dim)), jnp.array(0.0)

    real_apply_grads = jax_steps.apply_grads

    def counting_apply_grads(model, grads, opt_state, optimizer, num_accum):
        call_log.append("apply_grads")
        return real_apply_grads(model, grads, opt_state, optimizer, num_accum)

    def fake_eval_logits(backbone, loss_fn, images, labels, key, bf16=False):
        call_log.append("eval_logits")
        return jnp.zeros((images.shape[0], num_classes))

    # loss_fn.update is the CAVI head update -- the only thing that arm does
    # at step == 1, and it sets the leftmost point of every downstream curve.
    # Wrapped (not faked) at the CLASS level, same as apply_grads: it still
    # runs the real, CPU-safe computation, just counted.
    real_update = IBProbit.update

    def counting_update(self, *a, **kw):
        call_log.append("update")
        return real_update(self, *a, **kw)

    monkeypatch.setattr(jax_steps, "ibprobit_micro_forward", fake_forward)
    monkeypatch.setattr(jax_steps, "ibprobit_micro_forward_nograd", fake_forward_nograd)
    monkeypatch.setattr(jax_steps, "apply_grads", counting_apply_grads)
    monkeypatch.setattr(fj, "eval_logits_ibprobit", fake_eval_logits)
    monkeypatch.setattr(IBProbit, "update", counting_update)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ibprobit", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        # batch_size=5 requested, effective=4 achieved (see the CE test above
        # for why); 1.5 * 8 // 4 = 3 steps either way. OOD off -- see the note
        # in the CE test above.
        epochs=1.5, batch_size=4, micro_batch_size=2, num_update_iters=3,
        ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)

    fj.main(args, on_dump=None)

    # step 1: forward_nograd x num_accum(2), NO apply_grads, then update, then
    # 2 eval calls. step 2 and 3: forward x2, exactly one apply_grads, update,
    # then 2 eval calls. "update" appearing at EVERY step (including step 1,
    # where it is the only thing IBProbit does) is the invariant closing the
    # gap where a step-1 "skip loss_fn.update too" mutation previously
    # survived: only counting apply_grads could not tell "correctly skipped
    # the optimizer step" apart from "skipped everything at step 1".
    assert call_log == [
        "forward_nograd", "forward_nograd", "update", "eval_logits", "eval_logits",
        "forward", "forward", "apply_grads", "update", "eval_logits", "eval_logits",
        "forward", "forward", "apply_grads", "update", "eval_logits", "eval_logits",
    ]

    records = list(evaldump.iter_dumps(dump_root, "g", "u"))
    by_split = {}
    for r in records:
        by_split.setdefault(r.meta["split"], set()).add(r.meta["step"])
        # "softmax_logits", not the "moderated_score" this once asserted.
        # IBProbit.__call__(with_logits=True) returns _mixture_logp, which is
        # normalized by construction -- (1-eps)*1 + K*(eps/K) == 1 -- so
        # softmaxing it is a no-op and metrics.classification_metrics is
        # correct on it. It is the same kind of array the VBLL arm dumps under
        # the same label. Under the old label, curves.run_curve refused every
        # IBProbit dump, so a tuning campaign whose training all succeeded
        # would still have failed every trial at scoring time.
        assert r.meta["logit_kind"] == "softmax_logits"
    assert by_split["val"] == by_split["test"] == {1, 2, 3}


# --- M5: default uid must not collide with fig1_curves.UID_RE's dataset ----


def test_main_default_uid_includes_the_dataset_before_the_arm(tmp_path, monkeypatch):
    """fig1_curves.UID_RE parses a uid as {model}_{dataset}_seed{seed}. The
    old default {model}_{arm}_seed{seed} (e.g. dinov3_large_ce_seed137)
    parses CLEANLY under that pattern with dataset="ce" -- a silent mislabel,
    not a parse failure -- so this must check the actual VALUE main() assigns
    to args.uid, not just that SOME uid gets set.
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = [
        "--arm", "ce", "--model", "fake_model", "--dataset", "fake",
        "--img-size", str(img_size), "--allow-non-224",
        "--epochs", "1.0", "--batch-size", "4", "--micro-batch-size", "2",
        "--eval-quantum", "4", "--dump-every-k-evals", "1",
        "--dump-root", str(dump_root), "--group-id", "g",
        "--split-cache-dir", str(split_cache), "--seed", "137",
        "--ood-every-k-dumps", "0",
        # --uid deliberately omitted -- that's what triggers the default this
        # test targets.
    ]
    args = fj.build_argparser().parse_args(argv)
    assert args.uid is None

    fj.main(args, on_dump=None)

    assert args.uid == "fake_model_fake_ce_seed137"
    records = list(evaldump.iter_dumps(dump_root, "g", args.uid))
    assert records, "no dumps found under the assigned default uid"


# --- I7: --head-lr's effective default -------------------------------------


def test_main_ce_arm_head_lr_defaults_to_the_100x_trunk_rate(tmp_path, monkeypatch):
    """--head-lr's argparse default stays None (check_arm_flags needs that
    None sentinel to reject an explicit --head-lr on --arm ibprobit -- see
    test_head_lr_is_rejected_for_the_ibprobit_arm, which passes exactly 1e-2
    and must still be rejected), but the EFFECTIVE rate the CE optimizer's
    head part receives when --head-lr is omitted must be
    fj.DEFAULT_HEAD_LR (1e-2, matching full_finetune_comparison.py), not
    args.lr (which would give the head and trunk the same rate -- the worst
    end of the range this default is meant to avoid).
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    real_adamw = optax.adamw
    seen_lrs = []

    def capturing_adamw(lr, *a, **kw):
        seen_lrs.append(lr)
        return real_adamw(lr, *a, **kw)

    monkeypatch.setattr(fj.optax, "adamw", capturing_adamw)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)
    assert args.head_lr is None  # sentinel preserved for check_arm_flags

    fj.main(args, on_dump=None)

    assert fj.DEFAULT_HEAD_LR == pytest.approx(1e-2)
    # optax.multi_transform builds "backbone" then "head" (dict insertion
    # order in Python 3.7+), so seen_lrs[0] is the trunk rate and
    # seen_lrs[1] is the head rate.
    assert seen_lrs == [args.lr, fj.DEFAULT_HEAD_LR]


# --- I6: train throughput report --------------------------------------------


def test_main_reports_train_throughput_excluding_step_one(tmp_path, monkeypatch, capsys):
    """main() must time only the post-compile steps (excluding step 1 from
    both the accumulated total AND the divisor used for the mean) and report
    images/second computed from the ACHIEVED effective batch, not the
    requested --batch-size (batch_size=5 requested vs effective_batch=4
    achieved with this config -- see the sibling dumps test for why that gap
    matters and must not be conflated).

    Two independently-controlled, well-separated delays catch two different
    mutations: COMPILE_SLEEP (huge, step 1 only) would leak into the reported
    total if the `step > 1` accumulation guard were dropped; STEP_SLEEP
    (uniform, every post-compile step) would shift the reported mean/img-s if
    the divisor used steps instead of steps - 1 -- a bare self-consistency
    check (recomputing img/s from the SAME printed mean) cannot catch that
    second one, since both numbers would still agree with each other even if
    both are wrong the same way.
    """
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    real_ce_micro_grad = jax_steps.ce_micro_grad
    NUM_ACCUM = 2  # accum_plan(5, 2) -> micro=2, num_accum=2 (matches the config below)
    COMPILE_SLEEP = 1.0
    STEP_SLEEP = 0.15
    call_count = [0]

    def timed_ce_micro_grad(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            time.sleep(COMPILE_SLEEP)  # step 1's first micro-batch (compile)
        elif call_count[0] > NUM_ACCUM:
            time.sleep(STEP_SLEEP / NUM_ACCUM)  # every micro-batch of steps > 1
        return real_ce_micro_grad(*a, **kw)

    monkeypatch.setattr(jax_steps, "ce_micro_grad", timed_ce_micro_grad)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    # epochs=1.5, batch_size=5, micro=2 -> effective_batch=4, steps=3 (see
    # test_main_ibprobit_arm_skips_the_optimizer_step_at_step_one for the same
    # arithmetic on this arm's sibling). n_timed = steps - 1 = 2.
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.5, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)
    fj.main(args, on_dump=None)

    out = capsys.readouterr().out
    m = re.search(
        r"train time: ([\d.]+)s total, ([\d.]+)ms/step.*?([\d.]+) img/s", out
    )
    assert m, f"expected a throughput report line in stdout, got:\n{out}"
    total_time, mean_ms, img_per_sec = (
        float(m.group(1)), float(m.group(2)), float(m.group(3))
    )

    steps, n_timed = 3, 2

    # COMPILE_SLEEP must not be in the reported total -- if the `step > 1`
    # accumulation guard were dropped, total_time would be >= COMPILE_SLEEP.
    assert total_time < COMPILE_SLEEP, (
        f"reported train time {total_time}s includes step 1's "
        f"{COMPILE_SLEEP}s compile sleep -- step 1 was not excluded"
    )
    # ...and it should be close to n_timed * STEP_SLEEP (2 * 0.15s = 0.30s).
    assert total_time == pytest.approx(n_timed * STEP_SLEEP, rel=0.3)

    # The divisor must be n_timed = steps - 1 = 2, not steps = 3: dividing by
    # 3 instead of 2 would report a mean ~33% too low (0.30s / 3 = 100ms vs
    # the correct 0.30s / 2 = 150ms) -- well outside this tolerance.
    assert mean_ms == pytest.approx(STEP_SLEEP * 1000.0, rel=0.2)

    # effective_batch = 4: accum_plan(5, 2) -> micro=2, num_accum=2. A version
    # that used args.batch_size (5, the REQUESTED batch) instead of
    # plan["effective_batch"] (4, the ACHIEVED one) would fail this.
    effective_batch = 4
    expected_img_per_sec = effective_batch / (mean_ms / 1000.0)
    assert img_per_sec == pytest.approx(expected_img_per_sec, rel=0.05)


# --- M8: a diverged run must stop, not train silently to the full budget ---


def test_main_raises_when_val_nll_is_not_finite_at_a_dump(tmp_path, monkeypatch):
    embed_dim, num_classes, img_size = 5, 3, 4
    _install_fake_heavy_modules(
        monkeypatch, embed_dim=embed_dim, num_classes=num_classes, img_size=img_size,
        train_n=8, eval_n=10,
    )
    real_evaluate_split = fj.evaluate_split
    calls = {"n": 0}

    def poisoned_evaluate_split(arm, backbone, head_or_loss, *rest, **kw):
        logits, out = real_evaluate_split(arm, backbone, head_or_loss, *rest, **kw)
        # The val split is evaluated first at each point, so poisoning the
        # first call of each pair poisons val specifically.
        if calls["n"] % 2 == 0:
            out = dict(out)
            out["nll"] = float("nan")
        calls["n"] += 1
        return logits, out

    monkeypatch.setattr(fj, "evaluate_split", poisoned_evaluate_split)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=4, micro_batch_size=2, ood_every_k_dumps=0,
    )
    args = fj.build_argparser().parse_args(argv)

    with pytest.raises(RuntimeError, match=r"step 1.*val_nll"):
        fj.main(args, on_dump=None)


# --- Sharding is a pure speed change ----------------------------------------


def test_the_shard_path_and_the_jpeg_path_yield_identical_micro_batches(tmp_path):
    """Same images, same labels, same order -- byte for byte.

    A shard exists only to remove JPEG decode from the training loop (measured
    220 img/s synthetic against ~40 through the live path on an A100). If it
    changed the stream at all, the JAX and PyTorch arms would train on
    different data under the same seed, and both would still produce plausible
    loss curves -- there is no downstream check that could catch it.
    """
    from bllarse.finetune import shards

    n, size, micro = 16, 4, 4
    split = FakeDatasetSplit(_fake_images(n, size), [i % 3 for i in range(n)])
    ds_config = {"img_key": "image", "label_key": "label"}
    mean, std = np.zeros(3), np.ones(3)

    shards.build_shard(
        split, img_key="image", label_key="label", img_size=size,
        root=tmp_path, dataset="fake", split="train", num_workers=2, chunk=8)
    reader = shards.open_shard(tmp_path, "fake", "train", size)

    fake_vit = types.ModuleType("vit_classification")
    fake_vit.prefetch_iterator = _fake_prefetch_iterator
    sys.modules["vit_classification"] = fake_vit
    try:
        total = 6
        jpeg = list(fj.stream_micro_batches(
            split, ds_config, micro, size, mean, std, total, seed=137))
        sharded = list(fj.stream_micro_batches(
            split, ds_config, micro, size, mean, std, total, seed=137,
            shard=reader))
    finally:
        del sys.modules["vit_classification"]

    assert len(jpeg) == len(sharded) == total
    for (ji, jl), (si, sl) in zip(jpeg, sharded):
        np.testing.assert_array_equal(np.asarray(ji), np.asarray(si))
        np.testing.assert_array_equal(np.asarray(jl), np.asarray(sl))
