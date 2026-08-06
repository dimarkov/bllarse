"""Invariants the training loop must satisfy, checked without a GPU.

Deliberately NOT an end-to-end run against a real DINOv3 backbone or the real
ImageNet-1k dataset: that needs an A100 and is covered by the Raven smoke in
the plan's exit criteria. What IS exercised here, with fakes standing in for
equimo/datasets, is the actual control flow of `main`, `stream_micro_batches`
and `evaluate_and_dump` -- because every failure mode this module cares about
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

import inspect
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import jax
import jax.numpy as jnp
import equinox as eqx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "alternatives"))

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


def test_dump_steps_and_ood_steps_never_disagree_about_the_final_step():
    plan = fj.resolve_plan(
        fj.build_argparser().parse_args(["--arm", "ce", "--batch-size", "512"]),
        micro_batch=64,
    )
    # curves._id_ood_pair raises if the final test dump and the final ood dump
    # sit at different steps, so the schedules must agree at the endpoint.
    assert plan["ood_steps"][-1] == plan["dump_steps"][-1]


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


# --- auto-probe contract: build_argparser must cover what the prober reads -
#
# main()'s auto-probe path (when --micro-batch-size is omitted) calls
# full_finetune_comparison.probe_micro_batch_size, which unconditionally
# builds an IBProbit head via build_ibprobit -- regardless of --arm, since the
# prober always measures the heavier of the two arms' micro-steps. That
# function reads ten args.* attributes finetune_jax's build_argparser did not
# originally define, so the auto-probe path raised AttributeError (masked by
# an UnboundLocalError from build_ibprobit's `finally: del loss_fn`) on the
# documented default invocation. The list below is DERIVED from the actual
# source via a regex over `args.<name>` accesses, not hardcoded, so this test
# keeps working if that attribute set changes again.


def _args_attrs_touched(fn) -> set:
    """Every `args.<name>` attribute access textually present in fn's source."""
    return set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)\b", inspect.getsource(fn)))


def test_probe_micro_batch_size_args_are_all_defined_on_finetune_jax_namespace():
    import full_finetune_comparison as ffc  # noqa: E402 -- heavy, kept local to this test

    touched = _args_attrs_touched(ffc.probe_micro_batch_size) | _args_attrs_touched(
        ffc.build_ibprobit
    )
    # Sanity: the regex must actually have found the ten IBProbit knobs (and
    # bf16/optimizer/estimator), or this test would trivially pass on zero
    # attributes and catch nothing.
    assert {"q_init", "px", "bmr_gate", "learn_eps", "bf16", "optimizer"} <= touched

    args = fj.build_argparser().parse_args(["--arm", "ibprobit", "--batch-size", "4"])
    missing = sorted(name for name in touched if not hasattr(args, name))
    assert not missing, (
        f"finetune_jax build_argparser is missing attributes the auto-probe "
        f"path reads: {missing}"
    )


# --- evaluate_and_dump: exercises the REAL function, not a hand transcript -
#
# The brief's version of the "both splits" test called evaldump.write_dump
# directly with hand-picked arguments, which does not exercise
# evaluate_and_dump at all -- a broken evaluate_and_dump (e.g. one that
# swapped `split` or dropped a chunk) would still pass it. This version drives
# the real function.


def test_evaluate_and_dump_writes_both_splits_with_the_effective_batch_samples_seen(tmp_path):
    key = jax.random.PRNGKey(0)
    embed_dim, num_classes, n = 5, 4, 7
    backbone = DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))
    head = eqx.nn.Linear(embed_dim, num_classes, key=key)

    images_uint8 = np.random.default_rng(0).integers(0, 256, size=(n, 2, 2, 3), dtype=np.uint8)
    labels = np.random.default_rng(1).integers(0, num_classes, size=(n,)).astype(np.int64)
    mean, std = np.zeros(3), np.ones(3)

    args = fj.build_argparser().parse_args([
        "--arm", "ce", "--batch-size", "4", "--eval-chunk", "3",
        "--dump-root", str(tmp_path), "--group-id", "g", "--uid", "u",
    ])

    for step, samples_seen in ((1, 4), (3, 12)):
        for split in ("val", "test"):
            fj.evaluate_and_dump(
                "ce", backbone, head, images_uint8, labels,
                split=split, step=step, samples_seen=samples_seen,
                args=args, mean=mean, std=std, key=key,
            )

    by_split = {}
    seen_samples = {}
    for record in evaldump.iter_dumps(tmp_path, "g", "u"):
        by_split.setdefault(record.meta["split"], set()).add(record.meta["step"])
        seen_samples[(record.meta["split"], record.meta["step"])] = record.meta["samples_seen"]
        # eval_chunk=3 over n=7 forces 3 chunks (3, 3, 1); every image must be
        # scored exactly once, not dropped or duplicated by the chunk loop.
        assert record.logits.shape == (n, num_classes)

    assert by_split["val"] == by_split["test"] == {1, 3}
    assert seen_samples[("val", 1)] == 4
    assert seen_samples[("test", 3)] == 12


def test_evaluate_and_dump_subset_masks_logits_outside_the_subset(tmp_path):
    """The override this task adds: `subset` must actually mask logits, not
    just be accepted and ignored (Task 6 will only need to CALL this)."""
    key = jax.random.PRNGKey(1)
    embed_dim, num_classes, n = 6, 5, 6
    backbone = DummyBackbone(proj=jax.random.normal(key, (3, embed_dim)))
    head = eqx.nn.Linear(embed_dim, num_classes, key=key)
    images_uint8 = np.random.default_rng(2).integers(0, 256, size=(n, 2, 2, 3), dtype=np.uint8)
    labels = np.random.default_rng(3).integers(0, num_classes, size=(n,)).astype(np.int64)
    mean, std = np.zeros(3), np.ones(3)
    args = fj.build_argparser().parse_args([
        "--arm", "ce", "--batch-size", "4",
        "--dump-root", str(tmp_path), "--group-id", "g", "--uid", "u",
    ])

    fj.evaluate_and_dump(
        "ce", backbone, head, images_uint8, labels,
        split="val", step=1, samples_seen=4, args=args, mean=mean, std=std, key=key,
    )
    unmasked = next(
        r for r in evaldump.iter_dumps(tmp_path, "g", "u") if r.meta["split"] == "val"
    )
    full_logits = unmasked.logits.astype(np.float32)
    # Pick the subset to deliberately exclude row 0's top class, so masking
    # is guaranteed to move that row's prediction -- a no-op mask (e.g. one
    # that always includes the argmax) would let this pass by accident.
    excluded = int(np.argmax(full_logits[0]))
    subset = [c for c in range(num_classes) if c != excluded]

    out = fj.evaluate_and_dump(
        "ce", backbone, head, images_uint8, labels,
        split="test", step=1, samples_seen=4, args=args, mean=mean, std=std, key=key,
        subset=subset,
    )
    masked = next(
        r for r in evaldump.iter_dumps(tmp_path, "g", "u") if r.meta["split"] == "test"
    )
    masked_logits = masked.logits.astype(np.float32)

    # The excluded class is driven to a large negative value everywhere...
    assert np.all(masked_logits[:, excluded] < -1000)
    # ...every kept class is left alone (float16 round-trip tolerance)...
    for c in subset:
        np.testing.assert_allclose(masked_logits[:, c], full_logits[:, c], atol=0.05)
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
        # batch_size=5 (REQUESTED) deliberately differs from the ACHIEVED
        # effective_batch=4 (accum_plan(5, 2) -> micro=2, num_accum=2,
        # effective=4, since round(5/2)=2): with batch_size=4 == effective the
        # samples_seen assertions below cannot tell "step * requested" apart
        # from "step * achieved", so a bug that used args.batch_size directly
        # would pass silently.
        epochs=1.0, batch_size=5, micro_batch_size=2,
    )
    args = fj.build_argparser().parse_args(argv)

    dump_calls = []
    fj.main(args, on_dump=lambda i, s, sc: dump_calls.append((i, s, dict(sc))))

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

    assert [c[0] for c in dump_calls] == [0, 1]
    assert [c[1] for c in dump_calls] == [1, 2]
    for _, _, scalars in dump_calls:
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
    real_evaluate_and_dump = fj.evaluate_and_dump

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

    def capturing_evaluate_and_dump(arm, backbone, head_or_loss, *rest, **kw):
        dump_snapshots.append((backbone, head_or_loss))
        return real_evaluate_and_dump(arm, backbone, head_or_loss, *rest, **kw)

    monkeypatch.setattr(jax_steps, "ce_micro_grad", capturing_ce_micro_grad)
    monkeypatch.setattr(jax_steps, "apply_grads", capturing_apply_grads)
    monkeypatch.setattr(fj, "evaluate_and_dump", capturing_evaluate_and_dump)

    dump_root = tmp_path / "dumps"
    split_cache = tmp_path / "splits"
    argv = _base_argv(
        arm="ce", dump_root=dump_root, split_cache_dir=split_cache, img_size=img_size,
        epochs=1.0, batch_size=5, micro_batch_size=2,
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
        # for why); 1.5 * 8 // 4 = 3 steps either way.
        epochs=1.5, batch_size=5, micro_batch_size=2, num_update_iters=3,
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
        assert r.meta["logit_kind"] == "moderated_score"
    assert by_split["val"] == by_split["test"] == {1, 2, 3}
