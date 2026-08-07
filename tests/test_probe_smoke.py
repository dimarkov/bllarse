import sys
from pathlib import Path

import numpy as np
import pytest

ALT = Path(__file__).resolve().parents[1] / "scripts" / "dinov3"
sys.path.insert(0, str(ALT))

from bllarse import evaldump


@pytest.fixture
def fake_features(tmp_path):
    """Two well-separated Gaussian blobs -- a linear head must reach high accuracy."""
    rng = np.random.default_rng(0)
    cache = tmp_path / "features"
    cache.mkdir()
    for split, n in (("train", 600), ("test", 200)):
        labels = rng.integers(0, 2, size=n)
        feats = rng.normal(size=(n, 16)).astype(np.float32)
        feats[labels == 1, 0] += 6.0
        np.savez(cache / f"dinov3_small_fake_res224_{split}.npz",
                 features=feats, labels=labels)
    return cache


def test_probe_writes_dumps_and_learns(fake_features, tmp_path):
    import probe

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "smoke_0",
        # NOTE: raised from the brief's --epochs 8. At the default --batch-size
        # 512 with only 600 training rows, _ce_epoch does floor(600/512) = 1
        # gradient step per epoch, so 8 epochs is 8 Adam steps -- nowhere near
        # enough at lr=1e-3 despite the blobs being trivially separable. Loss
        # is monotonically decreasing throughout (verified by direct script:
        # nll goes 1.39 -> 0.24 from epoch 1 to 1000), so the training loop
        # itself is correct; it is just this many steps slow. 1000 epochs
        # reaches held-out test accuracy 0.944 with this fixture and seed
        # (fully deterministic -- both the fixture rng and --seed are fixed),
        # comfortably above the 0.9 bar.
        "--epochs", "1000",
        "--n-dumps", "5",
        "--seed", "137",
        "--no-hf-cache",
    ])
    probe.main(args)

    records = list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", "smoke_0"))
    assert len(records) >= 2, "expected several dump points"
    assert records[0].meta["arm"] == "adamw_ce"
    assert records[0].meta["logit_kind"] == "softmax_logits"
    assert records[0].logits.shape[1] == 2
    # Both val and test are dumped from the same forward pass at every
    # scheduled point (task constraint 2) -- they must not collapse onto one
    # another. This caught a real bug: evaldump.write_dump's filename used to
    # be keyed on step alone, so writing val then test at the same step
    # silently clobbered the val dump.
    assert any(r.meta["split"] == "val" for r in records), "val dump must survive alongside test"
    assert any(r.meta["split"] == "test" for r in records), "test dump must survive alongside val"

    from bllarse import metrics
    test_records = [r for r in records if r.meta["split"] == "test"]
    final = test_records[-1]
    assert metrics.accuracy(final.logits, final.labels) > 0.9


def test_probe_dumps_ood_scores_when_an_ood_dataset_is_given(fake_features, tmp_path):
    import probe

    rng = np.random.default_rng(1)
    for split in ("train", "test"):
        n = 150
        np.savez(fake_features / f"dinov3_small_fakeood_res224_{split}.npz",
                 features=rng.normal(loc=20.0, size=(n, 16)).astype(np.float32),
                 labels=rng.integers(0, 2, size=n))

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--ood-dataset", "fakeood",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "smoke_ood",
        "--epochs", "4",
        "--n-dumps", "3",
        "--seed", "137",
        "--no-hf-cache",
    ])
    probe.main(args)

    dumps = list(evaldump.iter_dumps(tmp_path / "dumps", "smoke", "smoke_ood"))
    ood_steps = sorted(r.meta["step"] for r in dumps if r.meta["split"] == "ood")
    test_steps = sorted(r.meta["step"] for r in dumps if r.meta["split"] == "test")

    # This asserted a single endpoint dump. That forced Fig 2's detection
    # metrics to be read at the last step -- the only one with an OOD dump --
    # even though these runs have no early stopping and the arms drift from
    # their tuned operating point by very different amounts (AdamW+CE on
    # cifar100 nearly triples its NLL after step 1,786 of 30,000; VBLL moves by
    # 0.0003). One OOD dump per eval step lets detection be scored at the same
    # step as every other metric.
    assert ood_steps == test_steps, (
        "OOD must be dumped at exactly the eval steps the test split is, so the "
        "two can always be paired without comparing different checkpoints"
    )
    assert len(ood_steps) == 3


def test_batch_size_larger_than_dataset_raises_instead_of_training_zero_steps(
    fake_features, tmp_path
):
    """Cross-cutting fix: _ce_epoch computed n_batches = len(feats) //
    batch_size and looped that many times -- with batch_size > len(feats),
    n_batches is 0 and the loop silently trains nothing. Every production
    sweep hard-codes batch_size=512; a dataset-size change alone would then
    dump an untrained head whose metrics look like a real (if poor) result.
    This must fail loudly, not clamp to one batch."""
    import probe

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce",
        "--model", "dinov3_small",
        "--dataset", "fake",
        "--num-classes", "2",
        "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "smoke",
        "--uid", "smoke_toobig",
        "--epochs", "2",
        "--n-dumps", "2",
        "--batch-size", "601",  # fake_features has 600 training rows
        "--seed", "137",
        "--no-hf-cache",
    ])
    with pytest.raises(ValueError, match="batch_size"):
        probe.main(args)


def _run(fake_features, tmp_path, uid, *, epochs, batch_size, n_dumps):
    import probe

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "sched", "--uid", uid,
        "--epochs", str(epochs), "--n-dumps", str(n_dumps),
        "--batch-size", str(batch_size), "--seed", "137", "--no-hf-cache",
    ])
    probe.main(args)
    return [r for r in evaldump.iter_dumps(tmp_path / "dumps", "sched", uid)
            if r.meta["split"] == "test"]


def test_dumps_are_indexed_on_optimiser_steps_not_epochs(fake_features, tmp_path):
    """The schedule used to run over epochs, so on ImageNet the first dump
    landed 1.28M samples in -- past the entire early regime the
    sample-efficiency claim is about. Steps put the first dump at one batch."""
    # 600 rows / 100 = 6 steps per epoch, 10 epochs = 60 steps.
    records = _run(fake_features, tmp_path, "steps", epochs=10, batch_size=100, n_dumps=8)
    steps = sorted(r.meta["step"] for r in records)

    assert steps[0] == 1, "first dump must be after a single optimiser step"
    assert steps[-1] == 60, "last dump must be the final step, not the final epoch"
    # If this were still epoch-indexed every step would be a multiple of 6.
    assert any(step % 6 != 0 for step in steps), f"no mid-epoch dump in {steps}"


def test_samples_seen_is_exact(fake_features, tmp_path):
    """Drop-remainder means every step is exactly batch_size samples. The old
    epoch * len(train_f) overcounted by the dropped tail -- 2.6% on
    oxford_pets -- which biased the compute axis the figures are plotted on."""
    records = _run(fake_features, tmp_path, "exact", epochs=10, batch_size=100, n_dumps=8)
    for record in records:
        assert record.meta["samples_seen"] == record.meta["step"] * 100


def test_the_samples_axis_is_comparable_across_batch_sizes(fake_features, tmp_path):
    """The reason the schedule moved to samples: two runs with different batch
    sizes must be placeable on one x-axis. Same sample budget, different batch
    size -> same final samples_seen, different step counts."""
    coarse = _run(fake_features, tmp_path, "bs100", epochs=10, batch_size=100, n_dumps=8)
    fine = _run(fake_features, tmp_path, "bs50", epochs=10, batch_size=50, n_dumps=8)

    coarse_final = max(r.meta["samples_seen"] for r in coarse)
    fine_final = max(r.meta["samples_seen"] for r in fine)
    assert coarse_final == 6000   # 10 epochs x 6 steps x 100
    assert fine_final == 6000     # 10 epochs x 12 steps x 50
    assert coarse_final == fine_final

    # Same sample budget, genuinely different step grids -- so `step` is NOT
    # comparable across the two and `samples_seen` is, which is the point.
    assert max(r.meta["step"] for r in coarse) == 60
    assert max(r.meta["step"] for r in fine) == 120


def test_ce_step_is_jitted_and_matches_the_eager_computation():
    """The AdamW step must be compiled, and compiling it must not change the maths.

    Measured on an A100 at the Fig-1 working shapes (D=1280, C=101, batch=512),
    the un-jitted step cost 58.5 ms against 2.2 ms jitted -- 26x, which was
    343 s of the 460 s a food101 run took. probe.py was the only JAX training
    script in the repo with no jit at all; every sibling uses eqx.filter_jit.

    The equivalence half of this test is the point: a speedup that perturbs the
    update is not a speedup, it is a silent change to every published number.
    """
    import equinox as eqx
    import jax.numpy as jnp
    import optax
    from jax import random as jr, vmap

    import probe
    from bllarse.losses import CrossEntropy

    D, C, BS = 16, 3, 8
    key = jr.PRNGKey(0)
    key, hk, fk = jr.split(key, 3)
    head = eqx.nn.Linear(D, C, key=hk)
    loss_fn = CrossEntropy(0.0, C)
    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
    fb = jr.normal(fk, (BS, D))
    yb = jr.randint(fk, (BS,), 0, C)

    # The step is jitted: equinox exposes .lower() only on wrapped callables.
    assert hasattr(probe._ce_step, "lower"), "_ce_step is not wrapped in eqx.filter_jit"

    def eager(head, opt_state):
        def loss(h):
            return loss_fn(vmap(h)(fb), yb).mean()
        grads = eqx.filter_grad(loss)(head)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(head, eqx.is_array)
        )
        return eqx.apply_updates(head, updates), opt_state

    want_head, _ = eager(head, opt_state)
    got_head, _ = probe._ce_step(head, opt_state, optimizer, loss_fn, fb, yb)

    assert jnp.allclose(got_head.weight, want_head.weight, atol=1e-6)
    assert jnp.allclose(got_head.bias, want_head.bias, atol=1e-6)


def test_head_params_counts_every_array_leaf():
    import equinox as eqx
    from jax import random as jr

    import probe

    head = eqx.nn.Linear(16, 3, key=jr.PRNGKey(0))
    # 16*3 weights + 3 biases
    assert probe.head_param_count(head) == 51


def test_head_params_is_logged_as_a_run_param(fake_features, tmp_path, monkeypatch):
    """The Pareto front plots calibration against head size. Recomputing that
    size in the plotting script would be a second implementation of the exact
    quantity being compared, and it would silently disagree the moment a head
    gains a parameter."""
    import contextlib

    import probe

    logged = {}

    class Recorder:
        active = False

        def log_params(self, params):
            logged.update(params)

        def log_metrics(self, metrics, step=None):
            pass

        def set_tags(self, tags):
            pass

    @contextlib.contextmanager
    def fake_child_run(config, *, run_name, tags=None):
        yield Recorder()

    monkeypatch.setattr(probe, "child_run", fake_child_run)

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "hp", "--uid", "hp_0",
        "--epochs", "2", "--n-dumps", "2", "--seed", "137", "--no-hf-cache",
    ])
    probe.main(args)

    # 16 feature dims x 2 classes + 2 biases
    assert logged["head_param_count"] == 34


def test_on_dump_is_called_once_per_dump_with_val_nll(fake_features, tmp_path):
    import probe

    seen = []
    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_0",
        "--epochs", "10", "--n-dumps", "5", "--batch-size", "100",
        "--seed", "137", "--no-hf-cache",
    ])
    probe.main(args, on_dump=lambda i, step, scalars: seen.append((i, step, scalars)))

    assert len(seen) == 5, "one callback per dumped step"
    assert [i for i, _, _ in seen] == [0, 1, 2, 3, 4], (
        "dump_index must count dumps, not optimiser steps -- it is the pruner's "
        "step key, and trials vary batch_size so raw steps are not comparable"
    )
    assert all("val_nll" in scalars for _, _, scalars in seen)


def test_an_exception_from_on_dump_propagates(fake_features, tmp_path):
    """optuna.TrialPruned is raised from inside the callback and must escape
    the training loop. Swallowing it would run every pruned trial to
    completion -- the pruner would cost time instead of saving it."""
    import probe

    class Sentinel(Exception):
        pass

    def boom(i, step, scalars):
        raise Sentinel()

    args = probe.build_argparser().parse_args([
        "--arm", "adamw_ce", "--model", "dinov3_small", "--dataset", "fake",
        "--num-classes", "2", "--cache-dir", str(fake_features),
        "--split-cache-dir", str(tmp_path / "splits"),
        "--dump-root", str(tmp_path / "dumps"),
        "--group-id", "cb", "--uid", "cb_boom",
        "--epochs", "4", "--n-dumps", "3", "--batch-size", "100",
        "--seed", "137", "--no-hf-cache",
    ])
    with pytest.raises(Sentinel):
        probe.main(args, on_dump=boom)
