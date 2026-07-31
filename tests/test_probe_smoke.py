import sys
from pathlib import Path

import numpy as np
import pytest

ALT = Path(__file__).resolve().parents[1] / "scripts" / "alternatives"
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

    ood = [r for r in evaldump.iter_dumps(tmp_path / "dumps", "smoke", "smoke_ood")
           if r.meta["split"] == "ood"]
    assert len(ood) == 1, "OOD is dumped at the final point only"


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
