import numpy as np
import pytest

from bllarse import evaldump
from bllarse.tuning import budgets, space


def _write_curve(root, group_id, uid, *, argmin_at, n_steps):
    """A synthetic run whose val_nll is minimised at `argmin_at`."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=64)
    for step in range(1, n_steps + 1):
        # Confidence rises to a peak at argmin_at then falls back, so val_nll
        # has its minimum exactly there.
        sharpness = 4.0 - abs(step - argmin_at) * 0.1
        logits = np.zeros((64, 2), dtype=np.float32)
        logits[np.arange(64), labels] = sharpness
        for split in ("val", "test"):
            evaldump.write_dump(
                root, group_id, uid, step, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=step * 512,
            )


def test_epochs_for_steps_rounds_up():
    # food101: 75,750 // 512 = 147 steps/epoch. 4,000 steps needs 28 epochs
    # (27 would be 3,969).
    assert budgets.epochs_for_steps("food101", 4_000, 512) == 28


def test_budget_covers_the_slowest_run_with_margin(tmp_path):
    """Max over the dataset's pilot runs, not mean: the budget has to be
    adequate for the slowest learning rate in the range, which is exactly the
    config a too-short budget would unfairly score."""
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_food101_lr0.0001", argmin_at=40, n_steps=60)
    _write_curve(root, "pilot", "adamw_ce_food101_lr0.03", argmin_at=10, n_steps=60)

    out = budgets.derive_budgets(
        root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
    )
    assert out["food101"]["steps"] == 50          # 40 * 1.25, not 10 or 25
    assert out["food101"]["epochs"] == 1          # 50 steps < 147 per epoch


def test_the_ceiling_bounds_a_pathological_run(tmp_path):
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_dtd_lr0.0001", argmin_at=40, n_steps=60)

    out = budgets.derive_budgets(
        root, "pilot", batch_size=512, margin=1.25, ceiling_steps=42
    )
    assert out["dtd"]["steps"] == 42


def test_a_truncated_pilot_raises_instead_of_returning_a_lower_bound(tmp_path):
    """If the argmin is at the final dumped step the curve was still improving,
    so the budget derived from it is a lower bound of unknown looseness. That
    must fail loudly -- it is precisely the truncation the pilot exists to
    rule out."""
    root = tmp_path / "dumps"
    _write_curve(root, "pilot", "adamw_ce_dtd_lr0.03", argmin_at=60, n_steps=60)

    with pytest.raises(ValueError, match="FINAL dumped step"):
        budgets.derive_budgets(
            root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
        )


def test_cifar100_is_not_read_as_cifar10():
    assert budgets._uid_dataset("vbll_cifar100_lr0.001") == "cifar100"
    assert budgets._uid_dataset("vbll_cifar10_lr0.001") == "cifar10"


def test_the_90pct_rule_accepts_one_truncated_run_out_of_twelve(tmp_path, capsys):
    """The rule is a GLOBAL rate over ALL pilot runs, not per-dataset
    zero tolerance: one truncated run out of twelve is 91.7% non-truncated,
    which clears the default 90% threshold and must not raise."""
    root = tmp_path / "dumps"
    for i in range(11):
        _write_curve(root, "pilot", f"adamw_ce_dtd_lrA{i}", argmin_at=40, n_steps=60)
    _write_curve(root, "pilot", "adamw_ce_dtd_lrB0", argmin_at=60, n_steps=60)

    out = budgets.derive_budgets(
        root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
    )
    assert out["dtd"]["provenance"] == "derived from pilot"

    # A truncation that survives the global rate must still be surfaced, not
    # silently dropped -- it can look exactly like universal convergence
    # otherwise.
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "adamw_ce_dtd_lrB0" in captured.out


def test_the_90pct_rule_rejects_half_truncated(tmp_path):
    """Half the pilot runs truncated is 50% non-truncated, well below the 90%
    threshold, and must raise."""
    root = tmp_path / "dumps"
    for i in range(6):
        _write_curve(root, "pilot", f"adamw_ce_dtd_lrA{i}", argmin_at=40, n_steps=60)
    for i in range(6):
        _write_curve(root, "pilot", f"adamw_ce_dtd_lrB{i}", argmin_at=60, n_steps=60)

    with pytest.raises(ValueError, match="FINAL dumped step"):
        budgets.derive_budgets(
            root, "pilot", batch_size=512, margin=1.25, ceiling_steps=10_000
        )


def test_fixed_budgets_is_uniform_across_all_datasets_with_correct_epochs():
    out = budgets.fixed_budgets(30_000, batch_size=256)
    assert set(out) == set(space.DATASETS)
    for dataset, budget in out.items():
        assert budget["steps"] == 30_000
        assert budget["batch_size"] == 256
        assert budget["provenance"] == "fixed by decision"
        assert budget["epochs"] == budgets.epochs_for_steps(dataset, 30_000, 256)

    # dtd: 3760 // 256 = 14 steps/epoch, ceil(30000 / 14) = 2143.
    assert out["dtd"]["epochs"] == 2143
