import numpy as np
import pytest

from bllarse.analysis import thresholds


def _curve(rows):
    return [dict(step=s, samples_seen=s * 100, acc=a, nll=n, brier=b) for s, a, n, b in rows]


def test_targets_are_a_fraction_of_converged_baseline():
    baseline = _curve([(1, 0.10, 3.0, 1.2), (10, 0.80, 0.5, 0.30)])
    t = thresholds.make_targets(baseline, fraction=0.95)
    assert t.acc == pytest.approx(0.76)      # 95% of 0.80
    assert t.nll == pytest.approx(0.5 / 0.95)  # NLL is lower-is-better
    assert t.brier == pytest.approx(0.30 / 0.95)


def test_crossing_requires_all_three_metrics_jointly():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    # Accuracy clears at step 2 but calibration does not until step 4.
    curve = _curve([
        (1, 0.10, 3.00, 1.20),
        (2, 0.80, 2.00, 0.90),   # acc only -- must NOT count
        (3, 0.81, 0.60, 0.40),   # still short on nll and brier
        (4, 0.82, 0.50, 0.30),   # all three
    ])
    result = thresholds.first_crossing(curve, targets)
    assert result.step == 4
    assert result.samples_seen == 400


def test_accuracy_early_calibration_late_does_not_score_as_fast():
    """The exact failure mode results-plan 0.4 exists to prevent."""
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    fast_but_miscalibrated = _curve([(1, 0.90, 5.0, 1.5), (2, 0.91, 4.0, 1.4)])
    result = thresholds.first_crossing(fast_but_miscalibrated, targets)
    assert result.censored is True
    assert result.step is None
    assert result.samples_seen is None


def test_non_crossing_run_is_right_censored_not_dropped():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curve = _curve([(1, 0.10, 3.0, 1.2), (10, 0.50, 1.0, 0.8)])
    result = thresholds.first_crossing(curve, targets)
    assert result.censored is True
    assert result.step is None
    assert result.samples_seen is None
    assert result.last_step == 10


def test_non_monotone_curve_takes_the_first_crossing_not_the_last():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curve = _curve([
        (1, 0.80, 0.50, 0.30),   # crosses here
        (2, 0.50, 2.00, 1.00),   # regresses
        (3, 0.85, 0.40, 0.25),   # crosses again
    ])
    assert thresholds.first_crossing(curve, targets).step == 1


def test_crossings_are_reported_per_seed_not_from_the_mean_curve():
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curves = {
        0: _curve([(1, 0.80, 0.50, 0.30)]),
        1: _curve([(1, 0.10, 3.00, 1.20), (2, 0.80, 0.50, 0.30)]),
    }
    out = thresholds.crossings_per_seed(curves, targets)
    assert out[0].step == 1
    assert out[1].step == 2


def test_make_targets_rejects_an_empty_curve():
    with pytest.raises(ValueError):
        thresholds.make_targets([], fraction=0.95)


def test_crossing_carries_samples_seen_not_just_step():
    """Cross-cutting fix: Crossing used to carry only `step`, an epoch --
    samples_seen is the compute axis the sample-efficiency figure actually
    needs, and first_crossing must populate it from the crossing row."""
    targets = thresholds.Targets(acc=0.76, nll=0.53, brier=0.32)
    curve = _curve([(1, 0.10, 3.00, 1.20), (4, 0.82, 0.50, 0.30)])
    result = thresholds.first_crossing(curve, targets)
    assert result.step == 4
    assert result.samples_seen == 400  # samples_seen = step * 100, per _curve


def _val_curve(rows):
    """(step, val_nll) -> rows carrying the keys early_stop needs."""
    return [dict(step=s, samples_seen=s * 100, val_nll=v) for s, v in rows]


def test_early_stop_picks_the_val_nll_minimum_not_the_endpoint():
    """The whole point: an overfitting run's best checkpoint is interior, and
    reporting the endpoint would understate it."""
    curve = _val_curve([(1, 2.0), (10, 0.8), (100, 0.5), (1000, 0.9), (2000, 1.4)])
    sp = thresholds.early_stop(curve)
    assert sp.step == 100
    assert sp.samples_seen == 10_000
    assert sp.value == pytest.approx(0.5)
    assert sp.at_last_step is False


def test_early_stop_flags_a_minimum_at_the_last_dumped_point():
    """Monotone-decreasing val NLL means the budget ran out before the optimum.
    Measured for real: a dinov3_small probe on oxford_pets was still improving
    at 300 epochs. Reporting that as 'the early-stopping point' without the flag
    would misrepresent a truncation as a converged choice."""
    curve = _val_curve([(1, 3.0), (10, 2.0), (100, 1.0), (1000, 0.4)])
    sp = thresholds.early_stop(curve)
    assert sp.step == 1000
    assert sp.at_last_step is True


def test_early_stop_selects_on_val_but_the_row_carries_test_metrics():
    curve = [
        dict(step=1, samples_seen=100, val_nll=1.0, acc=0.50, nll=1.1),
        dict(step=2, samples_seen=200, val_nll=0.4, acc=0.80, nll=0.5),
        dict(step=3, samples_seen=300, val_nll=0.7, acc=0.79, nll=0.8),
    ]
    sp = thresholds.early_stop(curve)
    assert sp.step == 2
    assert sp.row["acc"] == pytest.approx(0.80)


def test_early_stop_rejects_a_curve_without_the_criterion():
    """run_curve omits val_* when no val dump exists at a step. Silently
    falling back to the endpoint would be indistinguishable from a real
    selection after the fact."""
    curve = [dict(step=1, samples_seen=100, acc=0.5)]
    with pytest.raises(ValueError, match="absent from 1 of 1 rows"):
        thresholds.early_stop(curve)


def test_early_stop_accepts_an_alternative_criterion():
    curve = [
        dict(step=1, samples_seen=100, val_nll=0.4, val_brier=0.9),
        dict(step=2, samples_seen=200, val_nll=0.9, val_brier=0.2),
    ]
    assert thresholds.early_stop(curve, "val_nll").step == 1
    assert thresholds.early_stop(curve, "val_brier").step == 2


def test_early_stop_per_seed_is_per_run_not_on_the_mean_curve():
    curves = {
        137: _val_curve([(1, 2.0), (10, 0.5), (100, 0.9)]),
        138: _val_curve([(1, 2.0), (10, 1.2), (100, 0.3)]),
    }
    stops = thresholds.early_stop_per_seed(curves)
    assert stops[137].step == 10
    assert stops[138].step == 100


def test_truncation_comes_from_the_dump_not_from_the_curve_length(tmp_path):
    """A single-dump run must not report as truncated just for being single.

    Both probing scripts now keep only their argmin evaluation, so the old
    inference -- "is the argmin the last row of the curve?" -- is trivially
    True for every run. `tuning.budgets` raises the step budget until
    at_last_step is false for >=90% of runs, so left uncorrected this would
    reject every budget forever. The training loop records the answer instead,
    since it is the only thing that knows how many evaluations there were.
    """
    from bllarse import evaldump
    from bllarse.analysis.curves import run_curve
    from bllarse.analysis.thresholds import early_stop

    labels = np.zeros(64, dtype=np.int64)
    logits = np.zeros((64, 2), dtype=np.float32)
    logits[:, 0] = 4.0

    for uid, at_last in (("converged", False), ("truncated", True)):
        for split in ("val", "test"):
            evaldump.write_dump(
                tmp_path, "g", uid, 512, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=65536, at_last_eval=at_last,
            )
        stop = early_stop(run_curve(tmp_path, "g", uid), "val_nll")
        assert stop.at_last_step is at_last, (
            f"{uid}: at_last_step must come from the dump's at_last_eval, not "
            f"from the curve having one row"
        )


def test_truncation_falls_back_to_the_curve_when_unrecorded(tmp_path):
    """Dumps written before at_last_eval existed carry None, and multi-dump
    curves are still read the old way -- so trajectory runs and archived dumps
    keep working unchanged."""
    from bllarse import evaldump
    from bllarse.analysis.curves import run_curve
    from bllarse.analysis.thresholds import early_stop

    labels = np.zeros(64, dtype=np.int64)
    for step, sharpness in ((1, 1.0), (2, 5.0), (3, 2.0)):
        logits = np.zeros((64, 2), dtype=np.float32)
        logits[:, 0] = sharpness
        for split in ("val", "test"):
            evaldump.write_dump(
                tmp_path, "g", "old", step, logits, labels,
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=200, samples_seen=step * 128,
            )
    stop = early_stop(run_curve(tmp_path, "g", "old"), "val_nll")
    assert stop.step == 2
    assert stop.at_last_step is False, "argmin at step 2 of 3 is not truncation"


def test_a_nonfinite_row_cannot_win_the_argmin():
    """`min` keeps its first candidate unless a later one compares strictly
    less, and every comparison against nan is False -- so a nan row that sorts
    first WINS the argmin and the run scores nan. Two real ibprobit trials
    failed this way. A point the model could not evaluate is not a candidate
    optimum."""
    curve = [
        {"step": 1, "samples_seen": 100, "val_nll": float("nan")},
        {"step": 2, "samples_seen": 200, "val_nll": 0.9},
        {"step": 3, "samples_seen": 300, "val_nll": 0.4},
        {"step": 4, "samples_seen": 400, "val_nll": 0.7},
    ]
    sp = thresholds.early_stop(curve, "val_nll")
    assert sp.step == 3
    assert sp.value == pytest.approx(0.4)


def test_an_all_nonfinite_curve_raises_rather_than_scoring():
    """A run that produced no usable measurement must not be handed back a
    number -- silently scoring it would put a fabricated value into the
    campaign's comparison."""
    curve = [
        {"step": 1, "samples_seen": 100, "val_nll": float("nan")},
        {"step": 2, "samples_seen": 200, "val_nll": float("inf")},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        thresholds.early_stop(curve, "val_nll")
