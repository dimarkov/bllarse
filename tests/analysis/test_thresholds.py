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
