import numpy as np
import pytest

from bllarse.analysis import thresholds


def _curve(rows, val_nll=None):
    curve = [dict(step=s, samples_seen=s * 100, acc=a, nll=n, brier=b)
             for s, a, n, b in rows]
    # val_nll defaults to tracking test nll, so a curve written without it
    # early-stops at the same row it would have converged at.
    for row, v in zip(curve, val_nll if val_nll is not None
                      else [r["nll"] for r in curve]):
        row["val_nll"] = v
    return curve


def test_targets_are_a_fraction_of_the_baselines_early_stopped_value():
    baseline = _curve([(1, 0.10, 3.0, 1.2), (10, 0.80, 0.5, 0.30)])
    t = thresholds.make_targets(baseline, fraction=0.95)
    assert t.acc == pytest.approx(0.76)      # 95% of 0.80
    assert t.nll == pytest.approx(0.5 / 0.95)  # NLL is lower-is-better
    assert t.brier == pytest.approx(0.30 / 0.95)


def test_targets_come_from_the_val_argmin_not_the_last_row():
    # The baseline peaks at step 10 and then DEGRADES, which is what
    # dinov3_large fine-tuning does past its tuning budget. val_nll turns with
    # test nll, so the reference is step 10 and not the endpoint -- otherwise
    # 95% of a decayed endpoint spends the tolerance on the decay.
    baseline = _curve([
        (1, 0.10, 3.00, 1.20),
        (10, 0.80, 0.50, 0.30),   # the val-NLL argmin
        (20, 0.78, 0.62, 0.35),   # degraded endpoint
    ])

    early = thresholds.make_targets(baseline, fraction=0.95)
    last = thresholds.make_targets(baseline, fraction=0.95, reference="last")

    assert early.nll == pytest.approx(0.50 / 0.95)
    assert last.nll == pytest.approx(0.62 / 0.95)
    # The endpoint reference is the LOOSER of the two on every metric.
    assert last.nll > early.nll
    assert last.brier > early.brier
    assert last.acc < early.acc


def test_make_targets_rejects_an_unknown_reference():
    with pytest.raises(ValueError, match="reference"):
        thresholds.make_targets(_curve([(1, 0.8, 0.5, 0.3)]), reference="best")


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
    """The failure mode the joint threshold exists to prevent: accuracy reached
    early with bad calibration must not count as a crossing."""
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
    """samples_seen, not step, is the compute axis the sample-efficiency
    figure actually needs, and first_crossing must populate it from the
    crossing row."""
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

    Both probing scripts keep only their argmin evaluation, so inferring
    truncation from the curve -- "is the argmin the last row?" -- is trivially
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
    """Dumps written without at_last_eval carry None, and early_stop then
    infers truncation from the curve -- which is sound on a multi-dump
    trajectory."""
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
    first WINS the argmin and the run scores nan. A point the model could not
    evaluate is not a candidate optimum."""
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
    comparison."""
    curve = [
        {"step": 1, "samples_seen": 100, "val_nll": float("nan")},
        {"step": 2, "samples_seen": 200, "val_nll": float("inf")},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        thresholds.early_stop(curve, "val_nll")


def test_every_stage4_sweep_has_a_figure_group_that_reads_it():
    """A sweep whose GROUP_ID no script knows about burns its GPU time
    and then contributes nothing -- silently, because both figure scripts skip
    a missing group directory (`os.path.isdir`) rather than raising.
    """
    import importlib, sys
    sys.path.insert(0, "bllarse_sweeps")
    sys.path.insert(0, "scripts/analysis")

    consumed = set()
    for fig in ("fig1_curves", "fig2_detection"):
        consumed |= set(importlib.import_module(fig).GROUPS.values())

    sweeps = [
        "probe_adamw_ce", "probe_vbll", "probe_ibprobit_noeps",
        "probe_ood_adamw_ce", "probe_ood_vbll", "probe_ood_ibprobit_noeps",
    ]
    for sweep in sweeps:
        gid = importlib.import_module(sweep).GROUP_ID
        assert gid in consumed, (
            f"{sweep}.GROUP_ID={gid!r} is read by no figure and no analysis -- "
            f"its runs would be written and then never looked at"
        )


def test_single_metric_crossing_ignores_the_other_metrics():
    # Accuracy is still terrible at step 2, but the panel that uses this gates
    # on val_nll alone and REPORTS accuracy rather than requiring it.
    curve = [dict(step=s, samples_seen=s * 100, acc=a, val_nll=v)
             for s, a, v in [(1, 0.10, 3.0), (2, 0.20, 0.45), (3, 0.80, 0.40)]]

    c = thresholds.first_crossing_metric(curve, "val_nll", 0.50)

    assert c.step == 2
    assert c.samples_seen == 200
    assert not c.censored


def test_single_metric_crossing_censors_a_run_that_never_reaches_it():
    curve = [dict(step=s, samples_seen=s * 100, val_nll=v)
             for s, v in [(1, 3.0), (2, 2.0)]]

    c = thresholds.first_crossing_metric(curve, "val_nll", 0.5)

    assert c.censored
    assert c.step is None
    assert c.last_step == 2


def test_single_metric_crossing_handles_higher_is_better():
    curve = [dict(step=s, samples_seen=s * 100, acc=a)
             for s, a in [(1, 0.10), (2, 0.85)]]

    c = thresholds.first_crossing_metric(curve, "acc", 0.80,
                                         lower_is_better=False)

    assert c.step == 2
