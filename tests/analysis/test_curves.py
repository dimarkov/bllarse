import numpy as np
import pytest

from bllarse import evaldump, metrics
from bllarse.analysis import curves


def _write_run(root, uid, steps, *, n=2000, k=5):
    """Dumps whose logits grow steadily more overconfident as training proceeds.

    Labels are SAMPLED from softmax(base), which makes `base` calibrated at T=1,
    so scaling it by `s` puts the NLL-optimal temperature at `s`. The scale rises
    with the step, so each dump point has a genuinely different optimal T and a
    per-step refit is distinguishable from a single fit reused.

    Do NOT use labels = base.argmax(-1) here. That makes the model correct on
    every point, NLL then falls monotonically as T -> 0, and fit_temperature pins
    to its clip bound (T = 0.01) at every step -- so all the fitted temperatures
    come out identical and the refit assertion cannot fail.
    """
    for step in steps:
        rng = np.random.default_rng(step)
        base = rng.normal(size=(n, k)).astype(np.float32)
        probs = np.exp(base) / np.exp(base).sum(axis=-1, keepdims=True)
        labels = np.array([rng.choice(k, p=row) for row in probs])
        logits = base * (1.0 + step / 2.0)
        for split, sl in (("val", slice(0, n // 2)), ("test", slice(n // 2, n))):
            evaldump.write_dump(
                root, "g", uid, step, logits[sl], labels[sl],
                arm="adamw_ce", split=split, logit_kind="softmax_logits",
                seed=137, samples_seen=step * 100,
            )


def test_run_curve_has_one_row_per_step_with_all_metrics(tmp_path):
    _write_run(tmp_path, "u1", [1, 5, 20])
    curve = curves.run_curve(tmp_path, "g", "u1")
    assert [row["step"] for row in curve] == [1, 5, 20]
    assert set(curve[0]) >= {"step", "samples_seen", "seed", "arm", "acc", "ece", "nll", "brier"}
    assert curve[0]["samples_seen"] == 100


def test_run_curve_carries_seed_and_arm_from_the_dump_meta(tmp_path):
    """Cross-cutting fix: crossings_per_seed takes Mapping[int, Curve] keyed
    by seed, but a caller had no way to get at a row's seed short of
    re-parsing the uid string, since run_curve dropped meta["seed"] and
    meta["arm"]. Both must now come straight off each row."""
    _write_run(tmp_path, "u1b", [1, 5])
    curve = curves.run_curve(tmp_path, "g", "u1b")
    assert all(row["seed"] == 137 for row in curve)
    assert all(row["arm"] == "adamw_ce" for row in curve)


def test_temperature_scaling_is_refit_at_every_step(tmp_path):
    _write_run(tmp_path, "u2", [1, 5, 20])
    plain = curves.run_curve(tmp_path, "g", "u2", temperature_scaled=False)
    scaled = curves.run_curve(tmp_path, "g", "u2", temperature_scaled=True)
    # A per-point refit must reduce NLL at every point, not just the last.
    for a, b in zip(plain, scaled):
        assert b["nll"] <= a["nll"] + 1e-6
        assert b["temperature"] > 0
    # And the fitted temperatures must not all be identical -- that would mean
    # one fit was reused rather than refit.
    assert len({round(row["temperature"], 6) for row in scaled}) > 1


def test_temperature_scaling_does_not_change_accuracy(tmp_path):
    _write_run(tmp_path, "u3", [1, 5])
    plain = curves.run_curve(tmp_path, "g", "u3", temperature_scaled=False)
    scaled = curves.run_curve(tmp_path, "g", "u3", temperature_scaled=True)
    for a, b in zip(plain, scaled):
        assert a["acc"] == pytest.approx(b["acc"])


def test_curve_feeds_the_threshold_protocol_directly(tmp_path):
    from bllarse.analysis import thresholds

    _write_run(tmp_path, "u4", [1, 5, 20])
    curve = curves.run_curve(tmp_path, "g", "u4")
    targets = thresholds.make_targets(curve, fraction=0.95)
    assert thresholds.first_crossing(curve, targets).step is not None


def test_run_detection_computes_from_id_and_ood_dumps(tmp_path):
    rng = np.random.default_rng(3)
    n, k = 400, 5
    for split, loc in (("test", 6.0), ("ood", 0.0)):
        base = rng.normal(loc=0.0, size=(n, k)).astype(np.float32)
        base[:, 0] += loc  # ID is confident on class 0; OOD is not
        evaldump.write_dump(
            tmp_path, "g", "u5", 1, base, base.argmax(axis=-1),
            arm="adamw_ce", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=100,
        )
    out = curves.run_detection(tmp_path, "g", "u5", metrics.msp_ood_score)
    assert set(out) == {"auroc", "fpr95", "aupr"}
    assert out["auroc"] > 0.8, "confident ID vs diffuse OOD must be separable"


def test_run_detection_refuses_to_compare_different_checkpoints(tmp_path):
    """The ood dump is written once, at the final step. If it lands at a
    different step from the final test dump, the scores come from different
    checkpoints and any AUROC computed from them is meaningless."""
    rng = np.random.default_rng(4)
    for split, step in (("test", 20), ("ood", 5)):
        base = rng.normal(size=(200, 5)).astype(np.float32)
        evaldump.write_dump(
            tmp_path, "g", "u7", step, base, base.argmax(axis=-1),
            arm="adamw_ce", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=step * 100,
        )
    with pytest.raises(ValueError, match="different checkpoints"):
        curves.run_detection(tmp_path, "g", "u7", metrics.msp_ood_score)


def test_run_detection_raises_without_an_ood_dump(tmp_path):
    _write_run(tmp_path, "u6", [1])
    with pytest.raises(ValueError, match="ood"):
        curves.run_detection(tmp_path, "g", "u6", metrics.msp_ood_score)
