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


def test_run_curve_rejects_a_logit_kind_it_cannot_metricise(tmp_path):
    """run_curve hands dumped arrays straight to classification_metrics, which
    softmaxes them -- correct for softmax_logits, silently wrong for anything
    else. A future arm (e.g. IBProbit's 'moderated_score', q-update-spec.md
    5.8) must be rejected, not mis-metricised."""
    evaldump.write_dump(
        tmp_path, "g", "u8", 1, np.zeros((10, 3), dtype=np.float32), np.zeros(10, dtype=int),
        arm="ibprobit", split="test", logit_kind="moderated_score",
        seed=137, samples_seen=100,
    )
    with pytest.raises(ValueError, match="moderated_score"):
        curves.run_curve(tmp_path, "g", "u8")


def test_run_curve_emits_val_metrics_for_post_hoc_stopping(tmp_path):
    """early_stop takes an argmin over val_nll, so the curve has to carry it."""
    _write_run(tmp_path, "r", [1, 2, 4])
    rows = curves.run_curve(tmp_path, "g", "r")
    for row in rows:
        for key in ("val_acc", "val_ece", "val_nll", "val_brier"):
            assert key in row, key
        # Distinct splits, so the two sides must not be the same numbers.
        assert row["val_nll"] != row["nll"]


def test_val_metrics_are_scaled_by_the_same_temperature_as_test(tmp_path):
    """A val curve left unscaled while test is scaled would make the stopping
    criterion and the reported metric disagree about what run they describe."""
    _write_run(tmp_path, "r", [1, 2, 4])
    raw = curves.run_curve(tmp_path, "g", "r", temperature_scaled=False)
    ts = curves.run_curve(tmp_path, "g", "r", temperature_scaled=True)
    for raw_row, ts_row in zip(raw, ts):
        assert ts_row["temperature"] != pytest.approx(1.0)
        # T is fit on val, so val NLL under that T is the minimum attainable.
        assert ts_row["val_nll"] <= raw_row["val_nll"] + 1e-6


def test_run_curve_omits_val_metrics_when_no_val_dump_exists(tmp_path):
    """Rather than inventing them -- early_stop then raises, which is the
    intended loud failure."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=200)
    evaldump.write_dump(
        tmp_path, "g", "testonly", 1, logits, labels,
        arm="adamw_ce", split="test", logit_kind="softmax_logits",
        seed=137, samples_seen=512,
    )
    row = curves.run_curve(tmp_path, "g", "testonly")[0]
    assert "val_nll" not in row


def _write_pair_with_score(root, uid, *, id_score, ood_score, score_kind="predictive_variance",
                           step=1, k=5):
    """A VBLL-shaped run: log-predictive dumps carrying a native epistemic score."""
    rng = np.random.default_rng(7)
    for split, score in (("test", id_score), ("ood", ood_score)):
        n = len(score)
        base = rng.normal(size=(n, k)).astype(np.float32)
        evaldump.write_dump(
            root, "g", uid, step, base, base.argmax(axis=-1),
            arm="vbll", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=100,
            score=np.asarray(score, dtype=np.float32), score_kind=score_kind,
        )


def test_run_native_detection_uses_the_dumped_score_not_the_logits(tmp_path):
    """The logits here are pure noise and carry no ID/OOD signal at all, so a
    logit-derived score would score ~0.5. Only the dumped predictive variance
    separates the two sets -- which is what makes this a real test that the
    native array is being read rather than silently ignored."""
    _write_pair_with_score(
        tmp_path, "n1",
        id_score=np.linspace(0.0, 1.0, 300),
        ood_score=np.linspace(5.0, 6.0, 300),   # uniformly higher variance
    )
    out = curves.run_native_detection(tmp_path, "g", "n1")
    assert set(out) == {"auroc", "fpr95", "aupr"}
    assert out["auroc"] == pytest.approx(1.0), "a fully separated score must give AUROC 1"

    # ...and the shared-score path over the SAME dumps sees nothing, confirming
    # the separation came from the score array rather than the logits.
    shared = curves.run_detection(tmp_path, "g", "n1", metrics.msp_ood_score)
    assert abs(shared["auroc"] - 0.5) < 0.1


def test_run_native_detection_raises_when_the_arm_has_no_native_score(tmp_path):
    """AdamW+CE has no epistemic quantity by construction, so its dumps carry no
    score. That must be a clear error rather than a crash deep in numpy."""
    rng = np.random.default_rng(8)
    for split in ("test", "ood"):
        base = rng.normal(size=(200, 5)).astype(np.float32)
        evaldump.write_dump(
            tmp_path, "g", "n2", 1, base, base.argmax(axis=-1),
            arm="adamw_ce", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=100,
        )
    with pytest.raises(ValueError, match="no native score"):
        curves.run_native_detection(tmp_path, "g", "n2")


def test_run_native_detection_refuses_mismatched_score_kinds(tmp_path):
    rng = np.random.default_rng(9)
    for split, kind in (("test", "predictive_variance"), ("ood", "epistemic_term")):
        base = rng.normal(size=(100, 5)).astype(np.float32)
        evaldump.write_dump(
            tmp_path, "g", "n3", 1, base, base.argmax(axis=-1),
            arm="vbll", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=100,
            score=rng.random(100).astype(np.float32), score_kind=kind,
        )
    with pytest.raises(ValueError, match="score_kind differs"):
        curves.run_native_detection(tmp_path, "g", "n3")


def test_run_native_detection_keeps_the_checkpoint_guard(tmp_path):
    """Both detection paths share _final_id_ood_pair, so the guard against
    pairing two different checkpoints must hold for the native path too."""
    rng = np.random.default_rng(10)
    for split, step in (("test", 20), ("ood", 5)):
        base = rng.normal(size=(100, 5)).astype(np.float32)
        evaldump.write_dump(
            tmp_path, "g", "n4", step, base, base.argmax(axis=-1),
            arm="vbll", split=split, logit_kind="softmax_logits",
            seed=137, samples_seen=step * 100,
            score=rng.random(100).astype(np.float32), score_kind="predictive_variance",
        )
    with pytest.raises(ValueError, match="different checkpoints"):
        curves.run_native_detection(tmp_path, "g", "n4")
