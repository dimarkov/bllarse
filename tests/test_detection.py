import numpy as np
import pytest

from bllarse import detection


def test_auroc_perfect_separation():
    # Convention: higher score = more OOD. Perfectly separated -> 1.0
    assert detection.auroc(np.array([0.0, 0.1, 0.2]), np.array([1.0, 1.1])) == pytest.approx(1.0)


def test_auroc_inverted_separation():
    assert detection.auroc(np.array([1.0, 1.1]), np.array([0.0, 0.1])) == pytest.approx(0.0)


def test_auroc_identical_distributions_is_half():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert detection.auroc(a, b) == pytest.approx(0.5, abs=0.02)


def test_auroc_handles_ties_as_half_credit():
    # All scores equal -> no discrimination -> exactly 0.5
    assert detection.auroc(np.zeros(10), np.zeros(10)) == pytest.approx(0.5)


def test_fpr_at_tpr_perfect_separation_is_zero():
    fpr = detection.fpr_at_tpr(np.array([0.0, 0.1, 0.2]), np.array([1.0, 1.1]))
    assert fpr == pytest.approx(0.0)


def test_fpr_at_tpr_captures_required_tpr():
    # 100 OOD at 1.0, 100 ID uniform on [0, 2]. To capture 95% of OOD we need a
    # threshold at or below 1.0, which admits roughly half the ID points.
    rng = np.random.default_rng(1)
    scores_id = rng.uniform(0.0, 2.0, size=10000)
    scores_ood = np.full(1000, 1.0)
    assert detection.fpr_at_tpr(scores_id, scores_ood) == pytest.approx(0.5, abs=0.03)


def test_aupr_perfect_separation():
    assert detection.aupr(np.array([0.0, 0.1]), np.array([1.0, 1.1])) == pytest.approx(1.0)


def test_aupr_baseline_is_positive_rate():
    # With no signal, average precision tends to the positive (OOD) base rate.
    rng = np.random.default_rng(2)
    scores_id = rng.normal(size=9000)
    scores_ood = rng.normal(size=1000)
    assert detection.aupr(scores_id, scores_ood) == pytest.approx(0.1, abs=0.03)


def test_detection_metrics_returns_all_three_keys():
    out = detection.detection_metrics(np.array([0.0, 0.1]), np.array([1.0]))
    assert set(out) == {"auroc", "fpr95", "aupr"}


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        detection.auroc(np.array([]), np.array([1.0]))
