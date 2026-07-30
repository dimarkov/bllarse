import jax.numpy as jnp
import numpy as np
import pytest

from bllarse import metrics


def test_accuracy_perfect_and_zero():
    logits = jnp.array([[5.0, 0.0], [0.0, 5.0]])
    assert metrics.accuracy(logits, jnp.array([0, 1])) == pytest.approx(1.0)
    assert metrics.accuracy(logits, jnp.array([1, 0])) == pytest.approx(0.0)


def test_brier_is_full_vector_not_top_label():
    # Uniform predictive over 4 classes: each p_k = 0.25.
    # Full-vector Brier = (1-0.25)^2 + 3*(0.25)^2 = 0.5625 + 0.1875 = 0.75
    logits = jnp.zeros((1, 4))
    assert metrics.brier(logits, jnp.array([0])) == pytest.approx(0.75, abs=1e-6)


def test_brier_zero_for_perfect_confident_prediction():
    logits = jnp.array([[100.0, -100.0, -100.0]])
    assert metrics.brier(logits, jnp.array([0])) == pytest.approx(0.0, abs=1e-6)


def test_nll_matches_hand_computation_for_uniform():
    # Uniform over 10 classes -> NLL = log(10)
    logits = jnp.zeros((1, 10))
    assert metrics.nll(logits, jnp.array([3])) == pytest.approx(np.log(10.0), abs=1e-6)


def _calibrated_logits_and_labels(seed, n=4000, k=5):
    """Logits that are calibrated at T=1, by SAMPLING labels from softmax(base).

    Do not use labels = base.argmax(-1) here: that makes the model correct on
    every point, so NLL falls monotonically as T -> 0 and there is no interior
    optimum for fit_temperature to find.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, k)).astype(np.float32)
    probs = np.exp(base) / np.exp(base).sum(axis=-1, keepdims=True)
    labels = np.array([rng.choice(k, p=row) for row in probs])
    return base, jnp.asarray(labels)


def test_fit_temperature_recovers_a_known_scaling():
    # base is calibrated at T=1, so base * 4 is overconfident by exactly 4x and
    # the NLL-optimal temperature is 4.
    base, labels = _calibrated_logits_and_labels(0)
    overconfident = jnp.asarray(base * 4.0)

    T = metrics.fit_temperature(overconfident, labels)
    assert 3.0 < T < 5.0, f"expected T near 4, got {T}"

    # Applying it must not change accuracy — temperature is rank-preserving.
    before = metrics.accuracy(overconfident, labels)
    after = metrics.accuracy(metrics.apply_temperature(overconfident, T), labels)
    assert before == pytest.approx(after)


def test_fit_temperature_reduces_nll():
    base, labels = _calibrated_logits_and_labels(1)
    overconfident = jnp.asarray(base * 4.0)
    T = metrics.fit_temperature(overconfident, labels)
    assert metrics.nll(metrics.apply_temperature(overconfident, T), labels) < metrics.nll(
        overconfident, labels
    )


def test_msp_ood_score_is_higher_for_uniform_than_confident():
    confident = jnp.array([[10.0, 0.0, 0.0]])
    uniform = jnp.zeros((1, 3))
    assert metrics.msp_ood_score(uniform)[0] > metrics.msp_ood_score(confident)[0]


def test_entropy_ood_score_is_higher_for_uniform_than_confident():
    confident = jnp.array([[10.0, 0.0, 0.0]])
    uniform = jnp.zeros((1, 3))
    assert metrics.entropy_ood_score(uniform)[0] > metrics.entropy_ood_score(confident)[0]


def test_entropy_ood_score_maximum_is_log_k():
    uniform = jnp.zeros((1, 8))
    assert float(metrics.entropy_ood_score(uniform)[0]) == pytest.approx(np.log(8.0), abs=1e-6)


def test_classification_metrics_returns_all_four_keys():
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(500, 10)).astype(np.float32))
    labels = jnp.asarray(rng.integers(0, 10, size=500))
    out = metrics.classification_metrics(logits, labels)
    assert set(out) == {"acc", "ece", "nll", "brier"}
    assert all(isinstance(v, float) for v in out.values())


def test_ood_scores_do_not_collapse_to_ties_on_float16_input():
    """Dumps are fp16, and jax.nn.softmax returns fp16 for fp16 input. Without the
    upcast, MSP over 5000 points collapses to a few hundred distinct values --
    and AUROC gives ties half credit, so Fig 2's detection metrics would be
    measurably wrong. Overflow is NOT the risk here: softmax subtracts the max
    internally, so raw fp16 logits never actually overflow through this path."""
    rng = np.random.default_rng(1)
    logits16 = rng.normal(size=(5000, 100)).astype(np.float16)

    scores = np.asarray(metrics.msp_ood_score(logits16))
    assert scores.dtype == np.float32
    distinct = len(np.unique(scores))
    assert distinct > 4000, f"only {distinct} distinct scores -- fp16 ties will distort AUROC"


def test_metrics_accept_float16_input():
    logits = jnp.asarray(np.full((10, 4), 60.0, dtype=np.float16))
    labels = jnp.zeros((10,), dtype=jnp.int32)
    assert np.isfinite(metrics.nll(logits, labels))
    assert np.isfinite(metrics.brier(logits, labels))
