import jax.numpy as jnp
import numpy as np
import pytest
from jax import nn

from bllarse import detection, metrics


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


def test_a_converged_temperature_is_not_flagged_as_clipped():
    base, labels = _calibrated_logits_and_labels(2)
    T = metrics.fit_temperature(jnp.asarray(base * 4.0), labels)
    assert not metrics.is_clipped_temperature(T)


@pytest.mark.parametrize("bound", metrics.BETA_BOUNDS)
def test_a_boundary_temperature_is_flagged(bound):
    """Both clip boundaries must be recognised, expressed as T = 1/beta."""
    assert metrics.is_clipped_temperature(1.0 / bound)


def test_a_separable_problem_clips_and_says_so():
    """Labels taken as argmax make the problem perfectly separable, so NLL
    falls monotonically as T -> 0 and Newton runs beta to its upper clip. The
    returned T is then a boundary, not an optimum, and every temperature-scaled
    metric computed from it rests on a fit that did not converge -- which is
    the case the flag exists to surface."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2000, 5)).astype(np.float32) * 8.0
    labels = jnp.asarray(np.asarray(base).argmax(-1))

    T = metrics.fit_temperature(jnp.asarray(base), labels)
    assert metrics.is_clipped_temperature(T), f"expected a clipped fit, got T={T}"
    assert T == pytest.approx(1.0 / metrics.BETA_BOUNDS[1])


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


def test_magnitude_scores_separate_a_pure_shift_that_msp_cannot_see():
    """The reason max-logit and energy exist at all.

    OOD here is the ID logit vector shifted by a constant. softmax is
    shift-invariant, so the two sets have IDENTICAL predictive distributions and
    MSP is exactly uninformative. The level, which the softmax normalizes away,
    separates them. This is the concrete failure mode: an OOD input
    whose logits are sharply peaked but small in magnitude reads as confident
    under MSP.
    """
    rng = np.random.default_rng(11)
    id_logits = rng.normal(size=(500, 8)).astype(np.float32) * 3.0
    ood_logits = id_logits - 8.0  # same softmax, lower level

    # Identical predictive distributions: MSP cannot do better than chance.
    msp_id = np.asarray(metrics.msp_ood_score(id_logits))
    msp_ood = np.asarray(metrics.msp_ood_score(ood_logits))
    np.testing.assert_allclose(msp_id, msp_ood, atol=1e-5)
    # Not exactly 0.5: the shift is applied in float32, so the two score arrays
    # agree to 1e-5 without being bit-identical, and near-ties break arbitrarily
    # instead of each taking the half credit an exact tie would get. The claim
    # is "no better than chance", not "identical to the last bit".
    assert detection.auroc(msp_id, msp_ood) == pytest.approx(0.5, abs=0.02)

    # The shift is a constant, so it moves every point's level by the same
    # amount -- but the levels themselves are spread out, so the two sets
    # overlap and separation is strong rather than perfect. AUROC is the honest
    # assertion, and it is what Fig 2 actually reports.
    for score_fn in (metrics.energy_ood_score, metrics.max_logit_ood_score):
        auroc = detection.auroc(
            np.asarray(score_fn(id_logits)), np.asarray(score_fn(ood_logits))
        )
        assert auroc > 0.9, f"{score_fn.__name__} should see the level shift (got {auroc:.3f})"


def test_magnitude_scores_reject_an_already_normalized_array():
    """VBLL dumps log(predictive), not logits, under the same logit_kind as the
    CE arm. logsumexp is then identically 0, which makes energy constant (AUROC
    exactly 0.5) and max-logit a monotone transform of MSP. Both would return a
    figure-shaped number, so the degenerate input has to raise."""
    rng = np.random.default_rng(12)
    log_probs = np.asarray(
        nn.log_softmax(jnp.asarray(rng.normal(size=(200, 6)), dtype=jnp.float32), axis=-1)
    )
    for score_fn in (metrics.energy_ood_score, metrics.max_logit_ood_score):
        with pytest.raises(ValueError, match="unnormalized"):
            score_fn(log_probs)


def test_magnitude_scores_accept_ordinary_logits():
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(200, 6)).astype(np.float32) * 2.0
    assert np.all(np.isfinite(np.asarray(metrics.energy_ood_score(logits))))
    assert np.all(np.isfinite(np.asarray(metrics.max_logit_ood_score(logits))))
