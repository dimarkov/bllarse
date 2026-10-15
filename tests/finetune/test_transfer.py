"""The batch/duration hyperparameter transfer rule.

Mlodozeniec et al. 2025 (arXiv:2512.22382, Table 1), which derives batch-size
and training-duration transfer from a shared AdamW SDE limit. The rule is only
useful if its two invariants actually hold, so those are what most of these
assert -- an exponent typo would still produce plausible-looking numbers, but
would break `steps * lr * wd` or the beta horizon fraction immediately.
"""

import math

import pytest

from bllarse.finetune import protocol


def _steps(samples: int, batch: int) -> int:
    return protocol.steps_for_samples(samples, batch)


# --- betas_for ------------------------------------------------------------

def test_the_reference_point_is_the_library_defaults():
    # The whole rule is expressed relative to this point, so if it drifts every
    # transferred config silently moves with it.
    assert protocol.REFERENCE_BETAS == (0.9, 0.999)
    assert protocol.reference_steps() == 3906
    assert protocol.betas_for(protocol.reference_steps()) == protocol.REFERENCE_BETAS


def test_the_beta_horizon_is_a_constant_fraction_of_the_run():
    """The invariant the rule actually encodes: (1 - beta) * steps is fixed.

    Stated as the horizon rather than the beta because that is the quantity
    with meaning -- beta2's 1/(1-beta2) = 1000-step averaging window is 25.6%
    of the 3906-step reference run, and has to stay 25.6% of a longer one.
    """
    reference = protocol.reference_steps()
    for steps in (1000, 3906, 10_009, 40_000):
        b1, b2 = protocol.betas_for(steps)
        assert (1.0 - b1) * steps == pytest.approx(0.1 * reference)
        assert (1.0 - b2) * steps == pytest.approx(0.001 * reference)
        # ... and therefore the horizon is the same fraction of the run.
        assert (1.0 / (1.0 - b2)) / steps == pytest.approx(1000 / reference)


def test_the_four_epoch_betas():
    # The betas a 4-epoch final run gets. 4 epochs of ImageNet-1k is
    # 5,124,668 samples = 10,009 steps at batch 512, against the 3906-step
    # reference.
    steps = _steps(5_124_668, 512)
    assert steps == 10_009
    b1, b2 = protocol.betas_for(steps)
    assert b1 == pytest.approx(0.96097, abs=1e-5)
    assert b2 == pytest.approx(0.99961, abs=1e-5)


def test_a_longer_run_gets_a_longer_averaging_window():
    _, b2_short = protocol.betas_for(2000)
    _, b2_long = protocol.betas_for(20_000)
    assert b2_long > b2_short


def test_it_refuses_a_run_too_short_to_hold_the_beta1_horizon():
    """Raises rather than clamping.

    (1 - 0.9) * 3906 = 390.6, so under ~391 steps beta1 transfers to <= 0 --
    which is AdamW with no first moment at all. Silently clamping would ship a
    different optimizer with nothing in the config recording it.
    """
    protocol.betas_for(391)
    with pytest.raises(ValueError, match="beta1 transfers out of range"):
        protocol.betas_for(390)


def test_it_carries_non_default_betas_through():
    # A second transfer from an already-transferred config has to compose, or
    # the rule only works once.
    once = protocol.betas_for(10_009)
    twice = protocol.betas_for(20_018, betas=once, from_steps=10_009)
    direct = protocol.betas_for(20_018)
    assert twice == pytest.approx(direct)


# --- transfer -------------------------------------------------------------

# The reference operating point (2M samples at batch 512, i.e.
# protocol.REFERENCE_STEPS), written as a LITERAL. These tests are
# about the rule, not about whatever the current tuning budget happens to be --
# reading protocol.MAX_SAMPLES here would make them re-derive their own
# expectations and assert nothing when it moves.
TUNE_SAMPLES = 2_000_000

TUNED = {
    "batch_size": 512,
    "lr": 1.17e-5,
    "weight_decay": 8.13e-6,
    "head_lr_ratio": 31.7,
}
FOUR_EPOCH_SAMPLES = 5_124_668


def _to_final(params=None, **kwargs):
    return protocol.transfer(
        TUNED if params is None else params,
        from_samples=TUNE_SAMPLES, from_batch=512,
        to_samples=FOUR_EPOCH_SAMPLES, to_batch=512, **kwargs)


def test_the_duration_factor_is_the_inverse_root_of_the_data_ratio():
    out = _to_final()
    expected = math.sqrt(TUNE_SAMPLES / FOUR_EPOCH_SAMPLES)
    assert expected == pytest.approx(0.6247, abs=1e-4)
    assert out["lr"] == pytest.approx(TUNED["lr"] * expected)
    assert out["weight_decay"] == pytest.approx(TUNED["weight_decay"] * expected)


def test_total_decoupled_shrinkage_is_invariant():
    """steps * lr * wd, the quantity finetune_space.TRUNK_WD_BOX derives by hand.

    This is the check that catches a wrong exponent: AdamW's decay is decoupled
    so per-step shrinkage is lr * wd, and the rule is only self-consistent if
    the total over a run does not move.
    """
    out = _to_final()
    before = _steps(TUNE_SAMPLES, 512) * TUNED["lr"] * TUNED["weight_decay"]
    after = _steps(FOUR_EPOCH_SAMPLES, 512) * out["lr"] * out["weight_decay"]
    assert after == pytest.approx(before, rel=1e-3)


def test_weight_decay_moves_with_the_learning_rate_not_against_it():
    """The direction the paper's own table renders ambiguously.

    Their Eq. 1 discretises a drift of (1/(eta*sigma))(g + lambda*sigma*Theta),
    so invariance needs lambda/eta fixed -- the SAME exponent as the learning
    rate, under the PyTorch AdamW convention that both optax and torch use.
    Getting this backwards still passes a smoke test and quietly doubles the
    effective regularisation across the transfer.
    """
    out = _to_final()
    assert out["weight_decay"] / out["lr"] == pytest.approx(
        TUNED["weight_decay"] / TUNED["lr"])


def test_the_head_lr_ratio_is_invariant():
    # Trunk and head take the same factor on every axis, so the ratio is the
    # transfer-safe coordinate and must not be touched.
    assert _to_final()["head_lr_ratio"] == TUNED["head_lr_ratio"]


def test_an_absolute_head_lr_is_scaled_like_the_trunk():
    out = _to_final({**TUNED, "head_lr": 3.7e-4})
    assert out["head_lr"] / out["lr"] == pytest.approx(3.7e-4 / TUNED["lr"])


def test_it_sets_the_betas_and_the_target_batch():
    out = _to_final()
    assert out["batch_size"] == 512
    assert (out["beta1"], out["beta2"]) == protocol.betas_for(
        _steps(FOUR_EPOCH_SAMPLES, 512))


def test_the_batch_rule_is_the_square_root_at_fixed_data():
    # m_D = 1, m_B = 4: lr and wd both take sqrt(4) = 2.
    out = protocol.transfer(
        TUNED, from_samples=TUNE_SAMPLES, from_batch=512,
        to_samples=TUNE_SAMPLES, to_batch=2048)
    assert out["lr"] == pytest.approx(TUNED["lr"] * 2.0)
    assert out["weight_decay"] == pytest.approx(TUNED["weight_decay"] * 2.0)
    assert out["batch_size"] == 2048


def test_shrinkage_is_invariant_across_the_batch_axis_too():
    out = protocol.transfer(
        TUNED, from_samples=TUNE_SAMPLES, from_batch=512,
        to_samples=TUNE_SAMPLES, to_batch=2048)
    before = _steps(TUNE_SAMPLES, 512) * TUNED["lr"] * TUNED["weight_decay"]
    after = _steps(TUNE_SAMPLES, 2048) * out["lr"] * out["weight_decay"]
    assert after == pytest.approx(before, rel=1e-3)


def test_a_round_trip_is_the_identity():
    forward = _to_final()
    back = protocol.transfer(
        forward, from_samples=FOUR_EPOCH_SAMPLES, from_batch=512,
        to_samples=TUNE_SAMPLES, to_batch=512)
    assert back["lr"] == pytest.approx(TUNED["lr"])
    assert back["weight_decay"] == pytest.approx(TUNED["weight_decay"])
    assert back["beta1"] == pytest.approx(protocol.REFERENCE_BETAS[0])
    assert back["beta2"] == pytest.approx(protocol.REFERENCE_BETAS[1])


def test_unrecognised_keys_pass_through_untouched():
    out = _to_final({**TUNED, "kl_ratio": 3.06e-7, "prior_scale": 1.66e-3})
    # Loss terms, not optimizer state. The SDE rule says nothing about them and
    # inventing a factor would be worse than leaving them alone.
    assert out["kl_ratio"] == 3.06e-7
    assert out["prior_scale"] == 1.66e-3


def test_it_refuses_to_move_q_init_across_a_batch_change():
    """q0 = alpha^2 / B is IBProbit's OWN batch relation.

    Two rules claim the BATCH axis and there is no evidence for which wins, so
    a batch change on a config carrying q_init raises. The duration axis is a
    separate question with its own rule -- see the test below.
    """
    with_q = {**TUNED, "q_init": 3.72e-5}
    with pytest.raises(ValueError, match="q_init"):
        protocol.transfer(
            with_q, from_samples=TUNE_SAMPLES, from_batch=512,
            to_samples=TUNE_SAMPLES, to_batch=1024)


def test_it_scales_q_init_by_the_inverse_square_of_the_horizon():
    """q0 ~ T^-2, measured at three horizons on dinov3_large.

    A 7-point grid one decade apart put the optimum at 1.29e-4 / 1.29e-5 /
    1.29e-6 at 781 / 2502 / 5004 steps -- one decade per 3.2x. Transferred by
    IDENTITY instead, a tuned q_init would leave a 4-epoch run ~68x above its
    horizon's optimum.
    """
    with_q = {**TUNED, "q_init": 1.292e-4}
    out = _to_final(with_q)
    from_steps = protocol.steps_for_samples(TUNE_SAMPLES, 512)
    to_steps = protocol.steps_for_samples(FOUR_EPOCH_SAMPLES, 512)
    assert out["q_init"] == pytest.approx(
        1.292e-4 * (from_steps / to_steps) ** 2)
    # The direction matters more than the constant: a longer run wants LESS
    # process noise, because the head has more steps to drift over.
    assert out["q_init"] < with_q["q_init"]


def test_the_q_rule_is_an_identity_when_the_duration_does_not_change():
    with_q = {**TUNED, "q_init": 3.72e-5}
    out = protocol.transfer(
        with_q, from_samples=TUNE_SAMPLES, from_batch=512,
        to_samples=TUNE_SAMPLES, to_batch=512)
    assert out["q_init"] == pytest.approx(3.72e-5)


@pytest.mark.parametrize("kwargs", [
    {"from_samples": 0}, {"to_samples": 0}, {"from_batch": 0}, {"to_batch": 0},
])
def test_it_rejects_a_degenerate_operating_point(kwargs):
    base = dict(from_samples=TUNE_SAMPLES, from_batch=512,
                to_samples=FOUR_EPOCH_SAMPLES, to_batch=512)
    with pytest.raises(ValueError):
        protocol.transfer(TUNED, **{**base, **kwargs})


# --- horizons: the searched coordinate --------------------------------------


def test_the_horizon_coordinate_inverts():
    for steps in (1953, 3906, 10_009):
        for beta in (0.5, 0.9, 0.99, 0.999, 0.9999):
            h = protocol.horizons(beta, steps)
            assert protocol.beta_from_horizons(h, steps) == pytest.approx(beta)


def test_the_reference_betas_are_390_and_3_9_horizons():
    # The reference horizons the boxes in finetune_space are stated around.
    assert protocol.horizons(0.9, 3906) == pytest.approx(390.6)
    assert protocol.horizons(0.999, 3906) == pytest.approx(3.906)


def test_a_searched_horizon_survives_the_transfer_unchanged():
    """THE property that makes the beta axes worth searching.

    A trial draws H at its own step count and records it; `transfer` moves the
    resulting beta to the final budget. If H did not come out the other side
    identical, the searched coordinate would not be the transferable one and
    tuning the betas would buy nothing.
    """
    tune_steps = _steps(1_000_000, 512)
    final_steps = _steps(FOUR_EPOCH_SAMPLES, 512)
    for h1, h2 in ((39.0, 0.4), (390.6, 3.906), (780.0, 40.0)):
        params = {
            **TUNED,
            "beta1": protocol.beta_from_horizons(h1, tune_steps),
            "beta2": protocol.beta_from_horizons(h2, tune_steps),
        }
        out = protocol.transfer(
            params, from_samples=1_000_000, from_batch=512,
            to_samples=FOUR_EPOCH_SAMPLES, to_batch=512)
        assert protocol.horizons(out["beta1"], final_steps) == pytest.approx(h1)
        assert protocol.horizons(out["beta2"], final_steps) == pytest.approx(h2)


def test_it_refuses_a_horizon_longer_than_the_run():
    """The hard bound, and the reason the tuning budget has a floor.

    beta >= 0 requires the averaging window to be at least one step, so an
    N-step run cannot express more than N windows. The bound is asymmetric in
    the direction that matters: longer memory is free at any budget, shorter
    memory is what a short run cannot reach.
    """
    protocol.beta_from_horizons(1952, 1953)
    with pytest.raises(ValueError, match="do not fit in a 1953-step run"):
        protocol.beta_from_horizons(1953, 1953)
    # Longer memory is unconstrained at the same budget.
    assert protocol.beta_from_horizons(0.001, 1953) > 0.999999


def test_the_tuning_budget_can_express_the_whole_beta1_box():
    """The floor under protocol.MAX_SAMPLES, asserted rather than argued.

    finetune_space.BETA1_HORIZONS_BOX tops out at 780, which needs a run longer
    than 780 steps to represent at all. At 1M/512 = 1953 steps that leaves
    beta1 >= 0.6; halving the budget would put the box ceiling at
    beta1 = 0.2, and halving once more would make it unrepresentable.
    """
    from bllarse.tuning import finetune_space as fs

    steps = _steps(protocol.MAX_SAMPLES, fs.FIXED_BATCH_SIZE)
    assert steps == 1953
    for box in (fs.BETA1_HORIZONS_BOX, fs.BETA2_HORIZONS_BOX):
        assert box[1] < steps, f"{box} is not representable in {steps} steps"
    assert protocol.beta_from_horizons(
        fs.BETA1_HORIZONS_BOX[1], steps) == pytest.approx(0.6, abs=0.01)
