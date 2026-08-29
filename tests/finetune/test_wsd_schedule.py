"""The WSD learning-rate schedule. Default off, and provably so."""

import pytest

from bllarse.finetune import protocol


def test_decay_frac_zero_returns_the_scalar_unchanged():
    # Every reported run so far used a constant LR. The default path must be
    # the SAME OBJECT, not a schedule that happens to be flat -- an optax
    # schedule and a float take different code paths inside adamw.
    assert protocol.wsd_schedule(1e-5, 1000, decay_frac=0.0, decay_to=0.5) == 1e-5


def test_stable_phase_holds_the_peak_then_decays_to_the_target():
    sched = protocol.wsd_schedule(1e-5, 100, decay_frac=0.5, decay_to=0.5)

    assert float(sched(0)) == pytest.approx(1e-5)
    assert float(sched(49)) == pytest.approx(1e-5)
    assert float(sched(50)) == pytest.approx(1e-5)      # decay starts here
    assert float(sched(75)) == pytest.approx(7.5e-6)    # halfway down
    assert float(sched(100)) == pytest.approx(5e-6)     # the floor


def test_the_floor_holds_past_the_end():
    # A run that overruns its planned step count by a step must not resume
    # extrapolating the line downward past the floor.
    sched = protocol.wsd_schedule(1e-5, 100, decay_frac=0.5, decay_to=0.5)

    assert float(sched(200)) == pytest.approx(5e-6)


def test_mean_multiplier_reports_the_shrinkage_the_run_actually_gets():
    # optax.adamw scales the decoupled decay update by the learning rate, so a
    # decayed lr decays weight decay with it. The tuned configuration's
    # steps*lr*wd invariant no longer holds, and the caller has to be told by
    # how much rather than discovering it in a result.
    assert protocol.wsd_mean_multiplier(0.0, 0.5) == pytest.approx(1.0)
    assert protocol.wsd_mean_multiplier(0.5, 0.5) == pytest.approx(0.875)
    assert protocol.wsd_mean_multiplier(1.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("frac,to", [(-0.1, 0.5), (1.1, 0.5), (0.5, 0.0), (0.5, 1.5)])
def test_out_of_range_arguments_are_refused(frac, to):
    with pytest.raises(ValueError):
        protocol.wsd_schedule(1e-5, 100, decay_frac=frac, decay_to=to)


def test_the_two_realisations_agree():
    """The torch arm reads wsd_multiplier; the JAX arms get an optax schedule
    built from the same numbers. A flag that meant different things in the two
    frameworks would be invisible in every result."""
    peak, steps = 3e-5, 997
    sched = protocol.wsd_schedule(peak, steps, decay_frac=0.4, decay_to=0.25)
    for step in (0, 1, 500, 597, 598, 700, 900, 996, 997, 1200):
        mult = protocol.wsd_multiplier(step, steps, decay_frac=0.4, decay_to=0.25)
        # rel=1e-6, not tighter: optax schedules evaluate in float32 while
        # wsd_multiplier is float64 Python arithmetic, so agreement is bounded
        # by the dtype rather than by the two definitions.
        assert float(sched(step)) == pytest.approx(peak * mult, rel=1e-6)


def test_multiplier_is_one_everywhere_when_the_schedule_is_off():
    for step in (0, 10, 500, 10_000):
        assert protocol.wsd_multiplier(
            step, 1000, decay_frac=0.0, decay_to=0.5) == 1.0
