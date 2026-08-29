"""The WSD learning-rate schedule. Default off, and provably so.

`decay_start` is WHERE THE DECAY BEGINS as a fraction of the run, not how long
it lasts. The first version of this parameter took the length and was misread
as the start within a day, so the distinction is asserted here rather than left
to the docstring.
"""

import pytest

from bllarse.finetune import protocol


def test_decay_start_of_one_returns_the_scalar_unchanged():
    # Every reported run so far used a constant LR. The OFF path must be the
    # SAME OBJECT, not a schedule that happens to be flat -- an optax schedule
    # and a float take different code paths inside adamw.
    assert protocol.wsd_schedule(1e-5, 1000, decay_start=1.0, decay_to=0.5) == 1e-5


def test_decay_starts_where_decay_start_says_and_ramps_to_the_end():
    # 0.25 of 100 steps: flat to step 25, then down over the remaining 75.
    sched = protocol.wsd_schedule(1e-5, 100, decay_start=0.25, decay_to=0.2)

    assert float(sched(0)) == pytest.approx(1e-5)
    assert float(sched(25)) == pytest.approx(1e-5)        # last stable step
    assert float(sched(26)) < 1e-5                        # decay has begun
    # Halfway down the ramp is 0.6x, not 0.5x: the multiplier runs 1.0 -> 0.2,
    # so its midpoint is (1 + 0.2)/2.
    assert float(sched(62)) == pytest.approx(6.05e-6, rel=1e-2)
    assert float(sched(100)) == pytest.approx(2e-6)       # the floor


def test_decay_start_is_not_the_length_of_the_decay():
    """The regression this parameter was renamed for.

    decay_start=0.25 must decay over the LAST 75% of the run, not the last 25%.
    Under the old reading step 50 of 100 would still be at the peak.
    """
    sched = protocol.wsd_schedule(1e-5, 100, decay_start=0.25, decay_to=0.2)

    assert float(sched(50)) < 0.9e-5


def test_the_floor_holds_past_the_end():
    # A run that overruns its planned step count must not keep extrapolating
    # the line downward past the floor.
    sched = protocol.wsd_schedule(1e-5, 100, decay_start=0.5, decay_to=0.5)

    assert float(sched(200)) == pytest.approx(5e-6)


def test_mean_multiplier_reports_the_shrinkage_the_run_actually_gets():
    # optax.adamw and torch.optim.AdamW both scale the decoupled decay update
    # BY the learning rate, so a decayed lr decays weight decay with it. The
    # tuned configuration's steps*lr*wd invariant no longer holds, and the
    # caller has to be told by how much rather than discovering it in a result.
    assert protocol.wsd_mean_multiplier(1.0, 0.5) == pytest.approx(1.0)
    assert protocol.wsd_mean_multiplier(0.5, 0.5) == pytest.approx(0.875)
    assert protocol.wsd_mean_multiplier(0.25, 0.2) == pytest.approx(0.70)
    assert protocol.wsd_mean_multiplier(0.0, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("start,to", [(-0.1, 0.5), (1.1, 0.5), (0.5, 0.0), (0.5, 1.5)])
def test_out_of_range_arguments_are_refused(start, to):
    with pytest.raises(ValueError):
        protocol.wsd_schedule(1e-5, 100, decay_start=start, decay_to=to)


def test_the_two_realisations_agree():
    """The torch arm reads wsd_multiplier; the JAX arms get an optax schedule
    built from the same numbers. A flag that meant different things in the two
    frameworks would be invisible in every result."""
    peak, steps = 3e-5, 997
    sched = protocol.wsd_schedule(peak, steps, decay_start=0.6, decay_to=0.25)
    for step in (0, 1, 500, 597, 598, 700, 900, 996, 997, 1200):
        mult = protocol.wsd_multiplier(step, steps, decay_start=0.6, decay_to=0.25)
        # rel=1e-6, not tighter: optax schedules evaluate in float32 while
        # wsd_multiplier is float64 Python arithmetic, so agreement is bounded
        # by the dtype rather than by the two definitions.
        assert float(sched(step)) == pytest.approx(peak * mult, rel=1e-6)


def test_multiplier_is_one_everywhere_when_the_schedule_is_off():
    for step in (0, 10, 500, 10_000):
        assert protocol.wsd_multiplier(
            step, 1000, decay_start=1.0, decay_to=0.5) == 1.0
