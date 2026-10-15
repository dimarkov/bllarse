"""The arithmetic both arm scripts must agree on.

These are pure integer/RNG functions with no JAX or torch in them, which is the
point: the PyTorch arm imports the SAME module, so a divergence
between the two arms' step counts or batch order becomes impossible rather than
merely unlikely.
"""

import numpy as np
import pytest

from bllarse.finetune import protocol


def test_steps_for_epochs_matches_the_design_table():
    # A step table at a 4-epoch cap on ImageNet-1k, drop-remainder.
    n_train = 1_281_167
    assert protocol.steps_for_epochs(n_train, 4, 512) == 10_009
    assert protocol.steps_for_epochs(n_train, 4, 1024) == 5_004
    assert protocol.steps_for_epochs(n_train, 4, 2048) == 2_502
    assert protocol.steps_for_epochs(n_train, 4, 4096) == 1_251


def test_steps_for_epochs_accepts_a_fractional_epoch():
    # A run can ask for a fraction of an epoch, not only a whole number.
    assert protocol.steps_for_epochs(1_281_167, 0.5, 1024) == 625


def test_steps_for_epochs_rejects_a_batch_larger_than_the_dataset():
    with pytest.raises(ValueError, match="exceeds"):
        protocol.steps_for_epochs(1000, 4, 2000)


def test_steps_for_epochs_rejects_a_budget_yielding_no_steps():
    with pytest.raises(ValueError, match="zero steps"):
        protocol.steps_for_epochs(1000, 0.001, 512)


def test_steps_for_samples_matches_the_tuning_budget():
    # 1,000,000 samples, drop-remainder, at each rung of the ladder.
    at = protocol.steps_for_samples
    assert at(protocol.MAX_SAMPLES, 256) == 3906
    assert at(protocol.MAX_SAMPLES, 512) == 1953
    assert at(protocol.MAX_SAMPLES, 1024) == 976
    assert at(protocol.MAX_SAMPLES, 2048) == 488
    assert at(protocol.MAX_SAMPLES, 4096) == 244
    assert at(protocol.MAX_SAMPLES, 8192) == 122
    assert at(protocol.MAX_SAMPLES, 16384) == 61


def test_steps_for_samples_rejects_a_budget_yielding_no_steps():
    with pytest.raises(ValueError, match="zero steps"):
        protocol.steps_for_samples(100, 512)


def test_eval_every_steps_at_each_ladder_rung():
    # EVAL_QUANTUM / batch. Integer at every rung because the ladder is powers
    # of two and the quantum is 16,384.
    assert protocol.eval_every_steps(256) == 64
    assert protocol.eval_every_steps(512) == 32
    assert protocol.eval_every_steps(1024) == 16
    assert protocol.eval_every_steps(2048) == 8
    assert protocol.eval_every_steps(4096) == 4


def test_every_ladder_rung_yields_the_same_number_of_evaluations():
    # THE property the sample-spaced grid exists for: eval index i means
    # i * EVAL_QUANTUM samples for every trial in the study, so MedianPruner
    # compares trials at matching amounts of data rather than matching steps.
    counts = {
        batch: protocol.steps_for_samples(protocol.MAX_SAMPLES, batch)
        // protocol.eval_every_steps(batch)
        for batch in (256, 512, 1024, 2048, 4096, 8192, 16384)
    }
    assert set(counts.values()) == {61}, counts


def test_eval_every_steps_rejects_an_off_ladder_batch():
    # Rounding here would hand back a grid whose samples_seen values differ per
    # trial -- exactly the defect the quantum removes, but now invisible.
    with pytest.raises(ValueError, match="whole number"):
        protocol.eval_every_steps(768)


def test_accum_plan_divides_evenly_when_it_can():
    assert protocol.accum_plan(4096, 64) == (64, 64, 4096)


def test_accum_plan_rejects_a_micro_batch_that_does_not_divide_the_target():
    # Rounding would give 4096 / 100 -> 41 accumulation steps -> an effective
    # batch of 4100, not 4096. That silently mislabels every point of Fig 3's
    # samples-seen axis, so a pinned micro-batch (finetune_space.MICRO_BATCH)
    # must divide the target exactly or the plan is unrealisable.
    with pytest.raises(ValueError, match="does not divide"):
        protocol.accum_plan(4096, 100)


def test_accum_plan_clamps_micro_to_the_target():
    # A target smaller than what fits in memory needs no accumulation.
    assert protocol.accum_plan(256, 1024) == (256, 1, 256)


def test_accum_plan_rejects_a_nonpositive_micro_batch():
    with pytest.raises(ValueError, match="micro_batch"):
        protocol.accum_plan(4096, 0)


def test_shuffled_indices_is_a_permutation():
    idx = protocol.shuffled_indices(1000, seed=137, epoch=0)
    assert np.array_equal(np.sort(idx), np.arange(1000))


def test_shuffled_indices_is_deterministic_across_calls():
    # This is the property that keeps the JAX and PyTorch arms on the same
    # data stream. If it ever regresses, the two arms silently diverge.
    a = protocol.shuffled_indices(1000, seed=137, epoch=3)
    b = protocol.shuffled_indices(1000, seed=137, epoch=3)
    assert np.array_equal(a, b)


def test_shuffled_indices_differs_across_epochs_and_seeds():
    base = protocol.shuffled_indices(1000, seed=137, epoch=0)
    assert not np.array_equal(base, protocol.shuffled_indices(1000, seed=137, epoch=1))
    assert not np.array_equal(base, protocol.shuffled_indices(1000, seed=138, epoch=0))


def test_the_geometric_grid_is_gone():
    # No geometric dump grid anywhere, fine-tuning or linear probing. A
    # log-spaced grid cannot serve as a pruner key, a patience clock or an
    # argmin search -- one "count" would mean ~150 steps early and ~7,000
    # late -- so dumps are strides of the sample-spaced eval index instead.
    # This asserts the grid helpers cannot come back by accident.
    assert not hasattr(protocol, "dump_steps")
    assert not hasattr(protocol, "ood_dump_steps")
