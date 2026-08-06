"""The arithmetic both arm scripts must agree on.

These are pure integer/RNG functions with no JAX or torch in them, which is the
point: the PyTorch arm in Phase 2 imports the SAME module, so a divergence
between the two arms' step counts or batch order becomes impossible rather than
merely unlikely.
"""

import numpy as np
import pytest

from bllarse.finetune import protocol


def test_steps_for_epochs_matches_the_design_table():
    # The table in docs/superpowers/specs/2026-08-06-full-network-finetuning-design.md
    # section 3, at the fixed 4-epoch cap on ImageNet-1k.
    n_train = 1_281_167
    assert protocol.steps_for_epochs(n_train, 4, 512) == 10_009
    assert protocol.steps_for_epochs(n_train, 4, 1024) == 5_004
    assert protocol.steps_for_epochs(n_train, 4, 2048) == 2_502
    assert protocol.steps_for_epochs(n_train, 4, 4096) == 1_251


def test_steps_for_epochs_accepts_a_fractional_epoch():
    # Tuning trials get a fraction of an epoch, not a whole number of them.
    assert protocol.steps_for_epochs(1_281_167, 0.5, 1024) == 625


def test_steps_for_epochs_rejects_a_batch_larger_than_the_dataset():
    with pytest.raises(ValueError, match="exceeds"):
        protocol.steps_for_epochs(1000, 4, 2000)


def test_steps_for_epochs_rejects_a_budget_yielding_no_steps():
    with pytest.raises(ValueError, match="zero steps"):
        protocol.steps_for_epochs(1000, 0.001, 512)


def test_accum_plan_divides_evenly_when_it_can():
    assert protocol.accum_plan(4096, 64) == (64, 64, 4096)


def test_accum_plan_reports_the_effective_batch_it_actually_achieves():
    # 4096 / 100 rounds to 41 accumulation steps, so the effective batch is
    # 4100, not 4096. Callers log the returned value, never the request --
    # otherwise MLflow records a batch size the run did not use.
    micro, num_accum, effective = protocol.accum_plan(4096, 100)
    assert (micro, num_accum) == (100, 41)
    assert effective == 4100


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


def test_dump_steps_spans_one_to_num_steps():
    steps = protocol.dump_steps(5000)
    assert len(steps) == protocol.DEFAULT_N_DUMPS
    assert steps[0] == 1
    assert steps[-1] == 5000
    assert np.all(np.diff(steps) > 0)


def test_ood_dump_steps_subsamples_and_always_keeps_the_last():
    steps = np.array([1, 2, 4, 8, 16, 32, 64])
    chosen = protocol.ood_dump_steps(steps, every_k=2)
    assert chosen[0] == 1
    assert chosen[-1] == 64  # the final step is never dropped by the stride
    assert set(chosen).issubset(set(steps))


def test_ood_dump_steps_disabled_returns_empty():
    steps = np.array([1, 2, 4])
    assert len(protocol.ood_dump_steps(steps, every_k=0)) == 0
