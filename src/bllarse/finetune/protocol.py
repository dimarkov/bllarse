"""Budget arithmetic, the accumulation plan, and the shared batch order.

Fine-tuning inverts the probing campaign's convention. Probing fixed 30,000
optimizer STEPS and derived epochs (bllarse/tuning/budgets.py); here the cap is
4 EPOCHS and the step count falls out of the tuned batch size. That inversion is
what makes Fig 5 valid: results-plan puts samples-seen on its x-axis, so arms
tuned to different batch sizes stay comparable even though they take different
numbers of steps.
"""

import numpy as np

# The tuning campaign's sample budget. A budget in SAMPLES, not steps: at a
# fixed step budget `samples_seen = steps * batch`, so large batches would be
# handed more data and the tuned batch size would not transfer to the final run.
#
# 1,000,000 / 1,281,167 = 0.781 epochs, and 1e6 / EVAL_QUANTUM = 61 evaluations
# at every rung of the ladder.
#
# Halved from 2,000,000 once throughput was measured rather than assumed: at
# the measured ~40 img/s a 2M-sample trial is ~14 h of training alone, so the
# campaign did not fit its own compute budget. The risk this carries is
# directional and worth stating -- a shorter budget favours configs that move
# FAST, i.e. high trunk learning rates, which is also the catastrophic
# forgetting direction. Read `truncated` off the trials: if it is true almost
# everywhere, the ranking is being made on runs that were all still improving,
# and the budget is the binding constraint rather than the search.
MAX_SAMPLES = 1_000_000

# Samples between consecutive val_nll evaluations, for every trial.
#
# This replaces a log-spaced grid over STEPS. The old grid was sized as a figure
# artifact for 500-step runs and then reused at 30,000 steps for three jobs it
# was never sized for -- the pruner's key, the patience clock, and the argmin
# search -- where one "count" meant ~150 steps early and ~7,000 late.
#
# The quantum works because every rung of the batch ladder divides it: a trial
# at the top evaluates every 4 steps, one at the bottom every 64, and both land
# on exactly the same samples_seen values. Evaluation can only happen BETWEEN
# optimizer steps, so a quantum that is not a whole number of steps for some
# rung is unschedulable for that rung -- hence `eval_every_steps` raises.
EVAL_QUANTUM = 16_384


def steps_for_epochs(n_train: int, epochs: float, effective_batch: int) -> int:
    """Optimizer steps in `epochs` passes at `effective_batch`, drop-remainder.

    `epochs` is a float so a tuning trial can ask for a fraction of a pass --
    which is the whole tuning budget under this design, since trials are
    truncated by a samples-seen budget rather than run to convergence.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    if effective_batch > n_train:
        raise ValueError(
            f"effective_batch={effective_batch} exceeds n_train={n_train}: "
            f"a single batch cannot be drawn"
        )
    steps = int(epochs * n_train) // effective_batch
    if steps < 1:
        raise ValueError(
            f"epochs={epochs} at effective_batch={effective_batch} over "
            f"n_train={n_train} yields zero steps"
        )
    return steps


def steps_for_samples(max_samples: int, effective_batch: int) -> int:
    """Optimizer steps to consume `max_samples` at `effective_batch`.

    The tuning counterpart of `steps_for_epochs`, and the reason both exist:
    the final campaign is capped in EPOCHS (4) while tuning is capped in
    SAMPLES (MAX_SAMPLES), so the two derive their step counts from different
    quantities and neither is expressible as the other without knowing n_train.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    steps = int(max_samples) // effective_batch
    if steps < 1:
        raise ValueError(
            f"max_samples={max_samples} at effective_batch={effective_batch} "
            f"yields zero steps"
        )
    return steps


def eval_every_steps(effective_batch: int, quantum: int = EVAL_QUANTUM) -> int:
    """Optimizer steps between evaluations at `effective_batch`.

    Raises rather than rounding: a non-integer ratio means the requested
    schedule cannot be realised, and silently rounding it would hand back a
    grid whose samples_seen values differ per trial -- exactly the defect
    EVAL_QUANTUM exists to remove, but now invisible.
    """
    if effective_batch < 1:
        raise ValueError(f"effective_batch must be >= 1, got {effective_batch}")
    if quantum % effective_batch:
        raise ValueError(
            f"eval quantum {quantum} is not a whole number of "
            f"effective_batch={effective_batch} steps. The ladder is powers of "
            f"two, so this means the batch is off-ladder; see "
            f"tuning.finetune_space.BATCH_SIZES."
        )
    return quantum // effective_batch


def accum_plan(target_batch: int, micro_batch: int) -> tuple[int, int, int]:
    """Resolve a target batch into (micro, num_accum, effective).

    `micro_batch` is a memory artifact -- the largest per-forward batch that
    fits -- and differs between arms and frameworks. `target_batch` is the
    tuned quantity.

    The micro batch must DIVIDE the target exactly. This used to round, so a
    non-dividing pair silently produced an effective batch that was not the
    requested one (4096 over a micro batch of 100 gave 4100), which then
    mislabels every point of a samples-seen axis. Design A8 pins the micro
    batch to a power of two no larger than the ladder's gcd precisely so this
    is always satisfiable; a violation here means the pinned constant and the
    ladder have drifted apart, which is a configuration bug rather than
    something to paper over.
    """
    if micro_batch < 1:
        raise ValueError(f"micro_batch must be >= 1, got {micro_batch}")
    if target_batch < 1:
        raise ValueError(f"target_batch must be >= 1, got {target_batch}")
    micro = min(micro_batch, target_batch)
    if target_batch % micro:
        raise ValueError(
            f"micro batch {micro} does not divide target batch {target_batch}: "
            f"the effective batch would be {micro * round(target_batch / micro)}, "
            f"not the {target_batch} that was tuned."
        )
    num_accum = target_batch // micro
    return micro, num_accum, micro * num_accum


def shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray:
    """The training permutation for one pass.

    numpy `default_rng`, NOT `jax.random`: the PyTorch arm has to reproduce this
    exact order so both arms consume an identical data stream, which is the
    property full_finetune_comparison.py already guarantees within JAX and which
    a per-framework RNG would quietly break.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)
