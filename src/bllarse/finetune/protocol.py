"""Budget arithmetic, the accumulation plan, and the shared batch order.

Fine-tuning inverts the probing campaign's convention. Probing fixed 30,000
optimizer STEPS and derived epochs (bllarse/tuning/budgets.py); here the cap is
4 EPOCHS and the step count falls out of the tuned batch size. That inversion is
what makes Fig 5 valid: results-plan puts samples-seen on its x-axis, so arms
tuned to different batch sizes stay comparable even though they take different
numbers of steps.
"""

import numpy as np

from bllarse import evaldump

# Probing uses 40. Fine-tuning runs are 1-2 orders of magnitude shorter in
# steps, and every dump costs a full pass over 50k eval images, so 24 buys the
# same log-spaced resolution at a proportionate price.
DEFAULT_N_DUMPS = 24


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


def accum_plan(target_batch: int, micro_batch: int) -> tuple[int, int, int]:
    """Resolve a target batch into (micro, num_accum, effective).

    `micro_batch` is a memory artifact -- the largest per-forward batch that
    fits -- and differs between arms and frameworks. `target_batch` is the tuned
    quantity. The returned `effective` is what was ACTUALLY achieved and need
    not equal the request: callers must log the returned value, or MLflow will
    record a batch size the run did not use.
    """
    if micro_batch < 1:
        raise ValueError(f"micro_batch must be >= 1, got {micro_batch}")
    if target_batch < 1:
        raise ValueError(f"target_batch must be >= 1, got {target_batch}")
    micro = min(micro_batch, target_batch)
    num_accum = max(1, round(target_batch / micro))
    return micro, num_accum, micro * num_accum


def shuffled_indices(n: int, seed: int, epoch: int) -> np.ndarray:
    """The training permutation for one pass.

    numpy `default_rng`, NOT `jax.random`: the PyTorch arm has to reproduce this
    exact order so both arms consume an identical data stream, which is the
    property full_finetune_comparison.py already guarantees within JAX and which
    a per-framework RNG would quietly break.
    """
    return np.random.default_rng([seed, epoch]).permutation(n)


def dump_steps(num_steps: int, n_dumps: int = DEFAULT_N_DUMPS) -> np.ndarray:
    """Log-spaced eval steps, dense early. Always includes 1 and `num_steps`."""
    return evaldump.dump_schedule(num_steps, n_dumps)


def ood_dump_steps(steps: np.ndarray, every_k: int) -> np.ndarray:
    """Every k-th entry of `steps`, with the final step always retained.

    OOD evaluation forwards 47,500 images, so it rides a stride of the val/test
    schedule rather than every point. The last step is forced back in because
    Table 1's converged endpoint needs it and a stride will drop it whenever
    len(steps) is not 1 mod k.
    """
    steps = np.asarray(steps, dtype=int)
    if every_k <= 0:
        return np.asarray([], dtype=int)
    chosen = steps[::every_k]
    if len(steps) and steps[-1] not in chosen:
        chosen = np.append(chosen, steps[-1])
    return chosen
