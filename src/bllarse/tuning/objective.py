"""One Optuna trial: run a config on both tuning seeds, return its score.

The score IS analysis.thresholds.early_stop, not a reimplementation of it, so
the tuning criterion and the reported stopping rule cannot drift apart.
Selection is on val, reporting is on test, and bllarse.splits makes the two
disjoint by construction -- so this does not leak. The OOD split never enters
tuning at all: for Fig 2, studies are keyed on the ID dataset only, since both
scripts load OOD features after the training loop purely for a final dump.
"""

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bllarse.analysis.curves import run_curve
from bllarse.analysis.thresholds import early_stop
from bllarse.tools import adapters

# Disjoint from Fig 1/2's range(137, 147). Selecting hyperparameters on seeds
# that later produce the reported error bars would tune against the same random
# draws being reported.
TUNING_SEEDS = (200, 201)

SCRIPTS = {
    "adamw_ce": "scripts/dinov3/probe.py",
    # Same script as adamw_ce: the two JAX arms share the feature loading, the
    # pinned split and the dump schedule, and differ only in the training step.
    # `--arm` selects between them.
    "ibprobit": "scripts/dinov3/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

OnDump = Callable[[int, int, int, Mapping[str, float]], None]
OnConverged = Callable[[int, int, int], None]

# Warmup, in samples. Evaluations are `eval_quantum` samples apart and the
# quantum is at most half an epoch, so ten of them is at most five epochs --
# enough that a config which is merely slow to start is not killed for it.
# Expressed in samples rather than a count of evaluations for the same reason
# patience is: a count is only interpretable if you also know the quantum.
DEFAULT_WARMUP_SAMPLES = 10 * 16_384

# val_nll ranges from ~0.003 (flowers102) to ~5.2 (stanford_cars, VBLL) across
# this project's datasets, so a fixed absolute epsilon is wrong at one end or
# the other. It also drifts downward in the fourth decimal place essentially
# forever, so "any decrease counts" never plateaus. 0.1% relative improvement
# is comfortably above that drift at every scale this project uses, while
# still well under the gains a genuinely still-improving config produces
# between early dumps.
DEFAULT_MIN_RELATIVE_IMPROVEMENT = 1e-3


class EarlyConvergence(Exception):
    """Raised internally: this run's val_nll has plateaued, so its training
    loop should stop and scoring should proceed on the dumps written so far.

    Deliberately NOT optuna.TrialPruned. A pruned trial's objective value is
    discarded -- Optuna considers it a loser. A converged trial is the
    opposite: a config that finished doing everything it was going to do, and
    whose score is a perfectly good result. objective.py has no optuna
    import, so this stays a plain local exception; only worker.py needs to
    know it is unrelated to pruning.
    """

    def __init__(self, eval_index: int, step: int):
        super().__init__(f"converged at eval {eval_index} (step {step})")
        self.eval_index = eval_index
        self.step = step


def _patience_on_dump(
    on_dump: OnDump | None,
    *,
    patience_samples: int | None,
    patience_factor: float | None,
    warmup_samples: int,
    min_relative_improvement: float,
) -> OnDump:
    """Builds one seed's on_dump callback: the caller's own hook, THEN
    patience bookkeeping on top of it.

    Patience is measured in SAMPLES since the last improvement, never in a
    count of evaluations. Counting evaluations is what made the previous rule
    inert: a run had exactly 40 of them, so `patience=20` could only fire if the
    curve turned in the first half of the grid, and on five of seven datasets it
    never turned that early -- it fired on 46 of 133 trials, all cifar10 or dtd,
    and structurally could not fire on the rest.

    TWO forms, because one alone does not cover these datasets:

      * `patience_samples` -- absolute. Interpretable ("stop after 2M samples
        with no improvement") and the natural reading of the rule. It does not
        transfer across datasets on its own: the measured argmin sits at 43k
        samples on cifar10 and 9.0M on stanford_cars, so the smallest globally
        safe absolute threshold is 15M -- 350x past cifar10's optimum, i.e.
        disabled there and active only on the largest study.
      * `patience_factor` -- multiplicative, firing once samples_seen reaches
        `factor x` the samples at which the best was set. Scale-free, so it is
        the form that survives a 143x spread in where the optimum sits: the
        largest per-curve ratio between consecutive improvements is 1.30-4.88
        across all seven datasets, and a factor of 5 cuts none of the 170
        measured curves before its argmin.

    Set either, or both (whichever fires first). Both are `None` by default,
    which disables patience entirely.

    Caveat on the factor's calibration: it was measured on the old geometric
    grid, and every observed ratio is an exact power of that grid's 1.3026
    spacing -- they are quantised lower bounds, not resolved values. Re-derive
    it from the sample-spaced curves this protocol produces rather than
    tightening it on the numbers above; calibrating against a grid artifact is
    how `--patience 20` came about.

    Delegating first matters: if the caller's hook is Optuna's should_prune
    wired up to raise TrialPruned, that must win and propagate untouched --
    pruning a losing trial and stopping a converged one are different events,
    and only one of them can happen at a given evaluation.
    """
    best: float | None = None
    best_samples = 0

    def wrapped(
        eval_index: int, step: int, samples_seen: int, scalars: Mapping[str, float]
    ) -> None:
        nonlocal best, best_samples
        if on_dump is not None:
            on_dump(eval_index, step, samples_seen, scalars)

        value = float(scalars["val_nll"])
        # Relative improvement, not absolute: see DEFAULT_MIN_RELATIVE_IMPROVEMENT.
        if best is None or value < best * (1.0 - min_relative_improvement):
            best = value
            best_samples = samples_seen
            return

        if samples_seen < warmup_samples:
            return
        stalled = samples_seen - best_samples
        if patience_samples is not None and stalled >= patience_samples:
            raise EarlyConvergence(eval_index, step)
        if patience_factor is not None and samples_seen >= patience_factor * best_samples:
            raise EarlyConvergence(eval_index, step)

    return wrapped


def score_dumps(dump_root: Any, group_id: str, uids: Sequence[str]) -> float:
    """Mean over runs of the early-stopped val_nll. Lower is better."""
    values = [
        early_stop(run_curve(dump_root, group_id, uid), "val_nll").value
        for uid in uids
    ]
    if not values:
        raise ValueError("no runs to score")
    return float(sum(values) / len(values))


def evaluate(
    arm: str,
    config: Mapping[str, Any],
    *,
    dump_root: Any,
    group_id: str,
    uid: str,
    on_dump: OnDump | None = None,
    keep_dumps: bool = False,
    patience_samples: int | None = None,
    patience_factor: float | None = None,
    warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    on_converged: OnConverged | None = None,
) -> float:
    """Run `config` on both tuning seeds and return its objective value.

    `on_dump` fires during the FIRST seed only. Pruning a hopeless config before
    its second seed is where most of the saving comes from, and reporting from
    both would push two different trajectories through one trial's step keys --
    MedianPruner compares trials at matching keys, so that would corrupt every
    comparison it makes.

    `patience_samples` is a separate concern from `on_dump` and, unlike it,
    applies to BOTH seeds: it stops a run once that many samples have passed
    with no relative improvement in val_nll (see
    DEFAULT_MIN_RELATIVE_IMPROVEMENT), never before `warmup_samples`. `None`
    (the default) disables it entirely -- every run trains to its full step
    budget. `on_converged(seed, eval_index, step)`, if given, fires once per
    seed that stops early, so a caller (worker.py) can record how much a trial
    actually saved.

    Scoring always runs over whatever dump exists once each seed's run ends,
    whether that is the full budget or an early stop -- the training script
    buffers its argmin-val_nll evaluation and writes it from a `finally`, so
    the dump is there either way and the objective value is still
    `early_stop(run_curve(...), "val_nll")`.

    Dumps are removed unless `keep_dumps`: Stage 4 re-runs the winning config
    on the reported seeds, so nothing downstream reads a tuning run's dump.
    """
    if arm not in SCRIPTS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(SCRIPTS)}")

    module = adapters.load_script_module(SCRIPTS[arm])
    uids = [f"{uid}_seed{seed}" for seed in TUNING_SEEDS]

    try:
        for index, (seed, run_uid) in enumerate(zip(TUNING_SEEDS, uids)):
            args = adapters.parse_config(module, {
                **config,
                "seed": seed,
                "uid": run_uid,
                "group_id": group_id,
                "dump_root": str(dump_root),
            })
            seed_on_dump = on_dump if index == 0 else None
            if patience_samples is None and patience_factor is None:
                dump_cb = seed_on_dump
            else:
                dump_cb = _patience_on_dump(
                    seed_on_dump,
                    patience_samples=patience_samples,
                    patience_factor=patience_factor,
                    warmup_samples=warmup_samples,
                    min_relative_improvement=min_relative_improvement,
                )
            try:
                module.main(args, on_dump=dump_cb)
            except EarlyConvergence as converged:
                if on_converged is not None:
                    on_converged(seed, converged.eval_index, converged.step)
        return score_dumps(dump_root, group_id, uids)
    finally:
        if not keep_dumps:
            for run_uid in uids:
                shutil.rmtree(Path(dump_root) / group_id / run_uid, ignore_errors=True)
