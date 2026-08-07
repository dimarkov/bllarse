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
    "adamw_ce": "scripts/alternatives/probe.py",
    # Same script as adamw_ce: the two JAX arms share the feature loading, the
    # pinned split and the dump schedule, and differ only in the training step.
    # `--arm` selects between them.
    "ibprobit": "scripts/alternatives/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

OnDump = Callable[[int, int, Mapping[str, float]], None]
OnConverged = Callable[[int, int, int], None]

# A quarter of worker.py's 40-dump grid: the dump schedule is log-spaced, so
# the earliest dumps are a single optimiser step apart, and a raw patience
# counter would fire almost immediately on a run that is merely slow to start.
DEFAULT_WARMUP_DUMPS = 10

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

    def __init__(self, dump_index: int, step: int):
        super().__init__(f"converged at dump {dump_index} (step {step})")
        self.dump_index = dump_index
        self.step = step


def _patience_on_dump(
    on_dump: OnDump | None,
    *,
    patience: int,
    warmup_dumps: int,
    min_relative_improvement: float,
) -> OnDump:
    """Builds one seed's on_dump callback: the caller's own hook, THEN
    patience bookkeeping on top of it.

    Delegating first matters: if the caller's hook is Optuna's should_prune
    wired up to raise TrialPruned, that must win and propagate untouched --
    pruning a losing trial and stopping a converged one are different events,
    and only one of them can happen at a given dump.
    """
    best: float | None = None
    stall = 0
    seen = 0

    def wrapped(dump_index: int, step: int, scalars: Mapping[str, float]) -> None:
        nonlocal best, stall, seen
        if on_dump is not None:
            on_dump(dump_index, step, scalars)

        seen += 1
        value = float(scalars["val_nll"])
        # Relative improvement, not absolute: see DEFAULT_MIN_RELATIVE_IMPROVEMENT.
        if best is None or value < best * (1.0 - min_relative_improvement):
            best = value
            stall = 0
        else:
            stall += 1

        if seen >= warmup_dumps and stall >= patience:
            raise EarlyConvergence(dump_index, step)

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
    patience: int | None = None,
    warmup_dumps: int = DEFAULT_WARMUP_DUMPS,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    on_converged: OnConverged | None = None,
) -> float:
    """Run `config` on both tuning seeds and return its objective value.

    `on_dump` fires during the FIRST seed only. Pruning a hopeless config before
    its second seed is where most of the saving comes from, and reporting from
    both would push two different trajectories through one trial's step keys --
    MedianPruner compares trials at matching keys, so that would corrupt every
    comparison it makes.

    `patience` is a separate concern from `on_dump` and, unlike it, applies to
    BOTH seeds: it stops a run once `patience` consecutive dumps pass with no
    relative improvement in val_nll (see DEFAULT_MIN_RELATIVE_IMPROVEMENT),
    never before `warmup_dumps` dumps have been seen. `None` (the default)
    disables it entirely and reproduces today's behaviour exactly -- every
    run trains to its full step budget. `on_converged(seed, dump_index, step)`,
    if given, fires once per seed that stops early, so a caller (worker.py)
    can record how much a trial actually saved.

    Scoring always runs over whatever dumps exist once each seed's run ends,
    whether that is the full budget or an early stop -- the objective value is
    still `early_stop(run_curve(...), "val_nll")` either way.

    Dumps are removed unless `keep_dumps`: a full campaign is ~179k files that
    nothing downstream reads, since Stage 4 re-runs the winning config on the
    reported seeds.
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
            if patience is None:
                dump_cb = seed_on_dump
            else:
                dump_cb = _patience_on_dump(
                    seed_on_dump,
                    patience=patience,
                    warmup_dumps=warmup_dumps,
                    min_relative_improvement=min_relative_improvement,
                )
            try:
                module.main(args, on_dump=dump_cb)
            except EarlyConvergence as converged:
                if on_converged is not None:
                    on_converged(seed, converged.dump_index, converged.step)
        return score_dumps(dump_root, group_id, uids)
    finally:
        if not keep_dumps:
            for run_uid in uids:
                shutil.rmtree(Path(dump_root) / group_id / run_uid, ignore_errors=True)
