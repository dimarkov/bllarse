"""One Optuna trial: run a config on both tuning seeds, return its score.

The score IS analysis.thresholds.early_stop, not a reimplementation of it, so
the tuning criterion and the reported stopping rule cannot drift apart.
Selection is on val, reporting is on test, and bllarse.splits makes the two
disjoint by construction -- so this does not leak. The OOD split never enters
tuning at all: for Fig 2, studies are keyed on the ID dataset only, since both
scripts load OOD features after the training loop purely for a final dump.
"""

import math
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
    # The contamination ablation: same head, same script, eps_init pinned at
    # space.IBPROBIT_NOEPS_INIT instead of searched.
    "ibprobit_noeps": "scripts/dinov3/probe.py",
    "vbll": "scripts/vbll_pytorch/probing_vbll.py",
}

OnDump = Callable[[int, int, int, Mapping[str, float]], None]
OnConverged = Callable[[int, int, int], None]

# Evaluations that must pass before patience may fire, so a config that is
# merely slow to start is not killed for it. The quantum is at most half an
# epoch, so ten evaluations is at most five epochs.
DEFAULT_WARMUP_EVALS = 10

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
    patience: int,
    warmup_evals: int,
    min_relative_improvement: float,
) -> OnDump:
    """Builds one seed's on_dump callback: the caller's own hook, THEN
    patience bookkeeping on top of it.

    `patience` is consecutive evaluations with no relative improvement in
    val_nll (see DEFAULT_MIN_RELATIVE_IMPROVEMENT).

    A COUNT is a well-defined amount of training here, which is the whole
    difference from the rule this replaces. Evaluations are `eval_quantum`
    samples apart and uniformly spaced (see space.eval_quantum), so N
    evaluations is always N * quantum samples. The old `--patience 20` counted
    points on a GEOMETRIC grid bounded at 40, where one count meant ~150 steps
    early and ~7,000 steps late, and where 20 could only fire if the curve
    turned in the first half -- it fired on 46 of 133 trials, all of them
    cifar10 or dtd, and structurally could not fire on the other five datasets.

    The value is not derivable in advance. Two attempts to calibrate it from
    the previous campaign's curves both failed: an absolute sample threshold
    cannot serve datasets whose optima span 43k to 9.0M samples, and a
    multiplicative factor tuned on ibprobit (0 of 113 curves cut short) cut 125
    of 1,214 once adamw_ce and vbll were included. Set it, run, then read what
    actually fired off `patience_stopped` / `patience_stopped_at_eval`.

    Delegating first matters: if the caller's hook is Optuna's should_prune
    wired up to raise TrialPruned, that must win and propagate untouched --
    pruning a losing trial and stopping a converged one are different events,
    and only one of them can happen at a given evaluation.
    """
    best: float | None = None
    stall = 0
    seen = 0

    def wrapped(
        eval_index: int, step: int, samples_seen: int, scalars: Mapping[str, float]
    ) -> None:
        nonlocal best, stall, seen
        if on_dump is not None:
            on_dump(eval_index, step, samples_seen, scalars)

        seen += 1
        value = float(scalars["val_nll"])
        # Relative improvement, not absolute: see DEFAULT_MIN_RELATIVE_IMPROVEMENT.
        # `isfinite` guards the `best is None` arm specifically: without it a
        # non-finite FIRST evaluation becomes `best`, every later comparison
        # against nan is False, and the counter stalls for the rest of the run
        # -- firing at exactly `1 + patience`. A non-finite value carries no
        # information about improvement, so it counts as a stall and is never
        # adopted as the best.
        if math.isfinite(value) and (
            best is None or value < best * (1.0 - min_relative_improvement)
        ):
            best = value
            stall = 0
        else:
            stall += 1

        if seen >= warmup_evals and stall >= patience:
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
    patience: int | None = None,
    warmup_evals: int = DEFAULT_WARMUP_EVALS,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    on_converged: OnConverged | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
    scripts: Mapping[str, str] = SCRIPTS,
    score: Callable[[Any, str, Sequence[str]], float] = score_dumps,
) -> float:
    """Run `config` on every tuning seed and return its objective value.

    `on_dump` fires during the FIRST seed only. Pruning a hopeless config before
    its second seed is where most of the saving comes from, and reporting from
    both would push two different trajectories through one trial's step keys --
    MedianPruner compares trials at matching keys, so that would corrupt every
    comparison it makes. (With a single seed, as fine-tuning uses, "first seed
    only" is simply every seed.)

    `patience` is a separate concern from `on_dump` and, unlike it, applies to
    BOTH seeds: it stops a run once that many consecutive evaluations pass with
    no relative improvement in val_nll (see DEFAULT_MIN_RELATIVE_IMPROVEMENT),
    never within the first `warmup_evals`. `None` (the default) disables it
    entirely -- every run trains to its full step budget. `on_converged(seed, eval_index, step)`, if given, fires once per
    seed that stops early, so a caller (worker.py) can record how much a trial
    actually saved.

    Scoring always runs over whatever dump exists once each seed's run ends,
    whether that is the full budget or an early stop -- the training script
    buffers its argmin-val_nll evaluation and writes it from a `finally`, so
    the dump is there either way and the objective value is still
    `early_stop(run_curve(...), "val_nll")`.

    Dumps are removed unless `keep_dumps`: Stage 4 re-runs the winning config
    on the reported seeds, so nothing downstream reads a tuning run's dump.

    `seeds`, `scripts` and `score` exist for the fine-tuning campaign, which
    shares this run loop and differs from probing in all three. Every one
    defaults to the probing value, so the probing call path is bit-identical.

      * `seeds` -- fine-tuning runs ONE seed per trial. Two is right for a
        40-second probe; at ~6 hours a trial it would double a ~880 A100-hour
        campaign.
      * `scripts` -- a different arm -> script table
        (finetune_worker.SCRIPTS).
      * `score` -- fine-tuning scores the minimum val_nll observed in the run's
        own evaluation stream, not `early_stop(run_curve(...))`. That is forced
        by evaluating val alone: `curves.run_curve` reads `test_record.meta`,
        so it raises on a run with no test dump, and producing one would mean
        forwarding 30k images at a step nothing can identify until the run is
        over. The property this module's docstring cares about survives:
        `early_stop(curve, "val_nll")` IS the argmin of val_nll over the finite
        rows of the curve, and the evaluation stream is that curve -- the same
        quantity, minus one fp16 dump round-trip.
    """
    if arm not in scripts:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(scripts)}")

    module = adapters.load_script_module(scripts[arm])
    uids = [f"{uid}_seed{seed}" for seed in seeds]

    try:
        for index, (seed, run_uid) in enumerate(zip(seeds, uids)):
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
                    warmup_evals=warmup_evals,
                    min_relative_improvement=min_relative_improvement,
                )
            try:
                module.main(args, on_dump=dump_cb)
            except EarlyConvergence as converged:
                if on_converged is not None:
                    on_converged(seed, converged.eval_index, converged.step)
        return score(dump_root, group_id, uids)
    finally:
        if not keep_dumps:
            for run_uid in uids:
                shutil.rmtree(Path(dump_root) / group_id / run_uid, ignore_errors=True)
