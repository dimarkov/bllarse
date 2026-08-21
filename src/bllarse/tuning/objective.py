"""One Optuna trial: run a config on both tuning seeds, return its score.

The score IS analysis.thresholds.early_stop, not a reimplementation of it, so
the tuning criterion and the reported stopping rule cannot drift apart.
Selection is on val, reporting is on test, and bllarse.splits makes the two
disjoint by construction -- so this does not leak. The OOD split never enters
tuning at all: for Fig 2, studies are keyed on the ID dataset only, since both
scripts load OOD features after the training loop purely for a final dump.
"""

import collections
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
# forever, so "any decrease counts" never plateaus. A RELATIVE threshold is the
# only form that serves both ends of that range.
#
# 1e-4, lowered from 1e-3, because 1e-3 was calibrated against the probing
# campaign and is too coarse for fine-tuning. Measured on the fine-tuning
# studies: patience fired on 57 of 67 completed trials, cutting them at a
# median evaluation 88 of 122 with their argmin 20 evaluations earlier -- i.e.
# the instant the window elapsed. Trials that patience did NOT stop put their
# argmin at 91-100% of the budget, so those curves were still improving where
# the stopped ones were being called converged.
#
# The arithmetic that decides it. Patience needs `patience` CONSECUTIVE
# evaluations with no improvement, so on a curve falling at rate r per
# evaluation `best` refreshes every ceil(theta * best / r) of them and patience
# can only fire when
#
#     r < theta * best / patience
#
# -- a factor of `patience` smaller than the naive theta * best. At val_nll
# ~= 0.44 with patience 20 that critical rate is 2.2e-5 per evaluation at
# theta = 1e-3, so a run improving just under it is cut having given up as much
# as 1.1e-3 over the remaining 50 evaluations. THE ARMS ARE SEPARATED BY 0.0026
# (ibprobit_noeps 0.4415 against ce 0.4441), so patience could silently discard
# ~40% of the effect being measured, from whichever arm happened to sit on the
# wrong side of it.
#
# At 1e-4 the critical rate is 2.2e-6 and the most that can be discarded is
# ~1.1e-4, or 4% of the gap -- still above the fourth-decimal drift the
# constant exists to ignore.
#
# Cost: fewer early stops, so completed trials run closer to the full budget.
# Pruning is unaffected -- MedianPruner kills 73-89% of trials and that is
# where the campaign's compute saving actually comes from.
DEFAULT_MIN_RELATIVE_IMPROVEMENT = 1e-4


# Evaluations averaged over by the best-window score, as a FIXED COUNT.
#
# THE POINT. The default objective is `early_stop(curve, "val_nll")` -- the
# single-point MINIMUM of val_nll. On a curve that never settles that is the
# deepest dip of the noise, not the level the method reaches, so the search is
# rewarded for configurations that oscillate hard enough to dip low. Measured on
# probing/oxford_pets/dinov3_huge: the tuned lr 0.0182 has a late-phase sd 0.53x
# its own median, and the campaign chose it over lr 0.0024, which is 13x more
# stable AND 0.032 better on test NLL. Re-scored offline, ANY window of 2 or
# more evaluations displaces that unstable pick.
#
# WHY A COUNT, and not a fraction of the grid. An earlier version used 10% of
# `evals_planned` and argued a constant could not serve grids spanning 117 to
# 30,000 points. That argument was about PLANNED grids. Patience truncates runs
# long before their budget, so the trajectories that actually exist are far
# shorter and far more alike -- 71 to 507 evaluations on probing/dtd. A window
# sized off the plan therefore OVERRUNS the data: dtd's best config planned
# 3,750 evaluations, ran 79, and got a 375-evaluation window, which
# `best_window_mean` clamps to the trajectory length. The objective silently
# became "mean of the whole run", which charges a config for every point after
# its minimum. That cost dtd its genuinely best configuration -- one that
# reaches val_nll 0.4110 against the eventual winner's 0.4256, and is 0.02-0.035
# better on test at every backbone -- by ranking it 88th of 100. At a fixed
# width of 5 the same config ranks 6th, inside a top-10 spread of 0.001.
#
# 5, measured: width 1 is the unsmoothed minimum and keeps the oxford_pets
# pathology; 2 already removes it; 2-30 all behave alike on both cases; 50
# starts eroding real minima again. 5 sits in the middle of the flat region and
# can never exceed even the shortest trajectory seen (71).
DEFAULT_SMOOTH_WINDOW = 5


def best_window_mean(values: Sequence[float], window: int) -> float:
    """Lowest mean of any `window` consecutive evaluations. Lower is better.

    The stable counterpart to `min(values)`: a lone dip cannot carry a window,
    so a configuration must actually HOLD a low val_nll to score well.

    A window containing any non-finite value scores +inf rather than being
    skipped. A run that diverges anywhere inside a window has not held that
    level, and dropping the point would let a config launder a NaN excursion by
    having good neighbours. Implemented on cumulative sums with the non-finite
    positions counted separately, because inf inside a running sum turns into
    nan on the first subtraction.
    """
    import numpy as np

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("no evaluations to score")
    width = int(min(max(window, 1), array.size))

    finite = np.isfinite(array)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, array, 0.0))])
    bad = np.concatenate([[0], np.cumsum(~finite)])
    totals = sums[width:] - sums[:-width]
    spoiled = bad[width:] - bad[:-width]
    means = np.where(spoiled > 0, np.inf, totals / width)
    return float(means.min())




class RunningWindow:
    """The best-window score of a stream PREFIX, updated in O(1) per point.

    Exists so the PRUNER can be shown the same kind of statistic the trial is
    scored on. `best_window_mean` rescans the whole stream, which is O(n) per
    call and so O(n^2) over a run -- unusable when a trial reports up to 30,000
    evaluations.

    WHY THE PRUNER NEEDS THIS AT ALL. MedianPruner kills a trial by comparing
    what it reports against what other trials reported at the same eval index.
    Reporting the raw instantaneous val_nll while SCORING the windowed mean
    means the pruner selects on a different quantity from the objective -- and
    specifically on the dip-prone one the windowed objective exists to
    neutralise: a config holding a steady, mediocre value is killed, while a
    spiky one that dips low survives.

    Measured on the `smoothed` campaign, which ran with that mismatch: 11 of 21
    studies took their winner from the first 20 trials -- MedianPruner's
    un-prunable startup window -- and the median winner sat at 7% of the search
    against 42% under the previous campaign. Prune rates reached 89%. Roughly
    half the campaign was a best-of-20-random-draws, not a search.

    `best` tracks COMPLETE windows only, so at the end of a stream it equals
    `best_window_mean` exactly. Until the first full window exists, `push`
    returns the CURRENT prefix mean rather than the minimum over prefixes --
    deliberately, and the distinction matters: a running minimum over prefixes
    latches onto a single lucky first evaluation and holds that number for the
    rest of the run, which is the dip-sensitivity this class exists to remove,
    reintroduced at the start of every trial. The prefix mean cannot latch.

    Both forms are deterministic functions of the stream prefix, which is all
    the pruner requires -- trials are only ever compared at equal eval index.
    The OBJECTIVE remains `best_window_mean` over the finished stream; this is
    the pruner's view of it, not a second definition of the score.
    """

    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self.width = width
        self._buf: collections.deque[float] = collections.deque()
        self._total = 0.0
        self._nonfinite = 0
        self.best = math.inf

    def push(self, value: float) -> float:
        """Add one evaluation; return the score to report for this prefix."""
        finite = math.isfinite(value)
        self._buf.append(value)
        if finite:
            self._total += value
        else:
            self._nonfinite += 1
        if len(self._buf) > self.width:
            old = self._buf.popleft()
            if math.isfinite(old):
                self._total -= old
            else:
                self._nonfinite -= 1

        mean = math.inf if self._nonfinite else self._total / len(self._buf)
        if len(self._buf) < self.width:
            # No complete window yet: report where the run currently sits, and
            # do NOT record it. Recording would let evaluation 1 set a floor
            # nothing later could raise.
            return mean
        # A window spanning a non-finite value scores +inf, matching
        # `best_window_mean` -- a run that diverged has not held that level.
        self.best = min(self.best, mean)
        return self.best


def _recording_on_dump(on_dump: OnDump | None, stream: list[float]) -> OnDump:
    """Capture val_nll for every evaluation, then delegate.

    Recording FIRST so the stream survives whatever the inner hook raises --
    TrialPruned from the pruner, EarlyConvergence from patience. Applied to
    every seed, unlike `on_dump` itself, which is deliberately first-seed-only.
    """

    def wrapped(
        eval_index: int, step: int, samples_seen: int, scalars: Mapping[str, float]
    ) -> None:
        stream.append(float(scalars["val_nll"]))
        if on_dump is not None:
            on_dump(eval_index, step, samples_seen, scalars)

    return wrapped


def clear_jit_caches() -> None:
    """Drop XLA's compilation cache. Called between a trial's runs.

    BOTH arms reach JAX, which is the part that is easy to miss: the VBLL arm
    trains in PyTorch but scores through `bllarse.metrics`, whose
    `fit_temperature` is a jitted `lax.scan` invoked at every reported
    evaluation. Trials vary `batch_size`, so each one compiles kernels for
    shapes the last one did not use and the cache grows monotonically over a
    worker's lifetime -- nothing evicts it, because a worker is one process
    running hundreds of trials.

    Measured on the win5 campaign: all four `tune_vbll_flowers102` workers died
    after ~177 trials with `LLVM ERROR: Unable to allocate section memory` out
    of the CPU JIT, taking the study to 99 completed trials of the 100 it
    needed; `ibprobit_noeps_oxford_pets` lost 39 trials the same way. The
    failure is HOST memory, not GPU, so it is invisible in nvidia-smi.

    Best-effort by design: a jax that is missing or too old to expose
    `clear_caches` must not convert cleanup into a trial failure. The cost of
    clearing is recompilation on the next trial, which trials with differing
    batch sizes would pay anyway.
    """
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


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
    smooth_window: int | None = None,
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

    streams: list[list[float]] = []
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
            # Wrapped ONLY when smoothing is on, so the default call path stays
            # bit-identical -- the fine-tuning campaign shares this function and
            # must not acquire a per-evaluation side effect it did not have.
            if smooth_window is not None:
                stream: list[float] = []
                streams.append(stream)
                seed_on_dump = _recording_on_dump(seed_on_dump, stream)
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
            finally:
                # Per SEED, not per trial: a two-seed trial otherwise carries
                # the first seed's compiled kernels through the second.
                clear_jit_caches()
        if smooth_window is not None:
            # Mean over seeds of each seed's own best window, matching
            # `score_dumps`'s mean-over-seeds. Scored from the evaluation stream
            # rather than the dumps because the dump holds ONE point -- the
            # argmin -- which is exactly the statistic being replaced.
            per_seed = [best_window_mean(s, smooth_window) for s in streams if s]
            if not per_seed:
                raise ValueError(
                    "smoothed scoring was requested but no evaluation ran; the "
                    "step budget is shorter than one eval interval"
                )
            return float(sum(per_seed) / len(per_seed))
        return score(dump_root, group_id, uids)
    finally:
        if not keep_dumps:
            for run_uid in uids:
                shutil.rmtree(Path(dump_root) / group_id / run_uid, ignore_errors=True)
