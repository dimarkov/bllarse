"""The joint compute-to-threshold protocol (results-plan.md 0.4).

Fixed before looking at any results, because the researcher degrees of freedom
here are large. Four rules, all load-bearing:

1. The target is joint over accuracy, NLL and Brier. A method that reaches
   accuracy early with bad calibration must not score as fast -- that is the
   whole point of the figure.
2. Targets come from the strongest *baseline's* converged value, never from
   IBProbit.
3. Runs that never reach threshold are right-censored and reported as such,
   never dropped and never extrapolated.
4. Crossings are per seed, not the crossing of the seed-mean curve.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Curve = Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class Targets:
    acc: float
    nll: float
    brier: float


@dataclass(frozen=True)
class Crossing:
    step: int | None
    samples_seen: int | None
    censored: bool
    last_step: int | None


def make_targets(baseline_curve: Curve, fraction: float = 0.95) -> Targets:
    """Targets as a fraction of the baseline's converged value.

    Accuracy is higher-is-better so the target is `fraction * converged`. NLL and
    Brier are lower-is-better, so the target is `converged / fraction` -- a
    proportionally looser bound, not a tighter one.
    """
    if not baseline_curve:
        raise ValueError("baseline_curve is empty; cannot derive targets")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    converged = max(baseline_curve, key=lambda row: row["step"])
    return Targets(
        acc=converged["acc"] * fraction,
        nll=converged["nll"] / fraction,
        brier=converged["brier"] / fraction,
    )


def _meets(row: Mapping[str, Any], targets: Targets) -> bool:
    return (
        row["acc"] >= targets.acc
        and row["nll"] <= targets.nll
        and row["brier"] <= targets.brier
    )


def first_crossing(curve: Curve, targets: Targets) -> Crossing:
    """First step at which ALL THREE targets hold. Censored if never.

    `samples_seen` is carried alongside `step` because step is an epoch --
    samples_seen is the compute axis the sample-efficiency figure actually
    needs, and it is not always a fixed multiple of step (dataset size can
    differ per run).
    """
    if not curve:
        return Crossing(step=None, samples_seen=None, censored=True, last_step=None)

    ordered = sorted(curve, key=lambda row: row["step"])
    for row in ordered:
        if _meets(row, targets):
            return Crossing(
                step=row["step"],
                samples_seen=row["samples_seen"],
                censored=False,
                last_step=ordered[-1]["step"],
            )
    return Crossing(
        step=None, samples_seen=None, censored=True, last_step=ordered[-1]["step"]
    )


def crossings_per_seed(
    curves_by_seed: Mapping[int, Curve], targets: Targets
) -> dict[int, Crossing]:
    """Per-seed crossings. Deliberately not the crossing of the mean curve --
    averaging first would let one fast seed manufacture an early crossing that no
    individual run achieved."""
    return {seed: first_crossing(curve, targets) for seed, curve in curves_by_seed.items()}


@dataclass(frozen=True)
class StoppingPoint:
    step: int
    samples_seen: int
    criterion: str
    value: float
    row: Mapping[str, Any]
    at_last_step: bool


def early_stop(curve: Curve, criterion: str = "val_nll") -> StoppingPoint:
    """The dumped point that minimises `criterion` on the held-out val split.

    Early stopping is applied POST HOC rather than during training. The dumps
    hold the whole trajectory, so the stopping rule is a function of the curve,
    not of the run: it can be changed, or swapped for a different criterion,
    without re-running a single job, and -- more importantly -- the identical
    rule is provably applied to every arm, which a per-arm patience heuristic
    inside each training script could not guarantee.

    Selection is on val, reporting is on test, and the two are disjoint by
    construction (bllarse.splits), so this does not leak.

    `at_last_step` flags a minimum at the final EVALUATION: the curve was still
    improving when the budget ran out, so this is a truncation rather than a
    converged optimum, and the run is not evidence about where the optimum is.
    `tuning.budgets` reads it to decide a step budget is too small.

    It is taken from the dump's own `at_last_eval` metadata when present, and
    only inferred from the curve ("is the argmin the last row?") when it is
    not. The inference is unsound for a run that writes a single dump -- which
    is what both probing scripts now do, keeping only their argmin -- because
    the argmin is then trivially also the last row, and every run would report
    as truncated. Only the training loop knows how many evaluations there were.
    """
    if not curve:
        raise ValueError("curve is empty; nothing to select")
    missing = [row["step"] for row in curve if criterion not in row]
    if missing:
        raise ValueError(
            f"criterion {criterion!r} is absent from {len(missing)} of "
            f"{len(curve)} rows (first at step {missing[0]}). run_curve only "
            f"emits val_* metrics when a val dump exists at that step."
        )

    ordered = sorted(curve, key=lambda row: row["step"])
    best = min(ordered, key=lambda row: row[criterion])
    return StoppingPoint(
        step=best["step"],
        samples_seen=best["samples_seen"],
        criterion=criterion,
        value=best[criterion],
        row=best,
        at_last_step=(
            best["at_last_eval"]
            if best.get("at_last_eval") is not None
            else best["step"] == ordered[-1]["step"]
        ),
    )


def early_stop_per_seed(
    curves: Mapping[int, Curve], criterion: str = "val_nll"
) -> dict[int, StoppingPoint]:
    """`early_stop` per seed. Stopping is per run, exactly as it would have been
    live -- taking it on the seed-mean curve would understate the variance the
    error bars are supposed to show."""
    return {seed: early_stop(curve, criterion) for seed, curve in curves.items()}
