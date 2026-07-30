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
    """First step at which ALL THREE targets hold. Censored if never."""
    if not curve:
        return Crossing(step=None, censored=True, last_step=None)

    ordered = sorted(curve, key=lambda row: row["step"])
    for row in ordered:
        if _meets(row, targets):
            return Crossing(step=row["step"], censored=False, last_step=ordered[-1]["step"])
    return Crossing(step=None, censored=True, last_step=ordered[-1]["step"])


def crossings_per_seed(
    curves_by_seed: Mapping[int, Curve], targets: Targets
) -> dict[int, Crossing]:
    """Per-seed crossings. Deliberately not the crossing of the mean curve --
    averaging first would let one fast seed manufacture an early crossing that no
    individual run achieved."""
    return {seed: first_crossing(curve, targets) for seed, curve in curves_by_seed.items()}
