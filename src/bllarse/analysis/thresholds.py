"""The joint compute-to-threshold protocol.

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

import math

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


def targets_at_samples(curve: Curve, samples: int,
                       fraction: float = 1.0) -> tuple[Targets, Mapping[str, Any]]:
    """Targets from whatever this curve had reached by a FIXED sample budget.

    The other reference modes ask where a curve is best. This one asks where it
    is at a stated amount of training, which is the right question for "how long
    do the other arms need to reach what this one reaches in one epoch". It is
    also the reason this can be anchored on the method under test without the
    circularity rule 2 of the module docstring guards against: the reference is
    pinned by the BUDGET, not by the arm's own argmin, so it does not move when
    the arm gets better or worse. An arm that improves late cannot loosen its own bar.

    The nearest evaluation at or before `samples` is used -- never one after, so
    the reference is something the arm had actually achieved by that budget.

    Returns the targets and the row they came from, so a caller can report what
    the reference actually was rather than just the derived bound.
    """
    if not curve:
        raise ValueError("curve is empty; cannot derive targets")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    eligible = [row for row in curve if row["samples_seen"] <= samples]
    if not eligible:
        raise ValueError(
            f"no evaluation at or before {samples:,} samples; the curve starts "
            f"at {min(r['samples_seen'] for r in curve):,}")
    row = max(eligible, key=lambda r: r["samples_seen"])
    return Targets(
        acc=row["acc"] * fraction,
        nll=row["nll"] / fraction,
        brier=row["brier"] / fraction,
    ), row


# The points of a baseline curve a target can be read from; see make_targets.
REFERENCES = ("early_stop", "last")


def make_targets(baseline_curve: Curve, fraction: float = 0.95,
                 reference: str = "early_stop") -> Targets:
    """Targets as a fraction of the baseline's reference value.

    Accuracy is higher-is-better so the target is `fraction * reference`. NLL
    and Brier are lower-is-better, so the target is `reference / fraction` -- a
    proportionally looser bound, not a tighter one.

    `reference` picks WHICH point of the baseline curve that is:

    * "early_stop" (default) -- the val-NLL argmin, i.e. the point a
      practitioner would actually deploy. Selection is on val and reporting is
      on test, disjoint by construction, so it does not leak.
    * "last" -- the final evaluation. This assumes the curve is monotone,
      which it need not be: on dinov3_large fine-tuning the CE+TS baseline
      bottoms out around 70% of the budget and its endpoint
      sits 4.2% above its own best NLL, so "95% of the endpoint" spends most of
      its tolerance on the baseline's own late decay rather than on a
      deliberate margin.

    "best" is deliberately NOT offered. Taking the argmin of the TEST curve
    would set the target from the split the figure reports on.
    """
    if not baseline_curve:
        raise ValueError("baseline_curve is empty; cannot derive targets")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if reference not in REFERENCES:
        raise ValueError(
            f"reference must be one of {REFERENCES}, got {reference!r}")

    if reference == "early_stop":
        converged = early_stop(baseline_curve).row
    else:
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

    `samples_seen` is carried alongside `step` because it is the compute axis
    the sample-efficiency figure actually needs, and it is not always a fixed
    multiple of step (batch size can differ per run).
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


def first_crossing_metric(curve: Curve, metric: str, target: float,
                          lower_is_better: bool = True) -> Crossing:
    """First step at which ONE metric reaches `target`. Censored if never.

    The joint acc/NLL/Brier rule (`first_crossing`) answers "when is this run
    as good as the baseline in every respect". This answers "when does it reach
    a given value of one quantity", which is the right question when the
    quantity is the SELECTION criterion -- val_nll -- and accuracy is something
    you then report at that point rather than something you gate on.

    Gating on val_nll alone is not a weaker joint test; it is a different one.
    A run can cross here while still behind on accuracy, and the caller is
    expected to say what accuracy it had.
    """
    if not curve:
        return Crossing(step=None, samples_seen=None, censored=True,
                        last_step=None)
    ordered = sorted((r for r in curve if metric in r),
                     key=lambda row: row["step"])
    if not ordered:
        raise ValueError(f"metric {metric!r} is absent from every row")
    for row in ordered:
        value = row[metric]
        if value <= target if lower_is_better else value >= target:
            return Crossing(step=row["step"], samples_seen=row["samples_seen"],
                            censored=False, last_step=ordered[-1]["step"])
    return Crossing(step=None, samples_seen=None, censored=True,
                    last_step=ordered[-1]["step"])


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
    is what both probing scripts do, keeping only their argmin -- because
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
    # Non-finite points are dropped before the argmin rather than ranked with
    # it. `min` keeps its first candidate unless a later one compares strictly
    # less, and every comparison against nan is False -- so a single nan row,
    # if it sorts first, WINS the argmin and the run scores nan, with every good
    # value afterwards unable to displace it. A point the model could not
    # evaluate is not a candidate optimum.
    finite = [row for row in ordered if math.isfinite(row[criterion])]
    if not finite:
        raise ValueError(
            f"every one of {len(ordered)} rows has a non-finite {criterion!r}; "
            f"this run produced no usable measurement and must not be scored "
            f"as though it did."
        )
    best = min(finite, key=lambda row: row[criterion])
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
