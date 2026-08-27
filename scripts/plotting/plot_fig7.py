#!/usr/bin/env python3
"""Fig 7 -- compute and memory to reach the joint threshold.

    python scripts/plotting/plot_fig7.py --scores /ptmp/$USER/large_all.json

The claim is *faster to calibrated*, not just faster to accurate, so the x-axis
is cost and the bar is the cost of reaching a target that includes calibration.

Threshold protocol (results-plan 0.4, implemented in analysis.thresholds, fixed
before any of these numbers were looked at):

  - the target is JOINT on accuracy, NLL and Brier -- the first step where all
    three are at or better than target, never the first step where one is;
  - targets come from the STRONGEST BASELINE's converged value, never from
    IBProbit. The baseline here is whichever of CE and CE+TS converges to the
    lower NLL, chosen by that rule and printed, not hand-picked;
  - crossings are PER SEED. Crossing the seed-mean curve would let one fast
    seed manufacture a crossing no individual run achieved;
  - a run that never reaches the target is RIGHT-CENSORED and drawn as such --
    an open bar with an arrow at its budget -- never dropped and never
    extrapolated.

There is NO FLOPs panel, and its absence is deliberate. results-plan v2 asks
for one, derived as `samples-to-threshold x per-step cost from bench_step.py`.
But `bench_step.step_flops` is a BACKBONE model -- depth, width, token count --
and is therefore identical across arms by construction. Multiplying every arm
by one constant makes that panel a rescaled copy of samples-to-threshold with
the same ranking, presented as though it measured compute.

The arms do not cost the same. VBLL carries a covariance head and IBProbit runs
128 CAVI iterations per step; neither appears in a trunk-only FLOP count. The
measured wall clock says how much that matters: over the same 78-eval budget on
dinov3_large, CE takes 10.2-10.3 h, IBProbit 10.7-10.9 h and VBLL 12.7-12.9 h.
A model that assigns all three the same per-sample cost is wrong by ~25% on
VBLL, and wrong in the direction that flatters it.

Reinstating the panel needs a per-arm FLOP model -- head cost and, for
IBProbit, the Cholesky term at its own iteration count -- not a shared scalar.
`panel_cost` is general over a per-(arm, seed) cost function and will take one.

Both panels come from separate measurements -- the dump-mtime timeline and
`probe_micro_batch{,_torch}.py` -- passed in as JSON. A panel with no input
draws as empty and says so; it is never filled in from the trajectory, which
cannot support either number.

Peak memory is a property of (arm, model, batch), not of the trajectory: one
measurement per arm is the honest object and panel c carries no seed spread.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bllarse.analysis import final_json, thresholds  # noqa: E402

BASELINES = [("ce", "id"), ("ce", "ts")]


def _mean_curve(data, uids, pass_key="id"):
    """Seed-mean rows for one arm, on the x values any seed reached.

    The TARGETS come from this, so it has to be the mean for every baseline
    variant alike. Taking one arbitrary seed for the TS baseline and the mean
    for the raw one -- which an earlier version did -- makes the target depend
    on which run happened to be first in a dict.
    """
    per_metric = {}
    for metric in ("acc", "nll", "brier"):
        per_seed = {seed: [(x, y) for x, y in zip(
            data["runs"][uid][pass_key]["samples_seen"],
            data["runs"][uid][pass_key][metric])] for seed, uid in uids.items()}
        xs, mean, _lo, _hi, _n = final_json.band(per_seed)
        per_metric[metric] = dict(zip(xs, mean))
    xs = sorted(set.intersection(*(set(v) for v in per_metric.values())))
    return [{"step": x, "samples_seen": x,
             **{m: per_metric[m][x] for m in per_metric}} for x in xs]


def pick_targets(data, fraction):
    """Targets from the strongest of the CE variants, by converged NLL."""
    best = None
    for arm, pass_key in BASELINES:
        by_arm = final_json.uids_by_arm(data, pass_key)
        uids = by_arm.get(arm)
        if not uids:
            continue
        curve = _mean_curve(data, uids, pass_key)
        if not curve:
            continue
        converged = max(curve, key=lambda r: r["step"])
        if best is None or converged["nll"] < best[0]["nll"]:
            best = (converged, f"{arm}/{pass_key}", curve)
    if best is None:
        raise SystemExit("[fig7] no CE baseline in the scored JSON")
    converged, name, curve = best
    targets = thresholds.make_targets(curve, fraction=fraction)
    print(f"[fig7] baseline {name}: converged acc={converged['acc']:.4f} "
          f"nll={converged['nll']:.4f} brier={converged['brier']:.4f}")
    print(f"[fig7] targets at {fraction:g}: acc>={targets.acc:.4f} "
          f"nll<={targets.nll:.4f} brier<={targets.brier:.4f}")
    return targets, name


def crossings(data, targets):
    """[(label, colour, arm, {seed: Crossing})] in ARMS order."""
    entries = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        curves = {seed: final_json.rows(data, uid, pass_key)
                  for seed, uid in uids.items()}
        entries.append((label, colour, arm,
                        thresholds.crossings_per_seed(curves, targets)))
    return entries


def panel_cost(ax, entries, cost, title, ylabel):
    """Bars of mean cost to threshold, per-seed dots, censored runs as bounds.

    `cost(arm, seed, crossing) -> (value, is_bound)`. A censored run has no
    crossing, so its value is the cost of its WHOLE budget and `is_bound` says
    to draw it as a lower bound rather than as a measurement. Returning None
    drops the seed, for panels whose measurement is missing for that run.
    """
    costed = []
    for label, colour, arm, cross in entries:
        reached, censored = [], []
        for seed, c in cross.items():
            got = cost(arm, seed, c)
            if got is None:
                continue
            value, is_bound = got
            (censored if is_bound else reached).append(value)
        costed.append((label, colour, reached, censored, len(cross)))

    finite = [v for _, _, r, c, _ in costed for v in (r + c)]
    if not finite:
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "no measurement for these runs", ha="center",
                va="center", fontsize=8, color="#777", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    # Bars on a log axis are drawn from the axis floor, so the floor has to be
    # set from the data. Left to matplotlib it lands at a decade boundary far
    # below the smallest bar and every bar reads as full height.
    floor = min(finite) / 2.0

    for i, (label, colour, reached, censored, n_seeds) in enumerate(costed):
        if reached:
            # height is TOP MINUS BOTTOM. Passing the value itself with a
            # non-zero `bottom` draws the bar to floor+value, which on a log
            # axis put every bar ~1.5x above its own per-seed dots.
            mean = sum(reached) / len(reached)
            ax.bar(i, mean - floor, width=0.6, bottom=floor,
                   color=colour, alpha=0.75, zorder=2)
            ax.plot([i] * len(reached), reached, "o", ms=5, color="k",
                    mfc="white", mew=1.0, zorder=4)
        # A censored run is not a fast run and not a missing one: it is a lower
        # bound at its own budget. Open bar, arrow off the top, and a count.
        # The arrow and the count are placed in POINTS off the bar, never in
        # data units -- a multiplicative offset on a log axis rescales the
        # whole panel and squashes the bars it was meant to annotate.
        for x in censored:
            ax.bar(i, x - floor, width=0.6, bottom=floor, facecolor="none",
                   edgecolor=colour, hatch="///", lw=1.2, zorder=3)
            ax.annotate("", xy=(i, x), xytext=(0, 16),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=colour, lw=1.2))
        if censored:
            ax.annotate(f"{len(censored)}/{n_seeds} censored",
                        xy=(i, max(censored)), xytext=(0, 20),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=colour)

    ax.set_xticks(range(len(costed)))
    ax.set_xticklabels([e[0] for e in costed], rotation=20, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    # Headroom for the censored arrow and its count, which sit ~20 points above
    # the tallest bar. Without it matplotlib clips the label at the axes edge.
    any_censored = any(c for _, _, _, c, _ in costed)
    ax.set_ylim(floor, max(finite) * (1.9 if any_censored else 1.25))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)


def panel_memory(ax, values, title, ylabel, empty_note):
    """One measured scalar per arm -- peak memory, which has no seed spread."""
    if not values:
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, empty_note, ha="center", va="center", fontsize=8,
                color="#777", transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    # Peak memory is a property of the TRAINING run, and TS is post hoc -- so
    # the CE entry is keyed on the arm, not on which pass is plotted elsewhere.
    seen, labelled = set(), []
    for (arm, _pass), label, colour, _m in final_json.ARMS:
        if arm in seen or arm not in values:
            continue
        seen.add(arm)
        peak, micro = values[arm]
        labelled.append((f"{label.split(' (')[0]}\nmicro {micro}" if micro
                         else label, colour, peak))
    ax.bar(range(len(labelled)), [v for _, _, v in labelled], width=0.6,
           color=[c for _, c, _ in labelled], alpha=0.75, zorder=2)
    for i, (_lab, _c, v) in enumerate(labelled):
        ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8)
    ax.set_ylim(0, max(v for _, _, v in labelled) * 1.15)
    ax.set_xticks(range(len(labelled)))
    ax.set_xticklabels([lab for lab, _, _ in labelled], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)


def _elapsed_at(timeline, samples):
    """Wall-clock seconds at `samples`, from a measured [(samples, s)] list.

    Interpolated between the two bracketing eval points rather than snapped to
    one: a crossing lands at an eval point by construction, but the LAST point
    of a censored run is asked for its own budget and must return it exactly.
    """
    if not timeline:
        return None
    for x, secs in timeline:
        if x >= samples:
            return secs
    return timeline[-1][1]


def make_figure(data, out_path, *, targets, baseline, fraction,
                wallclock, memory):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    entries = crossings(data, targets)

    # {(arm, seed): [(samples, elapsed_s)]} -- a MEASURED timeline per run, so
    # panel b reads the clock at the crossing instead of assuming throughput is
    # flat across a run that pays compilation once and evaluation every dump.
    timelines = {}
    for run in (wallclock.get("runs") or {}).values():
        timelines[(run["arm"], int(run["seed"]))] = run["elapsed_s"]

    def wall_cost(arm, seed, c):
        timeline = timelines.get((arm, seed))
        if not timeline:
            return None
        if c.censored:
            return (_elapsed_at(timeline, c.last_step), True)
        return (_elapsed_at(timeline, c.samples_seen), False)

    panel_cost(axes[0], entries, wall_cost, "a — wall clock to threshold",
               "seconds to threshold (A100)")
    panel_memory(axes[1], memory, "b — peak memory (per arm, not per seed)",
                 "GiB", "no --memory given;\nrun probe_micro_batch.py")
    fig.suptitle(
        f"Fig 7 — cost to a JOINT acc/NLL/Brier target at "
        f"{fraction:g}x the {baseline} baseline's converged value\n"
        + final_json.provenance(data), fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig7] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="fig7_compute_to_threshold")
    p.add_argument("--fraction", type=float, default=0.95)
    p.add_argument("--wallclock", default=None,
                   help="scripts/analysis/wallclock_from_dumps.py output: a "
                        "measured (samples, elapsed) timeline per run.")
    p.add_argument("--memory", nargs="*", default=None,
                   help="probe_micro_batch{,_torch}.py --json-out files, one "
                        "per arm.")
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig7] {final_json.provenance(data)}")
    targets, baseline = pick_targets(data, args.fraction)
    wallclock = json.loads(Path(args.wallclock).read_text()) if args.wallclock else {}
    memory = {}
    for path in args.memory or ():
        probe = json.loads(Path(path).read_text())
        if probe.get("model") != data.get("model"):
            # Peak memory measured on a different backbone answers a different
            # question, and silently mixing them would make panel c a
            # comparison of model sizes wearing arm labels.
            raise SystemExit(
                f"[fig7] {path} is model={probe.get('model')!r} but the scores "
                f"are model={data.get('model')!r}")
        # The micro batch rides along because the arms do NOT share one --
        # vbll runs 64 where ce and ibprobit run 32 -- and a memory bar without
        # it invites the reader to attribute a batch difference to the method.
        # Batch is matched at the EFFECTIVE level (512) via accumulation.
        memory[probe["arm"]] = (probe["peak_gib"], probe.get("micro_batch"))

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    targets=targets, baseline=baseline, fraction=args.fraction,
                    wallclock=wallclock, memory=memory)


if __name__ == "__main__":
    main()
