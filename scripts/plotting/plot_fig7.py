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

Panel a is derived from the scored curves alone. Panels b (wall clock) and c
(peak memory) come from separate measurements -- MLflow run durations and
`scripts/finetune/probe_micro_batch.py` -- passed in as JSON. A panel with no
input draws as empty and says so; it is never filled in from the trajectory,
which cannot support either number.

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


def _mean_curve(data, uids):
    """Seed-mean rows for one arm, on the x values any seed reached."""
    per_metric = {}
    for metric in ("acc", "nll", "brier"):
        per_seed = {seed: [(x, y) for x, y in zip(
            data["runs"][uid]["id"]["samples_seen"],
            data["runs"][uid]["id"][metric])] for seed, uid in uids.items()}
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
        curve = _mean_curve(data, uids) if pass_key == "id" else [
            {"step": x, "samples_seen": x,
             "acc": data["runs"][uid][pass_key]["acc"][i],
             "nll": data["runs"][uid][pass_key]["nll"][i],
             "brier": data["runs"][uid][pass_key]["brier"][i]}
            for uid in [next(iter(uids.values()))]
            for i, x in enumerate(data["runs"][uid][pass_key]["samples_seen"])]
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


def panel_a(ax, data, targets, per_sample_flops):
    entries = []
    for (arm, pass_key), label, colour, _marker in final_json.ARMS:
        uids = final_json.uids_by_arm(data, pass_key).get(arm)
        if not uids:
            continue
        curves = {seed: final_json.rows(data, uid, pass_key)
                  for seed, uid in uids.items()}
        crossings = thresholds.crossings_per_seed(curves, targets)
        entries.append((label, colour, crossings))

    scale = per_sample_flops or 1.0
    # Bars on a log axis are drawn from the axis floor, so the floor has to be
    # set from the data. Left to matplotlib it lands at a decade boundary far
    # below the smallest bar and every bar reads as full height.
    finite = [c.samples_seen * scale for _, _, cr in entries
              for c in cr.values() if not c.censored]
    finite += [c.last_step * scale for _, _, cr in entries
               for c in cr.values() if c.censored]
    floor = min(finite) / 2.0 if finite else 1.0

    for i, (label, colour, crossings) in enumerate(entries):
        reached = [c.samples_seen * scale for c in crossings.values()
                   if not c.censored]
        censored = [c.last_step * scale for c in crossings.values() if c.censored]
        if reached:
            ax.bar(i, sum(reached) / len(reached), width=0.6, bottom=floor,
                   color=colour, alpha=0.75, zorder=2)
            ax.plot([i] * len(reached), reached, "o", ms=5, color="k",
                    mfc="white", mew=1.0, zorder=4)
        # A censored run is not a fast run and not a missing one: it is a lower
        # bound at its own budget. Open bar, arrow off the top, and a count.
        # The arrow and the count are placed in POINTS off the bar, never in
        # data units -- a multiplicative offset on a log axis rescales the
        # whole panel and squashes the bars it was meant to annotate.
        for x in censored:
            ax.bar(i, x, width=0.6, bottom=floor, facecolor="none",
                   edgecolor=colour, hatch="///", lw=1.2, zorder=3)
            ax.annotate("", xy=(i, x), xytext=(0, 16),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=colour, lw=1.2))
        if censored:
            ax.annotate(f"{len(censored)}/{len(crossings)} censored",
                        xy=(i, max(censored)), xytext=(0, 20),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=colour)

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], rotation=20, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    # Headroom for the censored arrow and its count, which sit ~20 points above
    # the tallest bar. Without it matplotlib clips the label at the axes edge.
    any_censored = any(c.censored for _, _, cr in entries for c in cr.values())
    ax.set_ylim(floor, max(finite) * (1.9 if any_censored else 1.25))
    ax.set_ylabel("FLOPs to threshold" if per_sample_flops
                  else "samples to threshold")
    ax.set_title("a — cost to joint threshold", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    return entries


def panel_bar(ax, values, title, ylabel, empty_note):
    """One measured scalar per arm. Shared by panels b and c."""
    if not values:
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, empty_note, ha="center", va="center", fontsize=8,
                color="#777", transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    labelled = [(label, colour, values[arm])
                for (arm, pass_key), label, colour, _m in final_json.ARMS
                if pass_key == "id" and arm in values]
    ax.bar(range(len(labelled)), [v for _, _, v in labelled], width=0.6,
           color=[c for _, c, _ in labelled], alpha=0.75, zorder=2)
    ax.set_xticks(range(len(labelled)))
    ax.set_xticklabels([lab for lab, _, _ in labelled], rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.set_axisbelow(True)


def make_figure(data, out_path, *, targets, baseline, fraction,
                per_sample_flops, wallclock, memory):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    panel_a(axes[0], data, targets, per_sample_flops)
    panel_bar(axes[1], wallclock, "b — wall clock to threshold",
              "seconds", "no --wallclock JSON given;\nMLflow run durations "
              "not read")
    panel_bar(axes[2], memory, "c — peak memory (per arm, not per seed)",
              "GiB", "no --memory JSON given;\nrun probe_micro_batch.py")
    fig.suptitle(
        f"Fig 7 — compute to a JOINT acc/NLL/Brier target at "
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
    p.add_argument("--flops-per-sample", type=float, default=None,
                   help="Forward+backward FLOPs for ONE sample, from "
                        "scripts/finetune/bench_step.py. Without it panel a is "
                        "in samples, which is the same ranking whenever the "
                        "arms share a backbone -- the arms differ only in the "
                        "head, so the per-sample cost differs between them and "
                        "a single scalar is an approximation the caption must "
                        "state.")
    p.add_argument("--wallclock", default=None,
                   help='JSON {"ce": seconds, ...} from MLflow run durations.')
    p.add_argument("--memory", default=None,
                   help='JSON {"ce": GiB, ...} from probe_micro_batch.py.')
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig7] {final_json.provenance(data)}")
    targets, baseline = pick_targets(data, args.fraction)
    wallclock = json.loads(Path(args.wallclock).read_text()) if args.wallclock else {}
    memory = json.loads(Path(args.memory).read_text()) if args.memory else {}

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    targets=targets, baseline=baseline, fraction=args.fraction,
                    per_sample_flops=args.flops_per_sample,
                    wallclock=wallclock, memory=memory)


if __name__ == "__main__":
    main()
