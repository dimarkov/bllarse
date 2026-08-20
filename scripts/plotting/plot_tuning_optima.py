#!/usr/bin/env python3
"""Validation NLL and Brier over training, for each arm's TUNED-OPTIMAL config.

    python scripts/plotting/plot_tuning_optima.py
    python scripts/plotting/plot_tuning_optima.py --from-json scripts/figures/tuning_optima.json

This is NOT Fig 5, and the distinction matters enough to state twice. Fig 5 is
the reported sample-efficiency figure: dinov3_huge, 4 epochs, 5 seeds, bands
over seeds. This is the TUNING campaign -- dinov3_large, 1M samples, ONE seed
(finetune_space.TUNING_SEED = 200), one trial per arm. It exists to show what
the search converged on while the final campaign has not run, and every number
on it is a single draw with no error bar.

Read it for the SHAPE, not the endpoints. The arms separate by roughly 7x in
NLL at the first evaluation and converge to within ~0.03 by the end, which is
the early-calibration behaviour Fig 5 will measure properly.

One trial per arm across both panels, never a different run per panel. The
trial chosen is each arm's best that is fully tracked; where that is not the
arm's overall best -- MLflow logging began partway through the campaign -- the
legend says so and `--json` records both.

Colours and markers are plot_fig1.LINES', so an arm is the same colour in every
figure in the paper.
"""

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT = "bllarse-finetuning-tuning"

# (study arm, label, colour, marker) -- colours from plot_fig1.LINES so an arm
# does not change colour between figures.
ARMS = [
    ("ibprobit_noeps", "IBProbit",  "#8172B2", "D"),
    ("ce",             "AdamW+CE",  "#4C72B0", "o"),
    ("vbll",           "VBLL",      "#C44E52", "^"),
]

METRICS = [("val_nll", "NLL ↓", True), ("val_brier", "Brier ↓", False)]

# Every metric the figure draws. A run missing any of them is skipped rather
# than drawn on some panels and not others.
REQUIRED = {"val_nll", "val_brier"}


def _arm_of(run):
    """The STUDY arm. The `arm` tag carries the SCRIPT flag, which collapses
    ibprobit_noeps onto ibprobit; group_id keeps the study identity."""
    gid = run.data.tags.get("group_id", "")
    return gid.replace("fttune_ft_", "").rsplit("_", 1)[0] if gid else "?"


def fetch(campaign, experiment=EXPERIMENT):
    from bllarse.mlflow_utils import load_mlflow_env_defaults
    load_mlflow_env_defaults()
    from mlflow import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"[plot] no MLflow experiment {experiment!r}")

    by_arm = collections.defaultdict(list)
    for run in client.search_runs([exp.experiment_id], max_results=50_000):
        gid = run.data.tags.get("group_id", "")
        if campaign and not gid.endswith(campaign):
            continue
        by_arm[_arm_of(run)].append(run)

    def curve(run, metric):
        hist = client.get_metric_history(run.info.run_id, metric)
        return [[p.step, p.value] for p in sorted(hist, key=lambda p: p.step)]

    out = {}
    for arm, runs in by_arm.items():
        full = [r for r in runs if r.info.status == "FINISHED"
                and REQUIRED <= set(r.data.metrics)]
        finished = [r for r in runs if r.info.status == "FINISHED"]
        if not full:
            print(f"[plot] {arm}: no fully-tracked run, skipping")
            continue
        scored = [(min(v for _, v in curve(r, "val_nll")), r) for r in full]
        best_nll, best = min(scored, key=lambda t: t[0])
        out[arm] = {
            "run_name": best.info.run_name,
            "min_val_nll": best_nll,
            # The SELECTION POOL, which is the only coverage claim MLflow can
            # support. A trial that never logged has no run here at all, so
            # `full` and `finished` are the same set by construction and their
            # ratio is vacuous -- an earlier version printed it as a coverage
            # figure, which read as "15 of 15" when the true pool was 15 of 89.
            # The arm's real completed count lives in the Optuna journal, and
            # this script deliberately does not reach for it.
            "n_tracked_complete": len(full),
            "n_runs_incl_pruned": len(runs),
            "series": {m: curve(best, m) for m, _, _ in METRICS},
        }
        print(f"[plot] {arm:16s} {best.info.run_name}  min NLL {best_nll:.5f}"
              f"  (best of {len(full)} tracked completions;"
              f" {len(runs)} MLflow runs incl. pruned)")
    return out


def make_figure(data, out_path, campaign):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 3.6))
    for ax, (metric, ylabel, logy) in zip(axes, METRICS):
        for arm, label, colour, marker in ARMS:
            d = data.get(arm)
            if not d:
                continue
            xy = d["series"].get(metric) or []
            if not xy:
                continue
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            ax.plot(xs, ys, color=colour, lw=1.8, label=label, zorder=3)
            # The argmin, which is the value the search actually optimises --
            # NOT the last point, which is what a curve invites you to read.
            i = min(range(len(ys)), key=lambda k: ys[k])
            ax.plot(xs[i], ys[i], marker=marker, ms=6, color=colour,
                    mec="white", mew=1.0, zorder=4)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("samples seen")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    # The selection pool, stated unconditionally. Tracking began partway
    # through the campaign, so an arm's pool can be far smaller than its
    # completed-trial count -- VBLL's especially -- and a reader comparing
    # arms needs to know the search behind each line was not equally deep.
    pool = [f"{lab} {data[a]['n_tracked_complete']}"
            for a, lab, _, _ in ARMS if a in data]
    note = "\nbest of tracked completions per arm: " + ", ".join(pool)
    fig.suptitle(
        f"Tuning optima, {campaign} — dinov3_large, 1M samples, single seed "
        f"(NOT the reported 4-epoch campaign){note}",
        fontsize=9, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default="betasearch1m",
                   help="Study suffix to select, matched on group_id.")
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="tuning_optima")
    p.add_argument("--from-json", default=None,
                   help="Re-plot from a previous run's JSON instead of "
                        "querying MLflow, so the figure is regenerable "
                        "without the tracking server.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{args.name}.json")

    if args.from_json:
        with open(args.from_json) as fh:
            data = json.load(fh)
    else:
        data = fetch(args.campaign)
        if not data:
            raise SystemExit("[plot] nothing to draw")
        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"[plot] wrote {json_path}")

    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    args.campaign)


if __name__ == "__main__":
    main()
