"""Fig 2: OOD detection from frozen DINOv3-H+ features.

Reads the JSON written by scripts/analysis/fig2_detection.py (which runs where
the dumps are). Three ID/OOD pairs across the columns, the three detection
metrics down the rows, and one point per (arm, score) with a band over seeds.

The x-axis is deliberately (arm x score) rather than just arm. Which scores an
arm can offer is a property of what it dumps, not a free choice -- AdamW+CE
dumps raw logits so energy and max-logit apply to it and not to VBLL, whose dump
is already normalized; VBLL dumps a native predictive variance that CE has no
counterpart for. Collapsing that to one bar per arm would silently pick a
winner per arm and hide the comparison the figure exists to make.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = [("cifar10_vs_svhn", "CIFAR-10 vs SVHN\n(far-OOD)"),
         ("cifar10_vs_cifar100", "CIFAR-10 vs CIFAR-100\n(near-OOD)"),
         ("cifar100_vs_svhn", "CIFAR-100 vs SVHN\n(far-OOD)")]

METRICS = [("auroc", "AUROC ↑"), ("fpr95", "FPR@95TPR ↓"), ("aupr", "AUPR ↑")]

# (arm, score, label, colour).
#
# MSP and entropy only. energy/max-logit were measured and are in the JSON, but
# they apply to AdamW+CE alone -- VBLL's dump is an already-normalized
# log-predictive, on which they are degenerate -- so putting them on a
# cross-arm figure compares four CE variants against two VBLL ones and invites
# reading the extra points as a method difference. VBLL's native predictive
# variance is likewise measured and excluded here; it belongs in the text, not
# on this figure, because it answers a different question (see the module
# docstring of scripts/analysis/fig2_detection.py) and is far off this scale.
SERIES = [
    ("adamw_ce", "msp",     "CE / MSP",       "#4C72B0"),
    ("adamw_ce", "entropy", "CE / entropy",   "#6C8EBF"),
    ("vbll",     "msp",     "VBLL / MSP",     "#C44E52"),
    ("vbll",     "entropy", "VBLL / entropy", "#D98C8E"),
    # Added with the campaign's third arm; this list had been left at two.
    ("ibprobit", "msp",     "IBProbit / MSP",     "#8172B2"),
    ("ibprobit", "entropy", "IBProbit / entropy", "#A093C7"),
]

# ID control: the four classification metrics, keyed by ID dataset. Undefined on
# the OOD side (an SVHN digit has no correct CIFAR label), so they cannot go on
# the detection panels -- but without them a detection number is uninterpretable.
ID_METRICS = [("acc", "Accuracy ↑"), ("ece", "ECE ↓"), ("nll", "NLL ↓"), ("brier", "Brier ↓")]
ID_DATASETS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")]
ID_ARMS = [("adamw_ce", "AdamW+CE", "#4C72B0"), ("vbll", "VBLL", "#C44E52"),
           ("ibprobit", "IBProbit", "#8172B2")]


def make_figure(summary, out_path):
    fig, axes = plt.subplots(len(METRICS), len(PAIRS),
                             figsize=(3.4 * len(PAIRS), 2.5 * len(METRICS)))

    for j, (pair_key, pair_label) in enumerate(PAIRS):
        for i, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            xs, means, stds, colours, labels = [], [], [], [], []
            for x, (arm, score, label, colour) in enumerate(SERIES):
                cell = summary.get(f"{arm}|{pair_key}|{score}")
                if not cell or metric_key not in cell:
                    continue
                xs.append(x)
                means.append(cell[metric_key]["mean"])
                stds.append(cell[metric_key]["std"])
                colours.append(colour)
                labels.append(label)

            if xs:
                ax.errorbar(xs, means, yerr=stds, fmt="none",
                            ecolor="#999999", elinewidth=1, capsize=3, zorder=2)
                ax.scatter(xs, means, c=colours, s=52, zorder=3,
                           edgecolors="white", linewidths=0.8)

            ax.set_xlim(-0.6, len(SERIES) - 0.4)
            ax.set_xticks(range(len(SERIES)))
            ax.set_xticklabels(
                [s[2] for s in SERIES] if i == len(METRICS) - 1 else [],
                rotation=45, ha="right", fontsize=7,
            )
            # Separates the arms; both offer the same two scores now.
            ax.axvline(1.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(pair_label, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=9)

    fig.suptitle(
        "Fig 2 — OOD detection from frozen DINOv3-H+ features\n"
        "hyperparameters tuned for in-distribution val_nll, not for OOD; "
        "bands are ±1 sd over seeds",
        fontsize=9.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def make_id_figure(id_summary, out_path):
    """The in-distribution control, at both the final and the val-optimal step.

    Two points per arm, joined: the val-optimal step is where the tuning
    campaign selected the configuration, the final step is where the detection
    metrics are computed. A long connector means that arm overfits without early
    stopping, and that its detection number is being read off a checkpoint its
    own tuning never endorsed.
    """
    fig, axes = plt.subplots(len(ID_METRICS), len(ID_DATASETS),
                             figsize=(3.0 * len(ID_DATASETS), 2.1 * len(ID_METRICS)))

    for j, (dataset, ds_label) in enumerate(ID_DATASETS):
        for i, (metric, metric_label) in enumerate(ID_METRICS):
            ax = axes[i, j]
            for x, (arm, arm_label, colour) in enumerate(ID_ARMS):
                cell = id_summary.get(f"{arm}|{dataset}")
                if not cell or metric not in cell:
                    continue
                final = cell[metric]["mean"]
                best = cell.get(f"at_best_{metric}", {}).get("mean")
                if best is not None:
                    ax.plot([x, x], [best, final], color=colour, alpha=0.35,
                            linewidth=1.4, zorder=2)
                    ax.scatter([x], [best], s=34, facecolors="white",
                               edgecolors=colour, linewidths=1.4, zorder=3,
                               label="val-optimal step" if (i == 0 and j == 0 and x == 0) else None)
                ax.scatter([x], [final], s=58, color=colour, zorder=4,
                           edgecolors="white", linewidths=0.8,
                           label="final step" if (i == 0 and j == 0 and x == 0) else None)

            ax.set_xlim(-0.6, len(ID_ARMS) - 0.4)
            ax.set_xticks(range(len(ID_ARMS)))
            ax.set_xticklabels([a[1] for a in ID_ARMS] if i == len(ID_METRICS) - 1 else [],
                               rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(ds_label, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        "DRAFT Fig 2 control — in-distribution performance of the same runs\n"
        "hollow = val-optimal step (where the config was tuned), "
        "filled = final step (where detection is scored)",
        fontsize=9.5, y=1.075,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default="scripts/results/fig2_detection.json")
    p.add_argument("--out-dir", default="scripts/figures")
    args = p.parse_args()

    with open(args.json) as fh:
        payload = json.load(fh)
    summary = payload["summary"]
    id_summary = payload.get("id_summary", {})

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(summary, os.path.join(args.out_dir, f"fig2_draft.{ext}"))
        if id_summary:
            make_id_figure(id_summary, os.path.join(args.out_dir, f"fig2_id_control.{ext}"))

    # The headline contrast, printed so it is not read off the figure by eye.
    print("\nNative epistemic score vs MSP on the same model (VBLL, AUROC):")
    for pair_key, label in PAIRS:
        native = summary.get(f"vbll|{pair_key}|native")
        msp = summary.get(f"vbll|{pair_key}|msp")
        if native and msp:
            delta = native["auroc"]["mean"] - msp["auroc"]["mean"]
            print(f"  {pair_key:<24s} native={native['auroc']['mean']:.4f}  "
                  f"msp={msp['auroc']['mean']:.4f}  delta={delta:+.4f}")


if __name__ == "__main__":
    main()
