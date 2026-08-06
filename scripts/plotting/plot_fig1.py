"""Fig 1: backbone scale x algorithm, linear probing on frozen DINOv3.

This is Fig 1 proper, not the single-backbone stand-in: the x-axis is the
backbone ladder, which is the claim the figure exists to support -- that the
advantage does not wash out as the backbone gets stronger. IBProbit is absent
pending the q change; the three lines here are the comparators it will be
measured against.

Every point is that run's VAL-OPTIMAL step, not its endpoint (see
scripts/analysis/fig1_curves.py). The companion figure plots the endpoint
penalty, which is where the two arms actually differ.

Only dinov3_huge was tuned; the other three rungs run huge's hyperparameters.
space.py accepts that deliberately -- the backbones emit 384/768/1024/1280-dim
features so a config optimal at one is not guaranteed optimal at another -- and
it has to be stated in the paper rather than buried.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LADDER = [("dinov3_small", "S"), ("dinov3_big", "B"),
          ("dinov3_large", "L"), ("dinov3_huge", "H+")]

DATASETS = [("oxford_pets", "Oxford-IIIT Pets"), ("food101", "Food-101"),
            ("flowers102", "Flowers-102"), ("stanford_cars", "Stanford Cars"),
            ("dtd", "DTD")]

METRICS = [("acc", "Accuracy ↑"), ("ece", "ECE (20 bin) ↓"),
           ("nll", "NLL ↓"), ("brier", "Brier ↓")]

# (label, arm, metric prefix, colour, marker)
LINES = [
    ("AdamW+CE",    "adamw_ce", "",    "#4C72B0", "o"),
    ("AdamW+CE+TS", "adamw_ce", "ts_", "#55A868", "s"),
    ("VBLL",        "vbll",     "",    "#C44E52", "^"),
]


def _series(summary, arm, dataset, key):
    xs, means, stds = [], [], []
    for x, (model, _) in enumerate(LADDER):
        cell = summary.get(f"{arm}|{model}|{dataset}")
        if not cell or key not in cell:
            continue
        xs.append(x)
        means.append(cell[key]["mean"])
        stds.append(cell[key]["std"])
    return xs, means, stds


def make_figure(summary, out_path):
    fig, axes = plt.subplots(len(METRICS), len(DATASETS),
                             figsize=(2.7 * len(DATASETS), 2.3 * len(METRICS)))

    for j, (dataset, ds_label) in enumerate(DATASETS):
        for i, (metric, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            for label, arm, prefix, colour, marker in LINES:
                xs, means, stds = _series(summary, arm, dataset, f"{prefix}{metric}")
                if not xs:
                    continue
                ax.errorbar(
                    xs, means, yerr=stds, color=colour, marker=marker,
                    markersize=5, linewidth=1.4, capsize=2.5, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=label if (i == 0 and j == 0) else None,
                )
            ax.set_xticks(range(len(LADDER)))
            ax.set_xticklabels([m[1] for m in LADDER] if i == len(METRICS) - 1 else [],
                               fontsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(ds_label, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=9)
            if i == len(METRICS) - 1:
                ax.set_xlabel("DINOv3 scale", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        "Fig 1 (IBProbit absent) — linear probing on frozen DINOv3, "
        "10 seeds, bands ±1 sd\n"
        "each point at its val-optimal step; only H+ was tuned, "
        "its config applied across the ladder",
        fontsize=9.5, y=1.06,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def make_overfit_figure(summary, out_path):
    """NLL penalty for training to the endpoint instead of stopping at the optimum.

    This is where the two arms genuinely differ, and it is invisible in the main
    figure because that one scores both at their optimum. A bar here is how much
    NLL an arm gives up by not early-stopping -- i.e. how much the KL term is
    worth when nobody is watching the validation curve.
    """
    fig, axes = plt.subplots(1, len(DATASETS),
                             figsize=(2.7 * len(DATASETS), 2.6), sharey=False)

    for j, (dataset, ds_label) in enumerate(DATASETS):
        ax = axes[j]
        for label, arm, prefix, colour, marker in LINES:
            if prefix:
                continue  # TS is a post-hoc fit; it has no separate trajectory
            xs, finals, _ = _series(summary, arm, dataset, "final_nll")
            _, bests, _ = _series(summary, arm, dataset, "nll")
            if not xs:
                continue
            ax.plot(xs, [f - b for f, b in zip(finals, bests)],
                    color=colour, marker=marker, markersize=5, linewidth=1.4,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=label if j == 0 else None)
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(range(len(LADDER)))
        ax.set_xticklabels([m[1] for m in LADDER], fontsize=8)
        ax.set_title(ds_label, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("DINOv3 scale", fontsize=8)
        if j == 0:
            ax.set_ylabel("NLL penalty at endpoint ↓", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Fig 1 companion — cost of training to the endpoint instead of the "
        "val optimum (0 = no penalty)",
        fontsize=9.5, y=1.16,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default="scripts/results/fig1_curves.json")
    p.add_argument("--out-dir", default="scripts/figures")
    args = p.parse_args()

    with open(args.json) as fh:
        summary = json.load(fh)["summary"]

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(summary, os.path.join(args.out_dir, f"fig1.{ext}"))
        make_overfit_figure(summary, os.path.join(args.out_dir, f"fig1_endpoint_penalty.{ext}"))

    print("\nVBLL minus AdamW+CE at each rung (NLL; positive = VBLL worse):")
    for dataset, _ in DATASETS:
        deltas = []
        for model, short in LADDER:
            ce = summary.get(f"adamw_ce|{model}|{dataset}")
            vb = summary.get(f"vbll|{model}|{dataset}")
            if ce and vb:
                deltas.append(f"{short}={vb['nll']['mean'] - ce['nll']['mean']:+.4f}")
        print(f"  {dataset:<16s} " + "  ".join(deltas))


if __name__ == "__main__":
    main()
