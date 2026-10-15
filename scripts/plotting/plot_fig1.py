"""Fig 1: backbone scale x algorithm, linear probing on frozen DINOv3.

The x-axis is the backbone ladder, which is the claim the figure exists to
support -- that the advantage does not wash out as the backbone gets stronger.

Four lines: AdamW+CE, AdamW+CE+TS, VBLL and IBProbit. TS is a series on CE
alone, not a per-arm variant -- see LINES.

Every point is that run's VAL-OPTIMAL step, not its endpoint (see
scripts/analysis/fig1_curves.py). A point ringed in black is one where at least
one seed's optimum fell at the very end of its budget -- training stopped before
the val curve turned, so that point is a lower bound, not a converged value.

The companion endpoint-penalty figure needs trajectory dumps, and the Fig 1
sweeps write one dump per run, so it refuses to draw rather than plotting the
zeros that follow. See `make_overfit_figure`.

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
from matplotlib.lines import Line2D

import paper_style
from paper_style import DOWN, UP

LADDER = [("dinov3_small", "S"), ("dinov3_big", "B"),
          ("dinov3_large", "L"), ("dinov3_huge", "H+")]

DATASETS = [("oxford_pets", "Oxford Pets"), ("food101", "Food-101"),
            ("flowers102", "Flowers-102"), ("stanford_cars", "Stanford Cars"),
            ("dtd", "DTD")]

# ECE is measured and stays in the JSON; it is off the figure by decision.
# NLL and Brier are both strictly proper, so they already price calibration
# together with sharpness -- a binned ECE beside them adds a third view of the
# same axis whose value moves with the bin count.
METRICS = [("acc", f"Accuracy {UP}"), ("nll", f"NLL {DOWN}"),
           ("brier", f"Brier {DOWN}")]

# (label, arm, metric prefix, colour, marker)
#
# The `ts_` prefix appears on adamw_ce ALONE, and that is the comparison the
# paper makes: temperature scaling is the cheap post-hoc fix available to a
# plain softmax head, so it belongs beside CE as its own comparator. VBLL and
# IBProbit are shown RAW because their claim is calibration by construction --
# temperature-scaling them would measure how much a one-parameter rescaling can
# rescue a method that says it does not need one.
#
# The ts_ numbers do not exist for the other arms at all: fig1_curves only
# computes them for the arms in curves.TS_ARMS. Numbers left in the JSON "for
# reference" get quoted -- and for VBLL and IBProbit they are not even the same
# transform, since those arms dump a normalized log-predictive rather than
# logits.
LINES = [
    ("AdamW+CE",    "adamw_ce", "",    "#4C72B0", "o"),
    ("AdamW+CE+TS", "adamw_ce", "ts_", "#55A868", "s"),
    ("VBLL",        "vbll",     "",    "#C44E52", "^"),
    # Every arm in fig1_curves.GROUPS needs an entry here; an arm missing from
    # this list is dropped from the figure without any error.
    ("IBProbit",    "ibprobit", "",    "#8172B2", "D"),
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


def _truncated(summary, arm, dataset, metric):
    """(x, y) for the rungs where some seed's val optimum sat at the budget end.

    Those runs were still improving when the budget ran out, so the plotted
    point is a lower bound on what the configuration reaches, not its converged
    value. Marked rather than dropped -- it is a real measurement of a run that
    has not finished improving, and dropping it would quietly shrink the cell.

    Keyed on `truncated`, not on `at_last_eval`: the exact flag misses the
    near-boundary case that dominates here. See analysis.curves.completeness.
    """
    xs, ys = [], []
    for x, (model, _) in enumerate(LADDER):
        cell = summary.get(f"{arm}|{model}|{dataset}")
        if not cell or metric not in cell:
            continue
        rate = cell.get("truncated")
        if rate and rate["mean"] > 0:
            xs.append(x)
            ys.append(cell[metric]["mean"])
    return xs, ys


def make_figure(summary, out_path):
    # Fifteen small panels, so one step below the paper context's default size.
    paper_style.use(font_scale=0.85)
    fig, axes = paper_style.figure(len(METRICS), len(DATASETS), height=4.1,
                                   sharex=True)
    any_truncated = False

    for j, (dataset, ds_label) in enumerate(DATASETS):
        for i, (metric, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            for label, arm, prefix, colour, marker in LINES:
                xs, means, stds = _series(summary, arm, dataset, f"{prefix}{metric}")
                if not xs:
                    continue
                ax.errorbar(
                    xs, means, yerr=stds, color=colour, marker=marker,
                    markersize=3.8, linewidth=1.3, capsize=1.8, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=label if (i == 0 and j == 0) else None,
                )
                # Overlaid, not styled into the errorbar call: matplotlib takes
                # ONE marker style per series, and only some rungs of a series
                # are truncated.
                tx, ty = _truncated(summary, arm, dataset, f"{prefix}{metric}")
                if tx:
                    any_truncated = True
                    ax.scatter(tx, ty, s=46, facecolors="none",
                               edgecolors="#333333", linewidths=0.8, zorder=5)
            ax.set_xticks(range(len(LADDER)))
            ax.set_xticklabels([m[1] for m in LADDER])
            ax.set_xlim(-0.45, len(LADDER) - 0.55)
            paper_style.few_ticks(ax, y=5)
            if i == 0:
                ax.set_title(ds_label)
            if j == 0:
                ax.set_ylabel(metric_label)
    # One x label for the row, under the middle column.
    axes[-1, len(DATASETS) // 2].set_xlabel("DINOv3 backbone size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if any_truncated:
        # A proxy, not a label on the first ring drawn: labels are harvested
        # from axes[0, 0] alone, so labelling the rings in place would put the
        # entry in the legend only when the top-left panel happens to contain a
        # truncated point.
        handles.append(Line2D([], [], linestyle="none", marker="o",
                              markersize=6.5, markerfacecolor="none",
                              markeredgecolor="#333333", markeredgewidth=0.8))
        # What a ring means is spelled out in the caption.
        labels.append("budget-truncated")
    paper_style.legend_below(fig, handles, labels)
    paper_style.save(fig, out_path)


def make_overfit_figure(summary, out_path):
    """NLL penalty for training to the endpoint instead of stopping at the optimum.

    This is where the arms genuinely differ, and it is invisible in the main
    figure because that one scores every arm at its own optimum. A point here is
    how much NLL an arm gives up by not early-stopping -- i.e. how much the KL
    term is worth when nobody is watching the validation curve.

    REQUIRES TRAJECTORY DUMPS. probe.py and probing_vbll.py write only the
    argmin evaluation, so on their dumps the endpoint IS the optimum and every
    difference here is exactly 0.0 -- a flat line through zero that would read
    as the measured result "no arm overfits", from no data at all.

    So it refuses. `fig1_curves.score_run` emits `final_*` only for runs that
    dumped more than one step; when nothing did, there is nothing to draw.
    """
    if not any("final_nll" in cell for cell in summary.values()):
        print(f"skipped {out_path}: no run dumped a trajectory, so the endpoint "
              f"is the val optimum and this figure would plot zeros. Re-run the "
              f"sweep with per-step dumps to measure it.")
        return

    fig, axes = plt.subplots(1, len(DATASETS),
                             figsize=(2.7 * len(DATASETS), 2.6), sharey=False)

    for j, (dataset, ds_label) in enumerate(DATASETS):
        ax = axes[j]
        for label, arm, prefix, colour, marker in LINES:
            if prefix:
                continue  # TS is a post-hoc fit; it has no separate trajectory
            xs, finals, _ = _series(summary, arm, dataset, "final_nll")
            if not xs:
                continue
            # Indexed by the SAME rungs the endpoint exists for. Taking a second
            # `_series` for "nll" and zipping would silently misalign whenever
            # only some rungs dumped a trajectory: zip stops at the shorter list,
            # so rung L's endpoint gets differenced against rung S's optimum.
            bests = [summary[f"{arm}|{LADDER[x][0]}|{dataset}"]["nll"]["mean"]
                     for x in xs]
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

    json_path = args.json

    with open(json_path) as fh:
        summary = json.load(fh)["summary"]

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(summary, os.path.join(args.out_dir, f"fig1_backbone_scale.{ext}"))
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
