"""Temporary diagnostic: IBProbit before vs after the re-tune.

Old data is restricted to seeds 137/138 so both sides rest on the same two
draws -- the new campaign has only those, and comparing 10 seeds against 2
would confound the config change with a change in sample size.

Rings mark budget-truncated points (val optimum in the last 10% of the budget):
those are lower bounds, not converged values.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo-relative, so this regenerates from a clean checkout. Both inputs are
# committed alongside it: the samples40 analysis (10 seeds, restricted below to
# 137/138) and the win5 2-seed slice.
#
# win5, NOT the earlier `smoothed` campaign: that one scored trials on a window
# sized as a FRACTION of the planned grid, which overran the patience-truncated
# trajectories and degenerated into the mean of the whole run. It cost dtd its
# best configuration. win5 uses a fixed 5-evaluation window.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "scripts" / "results"
OLD = json.load(open(RESULTS / "fig1_samples40.json"))["rows"]
NEW = json.load(open(RESULTS / "fig1_win5_2seed.json"))["rows"]
SEEDS = {"137", "138"}
LADDER = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
DATASETS = ["oxford_pets", "dtd", "flowers102", "stanford_cars", "food101"]
METRICS = [("acc", "Accuracy", True), ("nll", "NLL", False), ("brier", "Brier", False)]


def collect(rows, restrict):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] != "ibprobit":
            continue
        if restrict and str(r["seed"]) not in SEEDS:
            continue
        key = (r["dataset"], r["model"])
        for m, _, _ in METRICS:
            if m in r:
                d[key][m].append(r[m])
        d[key]["trunc"].append(bool(r.get("truncated")))
    return d


O, N = collect(OLD, True), collect(NEW, False)

fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True)
x = range(len(LADDER))
for row, (m, label, higher_better) in enumerate(METRICS):
    for col, ds in enumerate(DATASETS):
        ax = axes[row][col]
        for src, data, colour, name in (("old", O, "#888888", "before (samples40)"),
                                        ("new", N, "#c1440e", "after (win5)")):
            ys, es, tr = [], [], []
            for mo in LADDER:
                v = data.get((ds, mo), {}).get(m, [])
                ys.append(st.mean(v) if v else float("nan"))
                es.append((max(v) - min(v)) / 2 if len(v) > 1 else 0.0)
                t = data.get((ds, mo), {}).get("trunc", [])
                tr.append(any(t))
            ax.errorbar(x, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3,
                        color=colour, label=name if (row == 0 and col == 0) else None)
            for xi, (yi, ti) in enumerate(zip(ys, tr)):
                if ti:
                    ax.plot(xi, yi, "o", ms=13, mfc="none", mec=colour, mew=1.8)
        if row == 0:
            ax.set_title(ds, fontsize=11)
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels([b.split("_")[1] for b in LADDER], rotation=30)

handles, labels = axes[0][0].get_legend_handles_labels()
from matplotlib.lines import Line2D
handles.append(Line2D([], [], marker="o", ms=12, mfc="none", mec="black",
                      ls="none", label="budget-truncated (lower bound)"))
fig.suptitle("IBProbit (eps-free): samples40 vs win5 re-tune — seeds 137/138 only",
             y=0.985, fontsize=14)
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, 0.945))
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = ROOT / "scripts" / "figures" / "diag_ibprobit_before_after.png"
fig.savefig(out, dpi=130)
print("wrote", out)
