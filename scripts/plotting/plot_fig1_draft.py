"""Draft Fig 1: linear probing on frozen DINOv3, without IBProbit.

This is NOT Fig 1. Fig 1's x-axis is backbone scale (S -> B -> L -> H+); the
task datasets are only cached at `small` and `huge`, and only `huge` was tuned,
so the ladder collapses to a single rung and the x-axis here is the algorithm
instead. Its purpose is to show what the comparison looks like before spending
anything on the real run.

Data source is the tuning campaign itself, so every point is the val-selected
optimum of a search rather than a fresh run at that configuration. Selection is
on val_nll -- Optuna's objective -- and every plotted number is on the TEST
split at the dump where val_nll was minimal. Selecting and reporting on the same
split would produce a flattering draft that does not survive a real run.

Points, not bars: the arms differ in the third decimal, so the axes are zoomed
to the local range and a bar from a truncated baseline would overstate the gaps.
Per-seed values are drawn individually because with 2 seeds a mean alone hides
how little separates the arms.
"""
import argparse
import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEST_KEYS = ["test_acc", "test_ece", "test_nll", "test_brier"]
TS_KEYS = ["test_ts_acc", "test_ts_ece", "test_ts_nll", "test_ts_brier"]
SHORTLIST = 8

# (row label, better-direction) in results-plan 0.1 order.
METRICS = [("Accuracy", "up"), ("ECE (20 bin)", "down"),
           ("NLL", "down"), ("Brier", "down")]

# The five Fig 1 panels that the tuning space covers. ImageNet-1k is the sixth
# and is absent from bllarse.tuning.space entirely, so it has no tuned config.
FIG1_PANELS = ["oxford_pets", "food101", "flowers102", "stanford_cars", "dtd"]
# Carried along because the campaign tuned them and Fig 2 needs them as ID sets,
# but they are not Fig 1 panels and are drawn separated and greyed.
EXTRA_PANELS = ["cifar10", "cifar100"]

PRETTY = {"oxford_pets": "Oxford-IIIT Pets", "food101": "Food-101",
          "flowers102": "Flowers-102", "stanford_cars": "Stanford Cars",
          "dtd": "DTD", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

# (label, arm, metric-key set, colour, marker)
LINES = [
    ("AdamW+CE",    "adamw_ce", TEST_KEYS, "#4C72B0", "o"),
    ("AdamW+CE+TS", "adamw_ce", TS_KEYS,   "#55A868", "s"),
    ("VBLL",        "vbll",     TEST_KEYS, "#C44E52", "^"),
]


def extract(experiment_id="13"):
    """Pull the val-selected best trial per (dataset, arm) out of MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=100_000, output_format="list",
    )
    by_study = collections.defaultdict(list)
    for r in runs:
        g = r.data.tags.get("group_id", "")
        if g.startswith("tune_") and r.info.status == "FINISHED":
            by_study[g].append(r)

    def at_argmin(run_id):
        hist = client.get_metric_history(run_id, "val_nll")
        if not hist:
            return None
        best = min(hist, key=lambda h: h.value)
        out = {"val_nll": best.value, "step": best.step}
        for k in TEST_KEYS + TS_KEYS + ["temperature"]:
            h = {p.step: p.value for p in client.get_metric_history(run_id, k)}
            out[k] = h.get(best.step)
        return out

    record = {}
    for study, rs in sorted(by_study.items()):
        trials = collections.defaultdict(list)
        for r in rs:
            m = re.match(r"trial(\d+)_seed(\d+)", r.data.tags.get("uid", ""))
            if m:
                trials[int(m.group(1))].append(r)
        if not trials:
            continue
        proxy = {t: sum(s.data.metrics.get("val_nll", 9e9) for s in ss) / len(ss)
                 for t, ss in trials.items()}
        scored = {}
        for t in sorted(proxy, key=proxy.get)[:SHORTLIST]:
            seeds = [p for p in (at_argmin(s.info.run_id) for s in trials[t]) if p]
            if seeds:
                scored[t] = seeds
        if not scored:
            continue
        best_t = min(scored,
                     key=lambda t: sum(p["val_nll"] for p in scored[t]) / len(scored[t]))
        arm = "vbll" if "_vbll_" in study else "adamw_ce"
        ds = trials[best_t][0].data.params.get("dataset", "?")
        record[f"{ds}|{arm}"] = {
            "study": study, "trial": best_t, "n_complete_trials": len(trials),
            "params": dict(trials[best_t][0].data.params),
            "seeds": scored[best_t],
        }
    return record


def make_figure(record, out_path):
    panels = FIG1_PANELS + EXTRA_PANELS
    n_rows, n_cols = len(METRICS), len(panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.15 * n_rows))

    for j, ds in enumerate(panels):
        is_extra = ds in EXTRA_PANELS
        for i, (metric_label, direction) in enumerate(METRICS):
            ax = axes[i, j]
            vals_all = []
            for x, (label, arm, keys, colour, marker) in enumerate(LINES):
                cell = record.get(f"{ds}|{arm}")
                if not cell:
                    continue
                vals = [s[keys[i]] for s in cell["seeds"] if s.get(keys[i]) is not None]
                if not vals:
                    continue
                vals_all.extend(vals)
                ax.scatter([x] * len(vals), vals, s=14, color=colour, alpha=0.45,
                           zorder=3, linewidths=0)
                ax.scatter([x], [float(np.mean(vals))], s=64, color=colour,
                           marker=marker, zorder=4,
                           edgecolors="white", linewidths=0.8,
                           label=label if (i == 0 and j == 0) else None)

            ax.set_xlim(-0.6, len(LINES) - 0.4)
            ax.set_xticks(range(len(LINES)))
            ax.set_xticklabels([l[0] for l in LINES] if i == n_rows - 1 else [],
                               rotation=45, ha="right", fontsize=7)
            # Zoom to the local range: the differences live in the 3rd decimal.
            # Pad from the SPREAD, not the level -- a level-proportional pad is
            # ~0.02 on an accuracy near 1.0, two orders above the differences
            # being shown, and flattens every accuracy panel into a single line.
            # The absolute floor only applies when the arms agree exactly, where
            # there is no spread to scale from.
            if vals_all:
                lo, hi = min(vals_all), max(vals_all)
                pad = (hi - lo) * 0.4 or max(abs(hi), 1e-3) * 0.01
                bottom, top = lo - pad, hi + pad
                if metric_label == "Accuracy":
                    top = min(top, 1.0)   # accuracy has a hard ceiling
                ax.set_ylim(bottom, top)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if is_extra:
                ax.set_facecolor("#f4f4f4")
            if i == 0:
                ax.set_title(PRETTY.get(ds, ds) + ("\n(not a Fig 1 panel)" if is_extra else ""),
                             fontsize=8.5, color="#666666" if is_extra else "black")
            if j == 0:
                arrow = "↑" if direction == "up" else "↓"
                ax.set_ylabel(f"{metric_label} {arrow}", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        "DRAFT Fig 1 — linear probing, frozen DINOv3-H+, IBProbit absent\n"
        "val-selected best trial of the tuning campaign, scored on test at the "
        "val_nll-optimal dump; 2 seeds; axes zoomed to the local range",
        fontsize=9.5, y=1.075,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="scripts/results/fig1_draft.json")
    p.add_argument("--refresh", action="store_true",
                   help="re-query MLflow instead of reading the cache")
    p.add_argument("--out-dir", default="scripts/figures")
    args = p.parse_args()

    if args.refresh or not os.path.exists(args.cache):
        record = extract()
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"cached {len(record)} cells to {args.cache}")
    else:
        with open(args.cache) as fh:
            record = json.load(fh)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(record, os.path.join(args.out_dir, f"fig1_draft.{ext}"))


if __name__ == "__main__":
    main()
