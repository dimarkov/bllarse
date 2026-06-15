#!/usr/bin/env python3
"""Plot seed-averaged ACC / ECE / NLL trajectories for the full-network sweep.

Fetches the child runs of an MLflow parent (the sweep created by
scripts/run_fullnet_seed_sweep.sh) — or loads a previously cached CSV — and
plots per-epoch ACC, ECE and NLL averaged over seeds for each configuration
(dataset x batch size). A dashed vertical line marks the seed-averaged best
epoch (argmin validation NLL, logged as `best_epoch`), labelled at the top with
the seed-averaged wall-clock runtime taken to reach that epoch (estimated from
MLflow metric timestamps).

For now only two methods are shown:
    * CrossEntropy with data augmentation
    * IBProbit without data augmentation

Usage:
    # Fetch from MLflow (and optionally cache the long-form table):
    python scripts/plotting/plot_seed_sweep.py \
        --parent-run-name fullnet_seed_sweep --cache sweep_metrics.csv

    # Re-plot from a cached CSV without touching MLflow:
    python scripts/plotting/plot_seed_sweep.py --input sweep_metrics.csv
"""

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / server-friendly
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# Methods to display: (loss_fn, nodataaug, label).
METHODS = [
    ("CrossEntropy", False, "CrossEntropy + aug"),
    ("IBProbit", True, "IBProbit + no-aug"),
]
PALETTE = {
    "CrossEntropy + aug": "#1f77b4",
    "IBProbit + no-aug": "#d62728",
}

# Metrics to plot: (column, axis label).
METRICS = [
    ("acc", "Accuracy"),
    ("ece", "ECE"),
    ("nll", "NLL"),
]

DATASET_ORDER = ["cifar10", "cifar100"]
BS_ORDER = [512, 16384]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def fetch_mlflow_runs(experiment_name: str, parent_run_name: str) -> pd.DataFrame:
    """Fetch FINISHED child runs of the named parent and return per-epoch records."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from dotenv import load_dotenv

    load_dotenv(".env")
    load_dotenv(".env.secrets", override=False)
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    client = MlflowClient()

    parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"run_name = '{parent_run_name}'",
    )
    if not parents:
        print(
            f"ERROR: parent run '{parent_run_name}' not found in experiment "
            f"'{experiment_name}' (id={experiment_id})",
            file=sys.stderr,
        )
        sys.exit(1)
    parent_id = parents[0].info.run_id
    print(f"Parent run: {parent_id} ({parent_run_name})")

    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_id}' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=5000,
    )
    # Keep only the runs whose (loss_fn, nodataaug) matches a plotted method, so
    # we neither fetch nor display the other combinations.
    wanted = {(loss_fn, nodataaug) for loss_fn, nodataaug, _ in METHODS}

    def _matches(run) -> bool:
        rp = run.data.params
        key = (rp.get("loss_fn"), str(rp.get("nodataaug", "False")).lower() == "true")
        return key in wanted

    matching = [r for r in children if _matches(r)]
    print(
        f"Found {len(children)} FINISHED child runs; {len(matching)} match the "
        "selected methods. Fetching metric histories..."
    )

    records = []
    for i, run in enumerate(matching):
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  run {i + 1}/{len(matching)}: {run.info.run_name}")
            sys.stdout.flush()

        p = run.data.params
        base = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "dataset": p.get("dataset", "unknown"),
            "batch_size": int(p.get("batch_size", 0) or 0),
            "loss_fn": p.get("loss_fn", "unknown"),
            "nodataaug": str(p.get("nodataaug", "False")).lower() == "true",
            "seed": int(p.get("seed", 0) or 0),
            # best_epoch is logged once at the end of training.
            "best_epoch": run.data.metrics.get("best_epoch", np.nan),
        }

        per_epoch: dict[int, dict] = {}
        epoch_ts: dict[int, int] = {}  # step -> latest metric timestamp (ms)
        for col in ("acc", "ece", "nll"):
            try:
                history = client.get_metric_history(run.info.run_id, col)
            except Exception:
                history = []
            for point in history:
                per_epoch.setdefault(point.step, {})[col] = point.value
                epoch_ts[point.step] = max(epoch_ts.get(point.step, 0), point.timestamp)

        # Wall-clock runtime from run start until the best epoch was reached,
        # estimated from the metric timestamp at step == best_epoch.
        be = base["best_epoch"]
        runtime_to_best = np.nan
        if np.isfinite(be) and int(be) in epoch_ts:
            runtime_to_best = (epoch_ts[int(be)] - run.info.start_time) / 1000.0
        base["runtime_to_best_s"] = runtime_to_best

        for epoch, vals in per_epoch.items():
            rec = base.copy()
            rec.update(
                epoch=epoch,
                acc=vals.get("acc", np.nan),
                ece=vals.get("ece", np.nan),
                nll=vals.get("nll", np.nan),
            )
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"Done: {len(df)} per-epoch records from {len(matching)} runs.")
    return df


def assign_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the configured methods and add a 'method' label column."""
    df = df.copy()
    df["method"] = pd.NA
    for loss_fn, nodataaug, label in METHODS:
        mask = (df["loss_fn"] == loss_fn) & (df["nodataaug"].astype(bool) == nodataaug)
        df.loc[mask, "method"] = label
    kept = df[df["method"].notna()].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"Filtered out {dropped} records not matching the selected methods.")
    return kept


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '45 s', '12.3 min', '1.4 h'."""
    if not np.isfinite(seconds):
        return ""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


def plot_sweep(df: pd.DataFrame, output: str) -> None:
    """One figure per batch size: rows = dataset, columns = metric (ACC/ECE/NLL)."""
    import os

    sns.set_theme(style="white", context="paper")
    labels = [label for _, _, label in METHODS]

    batch_sizes = [bs for bs in BS_ORDER if not df[df["batch_size"] == bs].empty]
    if not batch_sizes:
        print("ERROR: no data for any batch size; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    base, ext = os.path.splitext(output)
    ext = ext or ".png"

    handles = [Line2D([0], [0], color=PALETTE[l], lw=2.5, label=l) for l in labels]
    handles.append(
        Line2D([0], [0], color="gray", lw=1.6, ls="--", label="mean best epoch")
    )

    for bs in batch_sizes:
        sub_bs = df[df["batch_size"] == bs]
        datasets = [ds for ds in DATASET_ORDER if not sub_bs[sub_bs["dataset"] == ds].empty]
        n_rows, n_cols = len(datasets), len(METRICS)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.8 * n_cols, 3.8 * n_rows),
            squeeze=False, sharex=True,
        )

        for i, ds in enumerate(datasets):
            cfg = sub_bs[sub_bs["dataset"] == ds]
            # seed-averaged best epoch and runtime-to-best per method
            # (both are constant within a run).
            best_epoch = {
                label: cfg[cfg["method"] == label]
                .groupby("seed")["best_epoch"].first().mean()
                for label in labels
            }
            runtime = {
                label: cfg[cfg["method"] == label]
                .groupby("seed")["runtime_to_best_s"].first().mean()
                if "runtime_to_best_s" in cfg.columns else np.nan
                for label in labels
            }

            for j, (col, mlabel) in enumerate(METRICS):
                ax = axes[i][j]
                sns.lineplot(
                    data=cfg, x="epoch", y=col, hue="method",
                    hue_order=labels, palette=PALETTE,
                    estimator="mean", errorbar=("ci", 95),
                    ax=ax, legend=False,
                )
                # vertical line at the seed-averaged best epoch for each method,
                # labelled at the top with the mean runtime to reach it.
                for k, label in enumerate(labels):
                    be = best_epoch.get(label, np.nan)
                    if not np.isfinite(be):
                        continue
                    ax.axvline(be, color=PALETTE[label], ls="--", lw=1.6, alpha=0.9)
                    rt = runtime.get(label, np.nan)
                    if np.isfinite(rt):
                        # Horizontal, boxed, color-matched; stack the two methods
                        # vertically so their labels never overlap.
                        ax.text(
                            be, 0.985 - 0.13 * k, _format_duration(rt),
                            transform=ax.get_xaxis_transform(),
                            color=PALETTE[label], fontsize=8, fontweight="bold",
                            va="top", ha="center", clip_on=False,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=PALETTE[label], lw=0.8, alpha=0.9),
                        )

                if i == 0:
                    ax.set_title(mlabel)            # column header = metric
                ax.set_xlabel("epoch" if i == n_rows - 1 else "")
                ax.set_ylabel(ds if j == 0 else "")  # row label = dataset

        fig.suptitle(f"batch size = {bs}", fontweight="bold")
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02), frameon=False,
        )
        sns.despine(fig=fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = f"{base}_bs{bs}{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Saved figure to {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-name", default="bllarse")
    ap.add_argument("--parent-run-name", default="fullnet_seed_sweep")
    ap.add_argument("--input", default=None,
                    help="Load a previously cached long-form CSV instead of MLflow.")
    ap.add_argument("--cache", default=None,
                    help="Save the fetched long-form table to this CSV.")
    ap.add_argument("--output", default="fullnet_seed_sweep.png")
    args = ap.parse_args()

    if args.input:
        print(f"Loading cached records from {args.input}")
        df = pd.read_csv(args.input)
    else:
        df = fetch_mlflow_runs(args.experiment_name, args.parent_run_name)
        if args.cache:
            df.to_csv(args.cache, index=False)
            print(f"Cached fetched records to {args.cache}")

    df = assign_methods(df)
    if df.empty:
        print("ERROR: no records matched the selected methods.", file=sys.stderr)
        sys.exit(1)
    plot_sweep(df, args.output)


if __name__ == "__main__":
    main()
