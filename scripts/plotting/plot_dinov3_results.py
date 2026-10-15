#!/usr/bin/env python3
"""Fetch MLflow sweep2a results and produce a figure + LaTeX table.

Usage:
    # Fetch from MLflow, save CSV, plot, and generate table:
    python scripts/dinov3/plot_sweep2a_results.py

    # Re-plot from cached CSV:
    python scripts/dinov3/plot_sweep2a_results.py --plot-only --input sweep2a_results.csv
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

import paper_style
from paper_style import DOWN, UP

load_dotenv()
load_dotenv(".env.secrets", override=False)

_KNOWN_MODELS = [
    "dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge",
]

def _load_dataset_info(cache_dir: str) -> dict[str, dict[str, int]]:
    """Read training set size and number of classes from cached feature .npz files.

    Filename format: {model}_{dataset}_res{N}_train.npz
    Returns {dataset: {"n_train": N, "n_classes": C}}.
    """
    info: dict[str, dict[str, int]] = {}
    for path in glob.glob(os.path.join(cache_dir, "*_train.npz")):
        fname = os.path.basename(path)
        stem = fname.split("_res")[0]  # e.g. "dinov3_large_food101"
        dataset = None
        for model in _KNOWN_MODELS:
            if stem.startswith(model + "_"):
                dataset = stem[len(model) + 1:]
                break
        if dataset and dataset not in info:
            data = np.load(path, mmap_mode="r")
            entry = {"n_train": int(data["features"].shape[0])}
            if "labels" in data.files:
                entry["n_classes"] = int(np.unique(data["labels"]).size)
            info[dataset] = entry
    return info


# ---------------------------------------------------------------------------
# MLflow fetching
# ---------------------------------------------------------------------------

def fetch_mlflow_runs(experiment_name: str, parent_run_name: str) -> pd.DataFrame:
    """Fetch child runs of the named parent from MLflow."""
    import mlflow
    from dotenv import load_dotenv

    load_dotenv()
    mlflow.set_experiment(experiment_name)

    # Find parent run
    parents = mlflow.search_runs(
        filter_string=f"run_name = '{parent_run_name}'",
        output_format="list",
    )
    if not parents:
        print(f"ERROR: parent run '{parent_run_name}' not found", file=sys.stderr)
        sys.exit(1)

    parent_id = parents[0].info.run_id
    print(f"Parent run: {parent_id}")

    # Fetch all child runs
    all_runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
        max_results=5000,
    )

    print(f"Found {len(all_runs)} child runs")

    records = []
    for run in all_runs:
        p = run.data.params
        m = run.data.metrics  # latest values
        records.append({
            "model": p.get("model", ""),
            "dataset": p.get("dataset", ""),
            "batch_size": int(p.get("batch_size", 0)),
            "num_update_iters": int(p.get("num_update_iters", 0)),
            "seed": int(p.get("seed", 0)),
            "test_acc": m.get("test_acc", np.nan),
            "test_ece": m.get("test_ece", np.nan),
            "test_nll": m.get("test_nll", np.nan),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS = ["test_acc", "test_ece", "test_nll"]
METRIC_LABELS = {"test_acc": f"Accuracy {UP}", "test_ece": f"ECE {DOWN}",
                 "test_nll": f"NLL {DOWN}"}
FIGURE_DATASETS = [
    "oxford_pets", "food101", "flowers102",
    "stanford_cars", "dtd", "imagenet1k",
]
DATASET_DISPLAY = {
    "oxford_pets":   "Oxford Pets",
    "food101":       "Food-101",
    "flowers102":    "Flowers-102",
    "stanford_cars": "Stanford Cars",
    "dtd":           "DTD",
    "imagenet1k":    "ImageNet-1k",
}
MODEL_DISPLAY = {
    "dinov3_small": "DINOv3-S",
    "dinov3_big":   "DINOv3-B",
    "dinov3_large": "DINOv3-L",
    "dinov3_huge":  "DINOv3-H",
}


def _batch_label(batch_size, _pos=None):
    """512 -> '512', 4096 -> '4k'."""
    batch_size = int(round(batch_size))
    return str(batch_size) if batch_size < 1024 else f"{batch_size // 1024}k"


def make_figure(df: pd.DataFrame, output_dir: str, prefix: str = "sweep2a", model: str = "dinov3_huge"):
    """One row per dataset, one column per metric.

    Six datasets across the page left each panel under an inch wide at print
    size, with no room for readable tick labels. Down the page, every panel is
    a third of the text width and all of them share one batch-size axis.
    """
    paper_style.use(font_scale=0.9)

    df_model = df[df["model"] == model].copy()
    datasets = [d for d in FIGURE_DATASETS if (df_model["dataset"] == d).any()]

    fig, axes = paper_style.figure(len(datasets), len(METRICS),
                                   height=0.95 * len(datasets) + 0.75,
                                   sharex=True)

    batch_sizes = sorted(df_model[df_model["dataset"].isin(datasets)]["batch_size"].unique())
    update_iters_list = sorted(df_model["num_update_iters"].unique())

    # The iteration count is ordered, so a sequential palette: light to dark
    # with more iterations. One extra colour is requested and the first
    # dropped, because viridis starts at a yellow that does not show on white.
    colors = dict(zip(update_iters_list,
                      sns.color_palette("viridis_r", len(update_iters_list) + 1)[1:]))

    for row_idx, dataset in enumerate(datasets):
        df_ds = df_model[df_model["dataset"] == dataset]

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            for ui in update_iters_list:
                df_ui = df_ds[df_ds["num_update_iters"] == ui]
                if df_ui.empty:
                    continue

                stats = df_ui.groupby("batch_size")[metric].agg(["mean", "std"])
                stats["std"] = stats["std"].fillna(0)
                stats = stats.dropna(subset=["mean"]).sort_index()
                if stats.empty:
                    continue

                x = np.array(stats.index)
                y = stats["mean"].values
                yerr = stats["std"].values

                ax.plot(x, y, marker="o", ms=3.0, lw=1.4, color=colors[ui],
                        label=f"{int(ui)}")
                if yerr.any():
                    ax.fill_between(x, y - yerr, y + yerr, alpha=0.18,
                                    color=colors[ui], lw=0)

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(ticker.FixedLocator(batch_sizes))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_batch_label))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            paper_style.few_ticks(ax, y=4)

            # A metric that does not move at all would otherwise get
            # matplotlib's default window around a constant, which for an
            # accuracy of 0.996 runs past 1.
            lo, hi = ax.dataLim.intervaly
            if hi - lo < 1e-9:
                ax.set_ylim(lo - 0.003, lo + 0.003)

            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric])
            if col_idx == 0:
                ax.set_ylabel(DATASET_DISPLAY.get(dataset, dataset))
    for ax in axes[-1]:
        ax.set_xlabel("Batch size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    paper_style.legend_below(fig, handles, labels, title="CAVI iterations per step")

    fname = os.path.join(output_dir, f"{prefix}_dinov3_huge.pdf")
    paper_style.save(fig, fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-name", default="bllarse")
    parser.add_argument("--parent-run-name", default="sweep5a-alt-classification-dinov3-IBProbit")
    parser.add_argument("--input", type=str, default=None, help="Path to existing CSV")
    parser.add_argument("--output", type=str, default="scripts/results/last_layer/dinov3_last_layer.csv")
    parser.add_argument("--output-dir", type=str, default="scripts/figures")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--csv-only", action="store_true")
    parser.add_argument("--cache-dir", default=".cache/features", help="Feature cache dir for dataset train sizes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        if not args.input:
            parser.error("--plot-only requires --input")
        df = pd.read_csv(args.input)
    else:
        if args.input and os.path.exists(args.input):
            print(f"Loading cached CSV: {args.input}")
            df = pd.read_csv(args.input)
        else:
            df = fetch_mlflow_runs(args.experiment_name, args.parent_run_name)
            df.to_csv(args.output, index=False)
            print(f"Saved {len(df)} records to {args.output}")

    if args.csv_only:
        return

    prefix = args.parent_run_name.split("-")[0]
    # Training-set size and class count go in the caption, not on the figure.
    dataset_info = _load_dataset_info(args.cache_dir)
    for dataset in FIGURE_DATASETS:
        info = dataset_info.get(dataset)
        if info:
            print(f"{DATASET_DISPLAY[dataset]}: n={info['n_train']:,}, "
                  f"C={info.get('n_classes', '?')}")
    make_figure(df, args.output_dir, prefix=prefix)


if __name__ == "__main__":
    main()
