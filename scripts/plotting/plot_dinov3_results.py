#!/usr/bin/env python3
"""Fetch MLflow sweep2a results and produce a figure + LaTeX table.

Usage:
    # Fetch from MLflow, save CSV, plot, and generate table:
    python scripts/alternatives/plot_sweep2a_results.py

    # Re-plot from cached CSV:
    python scripts/alternatives/plot_sweep2a_results.py --plot-only --input sweep2a_results.csv
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv

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
METRIC_LABELS = {"test_acc": "Accuracy", "test_ece": "ECE", "test_nll": "NLL"}
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


def make_figure(df: pd.DataFrame, output_dir: str, prefix: str = "sweep2a", model: str = "dinov3_huge", dataset_info: dict | None = None):
    """3-row (metrics) x 6-col (datasets) figure."""
    df_model = df[df["model"] == model].copy()

    n_rows = len(METRICS)
    n_cols = len(FIGURE_DATASETS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
        squeeze=False,
        sharex="col",
    )

    batch_sizes = sorted(df_model["batch_size"].unique())
    update_iters_list = sorted(df_model["num_update_iters"].unique())

    cmap = plt.cm.RdYlBu
    colors = {
        it: cmap(i / max(len(update_iters_list) - 1, 1))
        for i, it in enumerate(update_iters_list)
    }

    for col_idx, dataset in enumerate(FIGURE_DATASETS):
        df_ds = df_model[df_model["dataset"] == dataset]
        ds_batch_sizes = sorted(df_ds["batch_size"].unique())

        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            for ui in update_iters_list:
                df_ui = df_ds[df_ds["num_update_iters"] == ui]
                if df_ui.empty:
                    continue

                stats = df_ui.groupby("batch_size")[metric].agg(["mean", "std"])
                stats["std"] = stats["std"].fillna(0)
                stats = stats.reindex(ds_batch_sizes).dropna(subset=["mean"])
                if stats.empty:
                    continue

                x = np.array(stats.index)
                y = stats["mean"].values
                yerr = stats["std"].values

                ax.plot(x, y, marker="o", ms=3.5, color=colors[ui], label=f"iters={int(ui)}")
                if yerr.any():
                    ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=colors[ui])

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            if ds_batch_sizes:
                ax.xaxis.set_major_locator(ticker.FixedLocator(ds_batch_sizes))
            ax.tick_params(axis="x", rotation=45)



            if row_idx == 0:
                ds_label = DATASET_DISPLAY.get(dataset, dataset.replace("_", " ").title())
                info = (dataset_info or {}).get(dataset, {})
                n_train = info.get("n_train")
                n_classes = info.get("n_classes")
                parts = []
                if n_train:
                    parts.append(f"n={n_train:,}")
                if n_classes:
                    parts.append(f"C={n_classes}")
                title = f"{ds_label}\n({', '.join(parts)})" if parts else ds_label
                ax.set_title(title, fontsize=10, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Batch Size")
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric])

    # Single legend
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    fig.legend(
        handles, labels,
        loc="center left",
        ncol=1,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(1.01, 0.5),
    )

    model_label = MODEL_DISPLAY.get(model, model)
    fig.suptitle(f"IBProbit linear probing — {model_label}", fontsize=13, y=1.005)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fname = os.path.join(output_dir, f"{prefix}_dinov3_huge.{ext}")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge",
]
ALL_DATASETS = [
    "cifar10", "cifar100", "oxford_pets", "food101",
    "flowers102", "stanford_cars", "dtd", "imagenet1k",
]


def make_latex_table(df: pd.DataFrame, output_dir: str, prefix: str = "sweep2a"):
    """Table for largest batch_size and num_update_iters, all models x datasets."""
    max_bs = df["batch_size"].max()
    max_ui = df["num_update_iters"].max()
    df_filt = df[(df["batch_size"] == max_bs) & (df["num_update_iters"] == max_ui)]

    print(f"LaTeX table: batch_size={max_bs}, num_update_iters={max_ui}")

    # Aggregate over seeds
    grouped = (
        df_filt.groupby(["model", "dataset"])[METRICS]
        .agg(["mean", "std"])
    )

    def fmt(mean, std):
        if np.isnan(mean):
            return "—"
        if np.isnan(std) or std == 0:
            return f"{mean:.2f}"
        return f"{mean:.2f} $\\pm$ {std:.2f}"

    # Build table
    header_datasets = " & ".join(d.replace("_", "\\_") for d in ALL_DATASETS)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{IBProbit results (batch\_size=" + str(max_bs) + r", update\_iters=" + str(max_ui) + r")}",
        r"\label{tab:sweep2a}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(ALL_DATASETS) + "}",
        r"\toprule",
    ]

    for metric in METRICS:
        label = METRIC_LABELS[metric]
        lines.append(r"\multicolumn{" + str(len(ALL_DATASETS) + 1) + r"}{c}{\textbf{" + label + r"}} \\")
        lines.append(r"\midrule")
        lines.append(r"Model & " + header_datasets + r" \\")
        lines.append(r"\midrule")

        for model in ALL_MODELS:
            cells = [model.replace("_", "\\_")]
            for dataset in ALL_DATASETS:
                try:
                    row = grouped.loc[(model, dataset)]
                    m = row[(metric, "mean")]
                    s = row[(metric, "std")]
                    cells.append(fmt(m, s))
                except KeyError:
                    cells.append("—")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    fname = os.path.join(output_dir, f"{prefix}_table.tex")
    with open(fname, "w") as f:
        f.write(tex)
    print(f"Saved {fname}")


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
    dataset_info = _load_dataset_info(args.cache_dir)
    make_figure(df, args.output_dir, prefix=prefix, dataset_info=dataset_info)
    make_latex_table(df, args.output_dir, prefix=prefix)


if __name__ == "__main__":
    main()
