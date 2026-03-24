from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

from bllarse.mlflow_utils import load_mlflow_env_defaults


METRIC_LAYOUT = [
    ("acc", "Accuracy"),
    ("ece", "ECE"),
    ("nll", "NLL"),
]

SPLIT_LAYOUT = [
    ("matched", "Matched MNLI Validation"),
    ("mismatched", "Mismatched MNLI Validation"),
]

MLFLOW_KEYS = {
    "matched": {
        "acc": "acc",
        "ece": "ece",
        "nll": "nll",
    },
    "mismatched": {
        "acc": "val_mismatched_acc",
        "ece": "val_mismatched_ece",
        "nll": "val_mismatched_nll",
    },
}


def _int_param(run, key: str) -> int:
    return int(run.data.params[key])


def _str_param(run, key: str) -> str:
    return str(run.data.params[key])


def _summary_stat(values: list[float], stat: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "iqm":
        arr = np.sort(arr)
        trim = int(arr.size * 0.25)
        if trim > 0 and arr.size - 2 * trim > 0:
            arr = arr[trim:-trim]
        return float(np.mean(arr))
    raise ValueError(f"Unknown summary stat '{stat}'.")


def _spread_bounds(values: list[float], spread: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or spread == "none":
        return float("nan"), float("nan")
    if spread == "minmax":
        return float(np.min(arr)), float(np.max(arr))
    raise ValueError(f"Unknown spread mode '{spread}'.")


def _group_runs(
    runs: Iterable,
    *,
    min_batch_size: int,
    summary_stat: str,
) -> tuple[
    list[int],
    list[str],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, list[list[list[float]]]]],
]:
    run_list = [
        run for run in runs if _int_param(run, "train_batch_size") >= min_batch_size
    ]
    if not run_list:
        raise ValueError("No MLflow runs remain after applying the batch-size filter.")

    batch_sizes = sorted({_int_param(run, "train_batch_size") for run in run_list})
    optimizers = sorted({_str_param(run, "optimizer") for run in run_list})

    grids: dict[str, dict[str, np.ndarray]] = {
        split: {
            metric: np.full((len(optimizers), len(batch_sizes)), np.nan, dtype=np.float64)
            for metric, _ in METRIC_LAYOUT
        }
        for split, _ in SPLIT_LAYOUT
    }
    lowers: dict[str, dict[str, np.ndarray]] = {
        split: {
            metric: np.full((len(optimizers), len(batch_sizes)), np.nan, dtype=np.float64)
            for metric, _ in METRIC_LAYOUT
        }
        for split, _ in SPLIT_LAYOUT
    }
    uppers: dict[str, dict[str, np.ndarray]] = {
        split: {
            metric: np.full((len(optimizers), len(batch_sizes)), np.nan, dtype=np.float64)
            for metric, _ in METRIC_LAYOUT
        }
        for split, _ in SPLIT_LAYOUT
    }
    samples: dict[str, dict[str, list[list[list[float]]]]] = {
        split: {
            metric: [[[] for _ in batch_sizes] for _ in optimizers]
            for metric, _ in METRIC_LAYOUT
        }
        for split, _ in SPLIT_LAYOUT
    }

    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for run in run_list:
        grouped[(_str_param(run, "optimizer"), _int_param(run, "train_batch_size"))].append(run)

    optimizer_index = {value: idx for idx, value in enumerate(optimizers)}
    batch_index = {value: idx for idx, value in enumerate(batch_sizes)}

    for (optimizer, batch_size), members in grouped.items():
        row = optimizer_index[optimizer]
        col = batch_index[batch_size]
        for split, _ in SPLIT_LAYOUT:
            for metric, _ in METRIC_LAYOUT:
                key = MLFLOW_KEYS[split][metric]
                values = [
                    float(run.data.metrics[key])
                    for run in members
                    if key in run.data.metrics and np.isfinite(run.data.metrics[key])
                ]
                if values:
                    grids[split][metric][row, col] = _summary_stat(values, summary_stat)
                    low, high = _spread_bounds(values, "minmax")
                    lowers[split][metric][row, col] = low
                    uppers[split][metric][row, col] = high
                    samples[split][metric][row][col] = sorted(values)

    return batch_sizes, optimizers, grids, lowers, uppers, samples


def _scatter_seed_points(
    ax,
    *,
    batch_sizes: list[int],
    series_idx: int,
    num_series: int,
    sample_rows: list[list[float]],
    color,
    alpha: float,
    size: float,
    jitter_width: float,
) -> None:
    if num_series <= 1:
        series_center = 0.0
    else:
        series_center = np.linspace(-jitter_width, jitter_width, num_series)[series_idx]

    for batch_idx, batch_size in enumerate(batch_sizes):
        values = sample_rows[batch_idx]
        if not values:
            continue
        seed_offsets = np.linspace(-jitter_width * 0.35, jitter_width * 0.35, len(values))
        offsets = series_center + seed_offsets
        x = batch_size * np.power(2.0, offsets)
        ax.scatter(
            x,
            values,
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="none",
            zorder=3,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Plot seed-averaged RoBERTa MNLI deterministic probe metrics as a line grid."
    )
    parser.add_argument("--experiment-id", type=str, default="1")
    parser.add_argument("--parent-run-id", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--min-batch-size", type=int, default=512)
    parser.add_argument("--title", type=str, default="Adam/AdamW — RoBERTa-large MNLI")
    parser.add_argument("--legend-title", type=str, default="optimizer")
    parser.add_argument("--share-y-by-row", action="store_true")
    parser.add_argument("--summary-stat", choices=["mean", "iqm"], default="mean")
    parser.add_argument("--spread", choices=["none", "minmax"], default="none")
    parser.add_argument("--spread-alpha", type=float, default=0.18)
    parser.add_argument("--show-seed-points", action="store_true")
    parser.add_argument("--point-alpha", type=float, default=0.55)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--point-jitter-width", type=float, default=0.12)
    return parser


def main(args: argparse.Namespace) -> None:
    load_mlflow_env_defaults()
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{args.parent_run_id}"',
        max_results=1000,
    )
    if not runs:
        raise ValueError("No MLflow child runs matched the requested parent run.")

    batch_sizes, optimizers, grids, lowers, uppers, samples = _group_runs(
        runs,
        min_batch_size=args.min_batch_size,
        summary_stat=args.summary_stat,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 12),
        constrained_layout=True,
        sharey="row" if args.share_y_by_row else False,
    )
    fig.suptitle(args.title, fontsize=16)

    cmap = plt.get_cmap("viridis", len(optimizers))
    colors = {optimizer: cmap(idx) for idx, optimizer in enumerate(optimizers)}

    legend_handles = []
    legend_labels = []

    for row_idx, (metric, metric_label) in enumerate(METRIC_LAYOUT):
        for col_idx, (split, split_label) in enumerate(SPLIT_LAYOUT):
            ax = axes[row_idx, col_idx]
            values = grids[split][metric]
            lower_values = lowers[split][metric]
            upper_values = uppers[split][metric]
            sample_values = samples[split][metric]
            for opt_idx, optimizer in enumerate(optimizers):
                y = values[opt_idx]
                lower = lower_values[opt_idx]
                upper = upper_values[opt_idx]
                if args.spread != "none":
                    finite_mask = np.isfinite(lower) & np.isfinite(upper)
                    if np.any(finite_mask):
                        ax.fill_between(
                            np.asarray(batch_sizes)[finite_mask],
                            lower[finite_mask],
                            upper[finite_mask],
                            color=colors[optimizer],
                            alpha=args.spread_alpha,
                            linewidth=0,
                        )
                if args.show_seed_points:
                    _scatter_seed_points(
                        ax,
                        batch_sizes=batch_sizes,
                        series_idx=opt_idx,
                        num_series=len(optimizers),
                        sample_rows=sample_values[opt_idx],
                        color=colors[optimizer],
                        alpha=args.point_alpha,
                        size=args.point_size,
                        jitter_width=args.point_jitter_width,
                    )
                handle = ax.plot(
                    batch_sizes,
                    y,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=colors[optimizer],
                    label=optimizer,
                )[0]
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(handle)
                    legend_labels.append(optimizer)

            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(x) for x in batch_sizes], rotation=45, ha="right")
            ax.grid(alpha=0.25)

            if row_idx == 0:
                ax.set_title(split_label, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(METRIC_LAYOUT) - 1:
                ax.set_xlabel("Batch Size")

    fig.legend(
        legend_handles,
        legend_labels,
        title=args.legend_title,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[bllarse] wrote {png_path}")
    print(f"[bllarse] wrote {pdf_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
