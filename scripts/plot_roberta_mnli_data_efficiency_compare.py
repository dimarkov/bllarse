from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

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


def _parse_parent_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=RUN_ID, got '{spec}'.")
    label, run_id = spec.split("=", 1)
    label = label.strip()
    run_id = run_id.strip()
    if not label or not run_id:
        raise ValueError(f"Expected LABEL=RUN_ID, got '{spec}'.")
    return label, run_id


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


def _collect_history_parent(
    client: MlflowClient,
    *,
    experiment_id: str,
    parent_run_id: str,
) -> dict[str, dict[str, DefaultDict[int, list[float]]]]:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
        max_results=500,
    )
    if not runs:
        raise ValueError(f"No child runs found for history parent {parent_run_id}.")

    grouped: dict[str, dict[str, DefaultDict[int, list[float]]]] = {
        split: {metric: defaultdict(list) for metric, _ in METRIC_LAYOUT}
        for split, _ in SPLIT_LAYOUT
    }

    for run in runs:
        for split, _ in SPLIT_LAYOUT:
            for metric, _ in METRIC_LAYOUT:
                history = client.get_metric_history(run.info.run_id, MLFLOW_KEYS[split][metric])
                for point in history:
                    if np.isfinite(point.value):
                        grouped[split][metric][int(point.step)].append(float(point.value))
    return grouped


def _collect_subset_parent(
    client: MlflowClient,
    *,
    experiment_id: str,
    parent_run_id: str,
) -> dict[str, dict[str, DefaultDict[int, list[float]]]]:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
        max_results=1000,
    )
    if not runs:
        raise ValueError(f"No child runs found for subset parent {parent_run_id}.")

    grouped: dict[str, dict[str, DefaultDict[int, list[float]]]] = {
        split: {metric: defaultdict(list) for metric, _ in METRIC_LAYOUT}
        for split, _ in SPLIT_LAYOUT
    }

    for run in runs:
        sample_size = int(run.data.params["max_train_samples"])
        for split, _ in SPLIT_LAYOUT:
            for metric, _ in METRIC_LAYOUT:
                value = run.data.metrics.get(MLFLOW_KEYS[split][metric])
                if value is not None and np.isfinite(value):
                    grouped[split][metric][sample_size].append(float(value))
    return grouped


def _scatter_seed_points(
    ax,
    *,
    x_values: list[int],
    sample_lists: list[list[float]],
    series_idx: int,
    num_series: int,
    color,
    alpha: float,
    size: float,
) -> None:
    if num_series <= 1:
        series_offset_scale = 0.0
    else:
        series_offset_scale = np.linspace(-0.02, 0.02, num_series)[series_idx]

    for x, values in zip(x_values, sample_lists):
        if not values:
            continue
        seed_offsets = np.linspace(-0.01, 0.01, len(values))
        offsets = (series_offset_scale + seed_offsets) * x
        ax.scatter(
            x + offsets,
            values,
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="none",
            zorder=3,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot MNLI data-efficiency curves from MLflow parents.")
    parser.add_argument("--experiment-id", type=str, default="1")
    parser.add_argument("--history-parent", action="append", default=[], help="LABEL=RUN_ID")
    parser.add_argument("--subset-parent", action="append", default=[], help="LABEL=RUN_ID")
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--title", type=str, default="RoBERTa-large MNLI Data Efficiency")
    parser.add_argument("--summary-stat", choices=["mean", "iqm"], default="iqm")
    parser.add_argument("--share-y-by-row", action="store_true")
    parser.add_argument("--show-seed-points", action="store_true")
    parser.add_argument("--point-alpha", type=float, default=0.5)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--nll-ymax", type=float, default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    load_mlflow_env_defaults()
    client = MlflowClient()

    method_data: dict[str, dict[str, dict[str, DefaultDict[int, list[float]]]]] = {}
    method_order: list[str] = []

    for spec in args.history_parent:
        label, run_id = _parse_parent_spec(spec)
        method_order.append(label)
        method_data[label] = _collect_history_parent(
            client,
            experiment_id=args.experiment_id,
            parent_run_id=run_id,
        )

    for spec in args.subset_parent:
        label, run_id = _parse_parent_spec(spec)
        method_order.append(label)
        method_data[label] = _collect_subset_parent(
            client,
            experiment_id=args.experiment_id,
            parent_run_id=run_id,
        )

    if not method_order:
        raise ValueError("Provide at least one --history-parent or --subset-parent.")

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

    cmap = plt.get_cmap("tab10", len(method_order))
    colors = {label: cmap(idx) for idx, label in enumerate(method_order)}

    legend_handles = []
    legend_labels = []

    for row_idx, (metric, metric_label) in enumerate(METRIC_LAYOUT):
        for col_idx, (split, split_label) in enumerate(SPLIT_LAYOUT):
            ax = axes[row_idx, col_idx]
            for series_idx, label in enumerate(method_order):
                grouped = method_data[label][split][metric]
                x_values = sorted(grouped)
                y_values = [_summary_stat(grouped[x], args.summary_stat) for x in x_values]
                handle = ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=colors[label],
                    label=label,
                )[0]
                if args.show_seed_points:
                    _scatter_seed_points(
                        ax,
                        x_values=x_values,
                        sample_lists=[sorted(grouped[x]) for x in x_values],
                        series_idx=series_idx,
                        num_series=len(method_order),
                        color=colors[label],
                        alpha=args.point_alpha,
                        size=args.point_size,
                    )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(handle)
                    legend_labels.append(label)

            ax.grid(alpha=0.25)
            ax.set_xticks(sorted({x for label in method_order for x in method_data[label][split][metric]}))
            ax.tick_params(axis="x", rotation=45)
            if row_idx == 0:
                ax.set_title(split_label, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(METRIC_LAYOUT) - 1:
                ax.set_xlabel("Seen Training Examples")
            if metric == "nll" and args.nll_ymax is not None:
                ax.set_ylim(top=args.nll_ymax)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Method",
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
