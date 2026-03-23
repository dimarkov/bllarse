from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

from bllarse.mlflow_utils import load_mlflow_env_defaults


METRIC_SPECS = {
    "matched": {
        "acc": ("acc", "Matched Accuracy"),
        "nll": ("nll", "Matched NLL"),
        "ece": ("ece", "Matched ECE"),
    },
    "mismatched": {
        "acc": ("val_mismatched_acc", "Mismatched Accuracy"),
        "nll": ("val_mismatched_nll", "Mismatched NLL"),
        "ece": ("val_mismatched_ece", "Mismatched ECE"),
    },
}


def _float_param(run, key: str) -> float:
    return float(run.data.params[key])


def _int_param(run, key: str) -> int:
    return int(run.data.params[key])


def _group_runs(
    runs: Iterable,
    metric_keys: Dict[str, Tuple[str, str]],
) -> Tuple[List[int], List[int], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    run_list = list(runs)
    batch_sizes = sorted({_int_param(run_list[i], "train_batch_size") for i in range(len(run_list))})
    num_iters = sorted({_int_param(run_list[i], "num_update_iters") for i in range(len(run_list))})
    batch_index = {value: idx for idx, value in enumerate(batch_sizes)}
    iter_index = {value: idx for idx, value in enumerate(num_iters)}

    per_metric_values: Dict[str, Dict[Tuple[int, int], List[float]]] = {
        metric_name: defaultdict(list) for metric_name in metric_keys
    }
    counts: Dict[str, Dict[str, int]] = {}

    for run in run_list:
        batch_size = _int_param(run, "train_batch_size")
        num_update_iters = _int_param(run, "num_update_iters")
        for metric_name, (metric_key, _) in metric_keys.items():
            value = run.data.metrics.get(metric_key)
            if value is None or not np.isfinite(value):
                continue
            per_metric_values[metric_name][(batch_size, num_update_iters)].append(float(value))

    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}

    for metric_name in metric_keys:
        mean_grid = np.full((len(batch_sizes), len(num_iters)), np.nan, dtype=np.float64)
        std_grid = np.full((len(batch_sizes), len(num_iters)), np.nan, dtype=np.float64)
        count_grid: Dict[str, int] = {}
        for (batch_size, num_update_iters), values in per_metric_values[metric_name].items():
            row = batch_index[batch_size]
            col = iter_index[num_update_iters]
            mean_grid[row, col] = float(np.mean(values))
            std_grid[row, col] = float(np.std(values, ddof=0))
            count_grid[f"bs{batch_size}_iters{num_update_iters}"] = len(values)
        means[metric_name] = mean_grid
        stds[metric_name] = std_grid
        counts[metric_name] = count_grid

    return batch_sizes, num_iters, means, stds, counts


def _plot_metric(
    ax,
    values: np.ndarray,
    stds: np.ndarray,
    *,
    title: str,
    cmap: str,
    batch_sizes: List[int],
    num_iters: List[int],
    annotate: bool,
) -> None:
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("CAVI Iterations")
    ax.set_ylabel("Batch Size")
    ax.set_xticks(np.arange(len(num_iters)))
    ax.set_xticklabels([str(x) for x in num_iters])
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_yticklabels([str(x) for x in batch_sizes])

    if annotate:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                std = stds[row, col]
                if not np.isfinite(value):
                    label = "NA"
                else:
                    label = f"{value:.4f}\n±{std:.4f}"
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if np.isfinite(value) and image.norm(value) > 0.55 else "black",
                )

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _build_summary(
    *,
    batch_sizes: List[int],
    num_iters: List[int],
    means: Dict[str, np.ndarray],
    stds: Dict[str, np.ndarray],
    counts: Dict[str, Dict[str, int]],
    split: str,
    parent_run_id: str,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "parent_run_id": parent_run_id,
        "split": split,
        "batch_sizes": batch_sizes,
        "num_update_iters": num_iters,
        "metrics": {},
    }
    for metric_name in means:
        grid_summary = {}
        for row, batch_size in enumerate(batch_sizes):
            for col, num_update_iters in enumerate(num_iters):
                key = f"bs{batch_size}_iters{num_update_iters}"
                mean = means[metric_name][row, col]
                std = stds[metric_name][row, col]
                grid_summary[key] = {
                    "mean": None if not np.isfinite(mean) else float(mean),
                    "std": None if not np.isfinite(std) else float(std),
                    "count": int(counts[metric_name].get(key, 0)),
                }
        summary["metrics"][metric_name] = grid_summary
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot seed-averaged RoBERTa MNLI IBProbit heatmaps.")
    parser.add_argument("--experiment-id", type=str, default="1")
    parser.add_argument("--parent-run-id", type=str, required=True)
    parser.add_argument("--split", choices=["matched", "mismatched"], default="matched")
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument(
        "--title",
        type=str,
        default="RoBERTa-large MNLI IBProbit Single-Pass Sweep",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    load_mlflow_env_defaults()
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{args.parent_run_id}"',
        max_results=500,
    )
    if not runs:
        raise ValueError("No MLflow child runs matched the requested parent run.")

    metric_keys = METRIC_SPECS[args.split]
    batch_sizes, num_iters, means, stds, counts = _group_runs(runs, metric_keys)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(args.title, fontsize=14)

    for ax, metric_name in zip(axes, ["acc", "nll", "ece"]):
        _, title = metric_keys[metric_name]
        _plot_metric(
            ax,
            means[metric_name],
            stds[metric_name],
            title=title,
            cmap=args.cmap,
            batch_sizes=batch_sizes,
            num_iters=num_iters,
            annotate=args.annotate,
        )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = _build_summary(
        batch_sizes=batch_sizes,
        num_iters=num_iters,
        means=means,
        stds=stds,
        counts=counts,
        split=args.split,
        parent_run_id=args.parent_run_id,
    )
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[bllarse] wrote {png_path}")
    print(f"[bllarse] wrote {pdf_path}")
    print(f"[bllarse] wrote {json_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
