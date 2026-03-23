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


TRACKED_METRICS = {
    "loss": "Train Loss",
    "val_matched_acc": "Matched Accuracy",
    "val_matched_nll": "Matched NLL",
    "val_matched_ece": "Matched ECE",
    "val_mismatched_acc": "Mismatched Accuracy",
    "val_mismatched_nll": "Mismatched NLL",
    "val_mismatched_ece": "Mismatched ECE",
}


def _float_param(run, key: str) -> float:
    return float(run.data.params[key])


def _match_filter(run, learning_rate: float | None) -> bool:
    if learning_rate is None:
        return True
    return abs(_float_param(run, "learning_rate") - learning_rate) < 1e-12


def _metric_history(client: MlflowClient, run_id: str, key: str) -> Dict[int, float]:
    history = client.get_metric_history(run_id, key)
    return {int(item.step): float(item.value) for item in history}


def _aggregate_histories(
    client: MlflowClient,
    run_ids: Iterable[str],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_step: Dict[int, List[float]] = defaultdict(list)
    for run_id in run_ids:
        history = _metric_history(client, run_id, metric_key)
        for step, value in history.items():
            per_step[step].append(value)

    steps = np.array(sorted(per_step), dtype=np.int32)
    means = np.array([np.mean(per_step[step]) for step in steps], dtype=np.float64)
    stds = np.array([np.std(per_step[step], ddof=0) for step in steps], dtype=np.float64)
    return steps, means, stds


def _final_values(client: MlflowClient, run_ids: Iterable[str], metric_key: str) -> List[float]:
    values: List[float] = []
    for run_id in run_ids:
        history = _metric_history(client, run_id, metric_key)
        if not history:
            continue
        values.append(history[max(history)])
    return values


def _build_summary(client: MlflowClient, run_ids: List[str]) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_runs": len(run_ids), "metrics": {}}
    for metric_key in TRACKED_METRICS:
        values = _final_values(client, run_ids, metric_key)
        if not values:
            continue
        summary["metrics"][metric_key] = {
            "mean_final": float(np.mean(values)),
            "std_final": float(np.std(values, ddof=0)),
        }
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot epoch-wise RoBERTa MNLI MLflow histories.")
    parser.add_argument("--experiment-id", type=str, default="1")
    parser.add_argument("--parent-run-id", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--title", type=str, default="RoBERTa MNLI Seed-Averaged Metrics")
    return parser


def main(args: argparse.Namespace) -> None:
    load_mlflow_env_defaults()
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{args.parent_run_id}"',
        max_results=500,
    )
    runs = [run for run in runs if _match_filter(run, args.learning_rate)]
    if not runs:
        raise ValueError("No MLflow child runs matched the requested filters.")

    run_ids = [run.info.run_id for run in runs]
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(args.title, fontsize=14)

    panel_specs = [
        ("loss", None, axes[0, 0]),
        ("val_matched_acc", "val_mismatched_acc", axes[0, 1]),
        ("val_matched_nll", "val_mismatched_nll", axes[1, 0]),
        ("val_matched_ece", "val_mismatched_ece", axes[1, 1]),
    ]

    colors = {
        "loss": "#2f4858",
        "matched": "#15616d",
        "mismatched": "#ff7d00",
    }

    for primary_key, secondary_key, ax in panel_specs:
        steps, mean, std = _aggregate_histories(client, run_ids, primary_key)
        if secondary_key is None:
            ax.plot(steps, mean, color=colors["loss"], label=TRACKED_METRICS[primary_key], lw=2)
            ax.fill_between(steps, mean - std, mean + std, color=colors["loss"], alpha=0.18)
            ax.set_title(TRACKED_METRICS[primary_key])
        else:
            ax.plot(steps, mean, color=colors["matched"], label="Matched", lw=2)
            ax.fill_between(steps, mean - std, mean + std, color=colors["matched"], alpha=0.18)

            steps_2, mean_2, std_2 = _aggregate_histories(client, run_ids, secondary_key)
            ax.plot(steps_2, mean_2, color=colors["mismatched"], label="Mismatched", lw=2)
            ax.fill_between(steps_2, mean_2 - std_2, mean_2 + std_2, color=colors["mismatched"], alpha=0.18)
            title = TRACKED_METRICS[primary_key].replace("Matched ", "")
            ax.set_title(title)
            ax.legend(frameon=False)

        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = _build_summary(client, run_ids)
    summary.update(
        {
            "parent_run_id": args.parent_run_id,
            "learning_rate": args.learning_rate,
            "run_ids": run_ids,
        }
    )
    summary_path = output_prefix.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[bllarse] wrote {png_path}")
    print(f"[bllarse] wrote {pdf_path}")
    print(f"[bllarse] wrote {summary_path}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
