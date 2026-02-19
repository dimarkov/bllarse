#!/usr/bin/env python3
"""
Migrate grouped Weights & Biases runs to MLflow parent/child runs.

Each provided sweep/group name is created (or reused) as a parent MLflow run.
Each successful W&B run in that group is migrated as a nested child run.

This is a legacy/one-off migration utility and requires the optional `wandb`
package at runtime.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable

import mlflow
from mlflow.entities import Metric, ViewType
from mlflow.tracking import MlflowClient

from bllarse.mlflow_utils import load_mlflow_env_defaults


DEFAULT_METRICS = ("ece", "nll", "acc")
PARAM_VALUE_MAX_LEN = 5000
PARAM_KEY_MAX_LEN = 250
_INVALID_MLFLOW_KEY_CHARS = re.compile(r"[^A-Za-z0-9_./ -]")


@dataclass
class SweepMigrationStats:
    sweep_name: str
    finished_runs_found: int = 0
    migrated_runs: int = 0
    skipped_existing_runs: int = 0
    failed_runs: int = 0
    metric_rows_logged: int = 0
    metric_values_logged: int = 0


def _escape_filter_value(value: str) -> str:
    # MLflow filter values are single-quoted strings.
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _sanitize_param_key(key: Any) -> str:
    text = str(key).strip().replace(" ", "_")
    if not text:
        text = "unnamed"
    text = _INVALID_MLFLOW_KEY_CHARS.sub("_", text)
    if len(text) > PARAM_KEY_MAX_LEN:
        text = text[:PARAM_KEY_MAX_LEN]
    return text


def _coerce_param_value(value: Any) -> str:
    if isinstance(value, str):
        out = value
    elif value is None:
        out = "None"
    elif isinstance(value, (bool, int, float)):
        out = str(value)
    else:
        out = json.dumps(value, sort_keys=True, default=str)
    if len(out) > PARAM_VALUE_MAX_LEN:
        out = out[: PARAM_VALUE_MAX_LEN - 3] + "..."
    return out


def _configure_fast_fail_http() -> None:
    """
    Keep transient network issues from stalling migration for long periods.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "0")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _extract_step(row: dict[str, Any], fallback: int) -> int:
    for key in ("_step", "step", "global_step", "epoch"):
        value = row.get(key)
        if _is_finite_number(value):
            step = int(value)
            return max(step, 0)
    return max(fallback, 0)


def _prepare_config_params(config: dict[str, Any], sweep_name: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw_key, raw_value in (config or {}).items():
        if str(raw_key).startswith("_"):
            continue
        key = _sanitize_param_key(raw_key)
        suffix = 2
        while key in params:
            key = f"{_sanitize_param_key(raw_key)}_{suffix}"
            suffix += 1
        params[key] = _coerce_param_value(raw_value)
    params.setdefault("group_id", sweep_name)
    return params


def _iter_search_runs(
    client: MlflowClient,
    experiment_id: str,
    filter_string: str,
    order_by: list[str] | None = None,
    max_results_per_page: int = 5000,
) -> Iterable[Any]:
    page_token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ALL,
            max_results=max_results_per_page,
            order_by=order_by,
            page_token=page_token,
        )
        if not page:
            return
        for run in page:
            yield run
        page_token = getattr(page, "token", None)
        if not page_token:
            return


@contextmanager
def _start_run_quietly(
    *,
    experiment_id: str,
    run_name: str | None,
    tags: dict[str, str],
    nested: bool = False,
    parent_run_id: str | None = None,
):
    """
    Start an MLflow run while suppressing verbose per-run URL prints.
    """
    with io.StringIO() as out_buf, io.StringIO() as err_buf, redirect_stdout(out_buf), redirect_stderr(err_buf):
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
            parent_run_id=parent_run_id,
        )
    try:
        yield run
    except Exception:
        with io.StringIO() as out_buf, io.StringIO() as err_buf, redirect_stdout(out_buf), redirect_stderr(err_buf):
            mlflow.end_run(status="FAILED")
        raise
    else:
        with io.StringIO() as out_buf, io.StringIO() as err_buf, redirect_stdout(out_buf), redirect_stderr(err_buf):
            mlflow.end_run(status="FINISHED")


def _find_existing_parent_run(
    client: MlflowClient,
    experiment_id: str,
    *,
    sweep_name: str,
    wandb_entity: str,
    wandb_project: str,
) -> str | None:
    filter_string = (
        f"tags.is_parent = 'true' and "
        f"tags.migration_source = 'wandb' and "
        f"tags.sweep_name = '{_escape_filter_value(sweep_name)}' and "
        f"tags.wandb_entity = '{_escape_filter_value(wandb_entity)}' and "
        f"tags.wandb_project = '{_escape_filter_value(wandb_project)}'"
    )
    page = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=ViewType.ALL,
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not page:
        return None
    return page[0].info.run_id


def _create_parent_run(
    *,
    sweep_name: str,
    experiment_id: str,
    wandb_entity: str,
    wandb_project: str,
    dry_run: bool,
) -> str:
    if dry_run:
        return f"dryrun-parent-{sweep_name}"

    parent_tags = {
        "is_parent": "true",
        "group_id": sweep_name,
        "sweep_name": sweep_name,
        "migration_source": "wandb",
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
    }
    with _start_run_quietly(
        experiment_id=experiment_id,
        run_name=sweep_name,
        tags=parent_tags,
    ) as parent_run:
        return parent_run.info.run_id


def _existing_migrated_child_ids(
    client: MlflowClient,
    experiment_id: str,
    *,
    parent_run_id: str,
) -> set[str]:
    filter_string = f"tags.mlflow.parentRunId = '{_escape_filter_value(parent_run_id)}'"
    ids: set[str] = set()
    for run in _iter_search_runs(client, experiment_id, filter_string):
        run_id = run.data.tags.get("wandb_run_id")
        if run_id:
            ids.add(run_id)
    return ids


def _extract_commit(run: Any) -> str:
    commit = getattr(run, "commit", "") or ""
    if isinstance(commit, str) and commit.strip():
        return commit.strip()

    config = run.config or {}
    for key in ("git_sha", "git_hash", "github_sha", "git_commit", "commit"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _log_params_with_fallback(params: dict[str, str]) -> None:
    if not params:
        return
    try:
        mlflow.log_params(params)
    except Exception:
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as exc:
                print(f"[bllarse] WARNING: failed to log param '{key}': {exc}")


def _migrate_child_run(
    wandb_run: Any,
    *,
    mlflow_client: MlflowClient,
    sweep_name: str,
    experiment_id: str,
    parent_run_id: str,
    metric_names: tuple[str, ...],
    dry_run: bool,
) -> tuple[int, int]:
    if (wandb_run.state or "").lower() != "finished":
        return 0, 0

    params = _prepare_config_params(wandb_run.config or {}, sweep_name)
    commit = _extract_commit(wandb_run)
    tags = {
        "group_id": sweep_name,
        "migration_source": "wandb",
        "wandb_run_id": wandb_run.id,
        "wandb_run_name": wandb_run.name or "",
        "wandb_state": wandb_run.state or "",
        "wandb_url": getattr(wandb_run, "url", ""),
        "wandb_created_at": str(getattr(wandb_run, "created_at", "")),
    }
    if commit:
        tags["wandb_commit"] = commit

    if dry_run:
        return 0, 0

    metric_rows_logged = 0
    metric_values_logged = 0
    with _start_run_quietly(
        experiment_id=experiment_id,
        run_name=wandb_run.name or None,
        tags=tags,
        nested=True,
        parent_run_id=parent_run_id,
    ):
        _log_params_with_fallback(params)

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Expected an active MLflow run while migrating a W&B child run.")
        run_id = active_run.info.run_id
        metric_batch: list[Metric] = []
        timestamp_ms = int(time.time() * 1000)
        for idx, row in enumerate(wandb_run.scan_history(keys=["_step", *metric_names], page_size=1000)):
            step = _extract_step(row, fallback=idx)
            payload = [name for name in metric_names if _is_finite_number(row.get(name))]
            if not payload:
                continue
            for name in payload:
                metric_batch.append(
                    Metric(
                        key=name,
                        value=float(row[name]),
                        timestamp=timestamp_ms,
                        step=step,
                    )
                )
            if len(metric_batch) >= 1000:
                mlflow_client.log_batch(run_id=run_id, metrics=metric_batch, params=[], tags=[])
                metric_batch = []
            metric_rows_logged += 1
            metric_values_logged += len(payload)

        if metric_batch:
            mlflow_client.log_batch(run_id=run_id, metrics=metric_batch, params=[], tags=[])

    return metric_rows_logged, metric_values_logged


def migrate_sweep(
    *,
    wandb_api: Any,
    mlflow_client: MlflowClient,
    wandb_entity: str,
    wandb_project: str,
    sweep_name: str,
    experiment_id: str,
    metric_names: tuple[str, ...],
    reuse_parent: bool,
    skip_existing_children: bool,
    dry_run: bool,
    print_every: int,
    max_runs_per_sweep: int | None,
    index_offset: int,
    num_runs: int | None,
) -> SweepMigrationStats:
    stats = SweepMigrationStats(sweep_name=sweep_name)
    project_path = f"{wandb_entity}/{wandb_project}"
    runs = list(
        wandb_api.runs(
            project_path,
            filters={"group": sweep_name, "state": "finished"},
            per_page=200,
        )
    )
    runs = [run for run in runs if (run.state or "").lower() == "finished"]
    runs.sort(key=lambda run: str(getattr(run, "created_at", "")))
    total_finished = len(runs)

    if index_offset < 0:
        raise ValueError("index_offset must be >= 0")
    if index_offset > total_finished:
        raise ValueError(
            f"index_offset ({index_offset}) exceeds finished run count ({total_finished}) for '{sweep_name}'."
        )
    if num_runs is not None and num_runs <= 0:
        raise ValueError("num_runs must be > 0 when provided")

    if num_runs is None:
        runs = runs[index_offset:]
    else:
        runs = runs[index_offset : index_offset + num_runs]

    if max_runs_per_sweep is not None:
        runs = runs[:max_runs_per_sweep]
    stats.finished_runs_found = len(runs)

    if not runs:
        print(f"[bllarse] Sweep '{sweep_name}': no finished runs found.")
        return stats

    parent_run_id = None
    if reuse_parent and not dry_run:
        parent_run_id = _find_existing_parent_run(
            mlflow_client,
            experiment_id,
            sweep_name=sweep_name,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
    if parent_run_id is None:
        parent_run_id = _create_parent_run(
            sweep_name=sweep_name,
            experiment_id=experiment_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            dry_run=dry_run,
        )
        print(f"[bllarse] Sweep '{sweep_name}': created parent run {parent_run_id}.")
    else:
        print(f"[bllarse] Sweep '{sweep_name}': reusing parent run {parent_run_id}.")

    existing_ids: set[str] = set()
    if skip_existing_children and not dry_run:
        existing_ids = _existing_migrated_child_ids(
            mlflow_client,
            experiment_id,
            parent_run_id=parent_run_id,
        )
        if existing_ids:
            print(
                f"[bllarse] Sweep '{sweep_name}': {len(existing_ids)} child runs already migrated; "
                "these will be skipped."
            )

    for idx, run in enumerate(runs, start=1):
        if run.id in existing_ids:
            stats.skipped_existing_runs += 1
            continue
        try:
            row_count, value_count = _migrate_child_run(
                run,
                mlflow_client=mlflow_client,
                sweep_name=sweep_name,
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                metric_names=metric_names,
                dry_run=dry_run,
            )
            stats.migrated_runs += 1
            stats.metric_rows_logged += row_count
            stats.metric_values_logged += value_count
        except Exception as exc:
            stats.failed_runs += 1
            print(f"[bllarse] ERROR: failed to migrate W&B run {run.id} ({run.name}): {exc}")

        if print_every > 0 and idx % print_every == 0:
            print(
                f"[bllarse] Sweep '{sweep_name}': processed {idx}/{len(runs)} "
                f"(migrated={stats.migrated_runs}, skipped={stats.skipped_existing_runs}, "
                f"failed={stats.failed_runs})."
            )

    full_sweep_pass = index_offset == 0 and num_runs is None and max_runs_per_sweep is None
    if not dry_run and full_sweep_pass:
        mlflow_client.set_tag(parent_run_id, "source_finished_runs", str(stats.finished_runs_found))
        mlflow_client.set_tag(parent_run_id, "migrated_child_runs", str(stats.migrated_runs))
        mlflow_client.set_tag(parent_run_id, "skipped_existing_child_runs", str(stats.skipped_existing_runs))
        mlflow_client.set_tag(parent_run_id, "failed_child_runs", str(stats.failed_runs))

    return stats


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Migrate selected W&B sweeps/groups into MLflow parent/child runs. "
            "Only successful W&B runs are migrated."
        )
    )
    ap.add_argument(
        "--wandb-entity",
        type=str,
        default="verses_ai",
        help="Weights & Biases entity/team name.",
    )
    ap.add_argument(
        "--wandb-project",
        type=str,
        default="bllarse_experiments",
        help="Weights & Biases project name.",
    )
    ap.add_argument(
        "--sweep",
        dest="sweeps",
        action="append",
        required=True,
        help="Sweep/group name to migrate. Repeat this flag for multiple sweeps.",
    )
    ap.add_argument(
        "--mlflow-experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "bllarse"),
        help="Destination MLflow experiment name.",
    )
    ap.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="Optional MLflow tracking URI override.",
    )
    ap.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="Metric key to migrate from W&B history (default: ece, nll, acc).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to MLflow; only validate and count migratable runs.",
    )
    ap.add_argument(
        "--reuse-parent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing parent run for the same sweep if present (default: true).",
    )
    ap.add_argument(
        "--skip-existing-children",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip child runs already migrated under the parent by wandb_run_id tag (default: true).",
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Progress print interval (0 disables periodic progress prints).",
    )
    ap.add_argument(
        "--max-runs-per-sweep",
        type=int,
        default=None,
        help="Optional cap on finished runs migrated per sweep (for staged migrations).",
    )
    ap.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="Offset into the finished run list per sweep (applied after sorting by created_at).",
    )
    ap.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Optional number of runs to process from index-offset per sweep.",
    )
    ap.add_argument(
        "--wandb-timeout",
        type=int,
        default=30,
        help="W&B API timeout in seconds.",
    )
    return ap.parse_args()


def main() -> None:
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "This legacy migration tool requires the optional 'wandb' package. "
            "Install it in the active environment, then re-run this command."
        ) from exc

    args = _parse_args()
    metric_names = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking.fluent").setLevel(logging.WARNING)
    _configure_fast_fail_http()

    load_mlflow_env_defaults()
    tracking_uri = args.mlflow_tracking_uri.strip() or os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not args.dry_run:
        print(
            "[bllarse] WARNING: no MLflow tracking URI resolved; using MLflow default tracking store.",
        )

    experiment = mlflow.set_experiment(args.mlflow_experiment)
    experiment_id = experiment.experiment_id
    client = MlflowClient()
    wandb_api = wandb.Api(timeout=args.wandb_timeout)

    print(
        f"[bllarse] Starting migration for {len(args.sweeps)} sweep(s): {', '.join(args.sweeps)} "
        f"| W&B={args.wandb_entity}/{args.wandb_project} "
        f"| MLflow experiment={args.mlflow_experiment} ({experiment_id}) "
        f"| dry_run={args.dry_run}"
    )

    all_stats: list[SweepMigrationStats] = []
    for sweep_name in args.sweeps:
        stats = migrate_sweep(
            wandb_api=wandb_api,
            mlflow_client=client,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            sweep_name=sweep_name,
            experiment_id=experiment_id,
            metric_names=metric_names,
            reuse_parent=args.reuse_parent,
            skip_existing_children=args.skip_existing_children,
            dry_run=args.dry_run,
            print_every=args.print_every,
            max_runs_per_sweep=args.max_runs_per_sweep,
            index_offset=args.index_offset,
            num_runs=args.num_runs,
        )
        all_stats.append(stats)

    total_found = sum(s.finished_runs_found for s in all_stats)
    total_migrated = sum(s.migrated_runs for s in all_stats)
    total_skipped = sum(s.skipped_existing_runs for s in all_stats)
    total_failed = sum(s.failed_runs for s in all_stats)
    total_metric_rows = sum(s.metric_rows_logged for s in all_stats)
    total_metric_values = sum(s.metric_values_logged for s in all_stats)

    print("[bllarse] Migration summary:")
    for stats in all_stats:
        print(
            f"  - {stats.sweep_name}: "
            f"finished={stats.finished_runs_found}, migrated={stats.migrated_runs}, "
            f"skipped={stats.skipped_existing_runs}, failed={stats.failed_runs}, "
            f"metric_rows={stats.metric_rows_logged}, metric_values={stats.metric_values_logged}"
        )
    print(
        "[bllarse] Totals: "
        f"finished={total_found}, migrated={total_migrated}, skipped={total_skipped}, "
        f"failed={total_failed}, metric_rows={total_metric_rows}, metric_values={total_metric_values}"
    )


if __name__ == "__main__":
    main()
