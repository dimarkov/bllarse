#!/usr/bin/env python3
"""Submit a Python-defined experiment sweep as a SLURM array."""

import argparse
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bllarse.tools.module import get_module_from_source_path

DEFAULT_JOB_SCRIPT = Path("src/slurm/jobs/slurm_run_config_docker.sh")


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return default


def _sweep_name(configs: Sequence[Any], sweep_source: Path) -> str:
    group_ids = {
        value
        for config in configs
        if (value := _config_value(config, "group_id"))
    }
    if len(group_ids) == 1:
        return str(group_ids.pop())
    if len(group_ids) > 1:
        print(
            "[bllarse] WARNING: Multiple group_id values found; "
            f"using the sweep filename '{sweep_source.stem}' for the parent run."
        )
    return sweep_source.stem


def _create_mlflow_parent(
    configs: Sequence[Any],
    sweep_source: Path,
    *,
    total: int,
    chunk_size: int,
    chunk_offset: int,
) -> str:
    from bllarse.mlflow_utils import load_mlflow_env_defaults

    load_mlflow_env_defaults()
    import mlflow

    tracking_uri = (
        _config_value(configs[0], "mlflow_tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI")
    )
    experiment = (
        _config_value(configs[0], "mlflow_experiment")
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or "bllarse"
    )
    name = _sweep_name(configs, sweep_source)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    tags = {
        "sweep_source": str(sweep_source),
        "sweep_name": name,
        "group_id": name,
        "is_parent": "true",
        "sweep_size_total": str(total),
        "sweep_chunk_size": str(chunk_size),
        "sweep_chunk_offset": str(chunk_offset),
    }
    with mlflow.start_run(run_name=name, tags=tags) as parent:
        return parent.info.run_id


def run_sweep(
    sweep_source: str | Path,
    *,
    venv_name: str = ".venv",
    max_concurrent: int = 7,
    job_name: str | None = None,
    job_script: str | Path = DEFAULT_JOB_SCRIPT,
    index_offset: int = 0,
    num_jobs: int | None = None,
    cpus_per_task: int | None = None,
    parent_run_id: str | None = None,
    exclude_nodes: str | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Validate and submit a sweep, returning the completed sbatch process."""
    source = Path(sweep_source).expanduser().resolve()
    script = Path(job_script).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"SLURM job script not found: {script}")
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be > 0")
    if cpus_per_task is not None and cpus_per_task <= 0:
        raise ValueError("cpus_per_task must be > 0")

    sweep = get_module_from_source_path(source)
    configs = sweep.create_configs()
    total = len(configs)
    if total == 0:
        raise ValueError("create_configs() returned an empty sequence")
    if index_offset < 0 or index_offset >= total:
        raise ValueError(f"index_offset={index_offset} out of range (0..{total - 1})")

    chunk_size = total - index_offset if num_jobs is None else num_jobs
    if chunk_size <= 0:
        raise ValueError("num_jobs must be > 0")
    if index_offset + chunk_size > total:
        raise ValueError(
            f"index_offset + num_jobs exceeds the {total} available configs"
        )

    selected_configs = configs[index_offset:index_offset + chunk_size]
    mlflow_configs = [
        config
        for config in selected_configs
        if bool(_config_value(config, "enable_mlflow", False))
    ]
    uses_mlflow = bool(mlflow_configs)
    if parent_run_id and not uses_mlflow:
        print(
            "[bllarse] WARNING: --parent-run-id was supplied, but no selected "
            "configuration enables MLflow."
        )

    env = os.environ.copy()
    if uses_mlflow and not parent_run_id and not dry_run:
        parent_run_id = _create_mlflow_parent(
            mlflow_configs,
            source,
            total=total,
            chunk_size=chunk_size,
            chunk_offset=index_offset,
        )
        print(f"[bllarse] MLflow parent run id: {parent_run_id}")

    env.update(
        {
            "BLLARSE_SWEEP_SOURCE": str(source),
            "VENV_NAME": venv_name,
            "INDEX_OFFSET": str(index_offset),
        }
    )
    if parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        if value := os.environ.get(key):
            env[key] = value

    command = [
        "sbatch",
        "--export=ALL",
        f"--array=0-{chunk_size - 1}%{max_concurrent}",
        f"--job-name={job_name or source.stem}",
    ]
    if cpus_per_task is not None:
        command.append(f"--cpus-per-task={cpus_per_task}")
    if exclude_nodes:
        command.append(f"--exclude={exclude_nodes}")
    command.append(str(script))

    if dry_run:
        print(shlex.join(command))
        return None
    Path("slurm_logs").mkdir(exist_ok=True)
    return subprocess.run(command, env=env, check=True, text=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_source", type=Path)
    parser.add_argument("--venv", default=".venv")
    parser.add_argument("--max-concurrent", type=int, default=7)
    parser.add_argument("--job-name")
    parser.add_argument("--job-script", type=Path, default=DEFAULT_JOB_SCRIPT)
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--num-jobs", type=int)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--exclude-nodes")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_sweep(
        args.sweep_source,
        venv_name=args.venv,
        max_concurrent=args.max_concurrent,
        job_name=args.job_name,
        job_script=args.job_script,
        index_offset=args.index_offset,
        num_jobs=args.num_jobs,
        cpus_per_task=args.cpus_per_task,
        parent_run_id=args.parent_run_id,
        exclude_nodes=args.exclude_nodes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
