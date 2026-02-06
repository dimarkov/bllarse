#!/usr/bin/env python3
"""
Launch a SLURM array sweep from a Python sweep file.

The sweep file must define:
  - create_configs() -> List[object]
  - run(config: object) -> None
"""
import argparse
import os
import subprocess
from bllarse.tools.module import get_module_from_source_path

def _format_sweep_name(sweep_source: str) -> str:
    base = os.path.basename(sweep_source)
    if base.endswith(".py"):
        base = base[:-3]
    return base

def _resolve_group_id(configs, sweep_source: str) -> str:
    group_id = None
    for cfg in configs:
        cfg_group = cfg.get("group_id")
        if not cfg_group:
            continue
        if group_id is None:
            group_id = cfg_group
        elif cfg_group != group_id:
            print(
                "[bllarse] WARNING: Multiple group_id values detected in sweep configs; "
                f"using '{group_id}' from the first config."
            )
            break
    return group_id or _format_sweep_name(sweep_source)

def run_sweep(sweep_source: str, venv_name: str, max_concurrent: int, job_name: str | None, job_script: str):
    sweep = get_module_from_source_path(sweep_source)
    all_configs = sweep.create_configs()
    n = len(all_configs)
    if n == 0:
        raise ValueError("create_configs() returned an empty list.")
    env = os.environ.copy()
    env["BLLARSE_SWEEP_SOURCE"] = sweep_source
    env["VENV_NAME"] = venv_name
    name = job_name or "bllarse_sweep"

    # If MLflow is enabled in the sweep configs, create a single parent run and
    # pass its run_id to each array task via env var. Child runs are created in
    # finetuning.py using nested=True + parent_run_id.
    enable_mlflow = any(
        cfg.get("enable_mlflow", False) or cfg.get("enable_wandb", False)
        for cfg in all_configs
    )
    if enable_mlflow:
        from bllarse.mlflow_utils import load_mlflow_env_defaults

        load_mlflow_env_defaults()
        tracking_uri = (
            all_configs[0].get("mlflow_tracking_uri")
            or env.get("MLFLOW_TRACKING_URI")
        )
        experiment = (
            all_configs[0].get("mlflow_experiment")
            or env.get("MLFLOW_EXPERIMENT_NAME")
            or "bllarse"
        )
        sweep_name = _resolve_group_id(all_configs, sweep_source)
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)

            parent_tags = {
                "sweep_source": sweep_source,
                "sweep_name": sweep_name,
                "group_id": sweep_name,
                "is_parent": "true",
                "sweep_size": str(n),
            }
            with mlflow.start_run(run_name=sweep_name, tags=parent_tags) as parent:
                env["MLFLOW_PARENT_RUN_ID"] = parent.info.run_id
        except Exception as exc:
            print(f"[bllarse] WARNING: Failed to create MLflow parent run: {exc}")

    subprocess.run(
        [
            "sbatch",
            "--array", f"0-{n-1}%{max_concurrent}",
            "--job-name", name,
            job_script,
        ],
        env=env,
        check=True,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_source", type=str)
    ap.add_argument("--venv", type=str, default=".venv")
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--job-name", type=str, default=None)
    ap.add_argument("--job-script", type=str, default="slurm/jobs/slurm_run_config.sh")
    args = ap.parse_args()
    run_sweep(args.sweep_source, args.venv, args.max_concurrent, args.job_name, args.job_script)
if __name__ == "__main__":
    main()
