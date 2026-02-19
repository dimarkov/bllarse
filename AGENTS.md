# Agent Handoff Notes

## Repo context
- Active branch for this file/state: `experimental_infra`
- Primary goal on this branch: maintain shared experiment infrastructure (SLURM launch flow, sweep orchestration, and MLflow integration) for model-agnostic training workflows.
- Main training script for current baseline workflows: `scripts/finetuning.py`
- Sweep launchers:
  - `python src/bllarse/tools/run_sweep.py <sweep.py> ...` (SLURM array launcher)
  - `python src/bllarse/tools/run_config.py <sweep.py> <config_idx>` (single config run)
- Keep infra changes generic when possible; avoid coupling branch-level infra to one specific model implementation.

## MLflow conventions
- Nested child runs are wired through `MLFLOW_PARENT_RUN_ID`.
- Common config keys expected by launcher/training scripts:
  - `enable_mlflow`
  - `mlflow_tracking_uri` (optional)
  - `mlflow_experiment` (optional, default `bllarse`)
  - `group_id`
  - `uid` (optional run name)
- Credentials helper: `src/bllarse/mlflow_utils.py` reads `~/.mlflow/credentials`.

## Environment / install
- Typical setup:
  - `uv venv`
  - `uv sync`
  - `source .venv/bin/activate`
- If using a non-default venv, pass it via `--venv` when launching sweeps.

## SLURM/login-node workflow
- Preferred sweep submission flow:
  - `python -m bllarse.tools.run_sweep ... --job-script src/slurm/jobs/slurm_run_config_docker.sh`
- Useful options:
  - `--cpus-per-task <N>` to request/pin CPU resources for data loading.
  - `--index-offset` + `--num-jobs` for chunking large sweeps.
  - `--parent-run-id <RUN_ID>` to attach additional submissions to an existing MLflow parent run.
- Behavior notes:
  - If `--parent-run-id` is omitted and MLflow is enabled, `run_sweep.py` creates a parent run automatically.
  - `src/slurm/jobs/slurm_run_config_docker.sh` activates `${VENV_NAME}` in Docker and executes `run_config.py`.
  - `scripts/start_docker_sbatch.sh` handles container env passthrough and CPU/shm settings.

## Quick local smoke commands
- Single config via sweep runner:
  - `python src/bllarse/tools/run_config.py bllarse_sweeps/llf_demo_sweep.py 0`
- Direct baseline training script:
  - `python scripts/finetuning.py --tune-mode last_layer --epochs 1 --enable-mlflow`
