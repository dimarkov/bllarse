# Agent Handoff Notes

## Repo context
- Active branch for this file/state: `experimental_infra_vbll_baseline`
- Parent infra branch: `experimental_infra`
- Primary goal on this branch: run VBLL baseline finetuning/sweeps on SLURM and log to MLflow, while preserving compatibility with the existing infra.
- Main training script for JAX/IBProbit: `scripts/finetuning.py`
- Sweep launchers:
  - `python src/bllarse/tools/run_sweep.py <sweep.py> ...` (SLURM array, creates MLflow parent run)
  - `python src/bllarse/tools/run_config.py <sweep.py> <config_idx>` (single config)

## VBLL integration status
- VBLL work from `origin/vbll` was cherry-picked onto this branch:
  - `c14ebc6e7e29d22d79daec5eadb267d6dc751a57` (`scripts/vbll_pytorch/*`)
  - `b1c18129cd2438b3d412ee6bf71ba771d612e503` (`.gitmodules` + `scripts/vbll_pytorch/scaling_mlps` submodule)
- VBLL training script: `scripts/vbll_pytorch/finetuning_vbll.py`
- Adapter for config-based sweeps: `run_vbll_training_from_config` in `src/bllarse/tools/adapters.py`
- Smoke sweep config: `bllarse_sweeps/vbll_smoke_sweep.py`

## MLflow conventions
- Both JAX and VBLL scripts now support nested child runs via `MLFLOW_PARENT_RUN_ID`.
- Use shared keys in configs:
  - `enable_mlflow`
  - `mlflow_tracking_uri` (optional)
  - `mlflow_experiment` (optional, default `bllarse`)
  - `group_id`
  - `uid` (optional run name)
- Credentials helper: `src/bllarse/mlflow_utils.py` reads `~/.mlflow/credentials`.

## Environment / install
- Initialize submodule:
  - `git submodule update --init --recursive scripts/vbll_pytorch/scaling_mlps`
- Install VBLL package/deps into active venv:
  - `uv sync --group vbll`
- Optional for Lion:
  - `uv pip install lion-pytorch`

## SLURM/login-node workflow (VBLL)
- Compatible with existing launch flow via `python -m bllarse.tools.run_sweep ... --job-script src/slurm/jobs/slurm_run_config_docker.sh`.
- Optional CPU pinning for data-loader stability: pass `--cpus-per-task <N>` to `run_sweep.py` (recommended when using `num_workers > 0`).
- Required prerequisites before submitting:
  - Run from repo root on `experimental_infra_vbll_baseline`.
  - The venv passed in `--venv` (e.g. `.venv_bllarse_new`) already has VBLL deps installed via `uv sync --group vbll`.
  - `scripts/vbll_pytorch/scaling_mlps` submodule is initialized.
  - MLflow URI/auth is set via env or `~/.mlflow/credentials`.
- Notes:
  - `run_sweep.py` creates a parent MLflow run on the login node and passes `MLFLOW_PARENT_RUN_ID` to array jobs.
  - `src/slurm/jobs/slurm_run_config_docker.sh` activates `${VENV_NAME}` inside Docker, then runs `src/bllarse/tools/run_config.py`.
  - If your venv is not located at repo-relative path, pass an absolute path to `--venv`.

## Quick local smoke command
- Single VBLL config via shared runner:
  - `python src/bllarse/tools/run_config.py bllarse_sweeps/vbll_smoke_sweep.py 0`
- Direct VBLL script run:
  - `python scripts/vbll_pytorch/finetuning_vbll.py --dataset cifar10 --epochs 1 --batch-size 32 --enable-mlflow`
