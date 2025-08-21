#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x.%A_%a.out
#SBATCH --error=slurm/logs/%x.%A_%a.err
#SBATCH --job-name=bllarse_sweep

set -euo pipefail
set -x

# 1) cd to repo root as seen by the login node
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# 2) Make the repo root discoverable for the adapter inside the container
export BLLARSE_REPO_ROOT="$REPO_ROOT"

# 3) Call your container wrapper; inside, activate the venv and run the config
#    NOTE: VENV_NAME and BLLARSE_SWEEP_SOURCE are set by run_sweep.py
scripts/start_docker_sbatch.sh \
  bash -lc "set -euo pipefail; \
            cd \"$REPO_ROOT\"; \
            if [[ -n \"${VENV_NAME:-}\" && -f \"${VENV_NAME}/bin/activate\" ]]; then source \"${VENV_NAME}/bin/activate\"; \
            elif [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; \
            python -m bllarse.tools.run_config \"${BLLARSE_SWEEP_SOURCE}\" \"${SLURM_ARRAY_TASK_ID}\""
