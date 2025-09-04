#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -G 1                             # one GPU, cluster’s preferred flag
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err

set -euo pipefail
set -x

# 1) cd to repo root as seen by the login node
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# 2) Make the repo root discoverable for the adapter inside the container
export BLLARSE_REPO_ROOT="$REPO_ROOT"

INDEX_OFFSET="${INDEX_OFFSET:-0}"
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID+INDEX_OFFSET))
echo "[bllarse] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  INDEX_OFFSET=${INDEX_OFFSET}  CONFIG_IDX=${CONFIG_IDX}"
# 3) Call your container wrapper; inside, activate the venv and run the config
#    NOTE: VENV_NAME and BLLARSE_SWEEP_SOURCE are set by run_sweep.py
scripts/start_docker_sbatch.sh \
  bash -lc "set -euo pipefail; \
            export WANDB_DIR=\${WANDB_DIR:-\$HOME/wandb}; mkdir -p \"\$WANDB_DIR\"; \
            export WANDB_CACHE_DIR=\${WANDB_CACHE_DIR:-\$HOME/.cache/wandb}; mkdir -p \"\$WANDB_CACHE_DIR\"; \
            cd \"$REPO_ROOT\"; \
            if [[ -n \"${VENV_NAME:-}\" && -f \"${VENV_NAME}/bin/activate\" ]]; then source \"${VENV_NAME}/bin/activate\"; \
            elif [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi; \
            python src/bllarse/tools/run_config.py \"${BLLARSE_SWEEP_SOURCE}\" \"${CONFIG_IDX}\""
