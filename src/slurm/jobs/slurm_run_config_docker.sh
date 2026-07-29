#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -G 1
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"

export BLLARSE_REPO_ROOT="$repo_root"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export INDEX_OFFSET="${INDEX_OFFSET:-0}"
export CONFIG_IDX="$((SLURM_ARRAY_TASK_ID + INDEX_OFFSET))"

echo "[bllarse] task=${SLURM_ARRAY_TASK_ID} offset=${INDEX_OFFSET} config=${CONFIG_IDX}"

scripts/start_docker_sbatch.sh bash -lc '
  set -euo pipefail
  cd "$BLLARSE_REPO_ROOT"

  if [[ "$VENV_NAME" = /* ]]; then
    venv_dir="$VENV_NAME"
  else
    venv_dir="$BLLARSE_REPO_ROOT/$VENV_NAME"
  fi
  if [[ ! -f "$venv_dir/bin/activate" ]]; then
    echo "[bllarse] ERROR: virtual environment not found at $venv_dir" >&2
    exit 2
  fi

  source "$venv_dir/bin/activate"
  python -m bllarse.tools.run_config "$BLLARSE_SWEEP_SOURCE" "$CONFIG_IDX"
'
