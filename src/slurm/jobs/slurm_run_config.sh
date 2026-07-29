#!/usr/bin/env bash
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH -G 1
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$repo_root"

if [[ "$VENV_NAME" = /* ]]; then
  venv_dir="$VENV_NAME"
else
  venv_dir="$repo_root/$VENV_NAME"
fi
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "[bllarse] ERROR: virtual environment not found at $venv_dir" >&2
  exit 2
fi

source "$venv_dir/bin/activate"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
config_idx="$((SLURM_ARRAY_TASK_ID + ${INDEX_OFFSET:-0}))"
python -m bllarse.tools.run_config "$BLLARSE_SWEEP_SOURCE" "$config_idx"
