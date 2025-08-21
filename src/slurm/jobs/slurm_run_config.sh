#!/usr/bin/env bash
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH -G 1
#SBATCH --output=slurm/logs/%x.%A_%a.out
#SBATCH --error=slurm/logs/%x.%A_%a.err

set -euo pipefail
set -x

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

VENV_DIR="${REPO_ROOT}/${VENV_NAME}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
else
  echo "ERROR: venv not found at ${VENV_DIR}" >&2
  exit 1
fi

python -m bllarse.tools.run_config \
  "${BLLARSE_SWEEP_SOURCE}" \
  "${SLURM_ARRAY_TASK_ID}"
