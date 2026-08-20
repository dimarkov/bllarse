#!/bin/bash -l
# Prepare the OOD caches once, serially, off the critical path.
#SBATCH --job-name=prep_ood
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err
#
# ONE task, never an array: the whole point is that exactly one process fills
# the cache. An array here would reproduce the betas1m race precisely.
#
# No GPU. This downloads and builds arrow tables; it decodes nothing.

set -euo pipefail
repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
mkdir -p slurm_logs
export PYTHONPATH="${repo_root}/src"
export PYTHONUNBUFFERED=1
[[ -n "${BLLARSE_HF_HOME:-}" ]] && export HF_HOME="$BLLARSE_HF_HOME"
set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

echo "[prep-ood] node=$(hostname) HF_HOME=${HF_HOME:-<unset>}"
# BLLARSE_OOD_KEYS scopes the job. Defaulting to the whole suite would make a
# re-run download whatever the suite happens to contain, which is not what
# "prepare the two that are missing" means.
keys=(${BLLARSE_OOD_KEYS:-})
if (( ${#keys[@]} )); then
  exec "${BLLARSE_FT_VENV:?set BLLARSE_FT_VENV}"     scripts/finetune/prepare_ood_cache.py --keys "${keys[@]}"
fi
exec "${BLLARSE_FT_VENV:?set BLLARSE_FT_VENV}" scripts/finetune/prepare_ood_cache.py
