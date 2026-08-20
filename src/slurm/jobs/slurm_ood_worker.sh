#!/bin/bash -l
# One OOD evaluation worker: polls checkpoint slots, scores the suite, exits.
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err
#
# Runs BESIDE the training arrays, not after them. The trainer writes a
# checkpoint at each scheduled step and moves on; this consumes them. That takes
# ~1 h per run off the training gpu -- which is what keeps a dinov3_huge 4-epoch
# job inside its 24 h wall -- and lets OOD run on a finer stride than the
# trainer could otherwise afford.
#
# --mem matches the trainers': the suite is 98,389 images preloaded as uint8
# host arrays (~14.8 GB) and held for the worker's lifetime.
#
# ONE WORKER PER FRAMEWORK. jax[cuda] and torch each ship a full CUDA stack and
# cannot share an environment, so each polls the same root and consumes only the
# slots whose metadata names its own framework.

set -euo pipefail

for var in BLLARSE_FT_VENV BLLARSE_FT_CKPT_ROOT BLLARSE_EVAL_DUMP_ROOT; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ood-worker] ERROR: $var is not set." >&2
    exit 2
  fi
done

if [[ $(type -t module || true) == function ]]; then
  module purge >/dev/null 2>&1 || true
  module load python-waterboa/2025.06 >/dev/null 2>&1 || true
fi

repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
mkdir -p slurm_logs "$BLLARSE_EVAL_DUMP_ROOT" "$BLLARSE_FT_CKPT_ROOT"
export PYTHONPATH="${repo_root}/src${BLLARSE_FT_PYLIBS:+:${BLLARSE_FT_PYLIBS}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${BLLARSE_FT_JAX_CACHE:-/raven/ptmp/cheins/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
[[ -n "${BLLARSE_HF_HOME:-}" ]] && export HF_HOME="$BLLARSE_HF_HOME"

set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

echo "[ood-worker] framework=${BLLARSE_OOD_FRAMEWORK:-jax} node=$(hostname)"
exec srun "$BLLARSE_FT_VENV" scripts/finetune/ood_worker.py \
  --checkpoint-root "$BLLARSE_FT_CKPT_ROOT" \
  --dump-root "$BLLARSE_EVAL_DUMP_ROOT" \
  --framework "${BLLARSE_OOD_FRAMEWORK:-jax}" \
  --idle-exit-seconds "${BLLARSE_OOD_IDLE_EXIT:-3600}"
