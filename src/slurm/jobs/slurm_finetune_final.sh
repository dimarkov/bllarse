#!/bin/bash -l
# ONE reported fine-tuning run: one arm, one seed, 4 epochs on dinov3_huge.
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
# The counterpart of slurm_finetune_tune.sh, and it differs in one structural
# way: a tuning task runs MANY trials chosen by Optuna, while this runs exactly
# ONE run whose identity is (arm, seed). The array index selects the seed.
#
# The config is not templated in. The task calls final_config.py --format exec,
# which reads the arm's finished study, applies the batch/duration transfer and
# REPLACES this process with the training run. So what gets reported is derived
# from what the search found, at submit time, with no hand-copied numbers in
# between -- which is the whole reason bllarse.tuning.final_config exists.

set -euo pipefail

for var in BLLARSE_FT_ARM BLLARSE_FT_SEEDS BLLARSE_FT_STORAGE BLLARSE_FT_VENV \
           BLLARSE_FT_SHARDS BLLARSE_FT_CAMPAIGN; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ft-final] ERROR: $var is not set." >&2
    exit 2
  fi
done

# The array index picks the seed. Out-of-range is a submission error, not
# something to clamp: a clamped index would run a duplicate seed and report it
# as an independent draw.
read -r -a SEEDS <<< "$BLLARSE_FT_SEEDS"
idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( idx >= ${#SEEDS[@]} )); then
  echo "[ft-final] ERROR: array index $idx exceeds ${#SEEDS[@]} seeds" >&2
  exit 2
fi
SEED="${SEEDS[$idx]}"

shard_meta="$(readlink -f "$BLLARSE_FT_SHARDS")/${BLLARSE_FT_DATASET:-imagenet1k}_train_${BLLARSE_FT_IMG_SIZE:-224}.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "[ft-final] ERROR: no training shard at $shard_meta." >&2
  exit 2
fi
export BLLARSE_FT_SHARDS

if [[ $(type -t module || true) == function ]]; then
  module purge >/dev/null 2>&1 || true
  module load python-waterboa/2025.06 >/dev/null 2>&1 || true
fi

repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
mkdir -p slurm_logs
export PYTHONPATH="${repo_root}/src${BLLARSE_FT_PYLIBS:+:${BLLARSE_FT_PYLIBS}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${BLLARSE_FT_JAX_CACHE:-/raven/ptmp/cheins/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
[[ -n "${BLLARSE_HF_HOME:-}" ]] && export HF_HOME="$BLLARSE_HF_HOME"
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"

# See slurm_finetune_tune.sh: no XLA_FLAGS here on purpose. This runs in a venv
# rather than the container that pins the platform allocator, and disabling
# command buffers measured a 2.2x slowdown.

set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

echo "[ft-final] arm=${BLLARSE_FT_ARM} seed=${SEED} node=$(hostname)"

final_args=(
  --arm "$BLLARSE_FT_ARM"
  --seed "$SEED"
  --storage-dir "$(readlink -f "$BLLARSE_FT_STORAGE")"
  --campaign "$BLLARSE_FT_CAMPAIGN"
  --model "${BLLARSE_FT_MODEL:-dinov3_huge}"
  --epochs "${BLLARSE_FT_EPOCHS:-4.0}"
  --eval-quantum "${BLLARSE_FT_EVAL_QUANTUM:-65536}"
  --format exec
)
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && final_args+=(--dump-root "$BLLARSE_EVAL_DUMP_ROOT")
final_args+=(--dump-every-k-evals "${BLLARSE_FT_DUMP_EVERY_K:-1}")
final_args+=(--ood-every-k-dumps "${BLLARSE_FT_OOD_EVERY_K:-2}")
# Delegate: at those steps write a checkpoint instead of forwarding 98,389
# images on the training gpu.
if [[ -n "${BLLARSE_FT_CKPT_ROOT:-}" ]]; then
  final_args+=(--ood-checkpoint-root "$BLLARSE_FT_CKPT_ROOT")
  final_args+=(--ood-max-pending "${BLLARSE_FT_OOD_MAX_PENDING:-2}")
fi

# exec, so the training run replaces this shell and SLURM sees its exit code.
exec srun "$BLLARSE_FT_VENV" scripts/finetune/final_config.py "${final_args[@]}"
