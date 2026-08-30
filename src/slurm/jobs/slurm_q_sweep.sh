#!/bin/bash -l
# One point of the IBProbit q sweep: same config as the reported run, one change.
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
# Diagnostic, not a reported campaign: ONE seed per point, dumps in their own
# group, no OOD. The question is why IBProbit's NLL rises 34% past its own
# minimum while CE rises 4.6% and VBLL 2.1%, and the trajectory on the
# in-distribution split is the whole evidence.
#
# The config comes from final_config.py exactly as the reported run's does --
# same study, same transfer -- plus the point's own --set overrides. So a
# difference between a sweep point and the reported run is the override and
# nothing else; there are no hand-copied numbers to drift.
#
# The CONTROL is not in this array. It already exists as
# ftfinal/dinov3_large_imagenet1k_ibprobit_seed137, and re-running it would
# spend a GPU-day reproducing a curve that is already scored.

set -euo pipefail

for var in BLLARSE_FT_POINTS_FILE BLLARSE_FT_STORAGE BLLARSE_FT_VENV \
           BLLARSE_FT_SHARDS BLLARSE_FT_CAMPAIGN; do
  if [[ -z "${!var:-}" ]]; then
    echo "[qsweep] ERROR: $var is not set." >&2
    exit 2
  fi
done

# Each point is "label|set=expr[|set=expr...]", one per line of a FILE on
# shared storage. Not an --export variable: the list contains ';' and '|', and
# SLURM's export parsing is not somewhere to discover a quoting bug after a
# queue wait. The file must be on /ptmp, never /tmp, which is node-local here.
if [[ ! -f "$BLLARSE_FT_POINTS_FILE" ]]; then
  echo "[qsweep] ERROR: no points file at $BLLARSE_FT_POINTS_FILE" >&2
  exit 2
fi
mapfile -t POINTS < <(grep -v '^\s*$' "$BLLARSE_FT_POINTS_FILE")
idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( idx >= ${#POINTS[@]} )); then
  echo "[qsweep] ERROR: array index $idx exceeds ${#POINTS[@]} points" >&2
  exit 2
fi
IFS='|' read -r -a FIELDS <<< "${POINTS[$idx]}"
LABEL="${FIELDS[0]}"

shard_meta="$(readlink -f "$BLLARSE_FT_SHARDS")/${BLLARSE_FT_DATASET:-imagenet1k}_train_${BLLARSE_FT_IMG_SIZE:-224}.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "[qsweep] ERROR: no training shard at $shard_meta." >&2
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

set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

SEED="${BLLARSE_FT_SEED:-137}"
MODEL="${BLLARSE_FT_MODEL:-dinov3_large}"
GROUP="${BLLARSE_FT_GROUP:-qsweep}"
ARM="${BLLARSE_FT_ARM:-ibprobit_noeps}"
# The uid token is the SCRIPT arm, which is what the dumps' own meta records:
# the study is `ibprobit_noeps` but every dump it produces says `ibprobit`. A
# uid built from the study name would not match its own dumps.
ARM_TOKEN="${ARM%_noeps}"
UID_STR="${MODEL}_${BLLARSE_FT_DATASET:-imagenet1k}_${ARM_TOKEN}_${LABEL}_seed${SEED}"

echo "[qsweep] arm=$ARM point=$LABEL uid=$UID_STR node=$(hostname)"

final_args=(
  --arm "$ARM"
  --seed "$SEED"
  --storage-dir "$(readlink -f "$BLLARSE_FT_STORAGE")"
  --campaign "$BLLARSE_FT_CAMPAIGN"
  --model "$MODEL"
  --epochs "${BLLARSE_FT_EPOCHS:-4.0}"
  --eval-quantum "${BLLARSE_FT_EVAL_QUANTUM:-65536}"
  --dump-every-k-evals 1
  --group-id "$GROUP"
  --uid "$UID_STR"
)
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && final_args+=(--dump-root "$BLLARSE_EVAL_DUMP_ROOT")
for ((i = 1; i < ${#FIELDS[@]}; i++)); do
  final_args+=(--set "${FIELDS[$i]}")
done
final_args+=(--format exec)

# vbll is the torch arm and cannot share the JAX environment.
interp="$BLLARSE_FT_VENV"
if [[ "$ARM" == "vbll" && -n "${BLLARSE_FT_VBLL_VENV:-}" ]]; then
  interp="$BLLARSE_FT_VBLL_VENV"
fi
exec srun "$interp" scripts/finetune/final_config.py "${final_args[@]}"
