#!/bin/bash -l
# Optuna workers for ONE fine-tuning arm's study. A LOGIN shell (-l), not
# `env bash`: on MPCDF no modules are preloaded and the `module` function only
# exists once /etc/profile.d has been sourced, which a non-login shell skips.
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
# NOT the apptainer path that slurm_tune.sh uses. bllarse-jax.sif has no
# equimo -- correctly, since the probing campaign consumes CACHED features and
# never runs the trunk -- so it cannot fine-tune. This runs against a venv that
# does, with the repo's own src ahead of that venv's (older) editable install
# on PYTHONPATH. Replace with a SIF once one exists that carries equimo.
#
# --cpus-per-task=18 and --mem=125000, unlike slurm_tune.sh's 40G: probing
# loads a feature matrix, this decodes JPEGs for 2M training samples plus 20k
# eval images per trial, and the loader is a ThreadPoolExecutor that wants
# cores.

set -euo pipefail

for var in BLLARSE_FT_ARM BLLARSE_FT_TRIALS BLLARSE_FT_STORAGE BLLARSE_FT_VENV; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ft-tune] ERROR: $var is not set." >&2
    exit 2
  fi
done

if [[ $(type -t module || true) == function ]]; then
  module purge >/dev/null 2>&1 || true
  module load python-waterboa/2025.06 >/dev/null 2>&1 || true
fi

# readlink -f, not the raw path: on MPCDF /ptmp and /u are top-level symlinks
# (/ptmp -> /raven/ptmp), and an unresolved path has repeatedly cost a job by
# existing on the host and not where the process actually looks.
repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
storage_dir="$(readlink -f "$BLLARSE_FT_STORAGE")"
mkdir -p "$storage_dir" slurm_logs

# The repo's src FIRST. The venv carries an editable install of an older
# checkout, and without this shadowing every worker would import that instead
# -- silently, since the package name is the same.
export PYTHONPATH="${repo_root}/src${BLLARSE_FT_PYLIBS:+:${BLLARSE_FT_PYLIBS}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
[[ -n "${BLLARSE_HF_HOME:-}" ]] && export HF_HOME="$BLLARSE_HF_HOME"
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"

# See slurm_tune.sh for the full account: XLA's CUDA-graph command-buffer
# capture collides with the platform allocator and kills workers mid-trial with
# `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` at large batches. The empty value
# disables command buffers. Do NOT remove this as a no-op flag.
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

echo "[ft-tune] worker=${SLURM_ARRAY_TASK_ID:-0} arm=${BLLARSE_FT_ARM}" \
     "trials=${BLLARSE_FT_TRIALS} node=$(hostname)"
"$BLLARSE_FT_VENV" -c "import bllarse, jax; print('[ft-tune] bllarse from', bllarse.__file__); print('[ft-tune] jax', jax.__version__, jax.devices())"

# --patience 20 consecutive evaluations with no relative val_nll improvement.
#
# The count means something here in a way it did not on the probing side.
# Evaluations are uniformly spaced in SAMPLES (protocol.EVAL_QUANTUM = 16,384),
# so 20 of them is always 327,680 samples, and there are 122 per trial at every
# rung of the batch ladder. With WARMUP_EVALS=10 the earliest possible stop is
# eval 30, so a doomed config costs ~1.5 h instead of ~6.3 h.
#
# The probing campaign's `--patience 20` counted points on a GEOMETRIC grid
# bounded at 40 and could therefore only fire when a curve turned in its first
# half -- it fired on 46 of 133 trials, all cifar10 or dtd, and structurally
# never on the other five datasets. That defect is gone, not worked around.
#
# 20 is a starting value, not a derived one. Read what actually happened off
# the trials' own `patience_stopped_at_eval` and `truncated` attributes: many
# stops at eval 30 means 20 is too tight, `truncated` nearly everywhere means
# the 2M sample budget binds instead, and those call for opposite corrections.
worker_args=(
  --arm "$BLLARSE_FT_ARM"
  --n-trials "$BLLARSE_FT_TRIALS"
  --storage-dir "$storage_dir"
  --patience "${BLLARSE_FT_PATIENCE:-20}"
)
# Set whenever the search space changes: the worker opens studies with
# load_if_exists, so an unsuffixed re-run RESUMES the old study and has TPE
# model a space that no longer exists.
if [[ -n "${BLLARSE_FT_CAMPAIGN:-}" ]]; then
  worker_args+=(--campaign "$BLLARSE_FT_CAMPAIGN")
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  worker_args+=(--dump-root "$BLLARSE_EVAL_DUMP_ROOT")
fi
# Smoke tests only. The campaign runs at protocol.MAX_SAMPLES (2,000,000) and
# a shortened budget is not comparable to it -- never set this on a study whose
# trials are meant to be scored against each other.
if [[ -n "${BLLARSE_FT_MAX_SAMPLES:-}" ]]; then
  worker_args+=(--max-samples "$BLLARSE_FT_MAX_SAMPLES")
fi

exec srun "$BLLARSE_FT_VENV" -m bllarse.tuning.finetune_worker "${worker_args[@]}"
