#!/bin/bash -l
# Four Optuna workers for ONE (arm, dataset) study. A LOGIN shell (-l), not
# `env bash`: on MPCDF no modules are preloaded and the `module` function only
# exists once /etc/profile.d has been sourced, which a non-login shell skips.
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err
# `--gres=gpu:a100:1`, not `-G 1`: Raven's default partition is `interactive`,
# which has no GPUs, so a bare `-G 1` is rejected at submit time with
# "Requested node configuration is not available" and the array never queues.
# The array is fixed at 4 and NOT derived from a config count -- that is the
# whole reason this is not a run_sweep submission. TPE degrades under
# parallelism (each worker samples a posterior not yet updated by the trials in
# flight beside it), so four is a deliberate ceiling, not a resource limit.

set -euo pipefail

# apptainer is not on PATH on MPCDF until a module is loaded, but this script
# also runs on hosts that have it on PATH and no module system at all -- so try
# the module only when the binary is genuinely missing, and fail loudly after.
if ! command -v apptainer >/dev/null 2>&1; then
  if [[ $(type -t module || true) == function ]]; then
    module purge >/dev/null 2>&1 || true
    module load "${BLLARSE_APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true
  fi
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "[bllarse] ERROR: apptainer not found on PATH and no module supplied it." >&2
  exit 2
fi

for var in BLLARSE_SIF BLLARSE_TUNE_ARM BLLARSE_TUNE_DATASET \
           BLLARSE_TUNE_TRIALS BLLARSE_TUNE_STORAGE; do
  if [[ -z "${!var:-}" ]]; then
    echo "[bllarse] ERROR: $var is not set." >&2
    exit 2
  fi
done

# readlink -f, not the raw path: on MPCDF /ptmp and /u are top-level symlinks
# (/ptmp -> /raven/ptmp). Apptainer binds the RESOLVED directory and does NOT
# recreate that symlink inside the container, so an unresolved path exists on
# the host but not in the container and `import bllarse` fails on every task.
repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
cd "$repo_root"
storage_dir="$(readlink -f "$BLLARSE_TUNE_STORAGE")"
mkdir -p "$storage_dir"

echo "[bllarse] tune worker=${SLURM_ARRAY_TASK_ID} arm=${BLLARSE_TUNE_ARM}" \
     "dataset=${BLLARSE_TUNE_DATASET} trials=${BLLARSE_TUNE_TRIALS}"

bind_args=(--bind "$repo_root" --bind "$storage_dir")
if [[ -n "${BLLARSE_HF_HOME:-}" ]]; then
  mkdir -p "$BLLARSE_HF_HOME"
  bind_args+=(--bind "$BLLARSE_HF_HOME:$BLLARSE_HF_HOME")
  export HF_HOME="$BLLARSE_HF_HOME"
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"
  bind_args+=(--bind "$BLLARSE_EVAL_DUMP_ROOT:$BLLARSE_EVAL_DUMP_ROOT")
fi

env_args=(
  --env "BLLARSE_REPO_ROOT=$repo_root"
  --env "PYTHONPATH=${repo_root}/src:${repo_root}/scripts/vbll_pytorch/src${PYTHONPATH:+:${PYTHONPATH}}"
)
[[ -n "${HF_TOKEN:-}" ]] && env_args+=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${HF_HOME:-}" ]] && env_args+=(--env "HF_HOME=$HF_HOME")
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && env_args+=(--env "BLLARSE_EVAL_DUMP_ROOT=$BLLARSE_EVAL_DUMP_ROOT")
[[ -n "${MLFLOW_TRACKING_URI:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
[[ -n "${MLFLOW_TRACKING_USERNAME:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME")
[[ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD")
[[ -n "${MLFLOW_PARENT_RUN_ID:-}" ]] && env_args+=(--env "MLFLOW_PARENT_RUN_ID=$MLFLOW_PARENT_RUN_ID")

# --patience 20, not the worker's own default of "disabled": this value was
# measured by replaying all 28 real pilot curves through the exact patience
# bookkeeping. It is the SMALLEST value at which zero runs stop before their
# own val_nll argmin (at 15 one run stops early, at 8 two do). A trial that
# stops before its argmin is scored at the wrong point, which changes which
# hyperparameters the campaign selects -- do NOT lower this "to save time"
# without re-running that replay against fresh pilot curves.
exec apptainer exec --nv \
  "${env_args[@]}" \
  "${bind_args[@]}" \
  "$BLLARSE_SIF" \
  python -m bllarse.tuning.worker \
    --arm "$BLLARSE_TUNE_ARM" \
    --dataset "$BLLARSE_TUNE_DATASET" \
    --n-trials "$BLLARSE_TUNE_TRIALS" \
    --storage-dir "$storage_dir" \
    --patience "${BLLARSE_TUNE_PATIENCE:-20}" \
    --cache-dir "${BLLARSE_CACHE_DIR:-.cache/features}"
