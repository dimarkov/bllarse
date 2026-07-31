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
export INDEX_OFFSET="${INDEX_OFFSET:-0}"
export CONFIG_IDX="$((SLURM_ARRAY_TASK_ID + INDEX_OFFSET))"

sif="${BLLARSE_SIF:-}"
if [[ -z "$sif" ]]; then
  echo "[bllarse] ERROR: BLLARSE_SIF is not set; point it at the .sif to run." >&2
  exit 2
fi
if [[ ! -f "$sif" ]]; then
  echo "[bllarse] ERROR: SIF image not found: $sif" >&2
  exit 2
fi

if [[ -z "${BLLARSE_SWEEP_SOURCE:-}" ]]; then
  echo "[bllarse] ERROR: BLLARSE_SWEEP_SOURCE is not set; run_sweep.py should have exported it." >&2
  exit 2
fi

echo "[bllarse] task=${SLURM_ARRAY_TASK_ID} offset=${INDEX_OFFSET} config=${CONFIG_IDX} sif=${sif}"

# Apptainer binds $HOME, $PWD and /tmp by default and preserves UID/GID, so the
# Docker wrapper's -u and passthrough gymnastics are unnecessary. Only paths
# outside those roots need an explicit bind.
bind_args=()
if [[ -n "${BLLARSE_HF_HOME:-}" ]]; then
  mkdir -p "$BLLARSE_HF_HOME"
  bind_args+=(--bind "$BLLARSE_HF_HOME:$BLLARSE_HF_HOME")
  export HF_HOME="$BLLARSE_HF_HOME"
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"
  bind_args+=(--bind "$BLLARSE_EVAL_DUMP_ROOT:$BLLARSE_EVAL_DUMP_ROOT")
fi

# Only forward variables that are actually set. `--env VAR=` (an explicit
# empty value) is NOT the same as omitting VAR: os.environ.get("VAR", default)
# returns '' rather than falling back to `default`, so an unset-but-passed
# BLLARSE_EVAL_DUMP_ROOT silently redirects every dump into the repo root
# (outside the eval_dumps/ .gitignore entry), an unset-but-passed HF_HOME
# makes every task re-resolve and re-download its cache, and an
# unset-but-passed HF_TOKEN sends an empty Authorization header. Each `[[ ]]`
# check below is a no-op (not an error) under `set -e` when it evaluates
# false, so this is safe under `set -euo pipefail`.
env_args=(
  --env "BLLARSE_REPO_ROOT=$BLLARSE_REPO_ROOT"
  --env "BLLARSE_SWEEP_SOURCE=$BLLARSE_SWEEP_SOURCE"
  --env "CONFIG_IDX=$CONFIG_IDX"
  --env "PYTHONPATH=${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
)
[[ -n "${HF_TOKEN:-}" ]] && env_args+=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${HF_HOME:-}" ]] && env_args+=(--env "HF_HOME=$HF_HOME")
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && env_args+=(--env "BLLARSE_EVAL_DUMP_ROOT=$BLLARSE_EVAL_DUMP_ROOT")
[[ -n "${MLFLOW_TRACKING_URI:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
[[ -n "${MLFLOW_TRACKING_USERNAME:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME")
[[ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD")
[[ -n "${MLFLOW_PARENT_RUN_ID:-}" ]] && env_args+=(--env "MLFLOW_PARENT_RUN_ID=$MLFLOW_PARENT_RUN_ID")
[[ -n "${MLFLOW_EXPERIMENT_NAME:-}" ]] && env_args+=(--env "MLFLOW_EXPERIMENT_NAME=$MLFLOW_EXPERIMENT_NAME")

exec apptainer exec --nv \
  "${env_args[@]}" \
  "${bind_args[@]}" \
  "$sif" \
  python -m bllarse.tools.run_config "$BLLARSE_SWEEP_SOURCE" "$CONFIG_IDX"
