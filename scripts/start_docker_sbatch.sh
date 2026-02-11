#!/bin/bash
set -euo pipefail

HOST_USER="${USER:-$(id -un 2>/dev/null || echo user)}"
HOST_LOGNAME="${LOGNAME:-$HOST_USER}"
HOST_TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "$HOST_TORCHINDUCTOR_CACHE_DIR"

docker_args=(
  --rm
  --gpus="device=$SLURM_JOB_GPUS"
  -e "HOME=$HOME"
  -e "USER=$HOST_USER"
  -e "LOGNAME=$HOST_LOGNAME"
  -e "TORCHINDUCTOR_CACHE_DIR=$HOST_TORCHINDUCTOR_CACHE_DIR"
  -v "$HOME:$HOME"
  -v "$(pwd):$(pwd)"
  -w "$(pwd)"
  -u "$(id -u):$(id -g)"
)

# Pass selected job metadata/env through to the container when available.
passthrough_vars=(
  MLFLOW_PARENT_RUN_ID
  MLFLOW_TRACKING_URI
  MLFLOW_EXPERIMENT_NAME
  BLLARSE_REPO_ROOT
)
for var_name in "${passthrough_vars[@]}"; do
  if [[ -n "${!var_name:-}" ]]; then
    docker_args+=(-e "$var_name=${!var_name}")
  fi
done

# Respect SLURM CPU allocation inside the container when available.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  docker_args+=(--cpus "${SLURM_CPUS_PER_TASK}")
fi

docker run "${docker_args[@]}" bllarse-dev "$@"
