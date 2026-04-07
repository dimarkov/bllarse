#!/bin/bash
set -euo pipefail

HOST_USER="${USER:-$(id -un 2>/dev/null || echo user)}"
HOST_LOGNAME="${LOGNAME:-$HOST_USER}"
HOST_TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
HOST_DOCKER_SHM_SIZE="${BLLARSE_DOCKER_SHM_SIZE:-8g}"
HOST_DOCKER_IMAGE="${BLLARSE_DOCKER_IMAGE:-bllarse-dev}"
mkdir -p "$HOST_TORCHINDUCTOR_CACHE_DIR"

docker_args=(
  --rm
  --gpus="device=$SLURM_JOB_GPUS"
  --shm-size "$HOST_DOCKER_SHM_SIZE"
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
  HF_TOKEN
  PYTHONPATH
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

if ! docker image inspect "$HOST_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "[bllarse] ERROR: Docker image '$HOST_DOCKER_IMAGE' is not available on host '$(hostname)'." >&2
  echo "[bllarse] Build it locally on that node (for example: docker build -t $HOST_DOCKER_IMAGE .) or exclude the node from the sweep." >&2
  exit 125
fi

docker run "${docker_args[@]}" "$HOST_DOCKER_IMAGE" "$@"
