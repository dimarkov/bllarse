#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}" ]]; then
  echo "[bllarse] ERROR: SLURM_JOB_GPUS is not set; request a GPU for this job." >&2
  exit 2
fi

host_user="${USER:-$(id -un)}"
host_logname="${LOGNAME:-$host_user}"
docker_image="${BLLARSE_DOCKER_IMAGE:-bllarse-dev}"
docker_shm_size="${BLLARSE_DOCKER_SHM_SIZE:-8g}"

docker_args=(
  --rm
  --gpus "device=${SLURM_JOB_GPUS}"
  --shm-size "$docker_shm_size"
  -e "HOME=$HOME"
  -e "USER=$host_user"
  -e "LOGNAME=$host_logname"
  -v "$HOME:$HOME"
  -v "$(pwd):$(pwd)"
  -w "$(pwd)"
  -u "$(id -u):$(id -g)"
)

passthrough_vars=(
  BLLARSE_REPO_ROOT
  BLLARSE_SWEEP_SOURCE
  CONFIG_IDX
  HF_TOKEN
  INDEX_OFFSET
  MLFLOW_EXPERIMENT_NAME
  MLFLOW_PARENT_RUN_ID
  MLFLOW_TRACKING_PASSWORD
  MLFLOW_TRACKING_URI
  MLFLOW_TRACKING_USERNAME
  PYTHONPATH
  VENV_NAME
)
for variable in "${passthrough_vars[@]}"; do
  if [[ -n "${!variable:-}" ]]; then
    docker_args+=(--env "$variable")
  fi
done

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  docker_args+=(--cpus "$SLURM_CPUS_PER_TASK")
fi

if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
  docker_args+=(
    -e "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
    -v "$TORCHINDUCTOR_CACHE_DIR:$TORCHINDUCTOR_CACHE_DIR"
  )
fi

if ! docker image inspect "$docker_image" >/dev/null 2>&1; then
  echo "[bllarse] ERROR: Docker image '$docker_image' is unavailable on $(hostname)." >&2
  echo "[bllarse] Build it on that node or exclude the node from the sweep." >&2
  exit 125
fi

exec docker run "${docker_args[@]}" "$docker_image" "$@"
