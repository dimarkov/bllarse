#!/bin/bash
set -euo pipefail

docker_args=(
  --rm
  --gpus="device=$SLURM_JOB_GPUS"
  -e "HOME=$HOME"
  -v "$HOME:$HOME"
  -v "$(pwd):$(pwd)"
  -w "$(pwd)"
  -u "$(id -u):$(id -g)"
)

# Respect SLURM CPU allocation inside the container when available.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  docker_args+=(--cpus "${SLURM_CPUS_PER_TASK}")
fi

docker run "${docker_args[@]}" bllarse-dev "$@"
