#!/bin/bash -l
# A LOGIN shell (-l), not `env bash`: on MPCDF no modules are preloaded and the
# `module` function itself only exists once /etc/profile.d has been sourced,
# which a non-login shell skips.
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/%x.%A_%a.out
#SBATCH --error=slurm_logs/%x.%A_%a.err
# `--gres=gpu:a100:1`, not `-G 1`: Raven's default partition is `interactive`,
# which has no GPUs, so a bare `-G 1` is rejected at submit time with
# "Requested node configuration is not available" and the array never queues.
# Both directives are overridable from run_sweep.py's --partition/--gres, since
# sbatch command-line flags take precedence over #SBATCH lines.

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
  echo "[bllarse]        Set BLLARSE_APPTAINER_MODULE (e.g. apptainer/1.4.3)." >&2
  exit 2
fi

# readlink -f, not the raw path: SLURM reports the submit dir exactly as typed,
# and on MPCDF /ptmp and /u are top-level symlinks (/ptmp -> /raven/ptmp).
# Apptainer binds the RESOLVED directory and does NOT recreate that symlink
# inside the container, so an unresolved /ptmp/... path exists on the host but
# not in the container -- PYTHONPATH then points at a directory that isn't
# there and `import bllarse` fails on every single array task, with a
# ModuleNotFoundError that says nothing about symlinks.
repo_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$(pwd)}")"
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
bind_args=(--bind "$repo_root")
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
  # scripts/vbll_pytorch/src too, not just src: the VBLL arm's entry point does
  # `from vbll_experiments.models import build_vbll_head`, and vbll_experiments
  # is a LOCAL package (scripts/vbll_pytorch/pyproject.toml declares it) that is
  # never pip-installed into the image. Harmless for the JAX arm, which simply
  # never imports it.
  --env "PYTHONPATH=${repo_root}/src:${repo_root}/scripts/vbll_pytorch/src${PYTHONPATH:+:${PYTHONPATH}}"
  # Unconditional, not arm-gated: harmless on the torch/VBLL image, which never
  # touches XLA for training. On the JAX arm it is load-bearing. Without it,
  # runs die mid-training with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
  # operation not permitted when stream is capturing` inside jit__ce_step,
  # followed by `cuda_command_buffer.cc:862] CUDA error: Failed to destroy CUDA
  # graph` and a fatal cublasLtMatmul INTERNAL error -- XLA's CUDA-graph
  # command-buffer capture colliding with `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
  # from bllarse-jax.def's %environment.
  #
  # slurm_tune.sh carried this fix from 2026-08; THIS script did not, so every
  # probing sweep since has been exposed to it. It surfaced again on 2026-08-20,
  # killing 2 of 128 tasks in the win5 Fig1/Fig2 submission. Do NOT remove it
  # "because it looks like a no-op flag" -- the empty value disables command
  # buffers entirely, which is the point.
  --env "XLA_FLAGS=--xla_gpu_enable_command_buffer="
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
