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

# Resolved for the same reason as repo_root: /ptmp is a host symlink to
# /raven/ptmp, apptainer binds the RESOLVED target and does not recreate the
# symlink inside the container, so an unresolved /ptmp/... path exists in the
# container as a READ-ONLY stub. A cache dir given that way does not fail at
# startup -- load_cached_features simply does not find the features there,
# falls back to downloading from the Hub, and dies on mkdir with
# `OSError: [Errno 30] Read-only file system` once per trial. Resolving here
# rather than trusting the caller is deliberate: this is the third time that
# symlink has cost a job, and the failure mode looks like a data problem
# rather than a path problem.
cache_dir="$(readlink -f "${BLLARSE_CACHE_DIR:-$repo_root/.cache/features}")"
if [[ "$cache_dir" != "$repo_root"/* ]]; then
  bind_args+=(--bind "$cache_dir")
fi
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
  # Unconditional, not arm-gated: it is harmless on the torch/VBLL image, which
  # never touches XLA for training. On the JAX arm, large-batch trials (seen at
  # batch 8192 and 2048 -- exactly what TPE explores) crashed workers with
  # `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: operation not permitted when stream
  # is capturing` inside jit__ce_step, followed by
  # `cuda_command_buffer.cc:862] CUDA error: Failed to destroy CUDA graph` and
  # a fatal cublasLtMatmul INTERNAL error -- i.e. XLA's CUDA-graph
  # command-buffer capture colliding with `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
  # set in bllarse-jax.def's %environment. The empty value disables command
  # buffers entirely; this was verified live on a GPU node against the exact
  # failing config (food101, batch 8192, lr 0.028) before being added here. Do
  # NOT remove this "since it looks like a no-op flag" -- without it, large
  # batches on the JAX arm reliably kill the worker process mid-trial.
  --env "XLA_FLAGS=--xla_gpu_enable_command_buffer="
)
[[ -n "${HF_TOKEN:-}" ]] && env_args+=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${HF_HOME:-}" ]] && env_args+=(--env "HF_HOME=$HF_HOME")
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && env_args+=(--env "BLLARSE_EVAL_DUMP_ROOT=$BLLARSE_EVAL_DUMP_ROOT")
[[ -n "${MLFLOW_TRACKING_URI:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
[[ -n "${MLFLOW_TRACKING_USERNAME:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME")
[[ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]] && env_args+=(--env "MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD")
[[ -n "${MLFLOW_PARENT_RUN_ID:-}" ]] && env_args+=(--env "MLFLOW_PARENT_RUN_ID=$MLFLOW_PARENT_RUN_ID")

# --patience-factor 5.0, replacing `--patience 20`.
#
# The old value was calibrated by replaying 28 pilot curves for the smallest
# count of consecutive no-improvement DUMPS at which none stopped before its
# argmin. That calibration was sound and the rule it produced was still inert:
# a run has exactly 40 dumps, so a patience of 20 can only fire when the curve
# turns in the first half of the grid. Measured over the first campaign's 133
# completed trials it fired on 46, every one of them cifar10 or dtd, and could
# not fire at all on the other five datasets -- whose argmins land at dumps
# 26-37. Counting points on a bounded grid was the defect, not the threshold.
#
# The replacement is multiplicative in SAMPLES: stop once samples_seen reaches
# 5x the samples at which the best val_nll was set. Absolute thresholds were
# measured and rejected first -- the argmin sits at 43k samples on cifar10 and
# 9.0M on stanford_cars, so the smallest globally safe absolute value is 15M,
# which is 350x past cifar10's optimum. The multiplicative form is scale-free:
# the largest per-curve ratio between consecutive improvements is 1.30-4.88
# across all seven datasets, and 5.0 cuts none of the 170 measured curves short
# of its argmin.
#
# Same warning as before, for the same reason: a trial stopped before its
# argmin is scored at the wrong point and changes what the campaign selects. Do
# not lower this to save time without re-running the replay. Re-derive it from
# the sample-spaced curves this protocol now produces, NOT from the numbers
# above -- those were measured on the geometric grid, where every observed
# ratio is an exact power of its 1.3026 spacing and therefore a quantised lower
# bound. Calibrating against a grid artifact is how `--patience 20` happened.
worker_args=(
  --arm "$BLLARSE_TUNE_ARM"
  --dataset "$BLLARSE_TUNE_DATASET"
  --n-trials "$BLLARSE_TUNE_TRIALS"
  --storage-dir "$storage_dir"
  --patience-factor "${BLLARSE_TUNE_PATIENCE_FACTOR:-5.0}"
  --cache-dir "$cache_dir"
)
# Optional, and unset reproduces the first campaign's unsuffixed study names.
# Set it whenever the search space has changed: the worker opens studies with
# load_if_exists, so an unsuffixed re-run resumes the old study instead of
# starting a new one.
if [[ -n "${BLLARSE_TUNE_CAMPAIGN:-}" ]]; then
  worker_args+=(--campaign "$BLLARSE_TUNE_CAMPAIGN")
fi

exec apptainer exec --nv \
  "${env_args[@]}" \
  "${bind_args[@]}" \
  "$BLLARSE_SIF" \
  python -m bllarse.tuning.worker "${worker_args[@]}"
