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

# --patience 40 consecutive evaluations with no relative val_nll improvement.
#
# The count is meaningful now in a way it was not before. Evaluations are
# uniformly spaced in SAMPLES (space.eval_quantum), so 40 of them is 40 x the
# quantum, the same amount of training wherever in the run it lands. The old
# `--patience 20` counted points on a GEOMETRIC grid bounded at 40, so it could
# only fire when a curve turned in its first half: measured over the first
# campaign's 133 completed trials it fired on 46, every one cifar10 or dtd, and
# structurally never on the other five datasets.
#
# 40 is a starting value, NOT a derived one, and deliberately so. Two attempts
# to derive a threshold from the previous campaign's curves both failed, and
# both failed by generalising from too little: an absolute sample threshold
# cannot serve datasets whose optima span 43k (cifar10) to 9.0M (stanford_cars)
# samples, and a multiplicative factor calibrated on ibprobit alone -- where it
# cut 0 of 113 curves short -- cut 125 of 1,214 once adamw_ce and vbll were
# measured too. The quantity cannot be known before the results exist.
#
# 40 also clears the arithmetic ceiling that a larger count would hit. The
# evaluation count per run is `steps * batch / quantum`, so it scales with
# batch: at batch 64 a cifar10/cifar100/food101 run has only ~118 evaluations
# (against 30,000 at the top of the ladder). A patience of 100 could fire
# there only if the argmin landed by eval 18 -- the same bounded-grid defect
# as --patience 20 on 40 dumps. At 40 it fires whenever the argmin is at eval
# 78 or earlier. The count is only fully batch-invariant once the budget is
# denominated in samples rather than steps; that is not done here.
#
# So: run it, then read what actually happened off the trials' own
# `patience_stopped` and `patience_stopped_at_eval` attributes, and adjust. The
# risk to watch for is unchanged -- a trial stopped before its own val_nll
# argmin is scored at the wrong point, which changes which hyperparameters the
# campaign selects.
worker_args=(
  --arm "$BLLARSE_TUNE_ARM"
  --dataset "$BLLARSE_TUNE_DATASET"
  --n-trials "$BLLARSE_TUNE_TRIALS"
  --storage-dir "$storage_dir"
  --patience "${BLLARSE_TUNE_PATIENCE:-40}"
  --cache-dir "$cache_dir"
)
# Optional. Stops every worker once the STUDY reaches this many COMPLETED
# trials, counting what earlier submissions already left in the journal -- so a
# top-up submission is sized by where the study should END UP rather than by
# guessing how many of the trials it starts will survive the pruner. That guess
# is what went wrong in the first campaign: at a fixed 16 trials per worker,
# prune rates of 8%-72% turned the same request into 40 completed trials for
# adamw_ce/food101 and 103 for ibprobit/cifar10.
#
# BLLARSE_TUNE_TRIALS remains the per-worker cap and still bounds the job.
if [[ -n "${BLLARSE_TUNE_TARGET_COMPLETE:-}" ]]; then
  worker_args+=(--target-complete "$BLLARSE_TUNE_TARGET_COMPLETE")
fi

# Optional. Scores trials on the lowest MEAN val_nll over a window of this
# fraction of the evaluation grid, instead of on its single-point minimum.
#
# The minimum is the deepest dip of the trajectory. On a configuration that
# never converges that measures the noise, not the level reached -- measured on
# probing/oxford_pets/dinov3_huge, the tuned lr 0.0182 oscillates with a
# late-phase sd 0.53x its own median and the search preferred it to lr 0.0024,
# which is 13x more stable AND 0.032 better on test NLL. Only the arms with a
# learning rate can exploit that; IBProbit's closed-form update has no dips to
# mine, so the bias is not symmetric across the arms being compared.
#
# THIS CHANGES THE OBJECTIVE, so it demands a new BLLARSE_TUNE_CAMPAIGN -- see
# the guard in submit_tune.sh. Mixing two objectives inside one study gives TPE
# a value column where half the rows mean something different.
if [[ -n "${BLLARSE_TUNE_SMOOTH_WINDOW:-}" ]]; then
  worker_args+=(--smooth-window "$BLLARSE_TUNE_SMOOTH_WINDOW")
fi

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
