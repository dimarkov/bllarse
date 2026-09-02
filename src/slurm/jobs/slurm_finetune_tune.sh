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

for var in BLLARSE_FT_ARM BLLARSE_FT_TRIALS BLLARSE_FT_STORAGE BLLARSE_FT_VENV \
           BLLARSE_FT_SHARDS; do
  if [[ -z "${!var:-}" ]]; then
    echo "[ft-tune] ERROR: $var is not set." >&2
    exit 2
  fi
done

# The shard is checked HERE, before a GPU-hour is spent, because its absence is
# a silent SLOW failure rather than a loud wrong one. finetune_jax falls back to
# decoding JPEGs, which measured ~40 img/s against the shard's ~220: every trial
# would still complete and still be scored, but a 24 h worker would finish 1
# instead of 7 and the campaign would quietly take five times as long.
#
# open_shard's own refusal of an INCOMPLETE shard covers the other half (a
# partial 193 GB file is mostly zeros, and zeros train as valid black images);
# this only has to catch the path being wrong or unmounted.
# Defaulted, not required: the dataset and image size are pinned in
# finetune_worker.trial_config, and a second place to set them is a second
# place for them to disagree.
shard_meta="$(readlink -f "$BLLARSE_FT_SHARDS")/${BLLARSE_FT_DATASET:-imagenet1k}_train_${BLLARSE_FT_IMG_SIZE:-224}.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "[ft-tune] ERROR: no training shard at $shard_meta." >&2
  echo "[ft-tune] Build it once with scripts/finetune/build_shards.py, or" >&2
  echo "[ft-tune] unset BLLARSE_FT_SHARDS to accept the ~5x slower JPEG path." >&2
  exit 2
fi
export BLLARSE_FT_SHARDS

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

# Python block-buffers stdout when it is a file, so every [finetune] progress
# line sits in an 8 KB buffer instead of the log. On a ~6 h trial that makes a
# hung job indistinguishable from a slow one -- the first smoke run reached 44
# minutes with an empty log while it was in fact compiling normally.
export PYTHONUNBUFFERED=1

# Persistent XLA compilation cache, on the shared filesystem so every worker on
# every node hits the same entries.
#
# Compiling the fwd+bwd graph for a 32-layer ViT-H measured ~19 minutes, and
# finetune_jax calls jax.clear_caches() in its `finally`, so without this each
# of the 3 trials in a worker pays it again. Across 306 trials that is ~100
# A100-hours of pure compilation.
#
# It is cacheable because every trial has IDENTICAL shapes: the micro batch is
# pinned (design A8) and the eval chunk is fixed, so batch_size varies only
# num_accum, which is a Python loop count and not a traced dimension.
export JAX_COMPILATION_CACHE_DIR="${BLLARSE_FT_JAX_CACHE:-/raven/ptmp/cheins/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
[[ -n "${BLLARSE_HF_HOME:-}" ]] && export HF_HOME="$BLLARSE_HF_HOME"
[[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] && mkdir -p "$BLLARSE_EVAL_DUMP_ROOT"

# NO XLA_FLAGS here, deliberately.
#
# slurm_tune.sh sets `--xla_gpu_enable_command_buffer=` and must: the probing
# image (bllarse-jax.def) puts XLA_PYTHON_CLIENT_ALLOCATOR=platform in its
# %environment, and CUDA-graph capture collides with that allocator, killing
# workers mid-trial with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
#
# This path runs in a VENV, which sets no such allocator, so the collision
# cannot occur -- and disabling command buffers here cost a measured 2.2x:
# 18.29 img/s with them off against 40.27 img/s with them on, same config, same
# node, dinov3_huge at batch 512 / micro 32. A 32-layer ViT launches thousands
# of small kernels per accumulated step and CUDA graphs are what amortise the
# launches.
#
# If this ever moves back into a container that pins the platform allocator,
# the flag has to come back WITH it. The flag is a property of the allocator,
# not of the model.

set -a; [ -f .env.secrets ] && . ./.env.secrets; set +a

echo "[ft-tune] worker=${SLURM_ARRAY_TASK_ID:-0} arm=${BLLARSE_FT_ARM}" \
     "trials=${BLLARSE_FT_TRIALS} node=$(hostname)"
"$BLLARSE_FT_VENV" -c "import bllarse, jax; print('[ft-tune] bllarse from', bllarse.__file__); print('[ft-tune] jax', jax.__version__, jax.devices())"

# --patience 20 consecutive evaluations with no relative val_nll improvement.
#
# The count means something here in a way it did not on the probing side.
# Evaluations are uniformly spaced in SAMPLES (protocol.EVAL_QUANTUM = 16,384),
# so 20 of them is always 327,680 samples, and there are 122 per trial at every
# rung of the batch ladder. The earliest possible stop is eval 21: stalls
# accumulate from evaluation 1 and WARMUP_EVALS=10 gates only the FIRING, not
# the counting. So a doomed config costs ~1 h instead of ~6.3 h.
#
# The probing campaign's `--patience 20` counted points on a GEOMETRIC grid
# bounded at 40 and could therefore only fire when a curve turned in its first
# half -- it fired on 46 of 133 trials, all cifar10 or dtd, and structurally
# never on the other five datasets. That defect is gone, not worked around.
#
# 20 is a starting value, not a derived one. Read what actually happened off
# the trials' own `patience_stopped_at_eval` and `truncated` attributes: many
# stops at eval 21 means 20 is too tight, `truncated` nearly everywhere means
# the 2M sample budget binds instead, and those call for opposite corrections.
worker_args=(
  --arm "$BLLARSE_FT_ARM"
  --storage-dir "$storage_dir"
  --patience "${BLLARSE_FT_PATIENCE:-20}"
)
# --n-trials is NOT here: it is a per-CHUNK count now, appended at the bottom.
# The depth the STUDY should reach, invariant to the prune rate. Without it the
# first campaign searched the arms to depths nobody chose: the same 10 trials
# per worker landed at 36 completions on ibprobit_noeps and 16 on vbll, because
# MedianPruner killed 83% and 92% of them respectively -- and those two arms
# were then compared against each other. --n-trials still caps the worker, so a
# study that cannot reach the target terminates rather than holding a GPU to
# the 24 h wall.
if [[ -n "${BLLARSE_FT_TARGET_COMPLETE:-}" ]]; then
  worker_args+=(--target-complete "$BLLARSE_FT_TARGET_COMPLETE")
fi
# Set whenever the search space changes: the worker opens studies with
# load_if_exists, so an unsuffixed re-run RESUMES the old study and has TPE
# model a space that no longer exists.
if [[ -n "${BLLARSE_FT_CAMPAIGN:-}" ]]; then
  worker_args+=(--campaign "$BLLARSE_FT_CAMPAIGN")
fi
if [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]]; then
  worker_args+=(--dump-root "$BLLARSE_EVAL_DUMP_ROOT")
fi
# The tuning budget. Was "smoke tests only" while the campaign was pinned to
# protocol.MAX_SAMPLES; it is now a first-class campaign setting, since the
# dinov3_small study deliberately tunes at 100k. Trials from DIFFERENT budgets
# are still not comparable -- that is what --campaign is for.
if [[ -n "${BLLARSE_FT_MAX_SAMPLES:-}" ]]; then
  worker_args+=(--max-samples "$BLLARSE_FT_MAX_SAMPLES")
fi
# The four campaign-level settings that are NOT searched. They move together
# with the budget by necessity: the beta axes are drawn in horizon units and
# converted at the trial's own step count, which is samples / batch.
if [[ -n "${BLLARSE_FT_MODEL:-}" ]]; then
  worker_args+=(--model "$BLLARSE_FT_MODEL")
fi
if [[ -n "${BLLARSE_FT_BATCH:-}" ]]; then
  worker_args+=(--batch-size "$BLLARSE_FT_BATCH")
fi
if [[ -n "${BLLARSE_FT_EVAL_QUANTUM:-}" ]]; then
  worker_args+=(--eval-quantum "$BLLARSE_FT_EVAL_QUANTUM")
fi
if [[ -n "${BLLARSE_FT_LAYER_DECAY_SPAN:-}" ]]; then
  # The SPAN, resolved by the training script against the backbone it loaded.
  # Prefer this over a per-block decay for any campaign that spans backbones.
  worker_args+=(--layer-decay-span "$BLLARSE_FT_LAYER_DECAY_SPAN")
fi
if [[ -n "${BLLARSE_FT_LAYER_DECAY:-}" ]]; then
  worker_args+=(--layer-decay "$BLLARSE_FT_LAYER_DECAY")
fi

# ONE PROCESS PER CHUNK of trials, not one process for the whole worker.
#
# Every trial in a campaign asks for exactly the same memory -- the micro batch
# is pinned (design A8) and the eval chunk is fixed -- so a trial that cannot
# allocate at #32 when #1 could is not a bigger trial. The arena changed, not
# the demand. That is measured, not assumed: on ft_ce_small100k all six workers
# died, on six DIFFERENT nodes (ravg1017/1022/1096/...), at staggered wall-clock
# times two hours apart, after 32, 33 and 39 trials each -- every one of them
# failing on the same 6.55 GiB in jit_ce_micro_grad. A node-level or
# cluster-level cause cannot produce that pattern; a per-process one does. It
# cost 28 of that arm's 273 trials, and Optuna recorded most of them as
# "no evaluation ran", which reads like a config bug rather than an OOM.
#
# A fresh process gets a fresh BFC arena. The compilation cache above is
# persistent and shared, so the restart pays a process start (~20 s) rather
# than a recompile: at a ~22 min median trial, chunking at 10 costs ~0.4% to
# retire a failure mode that cost ~10% of the study.
#
# Deliberately NOT the other two candidates: XLA_PYTHON_CLIENT_ALLOCATOR=platform
# collides with CUDA-graph capture (see the XLA_FLAGS comment above, which
# measured a 2.2x tax for disabling command buffers), and raising
# XLA_PYTHON_CLIENT_MEM_FRACTION only buys headroom -- it delays a leak instead
# of clearing it, and cannot tell the two apart.
chunk="${BLLARSE_FT_TRIALS_PER_PROC:-10}"
remaining="$BLLARSE_FT_TRIALS"
failed_chunks=0
chunk_log="$(mktemp)"
trap 'rm -f "$chunk_log"' EXIT

while (( remaining > 0 )); do
  n=$(( remaining < chunk ? remaining : chunk ))
  echo "[ft-tune] chunk of $n trials in a fresh process" \
       "($remaining of $BLLARSE_FT_TRIALS left on this worker)"

  # tee, not a redirect: a trial runs ~22 min and PYTHONUNBUFFERED above exists
  # so the log is live. PIPESTATUS, not $?, because $? is tee's.
  set +e
  srun "$BLLARSE_FT_VENV" -m bllarse.tuning.finetune_worker \
       "${worker_args[@]}" --n-trials "$n" 2>&1 | tee "$chunk_log"
  rc=${PIPESTATUS[0]}
  set -e

  # The study reached its depth while this worker was mid-budget. Release the
  # GPU instead of starting chunks that would each pay a process start to
  # discover the same thing. The token is printed by finetune_worker.main and
  # exists FOR this grep -- keep the two in step.
  if grep -q "TARGET-MET" "$chunk_log"; then
    echo "[ft-tune] study is at its target depth; releasing the GPU."
    break
  fi

  if (( rc != 0 )); then
    failed_chunks=$(( failed_chunks + 1 ))
    if (( failed_chunks >= 2 )); then
      # Two fresh arenas in a row cannot both be fragmented. Something is
      # actually wrong with the arm, and abort_on_consecutive_failures has
      # already burned a run of trials proving it -- stop rather than spend
      # the rest of the budget confirming it a third time.
      echo "[ft-tune] ERROR: two consecutive chunks failed (rc=$rc);" \
           "this is not fragmentation." >&2
      exit "$rc"
    fi
    echo "[ft-tune] chunk failed (rc=$rc); restarting once in a fresh process." >&2
  else
    failed_chunks=0
  fi

  # Decremented by the chunk SIZE even when the chunk died early. The worker
  # cap is a bound on GPU time, not a quota to be made up; --target-complete
  # governs how deep the study actually goes.
  remaining=$(( remaining - n ))
done
