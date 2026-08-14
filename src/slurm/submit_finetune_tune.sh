#!/bin/bash
# Submit one worker array per fine-tuning arm, sized by where each study should
# END UP rather than by how many trials it adds.
#
# The fine-tuning counterpart of submit_tune.sh, and it exists for the same
# reason that one does. The FIRST fine-tuning campaign (studies
# ft_*_shards2m) was submitted as four ad-hoc `sbatch --export=...` calls with
# a uniform 10 trials per worker. Trials the MedianPruner kills still consume
# that count, and the prune rate is not a property anyone controls -- it ran
# 83% on ce, 92% on vbll. The same request therefore landed at:
#
#     ibprobit_noeps  36 completed      ce        32 completed
#     ibprobit        20 completed      vbll      16 completed
#
# and those arms were then compared with each other. BLLARSE_FT_TARGET_COMPLETE
# removes the coupling: every worker stops once the STUDY -- across all workers
# and every earlier submission -- holds that many COMPLETED trials. So this
# script is idempotent. Run it again and arms already at depth exit without
# claiming a GPU; arms short of it top up by exactly the shortfall.
#
# RESUMING, not restarting. The worker opens studies with load_if_exists, so an
# unchanged BLLARSE_FT_CAMPAIGN continues the existing journals and TPE keeps
# every trial it has seen. That is sound ONLY while the search space is
# unchanged. The second campaign pins batch_size and moves five boxes, so it
# takes a new name -- and note the two failure modes differ: optuna silently
# accepts a widened FLOAT range on an existing study (modelling old trials
# against a range they were never drawn from), while a changed CATEGORICAL --
# IBPROBIT_UPDATE_ITERS gained two rungs -- raises `CategoricalDistribution
# does not support dynamic value space` and fails every trial.
#
# Usage, from the repo root on the cluster:
#   BLLARSE_FT_TARGET_COMPLETE=60 src/slurm/submit_finetune_tune.sh
#   BLLARSE_FT_DRY_RUN=1 src/slurm/submit_finetune_tune.sh   # print only
#   BLLARSE_FT_ARMS="vbll" src/slurm/submit_finetune_tune.sh

set -euo pipefail

: "${BLLARSE_FT_STORAGE:=/ptmp/${USER}/ft_studies}"
# Moved from fixb512 because the SEARCH SPACE changed again: lr and weight
# decay are now drawn as `lr_per_sqrt_batch` and `weight_decay_per_sqrt_batch`,
# the batch-invariant coordinates of the AdamW SDE transfer rule
# (finetune_space._suggest_per_sqrt_batch). At the pinned batch that is an
# exact relabelling of the same interval, so the two campaigns are directly
# comparable -- but they cannot SHARE a study. Renaming a float axis is
# optuna's quiet failure: `load_if_exists` would leave TPE modelling `lr` from
# 500-odd prior trials while the objective samples a name none of them carry.
: "${BLLARSE_FT_CAMPAIGN:=sdelr512}"
# 60, against the first campaign's 32 / 36 / 16. The reason to re-run at all is
# that the arms came out 0.0026 apart while the spread across each arm's own
# top six was 0.003-0.005 -- the ranking sat inside the search's own noise. A
# deeper search is the only thing that resolves that.
: "${BLLARSE_FT_TARGET_COMPLETE:=60}"

# The per-worker CAP, sized to fill a 24 h job rather than to express a depth.
# Measured on the first campaign: ~85% of trials are pruned, a pruned trial
# costs ~0.4 h (it dies a few evaluations past the pruner's 10-evaluation
# warmup, and the compilation cache absorbs the ~19 min graph build), and a
# completed one ~2.8 h at this batch. That averages ~0.8 h, so a 24 h worker
# fits about 30. The previous campaign's 10 left workers exiting at ~9 h with
# the allocation still open.
: "${BLLARSE_FT_TRIALS:=30}"
: "${BLLARSE_FT_WORKERS:=16}"

# 6 per arm, 18 GPUs across the three. At ~85% pruning, 60 completions needs
# ~400 trials and ~320 GPU-hours per arm, so this is ~2 days wall-clock. The
# prune rate should FALL under the pinned batch and re-centred boxes, in which
# case the target is reached earlier and the remaining array tasks exit at the
# worker's pre-flight check without claiming a trial.
: "${BLLARSE_FT_THROTTLE:=6}"
# ibprobit -- the eps-SEARCHED arm -- is absent by default. The first campaign
# answered its question: across the top six trials eps_init spanned 5.3e-5,
# 1.0e-4, 8.4e-7, 3.0e-6, 4.0e-4 and 2.1e-2, five decades, with every score
# inside 0.004 of the others. The axis does nothing, and ibprobit_noeps -- the
# arm with the mixture floor numerically absent -- won outright at 0.44150
# against 0.44273. Re-searching it would spend a quarter of the campaign
# re-measuring an established null. Pass BLLARSE_FT_ARMS to put it back.
: "${BLLARSE_FT_ARMS:=ce ibprobit_noeps vbll}"
: "${BLLARSE_FT_VENV:=}"
: "${BLLARSE_FT_SHARDS:=/ptmp/${USER}/shards}"
: "${BLLARSE_FT_DRY_RUN:=}"

repo_root="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
job="${repo_root}/src/slurm/jobs/slurm_finetune_tune.sh"

if [[ -z "$BLLARSE_FT_VENV" ]]; then
  echo "ERROR: set BLLARSE_FT_VENV to the interpreter that has equimo." >&2
  echo "  ...the JAX arms and vbll need DIFFERENT ones -- see below." >&2
  exit 2
fi

# The INTERPRETER, not the venv directory. slurm_finetune_tune.sh runs
# "$BLLARSE_FT_VENV" -m bllarse.tuning.finetune_worker, so a directory here
# fails with `Is a directory` -- one line of stderr, exit 126, after the job
# has already been allocated a GPU. That cost a full submission: all 48 array
# tasks across three arms started and died inside 3 seconds, and because the
# arrays were then exhausted the whole campaign had to be resubmitted.
#
# Checked HERE rather than in the job script because the job script cannot
# refuse before SLURM hands it a node, and one bad path multiplies by
# BLLARSE_FT_WORKERS x the number of arms.
check_interpreter() {
  local path="$1" label="$2"
  if [[ -d "$path" ]]; then
    echo "ERROR: $label is a directory: $path" >&2
    echo "       Pass the interpreter, e.g. ${path%/}/bin/python" >&2
    exit 2
  fi
  if [[ ! -x "$path" ]]; then
    echo "ERROR: $label is not executable: $path" >&2
    exit 2
  fi
}

# One venv per framework, and they cannot be merged: jax[cuda] and torch each
# ship a full CUDA stack, and installing both into one environment is what
# broke the vbll smoke runs. BLLARSE_FT_VENV names the JAX one; vbll overrides
# it with BLLARSE_FT_VBLL_VENV, whose jax is CPU-only (bllarse.metrics imports
# jax, and a CPU build coexists with torch's CUDA safely).
: "${BLLARSE_FT_VBLL_VENV:=$BLLARSE_FT_VENV}"

check_interpreter "$BLLARSE_FT_VENV" BLLARSE_FT_VENV
check_interpreter "$BLLARSE_FT_VBLL_VENV" BLLARSE_FT_VBLL_VENV

# Same class of failure, same reason to catch it here: the job script exits 2
# on a missing shard, but only after taking a node.
shard_meta="${BLLARSE_FT_SHARDS%/}/imagenet1k_train_224.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "ERROR: no training shard at $shard_meta" >&2
  exit 2
fi

for arm in $BLLARSE_FT_ARMS; do
  venv="$BLLARSE_FT_VENV"
  [[ "$arm" == "vbll" ]] && venv="$BLLARSE_FT_VBLL_VENV"

  exports="BLLARSE_FT_ARM=${arm}"
  exports+=",BLLARSE_FT_TRIALS=${BLLARSE_FT_TRIALS}"
  exports+=",BLLARSE_FT_TARGET_COMPLETE=${BLLARSE_FT_TARGET_COMPLETE}"
  exports+=",BLLARSE_FT_STORAGE=${BLLARSE_FT_STORAGE}"
  exports+=",BLLARSE_FT_CAMPAIGN=${BLLARSE_FT_CAMPAIGN}"
  exports+=",BLLARSE_FT_SHARDS=${BLLARSE_FT_SHARDS}"
  exports+=",BLLARSE_FT_VENV=${venv}"
  [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] &&
    exports+=",BLLARSE_EVAL_DUMP_ROOT=${BLLARSE_EVAL_DUMP_ROOT}"
  [[ -n "${BLLARSE_HF_HOME:-}" ]] &&
    exports+=",BLLARSE_HF_HOME=${BLLARSE_HF_HOME}"
  [[ -n "${BLLARSE_FT_PYLIBS:-}" ]] &&
    exports+=",BLLARSE_FT_PYLIBS=${BLLARSE_FT_PYLIBS}"

  # ALL, not just the listed ones: --export replaces the environment wholesale
  # rather than adding to it, and the job script needs PATH and HOME to find
  # `module` and the venv at all.
  cmd=(sbatch
       --job-name "ft_${arm}_${BLLARSE_FT_CAMPAIGN}"
       --array "0-$((BLLARSE_FT_WORKERS - 1))%${BLLARSE_FT_THROTTLE}"
       --export "ALL,${exports}"
       "$job")

  if [[ -n "$BLLARSE_FT_DRY_RUN" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
