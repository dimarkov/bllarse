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
# betas1m was PURGED, not superseded: its studies failed 2,851 of 2,860 trials
# on the dataset-cache race that `check_dataset` below now catches, and the
# journals, logs and dumps were deleted rather than kept as a harness-fault
# record.
# The name is not reused, so nothing in a log or a journal can be ambiguous
# about which campaign it came from.
#
# Moved from fixb512 because the SEARCH SPACE changed, in three ways:
#
#   * lr and weight decay are drawn as `lr_per_sqrt_batch` and
#     `weight_decay_per_sqrt_batch`, the batch-invariant coordinates. At the
#     pinned batch that alone is an exact relabelling.
#   * `beta1_horizons` and `beta2_horizons` are NEW axes. The betas were
#     previously unsearchable -- a per-step timescale with no way to move it
#     across budgets -- and protocol.horizons makes them transferable, hence
#     worth tuning. This is what makes the campaign more than a relabelling.
#   * protocol.MAX_SAMPLES is halved to 1,000,000, which pays for the two extra
#     dimensions. Trials are ~1.4 h rather than ~2.8 h.
#
# They cannot SHARE a study with fixb512 in any case. Renaming a float axis is
# optuna's quiet failure: `load_if_exists` would leave TPE modelling `lr` from
# 500-odd prior trials while the objective samples a name none of them carry.
: "${BLLARSE_FT_CAMPAIGN:=betasearch1m}"
# 60, against the first campaign's 32 / 36 / 16. The reason to re-run at all is
# that the arms came out 0.0026 apart while the spread across each arm's own
# top six was 0.003-0.005 -- the ranking sat inside the search's own noise. A
# deeper search is the only thing that resolves that.
: "${BLLARSE_FT_TARGET_COMPLETE:=60}"

# The per-worker CAP, sized to fill a 24 h job rather than to express a depth.
#
# Re-derived for the halved budget. At protocol.MAX_SAMPLES = 1,000,000 a
# completed trial costs ~1.4 h and a pruned one ~0.2 h (it dies a few
# evaluations past the pruner's 10-evaluation warmup, and the compilation cache
# absorbs the ~19 min graph build). At the measured ~85% prune rate that
# averages ~0.38 h, so a 24 h worker fits about 60. The previous value of 30
# was correct at 2M and is exactly half of what this budget supports.
#
# The cap has to clear the TARGET, not merely fill the wall. fixb512 measured
# 16-25% of trials completing, so 90 completions needs ~360-560 trials per arm
# against WORKERS x TRIALS = 960 here. At 30 it would have been 480 -- under
# the requirement for two of the three arms, which is precisely how fixb512
# exhausted its arrays at 53/55/32 against a target of 60. Overshoot is free:
# the worker's --target-complete pre-flight exits without claiming a GPU once
# the study is at depth.
: "${BLLARSE_FT_TRIALS:=60}"
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

# The dataset cache, resolved OFFLINE, once, before any node is taken.
#
# This is the check whose absence cost the betas1m campaign: BLLARSE_HF_HOME
# pointed at a cache that had no PREPARED imagenet-1k, so all 18 workers called
# load_dataset at the same moment, raced to download and build it into the same
# directory, and destroyed each other's `.incomplete` staging dir. Every trial
# failed for three days -- 48 array tasks, 410 GPU-hours, 9 usable trials out of
# 2,860.
#
# The shard covers TRAINING only. finetune_jax still calls load_dataset for the
# eval split and the label metadata, so a good shard says nothing about whether
# the dataset resolves -- which is exactly why the existing shard check did not
# catch this.
#
# OFFLINE deliberately. Requiring the cache to resolve with no network is not
# excess strictness, it is the property that matters: a cache that would have to
# DOWNLOAD is precisely the one that must not be handed to 18 concurrent
# workers. Preparing it is a one-time serial job, and the error below says so.
#
# ~3 minutes on a login node against a warm 156 GB cache. There is no skip flag,
# on purpose: the campaign it protects costs three orders of magnitude more.
hf_path="$(PYTHONPATH="${repo_root}/src" "$BLLARSE_FT_VENV" -c \
  'from bllarse.finetune.registry import DATASET_CONFIGS as D; print(D["imagenet1k"]["hf_path"])' \
  2>/dev/null)" || hf_path=""
if [[ -z "$hf_path" ]]; then
  echo "ERROR: could not read the imagenet1k hf_path from the registry via" >&2
  echo "       $BLLARSE_FT_VENV." >&2
  exit 2
fi

# EVERY interpreter in use, not just the JAX one. `datasets` can differ in
# version between the two venvs, and the prepared-dataset path is keyed by a
# builder revision hash -- so the same cache can resolve for one interpreter and
# trigger a rebuild for the other. That is not hypothetical: vbll runs its own
# torch venv and it is the arm that took 0 completions out of 960.
# The BACKBONE has to be cached too, and for exactly the same reason the
# dataset does. dinov3_small was present as 7 KB of config with zero weight
# files when its campaign was first prepared -- every worker would have raced
# to fetch it. Checked per interpreter because the two frameworks pull from
# different places: equimo has its own archive cache, transformers has HF_HOME.
check_backbone() {
  local interp="$1" label="$2" model="$3"
  echo "Resolving backbone $model offline via $label ..." >&2
  # JAX_PLATFORMS=cpu: this runs on a LOGIN node with no GPU, and initialising
  # the CUDA backend there aborts with a core dump rather than falling back.
  # The check only has to load weights, so the backend is irrelevant to it.
  # Thread caps, because this runs on a LOGIN node. XLA's CPU compiler sizes
  # its pool from nproc -- 104 here -- and on the deeper backbones that trips
  # the node's per-user thread limit: `LLVM ERROR: pthread_create failed:
  # Resource temporarily unavailable`. dinov3_small squeaks under it and
  # dinov3_large does not, so without this the guard passes on the backbone
  # nobody needed it for and blocks the two that matter -- reporting a missing
  # checkpoint when the weights are in fact cached. The check only loads
  # weights, so one thread is plenty.
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     XLA_FLAGS="--xla_force_host_platform_device_count=1" \
     HF_HOME="${BLLARSE_HF_HOME:-$HOME/.cache/huggingface}" \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
     PYTHONPATH="${repo_root}/src:${repo_root}/scripts/dinov3" \
     "$interp" -c "
import sys
model = '${model}'
if '${label}' == 'vbll':
    sys.path.insert(0, '${repo_root}/scripts/finetune')
    from transformers import AutoModel
    from finetune_vbll import HF_CHECKPOINTS
    m = AutoModel.from_pretrained(HF_CHECKPOINTS[model])
    print('  OK', model, m.config.num_hidden_layers, 'layers')
else:
    import jax.random as jr
    from vit_classification import get_pretrained_backbone
    bb = get_pretrained_backbone(model, jr.PRNGKey(0))
    print('  OK', model, len(bb.blocks[0].blocks), 'blocks')
" >&2; then
    return 0
  fi
  echo "ERROR: backbone $model does not resolve OFFLINE for $label." >&2
  echo "       Fetch it ONCE, serially, then re-run this script. Submitting" >&2
  echo "       anyway would have every worker download it at the same time." >&2
  exit 2
}

check_dataset() {
  local interp="$1" label="$2"
  echo "Resolving $hf_path offline via $label ..." >&2
  if HF_HOME="${BLLARSE_HF_HOME:-$HOME/.cache/huggingface}" \
     HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
     "$interp" -c "
from datasets import load_dataset
n = {k: len(v) for k, v in load_dataset('${hf_path}').items()}
assert n.get('train', 0) > 1_000_000, n
print('  OK', n)
" >&2; then
    return 0
  fi
  echo "ERROR: $hf_path does not resolve OFFLINE for $label from" >&2
  echo "       HF_HOME=${BLLARSE_HF_HOME:-<unset>}." >&2
  echo "       Submitting anyway would have every worker try to prepare it at" >&2
  echo "       once. That is what killed the betas1m campaign: 48 array tasks," >&2
  echo "       410 GPU-hours, 9 usable trials out of 2,860." >&2
  echo "       Prepare it ONCE, serially, then re-run this script:" >&2
  echo "         HF_HOME=<cache> $interp -c \\" >&2
  echo "           \"from datasets import load_dataset; load_dataset('$hf_path')\"" >&2
  echo "       ...or point BLLARSE_HF_HOME at a cache that already has it." >&2
  exit 2
}

# An explicit `if`, not `[[ ... ]] && continue`: under `set -e` a false `&&`
# compound as a loop body's last statement is an exit, and the guard would then
# silently skip the rest of the arms.
declare -A _checked=()
for arm in $BLLARSE_FT_ARMS; do
  interp="$BLLARSE_FT_VENV"
  if [[ "$arm" == "vbll" ]]; then interp="$BLLARSE_FT_VBLL_VENV"; fi
  if [[ -z "${_checked[$interp]:-}" ]]; then
    _checked[$interp]=1
    check_dataset "$interp" "${arm}'s interpreter"
    # Only when the campaign names a backbone. Without BLLARSE_FT_MODEL the
    # worker uses space.TUNING_MODEL, whose weights every previous campaign
    # already pulled -- checking it would fail closed on a machine that is
    # simply running the old default.
    if [[ -n "${BLLARSE_FT_MODEL:-}" ]]; then
      label="jax"; [[ "$arm" == "vbll" ]] && label="vbll"
      check_backbone "$interp" "$label" "$BLLARSE_FT_MODEL"
    fi
  fi
done

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
  # Campaign-level and not searched. Forwarded unconditionally when set, so a
  # campaign cannot half-apply: the beta horizon axes convert at samples/batch,
  # so a budget without its batch searches a different space than it records.
  for var in BLLARSE_FT_MAX_SAMPLES BLLARSE_FT_MODEL BLLARSE_FT_BATCH \
             BLLARSE_FT_EVAL_QUANTUM BLLARSE_FT_LAYER_DECAY \
             BLLARSE_FT_LAYER_DECAY_SPAN BLLARSE_FT_TRIALS_PER_PROC \
             BLLARSE_FT_PATIENCE; do
    [[ -n "${!var:-}" ]] && exports+=",${var}=${!var}"
  done

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
