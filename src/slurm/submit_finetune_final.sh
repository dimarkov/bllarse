#!/bin/bash
# Submit the REPORTED fine-tuning campaign: one array per arm, one task per seed.
#
# This is results-plan section 2 -- Figs 5-7 and Table 1. Not a search: the
# hyperparameters are already chosen, and each task derives its own by reading
# the finished tuning study and applying the batch/duration transfer
# (bllarse.tuning.final_config). Nothing is hand-copied from a search result
# into a shell command, which is the failure this whole path exists to prevent.
#
# THREE comparators train, not four. results-plan 0.2 fixes them as CE,
# CE + post-hoc temperature scaling, VBLL and IBProbit -- but temperature
# scaling is fit on a held-out split from the CE run's dumps, so it costs no
# training. `ibprobit_noeps` is the IBProbit arm; the eps ablation is settled.
#
# WALL CLOCK is the binding constraint, and it is why BLLARSE_FT_EVAL_QUANTUM
# defaults COARSER than tuning's. A reported run forwards val+test (50k) at
# every evaluation instead of val alone (10k), over 2.56x the samples of a
# tuning trial. At protocol.EVAL_QUANTUM (16,384) that is 312 evaluations x 50k
# = 15.6M forward passes against ~15.4M fwd-equivalents of training --
# evaluation roughly DOUBLES the run, putting a dinov3_huge job near 36 h
# against this partition's 24 h limit. At 65,536 it is 78 evaluations, ~4.5 h of
# evaluation, and the job fits at roughly 22 h.
#
# That trade is Fig 5's x-resolution against fitting the wall, so it is a flag
# rather than a constant. 78 points on a log x-axis is ample for the curve; drop
# it to 32,768 only if a longer wall clock is available.
#
# Usage, from the repo root on the cluster:
#   BLLARSE_FT_VENV=/ptmp/$USER/venv_ft/bin/python \
#   BLLARSE_FT_VBLL_VENV=/ptmp/$USER/venv_vbll_ft/bin/python \
#   BLLARSE_HF_HOME=/ptmp/$USER/hf_cache \
#   src/slurm/submit_finetune_final.sh
#
#   BLLARSE_FT_DRY_RUN=1 src/slurm/submit_finetune_final.sh   # print only

set -euo pipefail

: "${BLLARSE_FT_STORAGE:=/ptmp/${USER}/ft_studies}"
# The campaign whose STUDIES are read -- not a new name. This launcher consumes
# a finished search; it does not start one.
: "${BLLARSE_FT_CAMPAIGN:=betasearch1m}"
: "${BLLARSE_FT_ARMS:=ce ibprobit_noeps vbll}"

# results-plan 0.5: "5 seeds for full fine-tuning". finetune_space.REPORTED_SEEDS
# holds ten, which is a POOL, not a prescription -- the plan is the governing
# document and the difference is 320 vs 640 GPU-hours on the single largest line
# item in the budget. Override to widen it.
: "${BLLARSE_FT_SEEDS:=137 138 139 140 141}"

: "${BLLARSE_FT_MODEL:=dinov3_huge}"
: "${BLLARSE_FT_EPOCHS:=4.0}"
: "${BLLARSE_FT_EVAL_QUANTUM:=65536}"
# Fig 5 is CURVES, so every evaluation is dumped. Fig 6 scores the same
# trajectory under shift, and OOD rides a stride of that dump grid.
: "${BLLARSE_FT_DUMP_EVERY_K:=1}"
# Every 2nd dump, which the trainer could not have afforded. OOD now runs on
# SEPARATE gpus (slurm_ood_worker.sh) reading checkpoint slots, so its cost is
# the eval workers' wall clock rather than the trainers'. 78 evaluations at the
# default quantum gives 39 OOD passes; inline that would have been ~4.5 h per
# run on top of a job already near 23 h.
: "${BLLARSE_FT_OOD_EVERY_K:=2}"

# Where trainers drop checkpoints for the workers. A dinov3_huge checkpoint is
# ~3.4 GB, and --ood-max-pending bounds how many can accumulate per run.
: "${BLLARSE_FT_CKPT_ROOT:=/ptmp/${USER}/ft_ckpt}"
: "${BLLARSE_FT_OOD_MAX_PENDING:=2}"
# One worker per framework, plus however many parallel consumers each needs.
# The JAX workers serve ce and ibprobit_noeps (10 runs), the torch worker vbll
# (5), which is why they are sized separately.
: "${BLLARSE_OOD_WORKERS_JAX:=2}"
: "${BLLARSE_OOD_WORKERS_TORCH:=1}"
: "${BLLARSE_FT_THROTTLE:=5}"
: "${BLLARSE_FT_VENV:=}"
: "${BLLARSE_FT_SHARDS:=/ptmp/${USER}/shards}"
: "${BLLARSE_FT_DRY_RUN:=}"

repo_root="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
job="${repo_root}/src/slurm/jobs/slurm_finetune_final.sh"

if [[ -z "$BLLARSE_FT_VENV" ]]; then
  echo "ERROR: set BLLARSE_FT_VENV to the interpreter that has equimo." >&2
  exit 2
fi

# Identical guards to submit_finetune_tune.sh, and for the same reason: the job
# script cannot refuse before SLURM hands it a node, and one bad path
# multiplies by seeds x arms.
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
: "${BLLARSE_FT_VBLL_VENV:=$BLLARSE_FT_VENV}"
check_interpreter "$BLLARSE_FT_VENV" BLLARSE_FT_VENV
check_interpreter "$BLLARSE_FT_VBLL_VENV" BLLARSE_FT_VBLL_VENV

shard_meta="${BLLARSE_FT_SHARDS%/}/imagenet1k_train_224.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "ERROR: no training shard at $shard_meta" >&2
  exit 2
fi

hf_path="$(PYTHONPATH="${repo_root}/src" "$BLLARSE_FT_VENV" -c \
  'from bllarse.finetune.registry import DATASET_CONFIGS as D; print(D["imagenet1k"]["hf_path"])' \
  2>/dev/null)" || hf_path=""
if [[ -z "$hf_path" ]]; then
  echo "ERROR: could not read the imagenet1k hf_path from the registry." >&2
  exit 2
fi

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
  echo "       HF_HOME=${BLLARSE_HF_HOME:-<unset>}. Prepare it ONCE, serially," >&2
  echo "       then re-run. See submit_finetune_tune.sh for why." >&2
  exit 2
}

# The STUDIES have to be at depth before anything reports from them. A launcher
# that happily reads a half-finished search would freeze whichever trial
# happened to be leading, and nothing downstream would show that it did.
check_study() {
  local arm="$1"
  PYTHONPATH="${repo_root}/src" "$BLLARSE_FT_VENV" - "$arm" <<'PY' || exit 2
import sys, os
from bllarse.tuning import finetune_worker as fw
from bllarse.tuning.worker import n_completed
arm = sys.argv[1]
target = int(os.environ.get("BLLARSE_FT_TARGET_COMPLETE", "90"))
study = fw.open_study(arm, os.environ["BLLARSE_FT_STORAGE"],
                      campaign=os.environ["BLLARSE_FT_CAMPAIGN"])
done = n_completed(study)
if done < target:
    sys.exit(f"ERROR: {study.study_name} has {done} completed trials, "
             f"under the target {target}. The search is still running; its "
             f"leader can still change. Wait, or lower "
             f"BLLARSE_FT_TARGET_COMPLETE deliberately.")
print(f"  {study.study_name}: {done} completed, best {study.best_value:.5f}")
PY
}

declare -A _checked=()
for arm in $BLLARSE_FT_ARMS; do
  interp="$BLLARSE_FT_VENV"
  if [[ "$arm" == "vbll" ]]; then interp="$BLLARSE_FT_VBLL_VENV"; fi
  if [[ -z "${_checked[$interp]:-}" ]]; then
    _checked[$interp]=1
    check_dataset "$interp" "${arm}'s interpreter"
  fi
done

export BLLARSE_FT_STORAGE BLLARSE_FT_CAMPAIGN
echo "Studies:" >&2
for arm in $BLLARSE_FT_ARMS; do check_study "$arm"; done

# The OOD suite has to be CACHED before anything runs. Only ImageNet-A was,
# once, and 15 trainers plus their workers racing to download the rest is the
# exact shape of the betas1m loss. prepare_ood_cache.py --check-only resolves
# offline and exits non-zero with what is missing.
echo "OOD cache:" >&2
if ! HF_HOME="${BLLARSE_HF_HOME:-$HOME/.cache/huggingface}" \
     PYTHONPATH="${repo_root}/src" "$BLLARSE_FT_VENV" \
     "${repo_root}/scripts/finetune/prepare_ood_cache.py" --check-only >&2; then
  echo "ERROR: the OOD suite is not fully cached. Prepare it ONCE with" >&2
  echo "       src/slurm/jobs/slurm_prepare_ood.sh, then re-run this." >&2
  exit 2
fi

read -r -a _seeds <<< "$BLLARSE_FT_SEEDS"
n_seeds=${#_seeds[@]}
echo "Submitting ${n_seeds} seeds x $(wc -w <<< "$BLLARSE_FT_ARMS") arms" \
     "on ${BLLARSE_FT_MODEL} for ${BLLARSE_FT_EPOCHS} epochs" >&2

# Workers FIRST, so they are already draining by the time the first checkpoint
# lands. They exit on their own after BLLARSE_OOD_IDLE_EXIT with nothing to do,
# so an over-provisioned worker costs idle time rather than a stuck allocation.
submit_workers() {
  local framework="$1" venv="$2" n="$3"
  (( n > 0 )) || return 0
  local wexports="BLLARSE_FT_VENV=${venv}"
  wexports+=",BLLARSE_FT_CKPT_ROOT=${BLLARSE_FT_CKPT_ROOT}"
  wexports+=",BLLARSE_OOD_FRAMEWORK=${framework}"
  wexports+=",BLLARSE_EVAL_DUMP_ROOT=${BLLARSE_EVAL_DUMP_ROOT:-${repo_root}/eval_dumps}"
  [[ -n "${BLLARSE_HF_HOME:-}" ]] && wexports+=",BLLARSE_HF_HOME=${BLLARSE_HF_HOME}"
  [[ -n "${BLLARSE_FT_PYLIBS:-}" ]] && wexports+=",BLLARSE_FT_PYLIBS=${BLLARSE_FT_PYLIBS}"
  local wcmd=(sbatch
    --job-name "ftood_${framework}_${BLLARSE_FT_MODEL#dinov3_}"
    --array "0-$((n - 1))"
    --export "ALL,${wexports}"
    "${repo_root}/src/slurm/jobs/slurm_ood_worker.sh")
  if [[ -n "$BLLARSE_FT_DRY_RUN" ]]; then printf '%q ' "${wcmd[@]}"; echo
  else "${wcmd[@]}"; fi
}

if (( BLLARSE_FT_OOD_EVERY_K > 0 )) && (( BLLARSE_FT_DUMP_EVERY_K > 0 )); then
  submit_workers jax "$BLLARSE_FT_VENV" "$BLLARSE_OOD_WORKERS_JAX"
  if grep -qw vbll <<< "$BLLARSE_FT_ARMS"; then
    submit_workers torch "$BLLARSE_FT_VBLL_VENV" "$BLLARSE_OOD_WORKERS_TORCH"
  fi
fi

for arm in $BLLARSE_FT_ARMS; do
  venv="$BLLARSE_FT_VENV"
  [[ "$arm" == "vbll" ]] && venv="$BLLARSE_FT_VBLL_VENV"

  exports="BLLARSE_FT_ARM=${arm}"
  exports+=",BLLARSE_FT_SEEDS=${BLLARSE_FT_SEEDS}"
  exports+=",BLLARSE_FT_STORAGE=${BLLARSE_FT_STORAGE}"
  exports+=",BLLARSE_FT_CAMPAIGN=${BLLARSE_FT_CAMPAIGN}"
  exports+=",BLLARSE_FT_SHARDS=${BLLARSE_FT_SHARDS}"
  exports+=",BLLARSE_FT_VENV=${venv}"
  exports+=",BLLARSE_FT_MODEL=${BLLARSE_FT_MODEL}"
  exports+=",BLLARSE_FT_EPOCHS=${BLLARSE_FT_EPOCHS}"
  exports+=",BLLARSE_FT_EVAL_QUANTUM=${BLLARSE_FT_EVAL_QUANTUM}"
  [[ -n "${BLLARSE_EVAL_DUMP_ROOT:-}" ]] &&
    exports+=",BLLARSE_EVAL_DUMP_ROOT=${BLLARSE_EVAL_DUMP_ROOT}"
  [[ -n "${BLLARSE_HF_HOME:-}" ]] &&
    exports+=",BLLARSE_HF_HOME=${BLLARSE_HF_HOME}"
  [[ -n "${BLLARSE_FT_PYLIBS:-}" ]] &&
    exports+=",BLLARSE_FT_PYLIBS=${BLLARSE_FT_PYLIBS}"
  exports+=",BLLARSE_FT_DUMP_EVERY_K=${BLLARSE_FT_DUMP_EVERY_K}"
  exports+=",BLLARSE_FT_OOD_EVERY_K=${BLLARSE_FT_OOD_EVERY_K}"
  exports+=",BLLARSE_FT_CKPT_ROOT=${BLLARSE_FT_CKPT_ROOT}"
  exports+=",BLLARSE_FT_OOD_MAX_PENDING=${BLLARSE_FT_OOD_MAX_PENDING}"

  cmd=(sbatch
       --job-name "ftfinal_${arm}_${BLLARSE_FT_MODEL#dinov3_}"
       --array "0-$((n_seeds - 1))%${BLLARSE_FT_THROTTLE}"
       --export "ALL,${exports}"
       "$job")

  if [[ -n "$BLLARSE_FT_DRY_RUN" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
