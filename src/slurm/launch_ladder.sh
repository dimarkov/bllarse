#!/bin/bash
# The dinov3 tuning ladder: one Optuna study per (arm, backbone rung).
#
#   src/slurm/launch_ladder.sh small
#   src/slurm/launch_ladder.sh big large huge
#   src/slurm/launch_ladder.sh all
#
# Run it FROM the code tree on the cluster ($HOME or /ptmp), not from a laptop:
# it calls submit_finetune_tune.sh, which pre-flights the dataset and the
# backbone before it will queue anything.
#
# flock, because this script takes minutes -- the offline dataset probe and the
# backbone pre-flight both run before the first sbatch -- and a launcher invoked
# twice submits two identical arrays into one study. That has happened twice by
# accident, both times because the invoking tool call outlived its own timeout
# and was retried. Concurrency is refused rather than trusted to timing.
set -euo pipefail
exec 9>"/ptmp/${USER}/.launch_ladder.lock"
flock -n 9 || { echo "ALREADY RUNNING -- refusing to double-submit"; exit 0; }

cd "$(dirname "$0")/../.."

# --- How many samples a trial sees at each rung -----------------------------
#
# samples = batch x steps, so a rung can hold TWO of the three fixed, never all
# three. The choice is not cosmetic: `finetune_space.beta1_horizons_box` caps
# H1 at 0.4 x steps, so the batch sets how much of the b1 axis the search can
# reach at all.
#
#   fixed-steps  (default) -- 781 steps at every rung, so every rung searches
#     the SAME box, H1 in (39, 312). This is also the honest batch experiment:
#     the AdamW SDE rule claims batch is irrelevant once lr/sqrt(B) and H are
#     held fixed, so a batch effect at fixed steps is a violation of the rule
#     rather than an artefact of a shorter run. Costs samples: 4x at batch 512.
#
#   fixed-samples -- 100k everywhere, as the small rung ran. Cheap, but steps
#     fall to 195 at batch 512 and the box collapses to H1 in (39, 78), i.e.
#     b1 in [0.60, 0.80]. The small rung's own optima sit at H1 = 196 (ce) and
#     148 (ibprobit_noeps): BOTH above that ceiling. The search would be pinned
#     against a wall it cannot see, which is exactly the failure the trunk-LR
#     ceiling just cost us a campaign to find.
: "${BLLARSE_LADDER_MODE:=fixed-steps}"
: "${BLLARSE_LADDER_STEPS:=781}"

samples_for() {  # $1 = batch
  case "$BLLARSE_LADDER_MODE" in
    fixed-steps)   echo $(( BLLARSE_LADDER_STEPS * $1 )) ;;
    fixed-samples) echo 100000 ;;
    *) echo "unknown BLLARSE_LADDER_MODE=$BLLARSE_LADDER_MODE" >&2; exit 2 ;;
  esac
}

# --- Everything a rung does not choose --------------------------------------
export BLLARSE_FT_EVAL_QUANTUM=2048
# 0.75^12: the per-block decay dinov3_small was tuned at, carried across the
# ladder as a SPAN so a 12-block and a 32-block trunk damp their input layer by
# the same factor rather than by the same per-block ratio. A rung may override
# it; 1.0 is the span that means "no LLRD at all".
DEFAULT_SPAN=0.0316763515
export BLLARSE_FT_TARGET_COMPLETE=50
# 50, down from 90. `tuning-search-saturates-at-n40` measured the incumbent
# stopping improving by ~40 completions on every arm of the last campaign, so 90
# was roughly 2x over-provisioned. The saving pays for the pruner warmup moving
# from 10 evaluations to 30% of the run (finetune_worker.PRUNER_WARMUP_FRACTION),
# which triples what a pruned trial costs: 50 completions at the new warmup is
# close to the same GPU-hours as 90 at the old one. The two constants were
# changed together and should move together.
#
# 16 x 60 = 960 trial SLOTS per arm remains the cap, not the plan: the pruner
# takes most trials (the small rung completed 24% of CE's and 17% of VBLL's),
# and workers past the target exit on TARGET-MET without taking a trial, so
# leaving the slack in place costs nothing and covers a worse prune rate.
export BLLARSE_FT_WORKERS=16
export BLLARSE_FT_THROTTLE=6
export BLLARSE_FT_TRIALS=60
# A fresh process every 10 trials. The arena fragments across a long campaign:
# ft_ce_small100k lost 28 of 273 trials to an OOM that arrived after 32-39
# trials in every one of its six workers. See slurm_finetune_tune.sh.
export BLLARSE_FT_TRIALS_PER_PROC=10
export BLLARSE_FT_VENV="/ptmp/${USER}/venv_ft/bin/python"
export BLLARSE_FT_VBLL_VENV="/ptmp/${USER}/venv_vbll_ft/bin/python"
# NO BLLARSE_EVAL_DUMP_ROOT, deliberately. A tuning trial does not need its
# dumps: `finetune_worker.on_dump` computes the objective from the in-loop
# scalars and says in its own comment that it does NOT read the value back off
# the dump (the run evaluates val only, and a dump round-trips through fp16).
# Optuna keeps the full per-trial val_nll trajectory in `intermediate_values`,
# so the curves survive in the journal either way. The only consumer of the
# files is scripts/plotting/plot_tuning_optima.py, a stopgap for "the final
# campaign has not run yet" that is in no figure of the results plan.
#
# What they cost is real: /raven/ptmp filled cluster-wide on 2026-09-03 and 27
# trials across five studies died inside a single hour with
# OSError(28, 'No space left on device') -- always in on_dump, never in the
# training step. At the top rungs a lost trial is ~2.8 h.
#
# Set BLLARSE_EVAL_DUMP_ROOT in the environment to turn them back on for a
# campaign that actually wants them; submit_finetune_tune.sh still forwards it.
export BLLARSE_HF_HOME="/ptmp/${USER}/hf_cache"
export BLLARSE_FT_SHARDS="/ptmp/${USER}/shards"

ALL_ARMS='ce ibprobit_noeps vbll'

# --- The rungs --------------------------------------------------------------
# rung | campaign | model | effective batch | arms
#
# The campaign name carries the BUDGET, because the name is the study's
# identity and a study is only comparable to trials drawn from its own space.
# `large400k` rather than `large100k`: under fixed-steps a batch-512 rung sees
# 781 x 512 = 399,872 samples, and a name that said 100k would be a standing
# invitation to compare it against the small rung's trials.
#
# Batch rises with the backbone by measurement, not by taste: it is the largest
# effective batch whose micro batch still divides it on a 40 GB A100 (see
# finetune_space.MICRO_BATCH). 1024 is not reachable at any rung.
#
# small512 is a CONTROL, not a rung of the ladder. The ladder moves backbone and
# batch together, so a gap that widens along it cannot be attributed to either.
# This holds the backbone at dinov3_small and moves ONLY the batch, against the
# small rung -- the one comparison that can answer whether IBProbit's advantage
# is a batch effect. Two arms, because CE is the comparator and VBLL is not part
# of the question.
# SAMPLES is pinned here rather than derived: ft_ibprobit_noeps_small100klr
# already holds 60 completed trials at exactly 100000, and this rung TOPS IT UP.
# 781 x 128 = 99,968 would search the identical space (both floor to 781 steps)
# while recording a different budget on half the study's trials.
rung_small()    { CAMPAIGN=small100klr   MODEL=dinov3_small BATCH=128 ARMS="$ALL_ARMS" SAMPLES=100000; }
rung_small512() { CAMPAIGN=small400kb512 MODEL=dinov3_small BATCH=512 ARMS='ce ibprobit_noeps'; }
# The OTHER control. Between the 1M campaign where IBProbit led CE by 4.21% and
# the 100k one where it trails, five things changed at once, and LLRD is one of
# them -- it went from off to on. LLRD is a trunk-CONDITIONING aid, and a
# conjugate head that absorbs a badly-scaled trunk update is plausibly giving
# IBProbit the same thing LLRD now hands CE for free. Identical to `small` in
# every other respect, so the pair isolates it. SPAN=1.0 is LLRD off.
rung_smallnollrd() { CAMPAIGN=small100knollrd MODEL=dinov3_small BATCH=128 ARMS='ce ibprobit_noeps' SAMPLES=100000 SPAN=1.0; }
rung_big()      { CAMPAIGN=big200k       MODEL=dinov3_big   BATCH=256 ARMS="$ALL_ARMS"; }
# v2, 2026-09-09: NEW STUDY NAMES, not a resume. `open_study` passes
# load_if_exists=True, so reusing `large400k` would either raise on the changed
# num_update_iters distribution or -- worse -- silently mix trials scored by two
# different objectives. Four things changed and every one of them redefines what
# a trial MEANS, so the old journals stay frozen as history:
#
#   1. q_init now transfers by protocol.transfer's q0 ~ T^-2 rule instead of by
#      identity. The tuned value is unchanged; what it becomes at the reported
#      horizon is not. This is the fix for the search picking q ~ 1.3e-3, which
#      is two decades high at 2 epochs.
#   2. num_update_iters is PINNED at 256 rather than searched over five rungs --
#      measured to be within 0.001 val_nll of 128 once q is right, and collapsed
#      to a single value across the last campaign's 20 best trials anyway.
#   3. The pruner's warmup is 30% of the run, not a flat 10 evaluations (0.8% of
#      samples), which was selecting FOR the early-dip-then-climb signature of a
#      too-large q0.
#   4. The objective is the mean of the last 5 evaluations, not the running
#      argmin, so a config has to still be good at the end to win.
rung_large()    { CAMPAIGN=large400kv2   MODEL=dinov3_large BATCH=512 ARMS="$ALL_ARMS"; }
rung_huge()     { CAMPAIGN=huge400kv2    MODEL=dinov3_huge  BATCH=512 ARMS="$ALL_ARMS"; }

launch() {
  local rung="$1"
  if ! declare -F "rung_${rung}" >/dev/null; then
    echo "unknown rung '${rung}'" >&2; exit 2
  fi
  SAMPLES="" SPAN=""
  "rung_${rung}"
  local samples; samples="${SAMPLES:-$(samples_for "$BATCH")}"
  export BLLARSE_FT_LAYER_DECAY_SPAN="${SPAN:-$DEFAULT_SPAN}"
  echo "##### ${rung}: ${MODEL} batch ${BATCH} x $(( samples / BATCH )) steps" \
       "= ${samples} samples | span ${BLLARSE_FT_LAYER_DECAY_SPAN} | arms: ${ARMS} | campaign ${CAMPAIGN}"
  BLLARSE_FT_ARMS="$ARMS" \
  BLLARSE_FT_CAMPAIGN="$CAMPAIGN" \
  BLLARSE_FT_MODEL="$MODEL" \
  BLLARSE_FT_BATCH="$BATCH" \
  BLLARSE_FT_MAX_SAMPLES="$samples" \
    src/slurm/submit_finetune_tune.sh
}

[[ $# -gt 0 ]] || { echo "usage: $0 <small|small512|smallnollrd|big|large|huge|all>..." >&2; exit 2; }

# Order is submission order, and submission order is effectively priority:
# every array carries the same fairshare, so what goes in first starts first.
# The small rung is what unblocks the 2-epoch finals, so it leads.
requested=("$@")
if [[ "${requested[0]}" == "all" ]]; then
  # The two dinov3_small CONTROLS go out right behind the small rung: they are
  # what the ladder cannot answer, since it moves backbone and batch together.
  requested=(small small512 smallnollrd big large huge)
fi
echo "mode=${BLLARSE_LADDER_MODE} rungs: ${requested[*]}"
for rung in "${requested[@]}"; do
  launch "$rung"
done
echo "##### DONE"
