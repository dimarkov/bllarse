#!/bin/bash
# Submit one 4-worker tuning array per (arm, dataset), sized by where each
# study should END UP rather than by how many trials it adds.
#
# WHY THIS EXISTS. The first campaign was submitted as a shell loop with a
# per-worker trial count chosen per arm -- 16 for adamw_ce, 24 for vbll, 32 for
# the two IBProbit arms. Trials that the MedianPruner kills still consume that
# count, and the prune rate is not a property anyone controls: measured across
# the 28 samples40 studies it runs from 8% (vbll/flowers102) to 72%
# (ibprobit/oxford_pets). The same request therefore landed at 40 completed
# trials on adamw_ce/food101 and 103 on ibprobit/cifar10 -- the arms were
# searched to depths nobody chose, and Fig 1 compares whatever came out.
#
# BLLARSE_TUNE_TARGET_COMPLETE removes that coupling: every worker stops once
# the STUDY -- across all four workers and every earlier submission -- holds
# that many COMPLETED trials. So this script is idempotent. Run it again and
# studies already at the target exit immediately without claiming a trial;
# studies short of it top up by exactly the shortfall.
#
# BLLARSE_TUNE_TRIALS stays the per-worker cap and still bounds the job, so a
# study whose trials are nearly all pruned terminates rather than holding a GPU
# to the 24h wall. It is deliberately uniform and generous rather than a
# per-study table computed from today's journals: such a table is stale the
# moment a study advances, and overshoot costs only PRUNED trials, which are
# killed within a few evaluations and are the cheap ones.
#
# RESUMING, not restarting. The worker opens studies with load_if_exists, so
# with BLLARSE_TUNE_CAMPAIGN unchanged this continues the existing journals and
# TPE keeps every trial it has already seen. That is only sound while the
# SEARCH SPACE is unchanged. If you widen a box in space.py, set a new campaign
# name -- and note the two cases differ: optuna silently accepts a widened
# FLOAT range on an existing study (modelling old trials against a range they
# were never drawn from), but a changed CATEGORICAL -- the batch_size ladder --
# raises `CategoricalDistribution does not support dynamic value space` and
# fails every trial.
#
# Usage, from the repo root on the cluster:
#   BLLARSE_TUNE_TARGET_COMPLETE=100 src/slurm/submit_tune.sh
#   BLLARSE_TUNE_DRY_RUN=1 src/slurm/submit_tune.sh      # print, submit nothing
#   BLLARSE_TUNE_ARMS="adamw_ce vbll" src/slurm/submit_tune.sh

set -euo pipefail

: "${BLLARSE_TUNE_STORAGE:=/ptmp/${USER}/optuna}"
: "${BLLARSE_TUNE_CAMPAIGN:=samples40}"
: "${BLLARSE_TUNE_TARGET_COMPLETE:=100}"
: "${BLLARSE_TUNE_TRIALS:=80}"
: "${BLLARSE_SIF_DIR:=/ptmp/${USER}}"
# The arms the PAPER reports, and deliberately not all of space.ARMS.
#
# `ibprobit` -- the with-eps arm -- is absent by design. Both figures read the
# eps-free sweep: fig1_curves.SWEEP_MODULES maps "ibprobit" -> the
# `probe_ibprobit_noeps` module, and fig2_detection reads
# `probe_ibprobit_noeps_fig2`. The full-network fine-tuning campaign made the
# same call. So tuning the with-eps arm produces a configuration nothing
# downstream reads.
#
# It is not merely wasted compute. The loop below submits arms in order, so a
# with-eps array claims GPUs AHEAD of the eps-free one and delays the arm that
# is actually needed -- measured on 2026-08-12, when 20 running
# `tune_ibprobit_*` tasks held all 7 `tune_noeps_*` arrays PENDING.
#
# The arm is still runnable on request (`BLLARSE_TUNE_ARMS="ibprobit"`), since
# the eps ablation compares the two arms and would need both tuned under one
# objective. It just must never be the default again.
: "${BLLARSE_TUNE_ARMS:=adamw_ce vbll ibprobit_noeps}"
: "${BLLARSE_TUNE_DATASETS:=oxford_pets dtd flowers102 stanford_cars food101 cifar10 cifar100}"
: "${BLLARSE_TUNE_DRY_RUN:=}"
: "${BLLARSE_TUNE_SMOOTH_WINDOW:=}"

# Smoothing REDEFINES the objective (single-point minimum -> lowest windowed
# mean), so its values are not comparable with anything already in a journal.
# The worker opens studies with load_if_exists, so reusing a campaign name here
# would append the new objective's trials to the old objective's study and hand
# TPE a value column meaning two different things. Refuse rather than warn: the
# damage is silent, lands in a shared journal, and cannot be undone by deleting
# a run.
#
# `smoothed` is barred for a second, separate reason: it was run with the
# pruner reporting RAW val_nll while the objective scored the windowed mean.
# MedianPruner judges a trial against other trials' stored intermediate values,
# so appending windowed reports to journals full of raw ones would have it
# compare two different statistics. (That mismatch is also why `smoothed`'s own
# results are unusable -- 11 of its 21 winners came from MedianPruner's
# un-prunable first 20 trials.)
for _bad in samples40 smoothed smoothed2; do
  if [[ -n "$BLLARSE_TUNE_SMOOTH_WINDOW" && "$BLLARSE_TUNE_CAMPAIGN" == "$_bad" ]]; then
    echo "[bllarse] ERROR: campaign '$_bad' cannot be resumed under the current" >&2
    echo "  objective/pruner contract -- its journals hold trials scored or" >&2
    echo "  pruned on a different statistic. Choose a new BLLARSE_TUNE_CAMPAIGN." >&2
    exit 2
  fi
done

# The CONVERSE guard, and it is the one that bites now that more than one
# windowed campaign exists. The loop above stops a windowed submission from
# resuming an UNWINDOWED journal; nothing stopped the reverse. Re-running
# `BLLARSE_TUNE_CAMPAIGN=win2` with the window forgotten -- or typed as 5 --
# appends single-point-minimum trials to journals that already hold ~100 trials
# scored on the 2-window mean, and TPE then models a value column meaning two
# different things. The damage is silent and irreversible: three win2 studies
# exist (adamw_ce/oxford_pets, ibprobit_noeps/cifar10, vbll/dtd), each ~100
# completed trials, and a journal cannot be un-appended.
#
# So the campaign NAME decides the window rather than the caller re-typing it.
# A campaign not listed here keeps whatever the caller passed -- new names are
# free, existing ones are not.
case "$BLLARSE_TUNE_CAMPAIGN" in
  win2)       _want_window=2 ;;
  win5|win5b) _want_window=5 ;;
  *)          _want_window="$BLLARSE_TUNE_SMOOTH_WINDOW" ;;
esac
if [[ "$BLLARSE_TUNE_SMOOTH_WINDOW" != "$_want_window" ]]; then
  echo "[bllarse] ERROR: campaign '$BLLARSE_TUNE_CAMPAIGN' was written with" >&2
  echo "  BLLARSE_TUNE_SMOOTH_WINDOW=${_want_window}, but this submission passes" >&2
  echo "  '${BLLARSE_TUNE_SMOOTH_WINDOW:-<unset>}'. Its journals would then hold" >&2
  echo "  trials scored on two different statistics. Pass the matching window," >&2
  echo "  or choose a new BLLARSE_TUNE_CAMPAIGN." >&2
  exit 2
fi

export BLLARSE_TUNE_STORAGE BLLARSE_TUNE_CAMPAIGN BLLARSE_TUNE_TARGET_COMPLETE \
       BLLARSE_TUNE_TRIALS BLLARSE_TUNE_SMOOTH_WINDOW

job_script="$(dirname "$0")/jobs/slurm_tune.sh"
if [[ ! -f "$job_script" ]]; then
  echo "[bllarse] ERROR: $job_script not found; run this from the repo." >&2
  exit 2
fi

submitted=0
for arm in $BLLARSE_TUNE_ARMS; do
  # The image, and the short name the job wears in squeue. VBLL is the torch
  # arm; the other three all run scripts/dinov3/probe.py under JAX.
  case "$arm" in
    vbll)            sif="bllarse-torch.sif"; tag="vbll" ;;
    adamw_ce)        sif="bllarse-jax.sif";   tag="ce" ;;
    ibprobit)        sif="bllarse-jax.sif";   tag="ibprobit" ;;
    ibprobit_noeps)  sif="bllarse-jax.sif";   tag="noeps" ;;
    *) echo "[bllarse] ERROR: unknown arm '$arm'." >&2; exit 2 ;;
  esac

  sif_path="${BLLARSE_SIF_DIR}/${sif}"
  if [[ ! -f "$sif_path" ]]; then
    echo "[bllarse] ERROR: $sif_path not found (set BLLARSE_SIF_DIR)." >&2
    exit 2
  fi

  for ds in $BLLARSE_TUNE_DATASETS; do
    name="tune_${tag}_${ds}"
    echo "[bllarse] $name  arm=$arm sif=$sif" \
         "target=${BLLARSE_TUNE_TARGET_COMPLETE} cap=${BLLARSE_TUNE_TRIALS}/worker"
    if [[ -n "$BLLARSE_TUNE_DRY_RUN" ]]; then
      continue
    fi
    BLLARSE_SIF="$sif_path" BLLARSE_TUNE_ARM="$arm" BLLARSE_TUNE_DATASET="$ds" \
      sbatch --export=ALL --job-name="$name" "$job_script"
    submitted=$((submitted + 1))
  done
done

if [[ -n "$BLLARSE_TUNE_DRY_RUN" ]]; then
  echo "[bllarse] dry run: nothing submitted."
else
  echo "[bllarse] submitted $submitted arrays x 4 workers."
fi
