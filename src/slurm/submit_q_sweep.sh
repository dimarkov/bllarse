#!/usr/bin/env bash
# Why does IBProbit's NLL rise 34% past its own minimum? A q sweep at the
# reported duration.
#
#   BLLARSE_FT_VENV=/ptmp/$USER/venv_ft/bin/python \
#   BLLARSE_EVAL_DUMP_ROOT=/ptmp/$USER/eval_dumps \
#   BLLARSE_HF_HOME=/ptmp/$USER/hf_cache \
#     src/slurm/submit_q_sweep.sh
#
# The measurement that motivates it: on dinov3_large, past its NLL minimum at
# 50% of the budget, IBProbit ends +34.2% while CE+TS ends +4.6% and VBLL
# +2.1%, and every arm ends OVER-confident (conf - acc: IBProbit +0.059, CE
# +0.045, VBLL +0.020). Over-confidence is the wrong sign for "q is too large"
# -- q is a process-noise floor, and too much of it under-confidences -- so the
# points bracket the tuned value on BOTH sides rather than only below it.
#
# The predicted optimum comes from the campaign's own two durations: at fixed
# batch 512 the top-8 geometric mean of q_init is 5.02e-5 at 1M samples and
# 7.16e-5 at 2M, i.e. q* ~ T^+0.5. Extrapolated to 5.12M that is ~1.2e-4.
#
# The CONTROL is the reported run itself (ftfinal, q=5.347e-5, no_learn_q), and
# it is deliberately NOT re-run.
set -euo pipefail

: "${BLLARSE_FT_STORAGE:=/ptmp/${USER}/ft_studies}"
: "${BLLARSE_FT_CAMPAIGN:=betasearch1m}"
: "${BLLARSE_FT_SHARDS:=/ptmp/${USER}/shards}"
: "${BLLARSE_FT_MODEL:=dinov3_large}"
: "${BLLARSE_FT_SEED:=137}"
: "${BLLARSE_FT_GROUP:=qsweep}"
: "${BLLARSE_FT_ARM:=ibprobit_noeps}"
: "${BLLARSE_FT_VENV:=}"
: "${BLLARSE_FT_DRY_RUN:=}"

# label|override[|override...]
#   q1.2e-4  the T^+0.5 prediction
#   q3.0e-4  a deliberate over-shoot, so a flat result is distinguishable from
#            "the step was too small to see"
#   q2.0e-5  BELOW the tuned value. If the curve is U-shaped around 5.3e-5 this
#            is worse; if the late rise is really a too-large q, this is better.
#            Without it the sweep only tests one direction and cannot fail.
#   learnq   the recursion switched ON at the tuned q. Measured inert at these
#            operating points before, so this is a check on that measurement as
#            much as a candidate fix.
: "${BLLARSE_FT_POINTS:=q1.2e-4|q_init=1.2e-4;q3.0e-4|q_init=3.0e-4;q2.0e-5|q_init=2.0e-5;learnq|no_learn_q=false}"

repo_root="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
job="${repo_root}/src/slurm/jobs/slurm_q_sweep.sh"

if [[ -z "$BLLARSE_FT_VENV" ]]; then
  echo "ERROR: set BLLARSE_FT_VENV to the interpreter that has equimo." >&2
  exit 2
fi
if [[ -d "$BLLARSE_FT_VENV" || ! -x "$BLLARSE_FT_VENV" ]]; then
  echo "ERROR: BLLARSE_FT_VENV is not an executable interpreter: $BLLARSE_FT_VENV" >&2
  exit 2
fi

shard_meta="${BLLARSE_FT_SHARDS%/}/imagenet1k_train_224.json"
if [[ ! -f "$shard_meta" ]]; then
  echo "ERROR: no training shard at $shard_meta" >&2
  exit 2
fi

# Every point must RESOLVE before any of them is submitted. --set refuses a key
# that is not in the transferred config, so a typo fails here rather than after
# a queue wait, and it fails for all four rather than for the one that ran first.
IFS=';' read -r -a POINTS <<< "$BLLARSE_FT_POINTS"
echo "Resolving ${#POINTS[@]} points:" >&2
for point in "${POINTS[@]}"; do
  IFS='|' read -r -a fields <<< "$point"
  args=(--arm "$BLLARSE_FT_ARM" --seed "$BLLARSE_FT_SEED"
        --storage-dir "$(readlink -f "$BLLARSE_FT_STORAGE")"
        --campaign "$BLLARSE_FT_CAMPAIGN" --model "$BLLARSE_FT_MODEL"
        --uid "check" --group-id "$BLLARSE_FT_GROUP")
  for ((i = 1; i < ${#fields[@]}; i++)); do args+=(--set "${fields[$i]}"); done
  args+=(--format json)
  if ! out="$(PYTHONPATH="${repo_root}/src" "$BLLARSE_FT_VENV" \
              "${repo_root}/scripts/finetune/final_config.py" "${args[@]}" 2>&1)"; then
    echo "ERROR: point '${fields[0]}' does not resolve:" >&2
    echo "$out" >&2
    exit 2
  fi
  # Echo what this point actually OVERRODE, not a fixed key list. A WSD point
  # sets lr_decay_*, a q point sets q_init, and printing the wrong pair made a
  # correctly-configured run look unconfigured -- which is worse than silence,
  # because this line exists to catch the opposite mistake.
  echo "  ${fields[0]}: $(echo "$out" | sed -n 's/^\[final\] override //p' \
        | tr '\n' ';')" >&2
done

# The points go through a FILE, not --export: the list contains ';' and '|',
# and SLURM's export parsing is not where to find out that one of them did not
# survive. On shared storage, since /tmp is node-local on this cluster.
points_file="${repo_root}/.jobscripts/qsweep_${BLLARSE_FT_GROUP}_points.txt"
mkdir -p "$(dirname "$points_file")"
printf '%s\n' "${POINTS[@]}" > "$points_file"
echo "Points file: $points_file" >&2

n=$(( ${#POINTS[@]} - 1 ))
cmd=(sbatch --job-name="sweep_${BLLARSE_FT_ARM%_noeps}_${BLLARSE_FT_MODEL#dinov3_}" --array="0-${n}"
     --export="ALL,BLLARSE_FT_ARM=${BLLARSE_FT_ARM},BLLARSE_FT_VBLL_VENV=${BLLARSE_FT_VBLL_VENV:-},BLLARSE_FT_POINTS_FILE=${points_file},BLLARSE_FT_STORAGE=${BLLARSE_FT_STORAGE},BLLARSE_FT_CAMPAIGN=${BLLARSE_FT_CAMPAIGN},BLLARSE_FT_SHARDS=${BLLARSE_FT_SHARDS},BLLARSE_FT_MODEL=${BLLARSE_FT_MODEL},BLLARSE_FT_SEED=${BLLARSE_FT_SEED},BLLARSE_FT_GROUP=${BLLARSE_FT_GROUP},BLLARSE_FT_VENV=${BLLARSE_FT_VENV}"
     "$job")
if [[ -n "$BLLARSE_FT_DRY_RUN" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
"${cmd[@]}"
