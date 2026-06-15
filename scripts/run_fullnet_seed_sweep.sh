#!/usr/bin/env bash
#
# Full-network finetuning sweep over:
#   datasets        : cifar10, cifar100
#   batch sizes     : 512, 16384
#   loss functions  : IBProbit, CrossEntropy
#   data augmentation: on (aug) and off (--nodataaug)
#   seeds           : 5 per configuration
#
# Base model (constant): largest backbone (12 blocks, embed dim 1024) starting
# from the ImageNet-21k checkpoint (--pretrained in21k).
#
# => 2 x 2 x 2 x 2 x 5 = 80 runs, dispatched in parallel across the available GPUs
#    (one training job per GPU at a time; a freed GPU immediately picks up the
#    next run).
#
# Hyperparameters (per the experiment design):
#   trunk lr (--learning-rate, both losses, depends on batch size):
#       bs=512   -> 1e-5
#       bs=16384 -> 1e-4
#   head lr (--head-lr, CrossEntropy only, depends on batch size):
#       bs=512   -> 1e-3
#       bs=16384 -> 5e-3
#   num update iters (--num-update-iters, IBProbit only, depends on dataset):
#       cifar10  -> 16
#       cifar100 -> 160
#   --val-fraction is always 0.1; --tune-mode is always full_network.
#
# All runs are grouped under a single MLflow parent run (RUN_NAME); each child
# run is nested under it via MLFLOW_PARENT_RUN_ID (consumed by finetuning.py).
#
# Usage:
#   source .venv/bin/activate        # see README; or set PYTHON_CMD="uv run python"
#   bash scripts/run_fullnet_seed_sweep.sh
#
# Knobs (override via environment), e.g.:
#   DRY_RUN=1 bash scripts/run_fullnet_seed_sweep.sh          # print commands only
#   GPUS=0,1,2,3 bash scripts/run_fullnet_seed_sweep.sh       # use 4 GPUs
#   GPUS=0 bash scripts/run_fullnet_seed_sweep.sh             # serial on one GPU
#   ENABLE_MLFLOW=0 EPOCHS=200 bash scripts/run_fullnet_seed_sweep.sh

set -euo pipefail

# 'wait -n -p' (used by the GPU scheduler) needs bash >= 5.1.
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "ERROR: this script needs bash >= 5.1 (for 'wait -n -p'); found $BASH_VERSION" >&2
  exit 1
fi

# ---- Configurable knobs ------------------------------------------------------
PYTHON_CMD="${PYTHON_CMD:-python}"          # assumes the uv venv is activated
SCRIPT="${SCRIPT:-scripts/finetuning.py}"
EPOCHS="${EPOCHS:-20}"
SAVE_EVERY="${SAVE_EVERY:-1}"               # save/checkpoint every N epochs; EPOCHS should be a multiple
DEVICE="${DEVICE:-gpu}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
OPTIMIZER="${OPTIMIZER:-adamw}"             # trunk/head LRs assume adamw (or lion)
PRETRAINED="${PRETRAINED:-in21k}"           # start from the ImageNet-21k checkpoint
NUM_BLOCKS="${NUM_BLOCKS:-12}"              # largest model: 12 blocks
EMBED_DIM="${EMBED_DIM:-1024}"             # largest model: embed dim 1024
GPUS="${GPUS:-0,1}"                         # comma-separated GPU ids to run on
ENABLE_MLFLOW="${ENABLE_MLFLOW:-1}"         # 1 -> log to MLflow, 0 -> disable
RUN_NAME="${RUN_NAME:-fullnet_seed_sweep}"  # shared MLflow parent run name (all runs nest under it)
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-bllarse}"
GROUP_ID="${GROUP_ID:-$RUN_NAME}"           # group_id tag applied to every child run
DRY_RUN="${DRY_RUN:-0}"                     # 1 -> print commands without running
EXTRA_ARGS="${EXTRA_ARGS:-}"               # appended verbatim to every run
PARENT_RUN_ID="${PARENT_RUN_ID:-}"          # reuse an existing MLflow parent instead of creating one
ONLY_UIDS="${ONLY_UIDS:-}"                  # if set, run only tasks whose uid is in this whitespace-separated list

# Sweep axes
SEEDS=(137 138 139 140 141)
DATASETS=(cifar10 cifar100)
BATCH_SIZES=(512 16384)
LOSSES=(IBProbit CrossEntropy)
AUGS=(aug noaug)                            # 'aug' -> data augmentation on; 'noaug' -> --nodataaug
# -----------------------------------------------------------------------------

# Allow multi-word PYTHON_CMD / EXTRA_ARGS (e.g. "uv run python").
read -r -a PY <<< "$PYTHON_CMD"
read -r -a EXTRA <<< "$EXTRA_ARGS"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}
if (( NUM_GPUS == 0 )); then
  echo "ERROR: no GPUs specified (set GPUS=0,1,...)." >&2
  exit 1
fi

total=$(( ${#DATASETS[@]} * ${#BATCH_SIZES[@]} * ${#LOSSES[@]} * ${#AUGS[@]} * ${#SEEDS[@]} ))
echo "Planned runs: $total  (gpus=[$GPUS], mlflow=$ENABLE_MLFLOW, dry_run=$DRY_RUN, epochs=$EPOCHS)"
echo

# Create a single MLflow parent run so every config/seed run is grouped under
# one named run on the server. finetuning.py reads MLFLOW_PARENT_RUN_ID and
# starts each run nested under it (nested=True + parent_run_id), mirroring the
# codex sweep tooling (src/bllarse/tools/run_sweep.py).
if [[ "$ENABLE_MLFLOW" == "1" && "$DRY_RUN" != "1" && -n "$PARENT_RUN_ID" ]]; then
  # Reuse an existing parent (e.g. to re-run failed/missing children into the
  # same group). The parent already exists on the server, and children resolve
  # the same tracking URI from .env.secrets, so they nest correctly.
  export MLFLOW_PARENT_RUN_ID="$PARENT_RUN_ID"
  echo "Reusing existing MLflow parent run id: $PARENT_RUN_ID"
  echo
elif [[ "$ENABLE_MLFLOW" == "1" && "$DRY_RUN" != "1" ]]; then
  echo "Creating MLflow parent run '$RUN_NAME' (experiment '$MLFLOW_EXPERIMENT')..."
  parent_out=$(
    BLLARSE_SWEEP_RUN_NAME="$RUN_NAME" \
    BLLARSE_SWEEP_EXPERIMENT="$MLFLOW_EXPERIMENT" \
    "${PY[@]}" - <<'PYEOF'
import os
# Mirror finetuning.py's config loading so the parent run is created on the SAME
# tracking server that the child runs use (the tracking URI / credentials live
# in .env / .env.secrets, loaded via dotenv).
try:
    # Explicit paths: this runs via `python -` (stdin), where dotenv's
    # find_dotenv() would fail to walk the call stack.
    from dotenv import load_dotenv
    load_dotenv(".env")
    load_dotenv(".env.secrets", override=False)
except Exception:
    pass
import mlflow
try:
    from bllarse.mlflow_utils import load_mlflow_env_defaults
    load_mlflow_env_defaults()
except Exception:
    pass
uri = os.environ.get("MLFLOW_TRACKING_URI")
exp = os.environ.get("BLLARSE_SWEEP_EXPERIMENT") or "bllarse"
name = os.environ.get("BLLARSE_SWEEP_RUN_NAME") or "sweep"
if uri:
    mlflow.set_tracking_uri(uri)
mlflow.set_experiment(exp)
run = mlflow.start_run(
    run_name=name,
    tags={"sweep_name": name, "group_id": name, "is_parent": "true"},
)
mlflow.end_run()
print(f"PARENT_RUN_ID={run.info.run_id}")
print(f"TRACKING_URI={mlflow.get_tracking_uri()}")
PYEOF
  ) || true
  PARENT_RUN_ID=$(printf '%s\n' "$parent_out" | sed -n 's/^PARENT_RUN_ID=//p' | tail -n1)
  TRACKING_URI=$(printf '%s\n' "$parent_out" | sed -n 's/^TRACKING_URI=//p' | tail -n1)
  if [[ -z "$PARENT_RUN_ID" ]]; then
    echo "WARNING: could not create MLflow parent run; child runs will not be nested." >&2
    printf '%s\n' "$parent_out" >&2
  else
    export MLFLOW_PARENT_RUN_ID="$PARENT_RUN_ID"
    # Pin children to the exact same tracking server the parent was created on.
    if [[ -n "$TRACKING_URI" ]]; then
      export MLFLOW_TRACKING_URI="$TRACKING_URI"
    fi
    echo "MLflow parent run id: $PARENT_RUN_ID"
    echo "MLflow tracking uri:  ${TRACKING_URI:-<resolved by child from .env>}"
  fi
  echo
fi

# ---- Build the task list -----------------------------------------------------
TASK_CMDS=()   # each entry is a printf %q-quoted command line (run via bash -c)
TASK_UIDS=()

for dataset in "${DATASETS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    # Trunk learning rate depends only on the batch size (constant over losses).
    case "$bs" in
      512)   trunk_lr=1e-5 ;;
      16384) trunk_lr=1e-4 ;;
      *) echo "ERROR: no trunk LR defined for batch size $bs" >&2; exit 1 ;;
    esac

    for loss in "${LOSSES[@]}"; do
      # Loss-specific arguments.
      extra_loss=()
      if [[ "$loss" == "CrossEntropy" ]]; then
        case "$bs" in
          512)   head_lr=1e-3 ;;
          16384) head_lr=5e-3 ;;
          *) echo "ERROR: no head LR defined for batch size $bs" >&2; exit 1 ;;
        esac
        extra_loss+=(--head-lr "$head_lr")
      elif [[ "$loss" == "IBProbit" ]]; then
        case "$dataset" in
          cifar10)  num_iters=16 ;;
          cifar100) num_iters=160 ;;
          *) echo "ERROR: no num-update-iters defined for $dataset" >&2; exit 1 ;;
        esac
        extra_loss+=(--num-update-iters "$num_iters")
      else
        echo "ERROR: unknown loss $loss" >&2; exit 1
      fi

      for aug in "${AUGS[@]}"; do
        # Data augmentation on/off.
        aug_args=()
        case "$aug" in
          aug)   ;;                      # augmentation enabled (default)
          noaug) aug_args+=(--nodataaug) ;;
          *) echo "ERROR: unknown aug option $aug" >&2; exit 1 ;;
        esac

        for seed in "${SEEDS[@]}"; do
        uid="${dataset}_bs${bs}_${loss}_${aug}_seed${seed}"

        # When ONLY_UIDS is set, build only the listed runs (e.g. re-running
        # specific failed seeds).
        if [[ -n "$ONLY_UIDS" && " $ONLY_UIDS " != *" $uid "* ]]; then
          continue
        fi

        cmd=( "${PY[@]}" "$SCRIPT"
              --dataset "$dataset"
              --batch-size "$bs"
              --loss-fn "$loss"
              --tune-mode full_network
              --pretrained "$PRETRAINED"
              --num-blocks "$NUM_BLOCKS"
              --embed-dim "$EMBED_DIM"
              --optimizer "$OPTIMIZER"
              --learning-rate "$trunk_lr"
              --val-fraction "$VAL_FRACTION"
              --epochs "$EPOCHS"
              --save-every "$SAVE_EVERY"
              --device "$DEVICE"
              --seed "$seed"
              "${extra_loss[@]}"
              "${aug_args[@]}" )

        if [[ "$ENABLE_MLFLOW" == "1" ]]; then
          cmd+=( --enable-mlflow
                 --mlflow-experiment "$MLFLOW_EXPERIMENT"
                 --group-id "$GROUP_ID"
                 --uid "$uid" )
        fi
        if [[ -n "$EXTRA_ARGS" ]]; then
          cmd+=( "${EXTRA[@]}" )
        fi

        TASK_CMDS+=( "$(printf '%q ' "${cmd[@]}")" )
        TASK_UIDS+=( "$uid" )
        done   # seed
      done     # aug
    done       # loss
  done         # bs
done           # dataset

# ---- Dry run: just print -----------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  for i in "${!TASK_CMDS[@]}"; do
    echo "==> [$((i + 1))/$total] ${TASK_UIDS[$i]}"
    echo "    ${TASK_CMDS[$i]}"
  done
  echo
  echo "Dry run: $total tasks would be distributed across GPUs [$GPUS]."
  if [[ "$ENABLE_MLFLOW" == "1" ]]; then
    echo
    echo "MLflow grouping: a parent run named '$RUN_NAME' would be created on the"
    echo "server first, its run id exported as MLFLOW_PARENT_RUN_ID, and each run"
    echo "nested under it via mlflow.parentRunId (this is what groups runs in the"
    echo "UI). --group-id is only an extra searchable tag, not the grouping key."
    echo "Both are skipped here because dry-run does not create a real parent run."
  fi
  exit 0
fi

# ---- Dispatch across GPUs ----------------------------------------------------
# Dynamic pool: keep one job running per GPU; when a job finishes, the freed GPU
# immediately picks up the next pending run.
echo "Dispatching $total runs across GPUs [$GPUS] (one job per GPU)..."
declare -A pid2gpu=()
free_gpus=( "${GPU_IDS[@]}" )
next=0
fail=0
N=${#TASK_CMDS[@]}

while (( next < N )) || (( ${#pid2gpu[@]} > 0 )); do
  # Launch on every currently free GPU.
  while (( next < N )) && (( ${#free_gpus[@]} > 0 )); do
    dev="${free_gpus[-1]}"
    unset 'free_gpus[-1]'
    echo "[gpu $dev] START [$((next + 1))/$N] ${TASK_UIDS[$next]}"
    CUDA_VISIBLE_DEVICES="$dev" bash -c "${TASK_CMDS[$next]}" &
    pid2gpu[$!]="$dev"
    next=$(( next + 1 ))
  done

  # Wait for any one job to finish, free its GPU.
  if (( ${#pid2gpu[@]} > 0 )); then
    done_pid=""
    if wait -n -p done_pid; then status=0; else status=$?; fi
    if [[ -n "$done_pid" && -n "${pid2gpu[$done_pid]:-}" ]]; then
      dev="${pid2gpu[$done_pid]}"
      unset "pid2gpu[$done_pid]"
      free_gpus+=( "$dev" )
      if (( status != 0 )); then
        echo "[gpu $dev] FAILED (pid $done_pid, exit $status)" >&2
        fail=$(( fail + 1 ))
      else
        echo "[gpu $dev] done (pid $done_pid)"
      fi
    fi
  fi
done

echo
echo "Done: $N runs, $fail failure(s)."
(( fail == 0 ))
