#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="bllarse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Shared defaults ──
EPOCHS=1
LOSS_FN="IBProbit"
SEEDS=(137 139 141 143 147 151 153 157 161)

# ── Sweep axes ──
MODELS=(dinov3_small dinov3_big)
DATASETS=(cifar10 cifar100)
BATCH_SIZES=(512 1024 2048)

# Loss-specific hyperparameters
IBPROBIT_NUM_UPDATE_ITERS=(16 32 64)
CE_LRS=(1e-3 5e-4)
CE_WEIGHT_DECAYS=(1e-4 1e-3)

# ── Get or create parent run for the loss function ──
PARENT_RUN_ID=$(python "$SCRIPT_DIR/get_mlflow_run.py" \
    --experiment-name "$EXPERIMENT_NAME" --run-name "alt-classification-dinov3-$LOSS_FN")
echo "Parent run for $LOSS_FN: $PARENT_RUN_ID"

# ── Sweep ──
for model in "${MODELS[@]}"; do
for dataset in "${DATASETS[@]}"; do
for batch_size in "${BATCH_SIZES[@]}"; do
for seed in "${SEEDS[@]}"; do

    if [[ "$LOSS_FN" == "IBProbit" ]]; then
        for nui in "${IBPROBIT_NUM_UPDATE_ITERS[@]}"; do
            echo "Running: $model / $dataset / bs=$batch_size / nui=$nui / seed=$seed"
            python "$SCRIPT_DIR/vit_classification.py" \
                --model "$model" --dataset "$dataset" --loss-fn "$LOSS_FN" \
                --run-id "$PARENT_RUN_ID" \
                --epochs "$EPOCHS" --batch-size "$batch_size" --seed "$seed" \
                --num-update-iters "$nui"
        done

    elif [[ "$LOSS_FN" == "CrossEntropy" ]]; then
        for lr in "${CE_LRS[@]}"; do
        for wd in "${CE_WEIGHT_DECAYS[@]}"; do
            echo "Running: $model / $dataset / bs=$batch_size / lr=$lr / wd=$wd / seed=$seed"
            python "$SCRIPT_DIR/vit_classification.py" \
                --model "$model" --dataset "$dataset" --loss-fn "$LOSS_FN" \
                --run-id "$PARENT_RUN_ID" \
                --epochs "$EPOCHS" --batch-size "$batch_size" --seed "$seed" \
                --lr "$lr" --weight-decay "$wd"
        done
        done

    else
        echo "Unknown loss function: $LOSS_FN"
        exit 1
    fi

done
done
done
done
