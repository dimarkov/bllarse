#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_NAME="bllarse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config ──
MODEL="dinov3_huge"
SEEDS=(137 139 141 143 145)

# Source sweeps to read best (model, imagenet1k) hyperparameters from.
IBPROBIT_SOURCE_SWEEP="sweep5a-alt-classification-dinov3-IBProbit"
CE_SOURCE_SWEEP="sweep10a-alt-classification-dinov3-CrossEntropy"

# Methods to evaluate. Comment out either to run only one arm.
METHODS=(IBProbit CrossEntropy)

for method in "${METHODS[@]}"; do

    if [[ "$method" == "IBProbit" ]]; then
        SOURCE_SWEEP="$IBPROBIT_SOURCE_SWEEP"
    else
        SOURCE_SWEEP="$CE_SOURCE_SWEEP"
    fi

    # ── Best hyperparameters for this (model, imagenet1k, method) ──
    echo "Fetching best hyperparameters for $method from $SOURCE_SWEEP ..."
    HPARAMS="$(uv run "$SCRIPT_DIR/get_best_hparams.py" \
        --parent-run-name "$SOURCE_SWEEP" \
        --model "$MODEL" --dataset imagenet1k --method "$method")"
    # Exports BATCH_SIZE and (LR | NUM_UPDATE_ITERS)
    eval "$HPARAMS"
    echo "  $method best: $HPARAMS" | tr '\n' ' '; echo

    # ── Get or create the OOD-eval parent run for this method ──
    PARENT_RUN_ID=$(uv run "$SCRIPT_DIR/get_mlflow_run.py" \
        --experiment-name "$EXPERIMENT_NAME" --run-name "ood-eval-$MODEL-$method")
    echo "Parent run for ood-eval-$MODEL-$method: $PARENT_RUN_ID"

    for seed in "${SEEDS[@]}"; do
        echo "Running: $method / $MODEL / seed=$seed"
        if [[ "$method" == "IBProbit" ]]; then
            uv run "$SCRIPT_DIR/ood_evaluation.py" \
                --model "$MODEL" --method "$method" --seed "$seed" \
                --batch-size "$BATCH_SIZE" --num-update-iters "$NUM_UPDATE_ITERS" \
                --run-id "$PARENT_RUN_ID" --log-mlflow
        else
            uv run "$SCRIPT_DIR/ood_evaluation.py" \
                --model "$MODEL" --method "$method" --seed "$seed" \
                --batch-size "$BATCH_SIZE" --lr "$LR" \
                --run-id "$PARENT_RUN_ID" --log-mlflow
        fi
    done

done

echo "All OOD evaluations complete!"
