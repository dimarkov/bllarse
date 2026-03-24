from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
OPTIMIZERS = ["adam", "adamw"]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_linear_probe_largebatch_adam_adamw",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    epochs=140,
    learning_rate=1.2e-3,
    dropout_rate=0.0,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
        weight_decay = 0.0 if optimizer == "adam" else 1e-2
        for train_batch_size in BATCH_SIZES:
            for seed in SEEDS:
                uid = (
                    "mnli_roberta_large_len512_"
                    f"{optimizer}_lr1.2e-3_bs{train_batch_size}_seed{seed}_ep140"
                )
                configs.append(
                    dict(
                        **BASE,
                        uid=uid,
                        optimizer=optimizer,
                        weight_decay=weight_decay,
                        train_batch_size=train_batch_size,
                        seed=seed,
                    )
                )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
