from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [4096, 8192, 16384]
LEARNING_RATES = [8e-4, 1e-3, 1.2e-3, 1.5e-3]
DROPOUT_RATES = [0.0, 0.02, 0.05]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_linear_probe_largebatch_adam_lr_dropout_refine",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    epochs=140,
    num_update_iters=0,
    weight_decay=0.0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for dropout_rate in DROPOUT_RATES:
        for learning_rate in LEARNING_RATES:
            for train_batch_size in BATCH_SIZES:
                for seed in SEEDS:
                    uid = (
                        "mnli_roberta_large_len512_"
                        f"adam_lr{learning_rate:.1e}_drop{dropout_rate:.2f}"
                        f"_bs{train_batch_size}_seed{seed}_ep140"
                    )
                    configs.append(
                        dict(
                            **BASE,
                            uid=uid,
                            learning_rate=learning_rate,
                            dropout_rate=dropout_rate,
                            train_batch_size=train_batch_size,
                            seed=seed,
                        )
                    )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
