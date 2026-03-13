from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

SEEDS = [2022, 2023, 2024]
LEARNING_RATES = [2e-5, 3e-5]
OPTIMIZERS = ["adam", "adamw"]
BATCH_SIZES = [8, 4]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_len256_linear_probe_multiseed",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=256,
    cache_dtype="float16",
    epochs=120,
    dropout_rate=0.1,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for optimizer in OPTIMIZERS:
        for learning_rate in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                for seed in SEEDS:
                    weight_decay = 0.0 if optimizer == "adam" else 1e-2
                    uid = (
                        "mnli_roberta_len256_"
                        f"{optimizer}_lr{learning_rate:.0e}_bs{batch_size}_seed{seed}"
                    )
                    configs.append(
                        dict(
                            **BASE,
                            uid=uid,
                            optimizer=optimizer,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            train_batch_size=batch_size,
                            seed=seed,
                        )
                    )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
