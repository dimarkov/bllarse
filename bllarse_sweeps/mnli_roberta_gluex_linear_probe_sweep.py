from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BACKBONES = ["FacebookAI/roberta-base"]
OPTIMIZERS = ["adam", "adamw"]
LEARNING_RATES = [2e-5, 3e-5]
BATCH_SIZES = [4, 8]
ADAMW_WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_gluex_linear_probe",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    extract_batch_size=32,
    epochs=120,
    dropout_rate=0.1,
    seed=2022,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for backbone in BACKBONES:
        for learning_rate in LEARNING_RATES:
            for train_batch_size in BATCH_SIZES:
                configs.append(
                    dict(
                        **BASE,
                        backbone=backbone,
                        optimizer="adam",
                        learning_rate=learning_rate,
                        train_batch_size=train_batch_size,
                        weight_decay=0.0,
                    )
                )
                for weight_decay in ADAMW_WEIGHT_DECAYS:
                    configs.append(
                        dict(
                            **BASE,
                            backbone=backbone,
                            optimizer="adamw",
                            learning_rate=learning_rate,
                            train_batch_size=train_batch_size,
                            weight_decay=weight_decay,
                        )
                    )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
