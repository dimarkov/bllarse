from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

LEARNING_RATES = [7e-4, 1e-3, 1.3e-3]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_len512_linear_probe_bs2048_refine",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    weight_decay=0.0,
    epochs=100,
    train_batch_size=2048,
    dropout_rate=0.1,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for learning_rate in LEARNING_RATES:
        for seed in SEEDS:
            uid = (
                "mnli_roberta_len512_"
                f"adam_lr{learning_rate:.1e}_bs2048_seed{seed}_ep100"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    learning_rate=learning_rate,
                    seed=seed,
                )
            )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
