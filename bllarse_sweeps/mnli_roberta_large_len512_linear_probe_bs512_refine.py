from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

LEARNING_RATES = [8e-4, 1e-3, 1.2e-3, 1.5e-3]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_linear_probe_bs512_refine",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    weight_decay=0.0,
    epochs=140,
    train_batch_size=512,
    dropout_rate=0.0,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for learning_rate in LEARNING_RATES:
        for seed in SEEDS:
            uid = (
                "mnli_roberta_large_len512_"
                f"adam_lr{learning_rate:.1e}_bs512_drop0.0_seed{seed}_ep140"
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
