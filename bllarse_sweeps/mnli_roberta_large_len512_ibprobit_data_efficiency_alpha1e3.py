from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_ibprobit_data_efficiency_alpha1e3",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="cavi",
    epochs=1,
    train_batch_size=16384,
    num_update_iters=16,
    ibprobit_alpha=1e-3,
    eval_every_batches=1,
    reset_loss_per_epoch=False,
    dropout_rate=0.0,
    learning_rate=1e-3,
    weight_decay=0.0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        uid = (
            "mnli_roberta_large_len512_"
            f"ibprobit_data_efficiency_bs16384_iters16_alpha1e3_seed{seed}"
        )
        configs.append(
            dict(
                **BASE,
                uid=uid,
                seed=seed,
            )
        )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
