from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_ibprobit_singlepass_multiseed_expanded_alpha1e2_iters6_only",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="cavi",
    epochs=1,
    learning_rate=1e-3,
    weight_decay=0.0,
    dropout_rate=0.0,
    reset_loss_per_epoch=False,
    ibprobit_alpha=1e-2,
    num_update_iters=6,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for train_batch_size in BATCH_SIZES:
        for seed in SEEDS:
            uid = (
                "mnli_roberta_large_len512_"
                f"cavi_alpha1e2_bs{train_batch_size}_iters6_seed{seed}_ep1"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    train_batch_size=train_batch_size,
                    seed=seed,
                )
            )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
