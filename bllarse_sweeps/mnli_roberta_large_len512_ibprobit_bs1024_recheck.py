from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

NUM_UPDATE_ITERS = [1, 4, 8, 16, 32, 64, 128]
SEEDS = [2022, 2023, 2024, 2025, 2026]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_ibprobit_bs1024_recheck",
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
    train_batch_size=1024,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for num_update_iters in NUM_UPDATE_ITERS:
        for seed in SEEDS:
            uid = (
                "mnli_roberta_large_len512_"
                f"cavi_bs1024_recheck_iters{num_update_iters}_seed{seed}_ep1"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    num_update_iters=num_update_iters,
                    seed=seed,
                )
            )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
