from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

SEEDS = [2022, 2023, 2024, 2025, 2026]
TOTAL_TRAIN_SAMPLES = 392702
SUBSET_SIZES = [16384 * k for k in range(1, 24)] + [TOTAL_TRAIN_SAMPLES]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_linear_probe_bs512_data_efficiency_retuned",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    weight_decay=0.0,
    learning_rate=2.8e-3,
    epochs=140,
    train_batch_size=512,
    dropout_rate=0.0,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for max_train_samples in SUBSET_SIZES:
        for seed in SEEDS:
            uid = (
                "mnli_roberta_large_len512_"
                f"adam_bs512_lr2.8e-3_subset{max_train_samples}_seed{seed}_ep140"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    max_train_samples=max_train_samples,
                    train_subset_seed=seed,
                    seed=seed,
                )
            )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
