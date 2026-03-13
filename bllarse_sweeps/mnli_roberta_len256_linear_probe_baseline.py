from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_len256_linear_probe_baseline",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=256,
    cache_dtype="float16",
    epochs=5,
    dropout_rate=0.1,
    seed=2022,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    return [
        dict(
            **BASE,
            uid="mnli_roberta_len256_adam_lr2e5_bs8",
            optimizer="adam",
            train_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.0,
        ),
        dict(
            **BASE,
            uid="mnli_roberta_len256_adam_lr3e5_bs8",
            optimizer="adam",
            train_batch_size=8,
            learning_rate=3e-5,
            weight_decay=0.0,
        ),
        dict(
            **BASE,
            uid="mnli_roberta_len256_adamw_lr2e5_bs8_wd1e2",
            optimizer="adamw",
            train_batch_size=8,
            learning_rate=2e-5,
            weight_decay=1e-2,
        ),
        dict(
            **BASE,
            uid="mnli_roberta_len256_adamw_lr3e5_bs8_wd1e2",
            optimizer="adamw",
            train_batch_size=8,
            learning_rate=3e-5,
            weight_decay=1e-2,
        ),
    ]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
