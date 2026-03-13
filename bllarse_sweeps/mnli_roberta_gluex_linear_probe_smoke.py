from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BASE: Dict[str, Any] = dict(
    stage="all",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_gluex_linear_probe_smoke",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="none",
    max_length=512,
    cache_dtype="float16",
    extract_batch_size=16,
    epochs=2,
    train_batch_size=8,
    dropout_rate=0.1,
    seed=2022,
    max_train_samples=128,
    max_val_samples=64,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    return [
        dict(**BASE, optimizer="adam", learning_rate=2e-5, weight_decay=0.0),
        dict(**BASE, optimizer="adamw", learning_rate=2e-5, weight_decay=1e-2),
    ]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
