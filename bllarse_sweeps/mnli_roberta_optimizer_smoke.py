from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BASE: Dict[str, Any] = dict(
    stage="all",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_optimizer_smoke",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull_push",
    max_length=256,
    cache_dtype="float16",
    extract_batch_size=16,
    epochs=1,
    train_batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-2,
    seed=0,
    max_train_samples=1024,
    max_val_samples=256,
)


def create_configs() -> List[Dict[str, Any]]:
    return [
        dict(**BASE, optimizer="adamw", num_update_iters=0),
        dict(**BASE, optimizer="cavi", num_update_iters=2),
    ]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
