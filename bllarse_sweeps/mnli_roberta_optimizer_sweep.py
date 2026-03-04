from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

OPTIMIZERS = ["adamw", "lion", "cavi"]
BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
CAVI_ITERS = [1, 2, 4, 8, 16, 32]
BACKBONES = ["FacebookAI/roberta-base", "FacebookAI/roberta-large"]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_optimizer_sweep",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=256,
    cache_dtype="float16",
    extract_batch_size=32,
    epochs=1,
    seed=0,
    learning_rate=1e-3,
    weight_decay=1e-2,
)


def _mk_cfg(
    *,
    backbone: str,
    optimizer: str,
    train_batch_size: int,
    num_update_iters: int,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(
        **BASE,
        backbone=backbone,
        optimizer=optimizer,
        train_batch_size=train_batch_size,
        num_update_iters=num_update_iters,
    )
    return cfg


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for backbone in BACKBONES:
        for train_batch_size in BATCH_SIZES:
            for optimizer in OPTIMIZERS:
                if optimizer == "cavi":
                    for num_update_iters in CAVI_ITERS:
                        configs.append(
                            _mk_cfg(
                                backbone=backbone,
                                optimizer=optimizer,
                                train_batch_size=train_batch_size,
                                num_update_iters=num_update_iters,
                            )
                        )
                else:
                    configs.append(
                        _mk_cfg(
                            backbone=backbone,
                            optimizer=optimizer,
                            train_batch_size=train_batch_size,
                            num_update_iters=0,
                        )
                    )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
