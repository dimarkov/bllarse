from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [1024, 2048, 4096]
NUM_UPDATE_ITERS = [16, 32]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_ibprobit_initial",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="cavi",
    weight_decay=0.0,
    epochs=20,
    dropout_rate=0.0,
    seed=2022,
    learning_rate=1e-3,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for train_batch_size in BATCH_SIZES:
        for num_update_iters in NUM_UPDATE_ITERS:
            uid = (
                "mnli_roberta_large_len512_"
                f"cavi_bs{train_batch_size}_iters{num_update_iters}_seed2022_ep20"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    train_batch_size=train_batch_size,
                    num_update_iters=num_update_iters,
                )
            )
    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
