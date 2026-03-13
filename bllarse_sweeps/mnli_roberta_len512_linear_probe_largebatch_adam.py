from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [1024, 2048, 4096]
LEARNING_RATES = [3e-5, 1e-4]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_len512_linear_probe_largebatch_adam",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    weight_decay=0.0,
    epochs=40,
    dropout_rate=0.1,
    num_update_iters=0,
    seed=2022,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            uid = (
                "mnli_roberta_len512_"
                f"adam_lr{learning_rate:.0e}_bs{batch_size}_seed2022"
            )
            configs.append(
                dict(
                    **BASE,
                    uid=uid,
                    train_batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
