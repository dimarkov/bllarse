from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [256, 512, 1024, 2048]
LEARNING_RATES = [3e-4, 7e-4, 1e-3]
DROPOUT_RATES = [0.0, 0.1]

BASE: Dict[str, Any] = dict(
    stage="train_eval",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=True,
    group_id="mnli_roberta_large_len512_linear_probe_regime_search",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull",
    max_length=512,
    cache_dtype="float16",
    optimizer="adam",
    weight_decay=0.0,
    epochs=120,
    seed=2022,
    num_update_iters=0,
)


def create_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    for train_batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            for dropout_rate in DROPOUT_RATES:
                uid = (
                    "mnli_roberta_large_len512_"
                    f"adam_lr{learning_rate:.1e}_bs{train_batch_size}"
                    f"_drop{dropout_rate:.1f}_seed2022_ep120"
                )
                configs.append(
                    dict(
                        **BASE,
                        uid=uid,
                        train_batch_size=train_batch_size,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                    )
                )

    return configs


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
