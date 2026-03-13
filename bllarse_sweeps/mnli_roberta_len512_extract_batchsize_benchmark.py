from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BATCH_SIZES = [32, 64, 128]

BASE: Dict[str, Any] = dict(
    stage="extract",
    backbone="FacebookAI/roberta-base",
    reuse_cache=False,
    enable_mlflow=False,
    group_id="mnli_roberta_len512_extract_batchsize_benchmark",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="none",
    max_length=512,
    cache_dtype="float16",
    max_train_samples=32768,
    max_val_samples=2048,
    seed=2022,
)


def create_configs() -> List[Dict[str, Any]]:
    return [
        dict(
            **BASE,
            uid=f"mnli_roberta_len512_extract_bs{batch_size}",
            extract_batch_size=batch_size,
        )
        for batch_size in BATCH_SIZES
    ]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
