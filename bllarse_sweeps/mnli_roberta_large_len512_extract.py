from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

BASE: Dict[str, Any] = dict(
    stage="extract",
    backbone="FacebookAI/roberta-large",
    reuse_cache=True,
    enable_mlflow=False,
    group_id="mnli_roberta_large_len512_extract",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull_push",
    max_length=512,
    cache_dtype="float16",
    extract_batch_size=64,
    seed=2022,
)


def create_configs() -> List[Dict[str, Any]]:
    return [dict(BASE)]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
