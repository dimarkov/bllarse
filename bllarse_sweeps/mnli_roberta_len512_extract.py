from typing import Any, Dict, List

from bllarse.tools.adapters import run_script_from_config

# Benchmarked on 1x A100 40GB with 8 CPUs per task. For this max_length=512
# extraction path, batch size 256 gave the best clean throughput in our
# allocator-matched probes while keeping plenty of memory headroom.
BASE: Dict[str, Any] = dict(
    stage="extract",
    backbone="FacebookAI/roberta-base",
    reuse_cache=True,
    enable_mlflow=False,
    group_id="mnli_roberta_len512_extract",
    cache_dir="data/feature_cache",
    hf_repo_id="dimarkov/bllarse-features",
    hf_subdir_prefix="roberta_activations/mnli_roberta_cls",
    hf_sync="pull_push",
    max_length=512,
    cache_dtype="float16",
    extract_batch_size=256,
    seed=2022,
)


def create_configs() -> List[Dict[str, Any]]:
    return [dict(**BASE, uid="mnli_roberta_len512_extract_full")]


def run(config: Dict[str, Any]) -> None:
    run_script_from_config("roberta_mnli.py", config)
