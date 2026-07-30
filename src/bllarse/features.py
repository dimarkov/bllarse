"""Loading cached backbone features.

Both probing arms read the same .npz files that precompute_features.py writes,
so the backbone is byte-identical across frameworks and Figs 1-4 carry no
cross-framework parity risk.

Deliberately free of JAX and PyTorch. The PyTorch container must import this
without the JAX stack, and the JAX arm must not pull equimo/mlpox/datasets in
just to call np.load. `huggingface_hub` is imported lazily for the same reason.
"""

import os
import shutil
from pathlib import Path

import numpy as np


def feature_cache_path(cache_dir, model: str, dataset: str, img_size: int, split: str) -> Path:
    """Local cache path, matching precompute_features.py's layout exactly."""
    return Path(cache_dir) / f"{model}_{dataset}_res{img_size}_{split}.npz"


def feature_hf_path(model: str, dataset: str, img_size: int, split: str) -> str:
    """Path within the HF dataset repo. Note the model is a directory there,
    where it is a filename prefix locally."""
    return f"{model}/{dataset}_res{img_size}_{split}.npz"


def _hf_hub_download(repo_id: str, filename: str, repo_type: str) -> str:
    """Indirection so tests can substitute the download without network access."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def load_cached_features(
    cache_dir, model: str, dataset: str, img_size: int, split: str, hf_repo: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) from the local cache, falling back to HF Hub.

    A successful HF fetch is copied into the local cache, so a sweep pays the
    download once per node rather than once per task.
    """
    local = feature_cache_path(cache_dir, model, dataset, img_size, split)
    if local.exists():
        data = np.load(local)
        return data["features"], data["labels"]

    if hf_repo:
        remote = feature_hf_path(model, dataset, img_size, split)
        try:
            downloaded = _hf_hub_download(repo_id=hf_repo, filename=remote, repo_type="dataset")
        except Exception as exc:  # network, auth, or a genuinely absent file
            raise FileNotFoundError(
                f"Features not in the local cache ({local}) and the HF fetch of "
                f"{hf_repo}/{remote} failed: {exc}. Run precompute_features.py "
                f"for {model} x {dataset}."
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local)
        data = np.load(local)
        return data["features"], data["labels"]

    raise FileNotFoundError(
        f"Features not found at {local} and no hf_repo was given. Run "
        f"precompute_features.py --models {model} --datasets {dataset} first."
    )
