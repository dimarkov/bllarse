"""Loading cached backbone features.

Both probing arms read the same .npz files that precompute_features.py writes,
so the backbone is byte-identical across frameworks and Figs 1-4 carry no
cross-framework parity risk.

Deliberately free of JAX and PyTorch. The PyTorch container must import this
without the JAX stack, and the JAX arm must not pull equimo/mlpox/datasets in
just to call np.load. `huggingface_hub` is imported lazily for the same reason.
"""

import shutil
from pathlib import Path

import numpy as np

# Hugging Face's "this does not exist" errors, matched by class name so this
# module stays importable without huggingface_hub installed. Anything else --
# timeouts, 401s, proxy errors -- is a transient/config problem, and telling the
# user to regenerate their features would send them down the wrong path.
_HF_ABSENT_ERRORS = frozenset(
    {"EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError"}
)


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
        except Exception as exc:
            if type(exc).__name__ in _HF_ABSENT_ERRORS:
                raise FileNotFoundError(
                    f"Features not in the local cache ({local}) and "
                    f"{hf_repo}/{remote} does not exist on the Hub "
                    f"({type(exc).__name__}). Run precompute_features.py "
                    f"--models {model} --datasets {dataset} to create them."
                ) from exc
            raise RuntimeError(
                f"Features not in the local cache ({local}) and fetching "
                f"{hf_repo}/{remote} failed with {type(exc).__name__}: {exc}. "
                f"That is a network or authentication problem, not missing data "
                f"-- check connectivity and HF_TOKEN before regenerating anything."
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local)
        data = np.load(local)
        return data["features"], data["labels"]

    raise FileNotFoundError(
        f"Features not found at {local} and no hf_repo was given. Run "
        f"precompute_features.py --models {model} --datasets {dataset} first."
    )
