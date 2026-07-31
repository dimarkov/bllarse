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
import tempfile
from pathlib import Path

import numpy as np


def _hub_status(exc: BaseException) -> int | None:
    """HTTP status behind a Hub failure, or None if it was not an HTTP error.

    Read by duck-typing on `.response.status_code` rather than by catching
    huggingface_hub's exception classes. Two reasons: importing them would break
    the lazy-import contract this module exists to satisfy, and the class names
    do not mean what they look like -- the 404 for an absent file is
    RemoteEntryNotFoundError (EntryNotFoundError is a bare base class), and
    GatedRepoError subclasses RepositoryNotFoundError, so name-matching
    'not found' silently swallows auth failures. The status code does not lie.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


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
            status = _hub_status(exc)
            if status == 404:
                raise FileNotFoundError(
                    f"Features not in the local cache ({local}) and "
                    f"{hf_repo}/{remote} does not exist on the Hub. Run "
                    f"precompute_features.py --models {model} --datasets {dataset} "
                    f"to create them."
                ) from exc
            if status in (401, 403):
                raise RuntimeError(
                    f"Features not in the local cache ({local}) and the Hub "
                    f"refused access to {hf_repo}/{remote} (HTTP {status}). Check "
                    f"HF_TOKEN and that you have access to the repo. Do NOT "
                    f"regenerate -- the data may well be there."
                ) from exc
            raise RuntimeError(
                f"Features not in the local cache ({local}) and fetching "
                f"{hf_repo}/{remote} failed with {type(exc).__name__}: {exc}. "
                f"That is a network or offline problem, not missing data -- do "
                f"NOT regenerate before checking connectivity."
            ) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        # The sweep runs several tasks concurrently against the same cold
        # cache, so a copy straight to `local` risks another task's np.load
        # observing a half-written file (BadZipFile). Write to a temp file in
        # the same directory, then os.replace -- atomic on POSIX -- so a
        # concurrent reader only ever sees the old (absent) or new (complete)
        # file, never a partial one.
        fd, tmp_name = tempfile.mkstemp(dir=local.parent, prefix=f".{local.name}.", suffix=".tmp")
        os.close(fd)
        try:
            shutil.copy2(downloaded, tmp_name)
            os.replace(tmp_name, local)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        data = np.load(local)
        return data["features"], data["labels"]

    raise FileNotFoundError(
        f"Features not found at {local} and no hf_repo was given. Run "
        f"precompute_features.py --models {model} --datasets {dataset} first."
    )
