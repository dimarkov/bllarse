"""Pinned, seed-independent val/test splits.

Temperature scaling is fit on a held-out split (results-plan.md 0.2) and, under
the per-dump-point refit schedule, the same split is reused ~40 times per run.
If the split moved with the run seed, every curve would carry 40 differently
conditioned fits and no two seeds would be comparable. So the partition is
derived from the dataset name alone.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

# Fixed project-wide constant. Changing it invalidates every existing dump.
SPLIT_SEED = 20260730


def _dataset_seed(dataset: str) -> int:
    """Stable per-dataset seed. Python's hash() is salted per process, so it
    cannot be used here -- the value must be identical across machines and runs."""
    digest = hashlib.sha256(f"{SPLIT_SEED}:{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_pinned_split(
    n: int, val_fraction: float, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """Partition range(n) into (val_idx, test_idx). No run seed, by design."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = int(round(n * val_fraction))
    if n_val == 0 or n_val == n:
        raise ValueError(
            f"val_fraction={val_fraction} yields a degenerate split for n={n}"
        )

    rng = np.random.default_rng(_dataset_seed(dataset))
    perm = rng.permutation(n)
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])


def pinned_split_path(cache_dir, dataset: str, split: str, val_fraction: float) -> Path:
    return Path(cache_dir) / f"{dataset}_{split}_val{val_fraction}.npz"


def load_or_make_pinned_split(
    cache_dir, dataset: str, split: str, n: int, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached split, or create and cache it. Fails loudly on size drift."""
    path = pinned_split_path(cache_dir, dataset, split, val_fraction)
    if path.exists():
        data = np.load(path)
        val_idx, test_idx = data["val_idx"], data["test_idx"]
        cached_n = val_idx.size + test_idx.size
        if cached_n != n:
            raise ValueError(
                f"cached split at {path} has size {cached_n} but {n} were "
                f"requested -- the underlying dataset changed; delete the cache "
                f"deliberately if that is intended"
            )
        return val_idx, test_idx

    val_idx, test_idx = make_pinned_split(n, val_fraction, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The sweep runs several tasks concurrently against the same cold cache,
    # so a plain np.savez(path, ...) followed by another task's np.load(path)
    # risks the reader seeing a half-written file (BadZipFile). Write to a
    # temp file in the same directory, then os.replace -- atomic on POSIX --
    # so a concurrent reader only ever sees the old (absent) or new (complete)
    # file, never a partial one. np.savez is given an open file object rather
    # than the temp path string so it does not append its own ".npz" suffix.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            np.savez(tmp_file, val_idx=val_idx, test_idx=test_idx)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return val_idx, test_idx
