import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bllarse import features


def test_cache_path_matches_the_precompute_layout():
    path = features.feature_cache_path(".cache/features", "dinov3_huge", "cifar10", 224, "test")
    assert Path(path).name == "dinov3_huge_cifar10_res224_test.npz"


def test_hf_path_matches_the_mirror_layout():
    assert features.feature_hf_path("dinov3_huge", "cifar10", 224, "test") == (
        "dinov3_huge/cifar10_res224_test.npz"
    )


def test_loads_from_the_local_cache(tmp_path):
    feats = np.arange(12, dtype=np.float32).reshape(4, 3)
    labels = np.array([0, 1, 0, 1])
    np.savez(tmp_path / "dinov3_small_fake_res224_train.npz", features=feats, labels=labels)

    got_f, got_l = features.load_cached_features(tmp_path, "dinov3_small", "fake", 224, "train")
    np.testing.assert_array_equal(got_f, feats)
    np.testing.assert_array_equal(got_l, labels)


def test_missing_features_raise_a_message_naming_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(tmp_path, "dinov3_small", "absent", 224, "train")


def test_falls_back_to_hf_when_the_local_cache_misses(tmp_path, monkeypatch):
    remote = tmp_path / "remote.npz"
    feats = np.ones((2, 3), dtype=np.float32)
    np.savez(remote, features=feats, labels=np.zeros(2, dtype=int))

    calls = {}

    def fake_download(repo_id, filename, repo_type):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["repo_type"] = repo_type
        return str(remote)

    monkeypatch.setattr(features, "_hf_hub_download", fake_download)

    cache = tmp_path / "cache"
    got_f, _ = features.load_cached_features(
        cache, "dinov3_small", "fake", 224, "train", hf_repo="dimarkov/bllarse-features"
    )
    np.testing.assert_array_equal(got_f, feats)
    assert calls["filename"] == "dinov3_small/fake_res224_train.npz"
    assert calls["repo_type"] == "dataset"
    # The download must be cached locally so the next run does not re-fetch.
    assert (cache / "dinov3_small_fake_res224_train.npz").exists()


def test_absent_file_on_the_hub_raises_file_not_found(tmp_path, monkeypatch):
    class EntryNotFoundError(Exception):
        pass

    def fake_download(repo_id, filename, repo_type):
        raise EntryNotFoundError("404")

    monkeypatch.setattr(features, "_hf_hub_download", fake_download)
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )


def test_network_failure_is_not_reported_as_missing_data(tmp_path, monkeypatch):
    """An auth or connectivity failure must not tell the user to spend GPU hours
    regenerating features that already exist."""

    def fake_download(repo_id, filename, repo_type):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(features, "_hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="network or authentication") as excinfo:
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )
    assert "precompute_features" not in str(excinfo.value)


def test_module_imports_without_jax_or_torch():
    """The torch container must import this without the JAX stack, and vice
    versa. A heavy import chain here would defeat the point of the module."""
    code = (
        "import sys, bllarse.features; "
        "heavy = [m for m in ('jax', 'torch', 'equimo', 'mlpox', 'datasets') "
        "if m in sys.modules]; "
        "assert not heavy, heavy; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
