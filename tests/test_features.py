import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


class _HubHTTPError(Exception):
    """Stands in for huggingface_hub's HfHubHTTPError, which carries .response."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def _raising(exc):
    def fake_download(repo_id, filename, repo_type):
        raise exc

    return fake_download


def test_404_on_the_hub_means_the_features_were_never_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_hf_hub_download", _raising(_HubHTTPError(404)))
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_does_not_tell_the_user_to_regenerate(tmp_path, monkeypatch, status):
    """A bad token against a private repo must not send someone to burn GPU
    hours regenerating features that are sitting on the Hub."""
    monkeypatch.setattr(features, "_hf_hub_download", _raising(_HubHTTPError(status)))
    with pytest.raises(RuntimeError, match="HF_TOKEN") as excinfo:
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )
    assert "precompute_features" not in str(excinfo.value)


def test_connection_failure_does_not_tell_the_user_to_regenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_hf_hub_download", _raising(ConnectionError("reset")))
    with pytest.raises(RuntimeError, match="network or offline") as excinfo:
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )
    assert "precompute_features" not in str(excinfo.value)


def test_classification_matches_the_real_huggingface_hub_hierarchy(tmp_path, monkeypatch):
    """Guard against the library's shape changing under us.

    The previous version of this module matched exception CLASS NAMES and got it
    wrong in both directions: the real 404 is RemoteEntryNotFoundError (plain
    EntryNotFoundError is a bare base class), and GatedRepoError -- an auth
    failure -- subclasses RepositoryNotFoundError. This test constructs the
    genuine article rather than a stand-in.
    """
    errors = pytest.importorskip("huggingface_hub.errors")

    class _Req:
        method, url = "GET", "https://huggingface.co/x"

    def _resp(status):
        return SimpleNamespace(status_code=status, headers={}, text="", request=_Req())

    real_404 = errors.RemoteEntryNotFoundError("nope", response=_resp(404))
    monkeypatch.setattr(features, "_hf_hub_download", _raising(real_404))
    with pytest.raises(FileNotFoundError, match="precompute_features"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )

    real_gated = errors.GatedRepoError("gated", response=_resp(403))
    monkeypatch.setattr(features, "_hf_hub_download", _raising(real_gated))
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        features.load_cached_features(
            tmp_path, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
        )


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
