import os
import re
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

    Matching exception CLASS NAMES gets this wrong in both directions: the real
    404 is RemoteEntryNotFoundError (plain EntryNotFoundError is a bare base
    class), and GatedRepoError -- an auth failure -- subclasses
    RepositoryNotFoundError. This test constructs the
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


def test_download_writes_via_temp_file_then_atomic_replace(tmp_path, monkeypatch):
    """The sweep runs several tasks concurrently against a cold cache, so a
    reader's np.load(local) must never observe a half-written file. Spy on
    os.replace to confirm the implementation writes a complete temp file in
    the SAME directory as the target (required for atomicity on POSIX) and
    only makes it visible under the final name via a single replace call --
    never via shutil.copy2 straight onto `local`."""
    remote = tmp_path / "remote.npz"
    feats = np.ones((2, 3), dtype=np.float32)
    np.savez(remote, features=feats, labels=np.zeros(2, dtype=int))

    monkeypatch.setattr(features, "_hf_hub_download", lambda repo_id, filename, repo_type: str(remote))

    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        assert Path(src).parent == Path(dst).parent, "replace must stay on one filesystem"
        assert Path(src) != Path(dst), "must write to a distinct temp file, not the final path"
        with np.load(src) as data:
            np.testing.assert_array_equal(data["features"], feats)
        return real_replace(src, dst)

    monkeypatch.setattr(features.os, "replace", spy_replace)

    cache = tmp_path / "cache"
    got_f, _ = features.load_cached_features(
        cache, "dinov3_small", "fake", 224, "train", hf_repo="r/f"
    )
    np.testing.assert_array_equal(got_f, feats)
    assert len(calls) == 1, "the download path must go through exactly one atomic replace"


def test_download_failure_leaves_no_partial_file_at_the_final_path(tmp_path, monkeypatch):
    remote = tmp_path / "remote.npz"
    np.savez(remote, features=np.ones((2, 3), dtype=np.float32), labels=np.zeros(2, dtype=int))
    monkeypatch.setattr(features, "_hf_hub_download", lambda repo_id, filename, repo_type: str(remote))

    def crashing_replace(src, dst):
        raise RuntimeError("simulated crash before the swap")

    monkeypatch.setattr(features.os, "replace", crashing_replace)

    cache = tmp_path / "cache"
    local = features.feature_cache_path(cache, "dinov3_small", "fake", 224, "train")
    with pytest.raises(RuntimeError, match="simulated crash"):
        features.load_cached_features(cache, "dinov3_small", "fake", 224, "train", hf_repo="r/f")

    assert not local.exists(), "a crash before the atomic replace must not leave a partial file"


def test_module_imports_without_jax_or_torch():
    """The torch container must import this without the JAX stack, and vice
    versa. A heavy import chain here would defeat the point of the module."""
    code = (
        "import sys, bllarse.features; "
        "heavy = [m for m in ('jax', 'torch', 'equimo', 'datasets') "
        "if m in sys.modules]; "
        "assert not heavy, heavy; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# flowers102's two pools are deliberately swapped relative to what the upstream
# repo calls them -- see features._TRAIN_SPLIT. Everything else must track
# DATASET_CONFIGS exactly, so the exception is listed here rather than being
# silently tolerated by a weakened assertion.
POOL_SWAPPED = {"flowers102"}


def test_eval_split_name_tracks_the_upstream_split_names():
    """precompute_features.py writes the held-out file under the dataset's own
    split name, so the two datasets whose upstream split is 'validation' must
    not be looked up as 'test'. Guarded against silent drift from
    vit_classification's DATASET_CONFIGS."""
    assert features.eval_split_name("food101") == "validation"
    assert features.eval_split_name("imagenet1k") == "validation"
    for dataset in ("cifar10", "cifar100", "svhn", "dtd", "oxford_pets",
                    "stanford_cars"):
        assert features.eval_split_name(dataset) == "test"


def test_flowers102_pools_are_swapped():
    """nelorth/oxford-flowers packs Flowers-102 as train=7,169 / test=1,020,
    which is the official protocol inverted: the official training split is the
    1,020 (10 per class) and the 7,169 is val+test. Reading the names literally
    trains on seven times the protocol's data and leaves 1,020 images to
    evaluate on, of which a 10% val fraction is 102 -- one per class to fit a
    temperature. So the two are swapped, and the swap is pinned here because
    getting it backwards is silent: both files load, both train, and only the
    numbers are wrong."""
    assert features.train_split_name("flowers102") == "test"
    assert features.eval_split_name("flowers102") == "train"


def test_train_split_name_defaults_to_train():
    for dataset in ("cifar10", "cifar100", "svhn", "dtd", "oxford_pets",
                    "stanford_cars", "food101", "imagenet1k"):
        assert features.train_split_name(dataset) == "train"


def test_eval_split_name_matches_dataset_configs():
    """The mapping above is a copy of vit_classification's DATASET_CONFIGS.
    Parse that file's test_split entries and assert the copy is still faithful,
    without importing it (it pulls in equimo/jax)."""
    src = Path(__file__).resolve().parents[1] / "scripts/dinov3/vit_classification.py"
    text = src.read_text()
    body = text[text.index("DATASET_CONFIGS"):]
    entries = re.findall(r'^\s{4}"(\w+)":\s*\{', body, flags=re.M)
    splits_found = re.findall(r'"test_split":\s*"(\w+)"', body)
    assert len(entries) == len(splits_found), (entries, splits_found)
    for dataset, split in zip(entries, splits_found):
        if dataset in POOL_SWAPPED:
            # Still a drift guard, just against the swap rather than the
            # identity: eval must read the file upstream calls "train", and
            # train must read the file upstream nominates as its test split.
            assert features.eval_split_name(dataset) == "train", dataset
            assert features.train_split_name(dataset) == split, dataset
            continue
        assert features.eval_split_name(dataset) == split, dataset
        assert features.train_split_name(dataset) == "train", dataset
