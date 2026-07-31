import os
from pathlib import Path

import numpy as np
import pytest

from bllarse import splits


def test_split_is_a_partition():
    val_idx, test_idx = splits.make_pinned_split(100, 0.1, "cifar10")
    assert len(val_idx) == 10
    assert len(test_idx) == 90
    assert set(val_idx).isdisjoint(test_idx)
    assert sorted(np.concatenate([val_idx, test_idx])) == list(range(100))


def test_split_is_independent_of_call_order_and_stable():
    a = splits.make_pinned_split(1000, 0.1, "cifar10")
    b = splits.make_pinned_split(1000, 0.1, "cifar10")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_datasets_get_different_splits():
    a, _ = splits.make_pinned_split(1000, 0.1, "cifar10")
    b, _ = splits.make_pinned_split(1000, 0.1, "cifar100")
    assert not np.array_equal(a, b)


def test_split_does_not_depend_on_any_run_seed():
    # There is no seed parameter to pass; the signature itself enforces this.
    import inspect

    params = inspect.signature(splits.make_pinned_split).parameters
    assert "seed" not in params, "run seed must not influence the pinned split"


def test_rejects_degenerate_fractions():
    with pytest.raises(ValueError):
        splits.make_pinned_split(100, 0.0, "cifar10")
    with pytest.raises(ValueError):
        splits.make_pinned_split(100, 1.0, "cifar10")


def test_load_or_make_round_trips_through_disk(tmp_path):
    first = splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    path = splits.pinned_split_path(tmp_path, "cifar10", "test", 0.1)
    assert path.exists()
    second = splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_load_or_make_rejects_a_cached_split_of_the_wrong_size(tmp_path):
    splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    with pytest.raises(ValueError, match="size"):
        splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 600, 0.1)


def test_writes_via_temp_file_then_atomic_replace(tmp_path, monkeypatch):
    """The sweep runs several tasks concurrently against a cold cache, so a
    reader's np.load(path) must never observe a half-written file. Spy on
    os.replace to confirm the implementation writes a complete temp file in
    the SAME directory as the target (required for atomicity on POSIX) and
    only makes it visible under the final name via a single replace call --
    never via np.savez(path, ...) straight onto the target."""
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        assert Path(src).parent == Path(dst).parent, "replace must stay on one filesystem"
        assert Path(src) != Path(dst), "must write to a distinct temp file, not the final path"
        with np.load(src) as data:
            assert "val_idx" in data.files and "test_idx" in data.files
        return real_replace(src, dst)

    monkeypatch.setattr(splits.os, "replace", spy_replace)

    splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)
    assert len(calls) == 1, "the split write must go through exactly one atomic replace"
    path = splits.pinned_split_path(tmp_path, "cifar10", "test", 0.1)
    assert path.exists()


def test_write_failure_leaves_no_partial_file_at_the_final_path(tmp_path, monkeypatch):
    def crashing_replace(src, dst):
        raise RuntimeError("simulated crash before the swap")

    monkeypatch.setattr(splits.os, "replace", crashing_replace)

    with pytest.raises(RuntimeError, match="simulated crash"):
        splits.load_or_make_pinned_split(tmp_path, "cifar10", "test", 500, 0.1)

    path = splits.pinned_split_path(tmp_path, "cifar10", "test", 0.1)
    assert not path.exists(), "a crash before the atomic replace must not leave a partial file"
