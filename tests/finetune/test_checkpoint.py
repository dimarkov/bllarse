"""Checkpoint slots between a training run and the OOD worker.

Two processes share a directory that one is writing into while the other polls
it, so the properties that matter are all about partial states: a reader must
never see a half-written slot, and a crash must never leave one that looks
ready but is not.
"""

import json

import pytest

from bllarse.finetune import checkpoint as ck


def _write(root, step, *, uid="u", group="g", body=b"weights", **kw):
    return ck.write(root, group_id=group, uid=uid, step=step, blob_suffix=".bin",
                    write_blob=lambda p: p.write_bytes(body),
                    meta={"arm": "ce", "samples_seen": step * 512}, **kw)


def test_a_slot_round_trips(tmp_path):
    stem = _write(tmp_path, 128)
    assert stem is not None
    meta = ck.read_meta(stem)
    assert meta["step"] == 128 and meta["arm"] == "ce"
    assert ck.blob_path(stem).read_bytes() == b"weights"


def test_done_is_written_last_so_a_partial_slot_is_invisible(tmp_path):
    """The reader keys on `.done` alone.

    If the marker existed before the blob, the worker could open a truncated
    checkpoint and evaluate garbage -- which would produce plausible OOD
    numbers rather than an error.
    """
    seen = {}

    def slow_blob(path):
        # Mid-write: nothing may be visible to `pending` yet.
        seen["pending_during_write"] = ck.pending(tmp_path)
        path.write_bytes(b"x" * 32)

    ck.write(tmp_path, group_id="g", uid="u", step=1, blob_suffix=".bin",
             write_blob=slow_blob, meta={})
    assert seen["pending_during_write"] == []
    assert len(ck.pending(tmp_path)) == 1


def test_pending_is_oldest_step_first(tmp_path):
    for step in (256, 16, 4096, 64):
        _write(tmp_path, step)
    steps = [ck.read_meta(s)["step"] for s in ck.pending(tmp_path)]
    assert steps == sorted(steps) == [16, 64, 256, 4096]


def test_zero_padding_is_what_makes_lexical_order_step_order(tmp_path):
    # Without padding, "step1000" sorts before "step64" and the worker consumes
    # newest-first at exactly the point it falls behind.
    _write(tmp_path, 64)
    _write(tmp_path, 1000)
    names = [s.name for s in ck.pending(tmp_path)]
    assert names == sorted(names)
    assert names[0].endswith("00000064")


def test_runs_do_not_collide(tmp_path):
    _write(tmp_path, 1, uid="a")
    _write(tmp_path, 1, uid="b")
    assert len(ck.pending(tmp_path)) == 2
    assert len(ck.pending(tmp_path, "g", "a")) == 1


def test_backpressure_skips_rather_than_blocking_or_evicting(tmp_path):
    """A lagging worker must not fill /ptmp, and must not lose a step it may
    already be reading.

    A dinov3_huge checkpoint is ~3.4 GB and 15 runs write concurrently.
    Skipping costs one point on an OOD curve; blocking stalls a gpu worth far
    more, and evicting would delete a slot mid-read.
    """
    assert _write(tmp_path, 1, max_pending=2) is not None
    assert _write(tmp_path, 2, max_pending=2) is not None
    assert _write(tmp_path, 3, max_pending=2) is None      # skipped
    assert len(ck.pending(tmp_path)) == 2
    ck.consume(ck.pending(tmp_path)[0])
    assert _write(tmp_path, 3, max_pending=2) is not None  # room again


def test_consume_removes_every_file(tmp_path):
    stem = _write(tmp_path, 7)
    ck.consume(stem)
    assert ck.pending(tmp_path) == []
    assert list(stem.parent.glob(stem.name + ".*")) == []


def test_consume_drops_the_marker_first(tmp_path, monkeypatch):
    """A crash mid-delete must not leave a slot that looks ready with no blob.

    The worker would fail on it, re-list it, and fail again forever.
    """
    stem = _write(tmp_path, 9)
    real_unlink = ck.Path.unlink
    calls = []

    def spy(self, *a, **kw):
        calls.append(self.suffix)
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(ck.Path, "unlink", spy)
    ck.consume(stem)
    assert calls[0] == ck.DONE_SUFFIX


def test_rewriting_an_existing_step_is_a_no_op(tmp_path):
    # A resumed trainer must not clobber a slot the worker may be reading.
    stem = _write(tmp_path, 5, body=b"first")
    again = _write(tmp_path, 5, body=b"second")
    assert again == stem
    assert ck.blob_path(stem).read_bytes() == b"first"


def test_metadata_is_json_and_carries_the_run_identity(tmp_path):
    stem = ck.write(tmp_path, group_id="ftfinal", uid="dinov3_huge_x_seed137",
                    step=32, blob_suffix=".eqx", write_blob=lambda p: p.write_bytes(b""),
                    meta={"mlflow_run_id": "abc123", "arm": "ibprobit"})
    meta = json.loads((stem.with_suffix(".json")).read_text())
    assert meta["mlflow_run_id"] == "abc123"
    assert meta["group_id"] == "ftfinal"
    assert meta["uid"] == "dinov3_huge_x_seed137"
    assert meta["blob"].endswith(".eqx")
