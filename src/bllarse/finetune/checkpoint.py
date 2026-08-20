"""Checkpoint slots handed from a training run to the OOD evaluation worker.

The reported campaign evaluates four OOD sets (98,389 images) on a stride of
the dump grid. Doing that inline costs the TRAINING gpu roughly an hour per
run and pushes a dinov3_huge 4-epoch job past its 24 h wall. Doing it in a
separate process on a separate gpu costs the trainer only the checkpoint write.

WHY SLOTS AND NOT A ROLLING `latest`. The obvious design -- trainer overwrites
one file, worker grabs whatever is there -- does not work here, and the reason
is downstream rather than aesthetic: `analysis.curves._id_ood_pair` looks up the
test dump and the OOD dump AT THE SAME STEP and raises when they differ. An
opportunistic worker would evaluate whatever step it happened to catch, so
almost no OOD dump would pair with an ID dump and Fig 6 would have nothing to
plot. Every scheduled step therefore gets its own slot, retained until consumed.

ATOMICITY. The worker polls a directory the trainer is writing into, so a
half-written blob must never look ready. Each slot writes blob and metadata to
temporary names, renames them into place (rename is atomic within a
filesystem), and only then creates `.done`. The worker keys exclusively on
`.done`, so it can never observe a partial checkpoint.

BACKPRESSURE. A worker that falls behind must not fill /ptmp: a dinov3_huge
checkpoint is ~3.4 GB and 15 runs write concurrently. `pending_count` lets the
trainer SKIP a scheduled checkpoint rather than block or evict. Skipping loses
one point on an OOD curve; blocking would stall a gpu that costs far more, and
evicting would delete a step the worker may already be reading.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

# Written last, and the only thing a reader keys on.
DONE_SUFFIX = ".done"
META_SUFFIX = ".json"


def run_dir(root, group_id: str, uid: str) -> Path:
    """One directory per run, so runs cannot collide or interleave."""
    return Path(root) / str(group_id) / str(uid)


def _stem(directory: Path, step: int) -> Path:
    # Zero-padded so lexical order is step order -- the worker consumes oldest
    # first, and `sorted()` on names is what makes that true without parsing.
    return directory / f"step{int(step):08d}"


def pending_count(root, group_id: str, uid: str) -> int:
    """Ready-but-unconsumed slots for one run."""
    directory = run_dir(root, group_id, uid)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(f"*{DONE_SUFFIX}"))


def write(
    root,
    *,
    group_id: str,
    uid: str,
    step: int,
    blob_suffix: str,
    write_blob: Callable[[Path], None],
    meta: Mapping[str, Any],
    max_pending: int | None = None,
) -> Path | None:
    """Write one slot atomically. Returns its stem, or None if skipped.

    `write_blob` receives a TEMPORARY path and must write the weights to it;
    keeping serialisation in the caller is what lets equinox and torch share
    this module without it importing either.

    Returns None when `max_pending` is already met -- the backpressure case.
    The caller should log it: a silently dropped OOD point looks identical to
    one the schedule never asked for.
    """
    directory = run_dir(root, group_id, uid)
    directory.mkdir(parents=True, exist_ok=True)

    if max_pending is not None and pending_count(root, group_id, uid) >= max_pending:
        return None

    stem = _stem(directory, step)
    done = stem.with_suffix(DONE_SUFFIX)
    if done.exists():
        return stem

    blob_tmp = stem.with_suffix(blob_suffix + ".tmp")
    meta_tmp = stem.with_suffix(META_SUFFIX + ".tmp")
    write_blob(blob_tmp)
    payload = dict(meta)
    payload.update({"step": int(step), "group_id": str(group_id), "uid": str(uid),
                    "blob": stem.with_suffix(blob_suffix).name,
                    "written_at": time.time()})
    meta_tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))

    os.replace(blob_tmp, stem.with_suffix(blob_suffix))
    os.replace(meta_tmp, stem.with_suffix(META_SUFFIX))
    # LAST. Everything above is invisible to a reader until this exists.
    done.write_text("")
    return stem


def pending(root, group_id: str | None = None, uid: str | None = None) -> list[Path]:
    """Every ready slot, oldest step first, across runs unless narrowed."""
    base = Path(root)
    if not base.is_dir():
        return []
    if group_id and uid:
        dirs = [run_dir(root, group_id, uid)]
    elif group_id:
        dirs = sorted(p for p in (base / group_id).glob("*") if p.is_dir())
    else:
        dirs = sorted(p for p in base.glob("*/*") if p.is_dir())
    out: list[Path] = []
    for d in dirs:
        out.extend(sorted(d.glob(f"*{DONE_SUFFIX}")))
    return [p.with_suffix("") for p in out]


def read_meta(stem: Path) -> dict[str, Any]:
    return json.loads(Path(stem).with_suffix(META_SUFFIX).read_text())


def blob_path(stem: Path) -> Path:
    """The weights file this slot's metadata names."""
    return Path(stem).parent / read_meta(stem)["blob"]


def consume(stem: Path) -> None:
    """Drop the slot. `.done` goes FIRST so a crash mid-delete cannot leave a
    slot that looks ready but has no blob -- the worker would then fail on it
    forever."""
    stem = Path(stem)
    stem.with_suffix(DONE_SUFFIX).unlink(missing_ok=True)
    for path in stem.parent.glob(stem.name + ".*"):
        path.unlink(missing_ok=True)
