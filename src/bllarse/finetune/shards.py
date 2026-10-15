"""Pre-decoded uint8 image shards: one JPEG decode shared by every run.

The measured problem this solves. On an A100, one dinov3_large optimizer step
runs at 220 img/s with synthetic inputs (scripts/finetune/bench_step.py, 33.7%
of the device's own matmul ceiling), and the same step inside finetune_jax runs
at ~40 img/s against the live JPEG path. The step is not the bottleneck; the
decode is.

Not because decode is slow in isolation -- measured standalone at ~750 img/s --
but because it is not in isolation. The main thread dispatches JAX ops while
PIL decodes on worker threads, and only the decode itself releases the GIL; the
`np.stack`, the generator plumbing and a fresh ThreadPoolExecutor PER
MICRO-BATCH all contend for it.

A shard replaces all of that with a gather out of a memmap. The layout is
deliberately trivial -- one contiguous (N, H, W, 3) uint8 array plus a labels
vector -- because ROW i OF THE SHARD IS ROW i OF THE HF SPLIT. That identity is
what lets `protocol.shuffled_indices` keep producing the batch order unchanged,
so the JAX arm and the PyTorch arm still consume an identical stream and the
shard is invisible to everything downstream.

Size: ImageNet-1k train at 224px is 1,281,167 x 224 x 224 x 3 = 193 GB. Built
once in a single pass at the decode rate (~28 minutes), then read by every
trial of every arm.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from bllarse.finetune.imagenet import decode_uint8

# Bumped whenever the on-disk layout changes in a way a reader cannot detect.
# A shard written by an older version is REFUSED rather than reinterpreted:
# silently reading a differently-shaped array as images produces a run that
# trains on garbage and reports plausible-looking losses.
SHARD_FORMAT = 1

# Threads issuing row reads inside one `take`.
#
# A shard row is 150,528 contiguous bytes and a micro-batch gathers 32 of them
# from random offsets in a 193 GB file, so the gather is LATENCY-bound, not
# bandwidth-bound: ~10 ms of GPFS round trip per row, serialised, is ~320 ms a
# micro-batch, which is the ~82 img/s measured with serial reads against a step
# that benchmarks at 220.
#
# Sixteen because the reads are pure wait -- serial reads showed 2.0% CPU
# utilisation across 36 cores while the GPU idled -- and sixteen threads fit
# within an 18-CPU training job.
# This is a concurrency knob for a filesystem, not a compute one, so it is not
# bounded by cores in the way a decode pool would be.
DEFAULT_READ_WORKERS = 16


def shard_paths(root, dataset: str, split: str, img_size: int):
    """(images, labels, meta) paths for one (dataset, split, size)."""
    base = Path(root) / f"{dataset}_{split}_{img_size}"
    return (base.with_suffix(".u8"),
            base.with_suffix(".labels.npy"),
            base.with_suffix(".json"))


def _write_meta(path, **fields):
    path.write_text(json.dumps(fields, indent=2))


def build_shard(hf_split, *, img_key: str, label_key: str, img_size: int,
                root, dataset: str, split: str, num_workers: int = 16,
                chunk: int = 1024, resume: bool = True, log_every: int = 50) -> Path:
    """Decode `hf_split` once into a uint8 memmap. Returns the images path.

    Resumable by construction: the meta file records how many rows are
    complete, and a restart continues from there. A 28-minute job that has to
    start over because a node hiccuped at minute 27 is a job that gets run at
    a worse time, so this is not a luxury.

    Rows are written IN SPLIT ORDER, never shuffled. Shuffling belongs to
    protocol.shuffled_indices at read time, where both arms share it; baking a
    permutation into the shard would silently desynchronise them.
    """
    images_path, labels_path, meta_path = shard_paths(root, dataset, split, img_size)
    images_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(hf_split)
    shape = (n, img_size, img_size, 3)

    done = 0
    if resume and meta_path.exists() and images_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("format") == SHARD_FORMAT and tuple(meta["shape"]) == shape):
            done = int(meta.get("rows_complete", 0))
            print(f"[shards] resuming at row {done}/{n}", flush=True)
        else:
            print("[shards] existing shard has a different shape or format; "
                  "rebuilding from scratch", flush=True)

    mode = "r+" if (done and images_path.exists()) else "w+"
    images = np.memmap(images_path, dtype=np.uint8, mode=mode, shape=shape)

    if done == 0:
        labels = np.asarray(hf_split[label_key], dtype=np.int16)
        if len(labels) != n:
            raise ValueError(
                f"{dataset}/{split}: {len(labels)} labels for {n} rows")
        np.save(labels_path, labels)

    worker = partial(decode_uint8, hf_split, img_key, img_size)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # ONE pool for the whole build, not one per chunk: creating a pool per
        # batch is part of what makes the live path slow.
        for offset in range(done, n, chunk):
            stop = min(offset + chunk, n)
            images[offset:stop] = np.stack(
                list(pool.map(worker, np.arange(offset, stop))))
            # Meta after the data, so a crash between them leaves rows_complete
            # UNDERSTATED -- the resume then redoes a chunk, which is harmless.
            # The other order would skip rows and leave holes of zeros that no
            # later check could distinguish from black images.
            images.flush()
            _write_meta(meta_path, format=SHARD_FORMAT, dataset=dataset,
                        split=split, img_size=img_size, shape=list(shape),
                        rows_complete=stop, dtype="uint8", layout="NHWC")
            if (stop // chunk) % log_every == 0 or stop == n:
                rate = (stop - done) / max(time.perf_counter() - start, 1e-9)
                remaining = (n - stop) / max(rate, 1e-9)
                print(f"[shards] {stop}/{n} rows ({rate:.0f} img/s, "
                      f"~{remaining / 60:.1f} min left)", flush=True)

    images.flush()
    del images
    return images_path


class ShardReader:
    """Random-access reader over a built shard.

    Rows are fetched with concurrent `os.pread`, not a numpy fancy-index into
    the memmap. The fancy-index was the obvious implementation and it is the
    slow one: numpy walks the requested rows in order, each miss traps into a
    GPFS read, and those round trips serialise. Measured at ~82 img/s against
    a training step that benchmarks at 220 -- so the GPU sat idle roughly 60%
    of the time, with 2.0% CPU utilisation across 36 allocated cores.

    `os.pread` is what makes the threading worth anything: it releases the GIL
    for the whole syscall, so N threads genuinely overlap N round trips. A
    threaded fancy-index would not -- the copy happens with the GIL held.

    The pool and the descriptor live as long as the reader. One pool per
    `take` would put pool construction on the per-micro-batch path, which is
    part of what made the JPEG loader slow in the first place.
    """

    def __init__(self, images, labels, *, path=None,
                 read_workers: int = DEFAULT_READ_WORKERS):
        self._images = images
        self.labels = labels
        self._row_shape = tuple(images.shape[1:])
        self._row_bytes = int(np.prod(self._row_shape))
        # `path is None` keeps the memmap path available for callers that
        # construct a reader directly (and for a filesystem where threading
        # buys nothing). open_shard always passes it.
        self._fd = os.open(os.fspath(path), os.O_RDONLY) if path is not None else None
        self._pool = (
            ThreadPoolExecutor(max_workers=read_workers,
                               thread_name_prefix="shard-read")
            if self._fd is not None and read_workers > 1 else None
        )

    def __len__(self):
        return len(self._images)

    def _read_row(self, out, position, row) -> None:
        offset = int(row) * self._row_bytes
        buf = os.pread(self._fd, self._row_bytes, offset)
        if len(buf) != self._row_bytes:
            # Short reads are legal for pread at EOF. Here that can only mean
            # the file is smaller than the meta claims, which open_shard has
            # already checked -- so it means the shard changed underneath us.
            raise OSError(
                f"short read at row {row}: {len(buf)} of {self._row_bytes} "
                f"bytes. The shard has been truncated or replaced while open."
            )
        out[position] = np.frombuffer(buf, dtype=np.uint8).reshape(self._row_shape)

    def take(self, indices) -> np.ndarray:
        """(len(indices), H, W, 3) uint8, in the order requested."""
        idx = np.asarray(indices)
        if self._pool is None:
            return np.asarray(self._images[idx])
        out = np.empty((len(idx), *self._row_shape), dtype=np.uint8)
        # Consumed rather than left lazy: pool.map returns a generator, and an
        # exception in a worker only surfaces when its result is read.
        for _ in self._pool.map(
                lambda pair: self._read_row(out, *pair), enumerate(idx)):
            pass
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def open_shard(root, dataset: str, split: str, img_size: int,
               read_workers: int = DEFAULT_READ_WORKERS) -> ShardReader:
    """Open a built shard, or raise saying exactly what is missing.

    Refuses an INCOMPLETE shard. A partially built 193 GB file is mostly
    zeros, and zeros decode as valid black images -- a run would train on them
    and report a loss curve that looks merely bad rather than broken.
    """
    images_path, labels_path, meta_path = shard_paths(root, dataset, split, img_size)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no shard for {dataset}/{split} at {img_size}px under {root}. "
            f"Build it with scripts/finetune/build_shards.py."
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != SHARD_FORMAT:
        raise ValueError(
            f"{meta_path} is format {meta.get('format')}, expected "
            f"{SHARD_FORMAT}. Rebuild the shard."
        )
    shape = tuple(meta["shape"])
    if meta.get("rows_complete") != shape[0]:
        raise ValueError(
            f"{images_path} is INCOMPLETE: {meta.get('rows_complete')} of "
            f"{shape[0]} rows written. The missing rows are zeros, which train "
            f"as valid black images -- finish the build before using it."
        )
    images = np.memmap(images_path, dtype=np.uint8, mode="r", shape=shape)
    labels = np.load(labels_path)
    return ShardReader(images, labels, path=images_path,
                       read_workers=read_workers)
