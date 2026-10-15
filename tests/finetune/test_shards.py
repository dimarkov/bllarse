"""The shard format and, above all, that it does not change the batch stream.

A shard is a pure speed change: the training loop must see byte-identical
images in an identical order whether it decodes JPEGs or gathers from a memmap.
If that ever stops being true the two arms desynchronise and nothing downstream
can tell -- both runs still produce plausible loss curves.
"""

import json

import numpy as np
import pytest
from PIL import Image

from bllarse.finetune import protocol, shards


class FakeSplit:
    """Minimal HF-split stand-in: split[i] -> row, split["label"] -> column."""

    def __init__(self, images, labels):
        self._images, self._labels = images, labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, key):
        if isinstance(key, str):
            return {"label": list(self._labels), "image": list(self._images)}[key]
        return {"image": self._images[key], "label": self._labels[key]}


def _split(n, size, seed=0):
    rng = np.random.default_rng(seed)
    images = [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]
    return FakeSplit(images, [i % 7 for i in range(n)])


def _build(tmp_path, n=24, size=8, **kw):
    split = _split(n, size)
    shards.build_shard(
        split, img_key="image", label_key="label", img_size=size,
        root=tmp_path, dataset="fake", split="train", num_workers=2,
        chunk=kw.pop("chunk", 8), **kw,
    )
    return split


# --- The property everything else rests on ----------------------------------


def test_row_i_of_the_shard_is_row_i_of_the_split(tmp_path):
    """The identity that makes a shard invisible to shuffled_indices.

    Both arms derive their batch order from protocol.shuffled_indices over row
    positions. If a shard permuted or dropped rows, the same index would name a
    different image on the shard path than on the JPEG path, and the JAX and
    PyTorch arms would train on different data while reporting the same seed.
    """
    from bllarse.finetune.imagenet import decode_uint8

    split = _build(tmp_path)
    reader = shards.open_shard(tmp_path, "fake", "train", 8)

    assert len(reader) == len(split)
    for i in (0, 1, 7, 8, 23):
        np.testing.assert_array_equal(
            reader.take([i])[0], decode_uint8(split, "image", 8, i))


def test_the_threaded_reader_agrees_with_the_memmap_byte_for_byte(tmp_path):
    """The pread path is a SPEED change and must be nothing else.

    `take` fetches rows with concurrent os.pread rather than one numpy
    fancy-index, because the fancy-index serialises a GPFS round trip per row
    (~82 img/s measured, against a 220 img/s step). Every row lands at the
    position its index occupied, and threads must not shuffle that.
    """
    _build(tmp_path, n=64)
    threaded = shards.open_shard(tmp_path, "fake", "train", 8, read_workers=8)
    serial = shards.open_shard(tmp_path, "fake", "train", 8, read_workers=1)
    assert threaded._pool is not None and serial._pool is None

    order = [63, 0, 17, 17, 5, 42, 1]
    np.testing.assert_array_equal(threaded.take(order), serial.take(order))
    np.testing.assert_array_equal(
        threaded.take(range(64)), np.asarray(serial._images[np.arange(64)]))
    threaded.close()


def test_a_truncated_shard_raises_rather_than_returning_short_rows(tmp_path):
    # open_shard's rows_complete check cannot see a file replaced AFTER it
    # opened. A short pread silently left as zeros would be a black image.
    _build(tmp_path)
    reader = shards.open_shard(tmp_path, "fake", "train", 8, read_workers=4)
    images_path, _, _ = shards.shard_paths(tmp_path, "fake", "train", 8)
    with open(images_path, "r+b") as fh:
        fh.truncate(8 * 8 * 8 * 3)
    with pytest.raises(OSError, match="short read"):
        reader.take([20])
    reader.close()


def test_take_returns_rows_in_the_order_requested(tmp_path):
    # Not sorted, not deduplicated -- the caller passes a permutation slice and
    # the labels are indexed with the same array.
    _build(tmp_path)
    reader = shards.open_shard(tmp_path, "fake", "train", 8)
    order = [5, 0, 23, 5, 11]
    got = reader.take(order)
    for position, row in enumerate(order):
        np.testing.assert_array_equal(got[position], reader.take([row])[0])


def test_labels_line_up_with_rows(tmp_path):
    split = _build(tmp_path)
    reader = shards.open_shard(tmp_path, "fake", "train", 8)
    np.testing.assert_array_equal(reader.labels, np.asarray(split["label"]))


def test_the_shuffle_is_unchanged_by_sharding(tmp_path):
    # shuffled_indices depends only on (n, seed, epoch) -- so as long as the
    # shard has the same row count, the order is identical by construction.
    # Pinned because it is the invariant, not an implementation detail.
    _build(tmp_path)
    reader = shards.open_shard(tmp_path, "fake", "train", 8)
    np.testing.assert_array_equal(
        protocol.shuffled_indices(len(reader), 137, 0),
        protocol.shuffled_indices(24, 137, 0),
    )


# --- Refusing to hand back something that trains as valid data --------------


def test_open_shard_refuses_an_incomplete_build(tmp_path):
    """A partial 193 GB shard is mostly zeros, and zeros are valid black images.

    A run over them trains and reports a loss curve that looks merely bad. This
    is the one failure the format must make impossible to miss.
    """
    _build(tmp_path)
    _, _, meta_path = shards.shard_paths(tmp_path, "fake", "train", 8)
    meta = json.loads(meta_path.read_text())
    meta["rows_complete"] = 10
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="INCOMPLETE"):
        shards.open_shard(tmp_path, "fake", "train", 8)


def test_open_shard_refuses_a_foreign_format(tmp_path):
    _build(tmp_path)
    _, _, meta_path = shards.shard_paths(tmp_path, "fake", "train", 8)
    meta = json.loads(meta_path.read_text())
    meta["format"] = shards.SHARD_FORMAT + 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="format"):
        shards.open_shard(tmp_path, "fake", "train", 8)


def test_open_shard_says_how_to_build_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_shards"):
        shards.open_shard(tmp_path, "fake", "train", 8)


# --- Resume -----------------------------------------------------------------


def test_a_resumed_build_produces_the_same_bytes(tmp_path):
    """Interrupting a 28-minute build must not silently corrupt it.

    Meta is written AFTER the data, so a crash between them understates
    rows_complete and the resume redoes a chunk. The opposite order would skip
    rows and leave holes of zeros no later check could distinguish from black
    images.
    """
    split = _split(24, 8)
    common = dict(img_key="image", label_key="label", img_size=8,
                  root=tmp_path, dataset="fake", split="train",
                  num_workers=2, chunk=8)
    shards.build_shard(split, **common)
    full = np.array(shards.open_shard(tmp_path, "fake", "train", 8).take(range(24)))

    # Rewind to a partial state, then resume.
    _, _, meta_path = shards.shard_paths(tmp_path, "fake", "train", 8)
    meta = json.loads(meta_path.read_text())
    meta["rows_complete"] = 8
    meta_path.write_text(json.dumps(meta))
    shards.build_shard(split, resume=True, **common)

    resumed = np.array(shards.open_shard(tmp_path, "fake", "train", 8).take(range(24)))
    np.testing.assert_array_equal(full, resumed)


def test_build_rejects_a_label_count_that_disagrees_with_the_rows(tmp_path):
    split = FakeSplit([Image.new("RGB", (8, 8))] * 4, [0, 1, 2])
    with pytest.raises(ValueError, match="labels"):
        shards.build_shard(
            split, img_key="image", label_key="label", img_size=8,
            root=tmp_path, dataset="fake", split="train", num_workers=1)
