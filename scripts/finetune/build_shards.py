"""Decode a dataset split once into a uint8 shard, for every trial to reuse.

Run this ONCE per (dataset, split, img_size) before fine-tuning.
It is the difference between a dinov3_large trial costing ~7 hours and ~1.5:
the training step measures 220 img/s on an A100 with synthetic inputs and ~40
against the live JPEG path, because decode contends with JAX dispatch for the
GIL rather than because decode is slow.

ImageNet-1k train at 224px is 193 GB and takes one pass (~28 min at the
measured decode rate). Resumable -- a restart continues from the last complete
chunk.

    python scripts/finetune/build_shards.py --split train --root $SCRATCH/shards
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

from bllarse.finetune import shards  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--split", default="train", choices=["train", "validation"],
                   help="`train` is what the training loop streams; "
                        "`validation` is already preloaded once per run and "
                        "gains far less from sharding.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--root", required=True,
                   help="Directory for the shard. Needs ~193 GB for ImageNet-1k "
                        "train at 224px.")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=1024,
                   help="Rows decoded and flushed per step. Larger amortises "
                        "the flush; smaller loses less work to a restart.")
    p.add_argument("--no-resume", action="store_true",
                   help="Rebuild from row 0 even if a partial shard exists.")
    return p


def main():
    args = build_argparser().parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dinov3"))
    from vit_classification import DATASET_CONFIGS
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS[args.dataset]
    split_name = ("train" if args.split == "train"
                  else ds_config.get("test_split", "validation"))
    print(f"[shards] loading {args.dataset}/{split_name} ...", flush=True)
    hf_split = load_dataset(
        ds_config["hf_path"], name=ds_config.get("hf_config"))[split_name]

    n = len(hf_split)
    gib = n * args.img_size * args.img_size * 3 / 2**30
    print(f"[shards] {n} rows -> {gib:.1f} GiB at {args.img_size}px", flush=True)

    free = os.statvfs(args.root).f_bavail * os.statvfs(args.root).f_frsize / 2**30
    if free < gib * 1.05:
        raise SystemExit(
            f"[shards] {args.root} has {free:.0f} GiB free, need ~{gib:.0f}. "
            f"Refusing to start a build that cannot finish -- a partial shard "
            f"is mostly zeros, and zeros train as valid black images."
        )

    path = shards.build_shard(
        hf_split, img_key=ds_config["img_key"], label_key=ds_config["label_key"],
        img_size=args.img_size, root=args.root, dataset=args.dataset,
        split=args.split, num_workers=args.num_workers, chunk=args.chunk,
        resume=not args.no_resume,
    )
    # Re-open through the reader, which refuses an incomplete shard: the build
    # is only done when the thing that will consume it agrees.
    reader = shards.open_shard(args.root, args.dataset, args.split, args.img_size)
    print(f"[shards] wrote {path} ({len(reader)} rows), verified readable")


if __name__ == "__main__":
    main()
