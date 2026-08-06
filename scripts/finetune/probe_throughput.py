"""Measure decode and train throughput SEPARATELY, before committing compute.

The fine-tuning campaign is ~117 A100-hours at dinov3_large. If the data loader
cannot sustain the model's consumption rate, every one of those hours runs at
the loader's speed instead -- and that is invisible in a training log, which
shows only the combined step time.
"""

import argparse
import time

import numpy as np

from bllarse.finetune import imagenet

# Within this ratio, the loader intermittently stalls the GPU even though its
# mean rate looks adequate.
MARGIN = 1.2


def measure_decode_rate(split, ds_config, img_size, n_images, micro, num_workers):
    """Images per second through decode+resize alone, no model involved."""
    order = np.random.default_rng(0).permutation(len(split))[:n_images]
    start = time.perf_counter()
    decoded = 0
    for offset in range(0, n_images - micro + 1, micro):
        imagenet.load_images_uint8(
            split, ds_config["img_key"], img_size,
            order[offset:offset + micro], num_workers=num_workers,
        )
        decoded += micro
    elapsed = time.perf_counter() - start
    return decoded / elapsed if elapsed > 0 else float("inf")


def verdict(decode_rate: float, train_rate: float) -> str:
    """What to do about the measured pair. This is the probe's actual output."""
    if decode_rate < train_rate:
        return (
            f"LOADER-BOUND: decode {decode_rate:.0f} img/s < train "
            f"{train_rate:.0f} img/s. The GPU will idle. Remedy: pre-resized "
            f"224px shards, or raise --num-workers if cores are free."
        )
    if decode_rate < train_rate * MARGIN:
        return (
            f"MARGINAL: decode {decode_rate:.0f} img/s barely clears train "
            f"{train_rate:.0f} img/s. Expect intermittent stalls; pre-resized "
            f"shards are still worth it."
        )
    return (
        f"COMPUTE-BOUND: decode {decode_rate:.0f} img/s comfortably clears "
        f"train {train_rate:.0f} img/s. The loader is not the ceiling."
    )


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="imagenet1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-images", type=int, default=4096)
    p.add_argument("--micro", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--train-rate", type=float, default=None,
                   help="Measured train img/s to compare against. Omit to "
                        "report the decode rate alone.")
    return p


def main():
    args = build_argparser().parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))
    from vit_classification import DATASET_CONFIGS
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS[args.dataset]
    split = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))["train"]

    rate = measure_decode_rate(split, ds_config, args.img_size, args.n_images,
                               args.micro, args.num_workers)
    print(f"[probe] decode: {rate:.0f} img/s at {args.img_size}px "
          f"with {args.num_workers} workers")
    if args.train_rate is not None:
        print(f"[probe] {verdict(rate, args.train_rate)}")


if __name__ == "__main__":
    main()
