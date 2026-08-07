"""Precompute and cache ViT features for out-of-distribution ImageNet variants.

Mirrors precompute_features.py but for evaluation-only OOD datasets used to
measure calibration under distribution shift:
  - ImageNet-C (severity 5; 8 corruptions available as clean HF datasets)
  - ImageNet-R (axiong/imagenet-r)
  - ImageNet-A (barkermrl/imagenet-a)
  - ImageNet-V2 (vaishaal/ImageNetV2, matched-frequency variant)

Each source is normalized to a uniform access pattern with integer labels in
the *canonical ImageNet-1k* class index space (sorted-WNID order, which equals
ILSVRC/imagenet-1k's native label order). Features are extracted with the same
frozen backbone and resolution as the in-distribution ImageNet-1k features
(res512) so they share the head's feature space, then cached locally and
uploaded to HF Hub.

Usage:
    python scripts/dinov3/precompute_ood_features.py
    python scripts/dinov3/precompute_ood_features.py --model dinov3_huge
    python scripts/dinov3/precompute_ood_features.py --ood-keys imagenet_r imagenet_a
    python scripts/dinov3/precompute_ood_features.py --no-hf-cache  # local-only
"""

import gc
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from jax import random as jr, config

from datasets import load_dataset, load_dataset_builder

from vit_classification import (
    EQUIMO_MODELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_pretrained_backbone,
    load_or_compute_features,
    download_features_from_hf,
    upload_features_to_hf,
)

# OOD features are extracted at imagenet1k's resolution so the head (trained on
# imagenet1k_res512 features) sees inputs from the same feature space.
OOD_IMG_SIZE = 512

# ImageNet-C corruptions available as clean per-corruption HF datasets at
# severity 5 (the full 15-corruption set only exists as an opaque webdataset).
IMAGENET_C_CORRUPTIONS = [
    "brightness", "fog", "gaussian_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "zoom_blur",
]

# Any Tsomaros ImageNet-C dataset's ClassLabel gives the 1000 sorted WNIDs,
# i.e. the canonical ImageNet-1k index ordering.
CANONICAL_LABELS_SOURCE = "Tsomaros/ImageNet-C-gaussian_noise-severity_5"


class RelabeledSplit:
    """Wrap a HF dataset split, exposing canonical integer labels under 'label'.

    Supports exactly the access patterns iterate_batches relies on:
      - len(split)
      - split['label']  -> the canonical-index label array
      - split[int(i)]   -> the underlying row dict (for image decode)
    This avoids rewriting image data (no dataset.map / re-encode).
    """

    def __init__(self, split, labels):
        self._split = split
        self._labels = np.asarray(labels)
        assert len(self._labels) == len(self._split), "labels/split length mismatch"

    def __len__(self):
        return len(self._split)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return self._labels
            return self._split[key]
        return self._split[int(key)]


def load_wnid_to_idx():
    """Canonical WNID -> ImageNet-1k index map (sorted-WNID order)."""
    names = load_dataset_builder(CANONICAL_LABELS_SOURCE).info.features["label"].names
    assert len(names) == 1000, f"expected 1000 classes, got {len(names)}"
    return {wnid: i for i, wnid in enumerate(names)}


# ── Per-dataset preparation: return (split, img_key) with canonical labels ────

def prepare_imagenet_c(corruption, wnid_to_idx):
    ds = load_dataset(f"Tsomaros/ImageNet-C-{corruption}-severity_5", split="validation")
    # ClassLabel codes already equal canonical indices (names are sorted WNIDs).
    labels = np.asarray(ds["label"])
    return RelabeledSplit(ds, labels), "image"


def prepare_imagenet_r(wnid_to_idx):
    ds = load_dataset("axiong/imagenet-r", split="test")
    labels = np.array([wnid_to_idx[w] for w in ds["wnid"]])
    return RelabeledSplit(ds, labels), "image"


def prepare_imagenet_a(wnid_to_idx):
    ds = load_dataset("barkermrl/imagenet-a", split="train")
    subset_wnids = ds.features["label"].names  # 200 WNIDs in subset order
    codes = np.asarray(ds["label"])
    labels = np.array([wnid_to_idx[subset_wnids[c]] for c in codes])
    return RelabeledSplit(ds, labels), "image"


def prepare_imagenet_v2(wnid_to_idx):
    ds = load_dataset("vaishaal/ImageNetV2", split="train")
    # The 'train' split bundles 3 variants; keep the standard matched-frequency.
    keys = ds["__key__"]
    keep = [i for i, k in enumerate(keys) if k.startswith("imagenetv2-matched-frequency")]
    labels = np.array([int(keys[i].split("/")[1]) for i in keep])
    return RelabeledSplit(ds.select(keep), labels), "jpeg"


def build_ood_registry():
    """Map each OOD key to its preparation callable."""
    reg = {f"imagenet_c_{c}": (lambda w, c=c: prepare_imagenet_c(c, w))
           for c in IMAGENET_C_CORRUPTIONS}
    reg["imagenet_r"] = prepare_imagenet_r
    reg["imagenet_a"] = prepare_imagenet_a
    reg["imagenet_v2"] = prepare_imagenet_v2
    return reg


def main():
    registry = build_ood_registry()
    all_keys = list(registry.keys())

    parser = argparse.ArgumentParser(description="Precompute ViT features for OOD ImageNet variants")
    parser.add_argument("--model", type=str, default="dinov3_huge",
                        choices=list(EQUIMO_MODELS.keys()), help="Backbone model")
    parser.add_argument("--ood-keys", nargs="+", default=all_keys, choices=all_keys,
                        help="OOD datasets to process (default: all)")
    parser.add_argument("--cache-dir", type=str, default=".cache/features", help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Feature-extraction batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation + re-upload")
    parser.add_argument("--hf-repo", type=str, default="dimarkov/bllarse-features", help="HF dataset repo")
    parser.add_argument("--no-hf-cache", action="store_true", help="Disable HF Hub caching (local-only)")
    args = parser.parse_args()

    config.update("jax_platform_name", args.device)
    key = jr.PRNGKey(args.seed)

    model_config = EQUIMO_MODELS[args.model]
    channels_first = model_config.get("channels_first", True)
    img_size = OOD_IMG_SIZE
    hf_repo = None if args.no_hf_cache else args.hf_repo

    wnid_to_idx = load_wnid_to_idx()

    # Load backbone once for all OOD datasets.
    print(f"Loading pretrained backbone: {args.model}")
    key, model_key = jr.split(key)
    backbone = get_pretrained_backbone(args.model, model_key)

    total = len(args.ood_keys)
    for i, ood_key in enumerate(args.ood_keys, 1):
        cache_path = os.path.join(args.cache_dir, f"{args.model}_{ood_key}_res{img_size}_test.npz")
        hf_path = f"{args.model}/{ood_key}_res{img_size}_test.npz" if hf_repo else None

        print(f"\n[{i}/{total}] {ood_key} @ {img_size}px")

        if args.no_cache and os.path.exists(cache_path):
            os.remove(cache_path)

        # 1. Local cache hit — skip without touching the dataset.
        if not args.no_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            print(f"  cached locally ({data['features'].shape[0]} samples)")
            continue

        # 2. HF cache hit — download without building the (large) dataset.
        if not args.no_cache and hf_repo and download_features_from_hf(hf_repo, hf_path, cache_path):
            data = np.load(cache_path)
            print(f"  downloaded from HF ({data['features'].shape[0]} samples)")
            continue

        # 3. Compute: build the normalized dataset, extract, cache, upload.
        print("  extracting features...")
        split, img_key = registry[ood_key](wnid_to_idx)
        ds_config = {"img_key": img_key, "label_key": "label"}
        key, feat_key = jr.split(key)
        # load_or_compute_features re-checks local/HF (cheap misses here) then
        # computes, saves locally, and uploads to HF when hf_repo is set.
        features, labels = load_or_compute_features(
            cache_path, backbone, split, ds_config, img_size,
            IMAGENET_MEAN, IMAGENET_STD, args.batch_size, feat_key,
            hf_repo=(None if args.no_cache else hf_repo), hf_path=hf_path,
            channels_first=channels_first,
        )
        if args.no_cache and hf_repo and hf_path:
            upload_features_to_hf(hf_repo, cache_path, hf_path)
        print(f"  done ({features.shape[0]} samples, {features.shape[1]}d)")

        del split
        gc.collect()

    print(f"\nAll OOD features cached in {args.cache_dir}/")


if __name__ == "__main__":
    main()
