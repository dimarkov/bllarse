"""Precompute and cache ViT features for all model/dataset combinations.

Usage:
    python scripts/alternatives/precompute_features.py
    python scripts/alternatives/precompute_features.py --models dinov3_small dinov3_big
    python scripts/alternatives/precompute_features.py --datasets cifar10 oxford_pets
    python scripts/alternatives/precompute_features.py --no-cache  # force recompute + re-upload
    python scripts/alternatives/precompute_features.py --no-hf-cache  # local-only
"""

import gc
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from jax import random as jr, config

from datasets import load_dataset

from vit_classification import (
    EQUIMO_MODELS,
    DATASET_CONFIGS,
    NORM_STATS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_img_size,
    get_pretrained_backbone,
    load_or_compute_features,
    upload_features_to_hf,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute ViT features for all model/dataset combos")
    parser.add_argument(
        "--models", nargs="+", default=list(EQUIMO_MODELS.keys()),
        choices=list(EQUIMO_MODELS.keys()), help="Models to process (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()), help="Datasets to process (default: all)",
    )
    parser.add_argument("--cache-dir", type=str, default=".cache/features", help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation of all features")
    parser.add_argument("--hf-repo", type=str, default="dimarkov/bllarse-features", help="HF dataset repo for feature cache")
    parser.add_argument("--no-hf-cache", action="store_true", help="Disable HF Hub caching (local-only)")
    args = parser.parse_args()

    config.update("jax_platform_name", args.device)
    key = jr.PRNGKey(args.seed)

    hf_repo = None if args.no_hf_cache else args.hf_repo
    total = len(args.models) * len(args.datasets)
    done = 0

    for dataset_name in args.datasets:
        ds_config = DATASET_CONFIGS[dataset_name]

        # Determine splits
        splits = ["train", ds_config["test_split"]]
        splits = list(dict.fromkeys(splits))

        # Load dataset once per dataset
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")
        ds = load_dataset(ds_config["hf_path"])
        mean, std = NORM_STATS.get(dataset_name, (IMAGENET_MEAN, IMAGENET_STD))

        for model_name in args.models:
            done += 1
            model_config = EQUIMO_MODELS[model_name]
            img_size = get_img_size(model_name, dataset_name)
            print(f"\n[{done}/{total}] {dataset_name} x {model_name} @ {img_size}px")

            # Load backbone once per model
            key, model_key = jr.split(key)
            backbone = get_pretrained_backbone(model_name, model_key)

            channels_first = model_config.get("channels_first", True)
            key, feat_key = jr.split(key)

            for split in splits:
                cache_path = os.path.join(
                    args.cache_dir, f"{model_name}_{dataset_name}_res{img_size}_{split}.npz"
                )
                hf_path = f"{model_name}/{dataset_name}_res{img_size}_{split}.npz" if hf_repo else None

                if args.no_cache and os.path.exists(cache_path):
                    os.remove(cache_path)

                if not args.no_cache and os.path.exists(cache_path):
                    data = np.load(cache_path)
                    print(f"  {split}: cached ({data['features'].shape[0]} samples)")
                    continue

                print(f"  {split}: extracting features...")
                # When forcing recompute, skip HF Hub download but still upload after
                effective_hf_repo = None if args.no_cache else hf_repo
                features, labels = load_or_compute_features(
                    cache_path, backbone, ds[split], ds_config, img_size, mean, std,
                    args.batch_size, feat_key,
                    hf_repo=effective_hf_repo,
                    hf_path=hf_path,
                    channels_first=channels_first,
                )
                if args.no_cache and hf_repo and hf_path:
                    upload_features_to_hf(hf_repo, cache_path, hf_path)
                print(f"  {split}: done ({features.shape[0]} samples, {features.shape[1]}d)")

            del backbone
            gc.collect()

        del ds
        gc.collect()

    print(f"\nAll done! Features cached in {args.cache_dir}/")


if __name__ == "__main__":
    main()
