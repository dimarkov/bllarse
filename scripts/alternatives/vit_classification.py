"""
Vision Transformer Classification with Equimo

Fine-tunes ViT models from the equimo library on image classification tasks
using Bayesian last-layer methods (IBProbit, IBPolyaGamma).

Dependencies:
    pip install equimo>=0.3.0

Example usage:
    python scripts/alternatives/vit_classification.py \
        --model dinov3_small \
        --dataset cifar10 \
        --loss-fn IBProbit \
        --epochs 10 \
        --batch-size 64
"""

import os
import shutil
import argparse
import warnings
from itertools import product
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow
from huggingface_hub import hf_hub_download, HfApi

from jax import random as jr, config, vmap, image as jimg
from datasets import load_dataset

try:
    from equimo.io import load_model as equimo_load_model
    NO_EQUIMO = False
except ImportError:
    NO_EQUIMO = True
    warnings.warn("equimo not installed. Run: pip install equimo>=0.3.0")

from mlpox.load_models import load_model as mlpox_load_model

from bllarse.losses import IBProbit, CrossEntropy

from calibration import evaluate_classification


# Model configurations available in equimo
EQUIMO_MODELS = {
    # DINOv2 models (ViT with LayerNorm only)
    "dinov3_small": {"img_size": 224, "embed_dim": 384},
    "dinov3_big": {"img_size": 224, "embed_dim": 768},
    "dinov3_large": {"img_size": 224, "embed_dim": 1024},
    "dinov3_huge": {"img_size": 224, "embed_dim": 1280},
    "dinov3_max": {"img_size": 224, "embed_dim": 4096},
    "deepMLP_big": {"img_size": 64, "embed_dim": 512},
    "deepMLP_large": {"img_size": 64, "embed_dim": 1024}
}

# Dataset configurations
DATASET_CONFIGS = {
    "cifar10": {
        "hf_path": "cifar10",
        "num_classes": 10,
        "label_key": "label",
        "img_key": "img",
        "test_split": "test",
    },
    "cifar100": {
        "hf_path": "cifar100",
        "num_classes": 100,
        "label_key": "fine_label",
        "img_key": "img",
        "test_split": "test",
    },
    "oxford_pets": {
        "hf_path": "timm/oxford-iiit-pet",
        "num_classes": 37,
        "label_key": "label",
        "img_key": "image",
        "test_split": "test",
    },
    "food101": {
        "hf_path": "ethz/food101",
        "num_classes": 101,
        "label_key": "label",
        "img_key": "image",
        "test_split": "validation",
    },
    "flowers102": {
        "hf_path": "nelorth/oxford-flowers",
        "num_classes": 102,
        "label_key": "label",
        "img_key": "image",
        "test_split": "test",
    },
    "stanford_cars": {
        "hf_path": "tanganke/stanford_cars",
        "num_classes": 196,
        "label_key": "label",
        "img_key": "image",
        "test_split": "test",
    },
    "dtd": {
        "hf_path": "tanganke/dtd",
        "num_classes": 47,
        "label_key": "label",
        "img_key": "image",
        "test_split": "test",
    },
    "imagenet1k": {
        "hf_path": "ILSVRC/imagenet-1k",
        "num_classes": 1000,
        "label_key": "label",
        "img_key": "image",
        "test_split": "validation",
    },
}


def get_pretrained_backbone(model_name: str, key):
    """Load pretrained ViT backbone from equimo."""
    if NO_EQUIMO:
        raise ImportError("equimo not installed. Run: pip install equimo>=0.3.0")

    # Map to equimo model identifiers
    model_id_map = {
        "dinov3_small": ("vit", "dinov3_vits16plus_pretrain_lvd1689m"),
        "dinov3_big": ("vit", "dinov3_vitb16_pretrain_lvd1689m"),
        "dinov3_large": ("vit", "dinov3_vitl16_pretrain_lvd1689m"),
        "dinov3_huge": ("vit", "dinov3_vith16plus_pretrain_lvd1689m"),
        "dinov3_max": ("vit", "dinov3_vit7b16_pretrain_lvd1689m"),
        "deepMLP_big": ("mlp", "B_12-Wi_512_res_64_in21k"),
        "deepMLP_large": ("mlp", "B_12-Wi_1024_res_64_in21k")
    }

    if model_name not in model_id_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_id_map.keys())}")

    arch_type, model_id = model_id_map[model_name]

    if 'dinov3' in model_name:
        model = equimo_load_model(arch_type, model_id)
        return eqx.nn.inference_mode(model, True)
    else:
        model = mlpox_load_model(model_id)
        return eqx.nn.inference_mode(model, True)


NORM_STATS = {
    "cifar10": (np.array([0.49139968, 0.48215827, 0.44653124]),
                np.array([0.24703233, 0.24348505, 0.26158768])),
    "cifar100": (np.array([0.49139968, 0.48215827, 0.44653124]),
                 np.array([0.24703233, 0.24348505, 0.26158768])),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def iterate_batches(dataset_split, img_key, label_key, batch_size, img_size, mean, std, key=None):
    """Yield (images, labels) batches from a HuggingFace dataset split.

    If key is provided, shuffles the dataset indices. Otherwise iterates sequentially.
    Images are loaded, converted, resized, and normalized per batch to avoid OOM.
    """
    n_samples = len(dataset_split)

    if key is not None:
        indices = jr.permutation(key, n_samples)
        indices = np.array(indices)
    else:
        indices = np.arange(n_samples)

    for start in range(0, n_samples - batch_size + 1, batch_size):
        batch_idx = indices[start:start + batch_size]
        subset = dataset_split.select(batch_idx)

        # Load images
        images = []
        for example in subset:
            img = example[img_key]
            # Handle PIL images
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                img = jnp.array(img, dtype=np.float32) / 255.0
            else:
                img = jnp.array(img, dtype=np.float32)
                if img.max() > 1.0:
                    img /= 255.0
            # Ensure 3 channels (H, W, C)
            if img.ndim == 2:
                img = jnp.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = jnp.concatenate([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            # Resize to target size (per-image to handle variable sizes)
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = jimg.resize(
                    jnp.array(img),
                    (img_size, img_size, 3),
                    method="bilinear",
                    antialias=True,
                )
            images.append(img)

        images = jnp.stack(images)
        labels = jnp.asarray(subset[label_key])

        # Normalize and convert to channel-first (B, C, H, W)
        images = (images - mean) / std
        images = jnp.transpose(images, (0, 3, 1, 2))

        yield images, labels


def extract_features(model, images, key):
    """Extract features from ViT model (CLS token or global average)."""
    def single_forward(x):
        # equimo models typically return (class_token, patch_tokens)
        # or just features depending on architecture
        out = model(x, key=key)
        if isinstance(out, tuple):
            return out[0]  # CLS token
        return out

    return vmap(single_forward)(images)


def precompute_features(model, dataset_split, ds_config, img_size, mean, std, batch_size, key):
    """Extract features for an entire split once; return CPU numpy arrays."""
    all_features = []
    all_labels = []
    for images, labels in iterate_batches(
        dataset_split, ds_config["img_key"], ds_config["label_key"],
        batch_size=batch_size, img_size=img_size, mean=mean, std=std, key=None,
    ):
        features = extract_features(model, images, key)
        all_features.append(np.asarray(features))
        all_labels.append(np.asarray(labels))
    return np.concatenate(all_features), np.concatenate(all_labels)


def download_features_from_hf(repo_id, repo_path, local_path):
    """Try to download cached features from HF Hub. Returns True if found."""
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(downloaded, local_path)
        print(f"Downloaded features from HF Hub: {repo_id}/{repo_path}")
        return True
    except Exception:
        return False


def upload_features_to_hf(repo_id, local_path, repo_path):
    """Upload features to HF Hub."""
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded features to HF Hub: {repo_id}/{repo_path}")


def load_or_compute_features(
    cache_path, model, dataset_split, ds_config, img_size, mean, std, batch_size, key,
    hf_repo=None, hf_path=None,
):
    """3-tier feature cache: local file → HF Hub → compute.

    If hf_repo/hf_path are provided, also checks/stores on HF Hub.
    """
    # 1. Local cache
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["features"], data["labels"]

    # 2. HF Hub cache
    if hf_repo and hf_path:
        if download_features_from_hf(hf_repo, hf_path, cache_path):
            data = np.load(cache_path)
            return data["features"], data["labels"]

    # 3. Compute, save locally, upload to HF
    features, labels = precompute_features(
        model, dataset_split, ds_config, img_size, mean, std, batch_size, key,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, features=features, labels=labels)
    print(f"Cached features saved to {cache_path}")

    if hf_repo and hf_path:
        upload_features_to_hf(hf_repo, cache_path, hf_path)

    return features, labels


def iterate_feature_batches(features, labels, batch_size, key=None):
    """Yield (features, labels) batches from cached CPU numpy arrays."""
    n = len(features)
    indices = np.array(jr.permutation(key, n)) if key is not None else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield jnp.array(features[batch_idx]), jnp.array(labels[batch_idx])


def evaluate_model(loss_fn, test_features, test_labels, batch_size, loss_type, head=None):
    """Evaluate model on cached test features."""
    all_logits = []
    all_labels = []

    for features, labels in iterate_feature_batches(
        test_features, test_labels, batch_size=batch_size, key=None
    ):
        if head is not None:
            logits = vmap(head)(features)
            _, logits = loss_fn(logits, labels, with_logits=True, loss_type=loss_type)
        else:
            _, logits = loss_fn(features, labels, with_logits=True, loss_type=loss_type)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = jnp.concatenate(all_logits, axis=0)
    all_labels = jnp.concatenate(all_labels, axis=0)
    return evaluate_classification(all_logits, all_labels)


def train_epoch(
    loss_fn,
    train_features,
    train_labels,
    batch_size,
    num_update_iters,
    loss_type,
    key,
    head=None,
    opt_state=None,
    optimizer=None,
):
    """Train for one epoch on cached features.

    For IBProbit: uses CAVI updates (head/opt_state/optimizer are None).
    For CrossEntropy: runs gradient-based updates on the linear head.
    """
    key, shuffle_key = jr.split(key)

    current_loss_fn = loss_fn
    total_loss = 0.0
    n_batches = 0

    for features, labels in iterate_feature_batches(
        train_features, train_labels, batch_size=batch_size, key=shuffle_key
    ):
        if head is not None:
            # CrossEntropy path: gradient-based training of linear head
            @eqx.filter_value_and_grad
            def ce_loss(head):
                logits = vmap(head)(features)
                return current_loss_fn(logits, labels, loss_type=loss_type).mean()

            loss, grads = ce_loss(head)
            updates, opt_state = optimizer.update(grads, opt_state, head)
            head = eqx.apply_updates(head, updates)
        else:
            # IBProbit path: CAVI update
            if hasattr(current_loss_fn, 'update'):
                current_loss_fn = current_loss_fn.update(
                    features, labels, num_iters=num_update_iters
                )
            loss = current_loss_fn(features, labels, loss_type=loss_type).mean()

        total_loss += loss
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return current_loss_fn, head, opt_state, avg_loss


def main(args):
    if NO_EQUIMO:
        raise ImportError("equimo is required. Install with: pip install equimo>=0.3.0")

    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS.get(args.model, {"img_size": 224, "embed_dim": 384})
    img_size = model_config["img_size"]
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]

    key = jr.PRNGKey(args.seed)

    # --- Load/cache features (shared across all param combos) ---
    test_split_name = ds_config["test_split"]
    train_cache = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_train.npz")
    test_cache = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_{test_split_name}.npz")

    hf_repo = None if args.no_hf_cache else args.hf_repo
    train_hf_path = f"{args.model}/{args.dataset}_train.npz" if hf_repo else None
    test_hf_path = f"{args.model}/{args.dataset}_{test_split_name}.npz" if hf_repo else None

    if args.no_cache:
        for p in (train_cache, test_cache):
            if os.path.exists(p):
                os.remove(p)

    cache_exists = os.path.exists(train_cache) and os.path.exists(test_cache)

    if cache_exists:
        print("Loading cached features...")
        train_data = np.load(train_cache)
        train_features, train_labels = train_data["features"], train_data["labels"]
        test_data = np.load(test_cache)
        test_features, test_labels = test_data["features"], test_data["labels"]
        print(f"Train features: {train_features.shape}, Test features: {test_features.shape}")
    else:
        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset(ds_config["hf_path"])
        train_split = ds["train"]
        test_split = ds[test_split_name]
        print(f"Train samples: {len(train_split)}, Test samples: {len(test_split)}")

        mean, std = NORM_STATS.get(args.dataset, (IMAGENET_MEAN, IMAGENET_STD))

        print(f"Loading pretrained model: {args.model}")
        key, model_key = jr.split(key)
        backbone = get_pretrained_backbone(args.model, model_key)

        key, feat_key = jr.split(key)
        print("Precomputing train features...")
        train_features, train_labels = load_or_compute_features(
            train_cache, backbone, train_split, ds_config, img_size, mean, std,
            max(args.batch_size), feat_key, hf_repo=hf_repo, hf_path=train_hf_path,
        )
        print("Precomputing test features...")
        test_features, test_labels = load_or_compute_features(
            test_cache, backbone, test_split, ds_config, img_size, mean, std,
            max(args.batch_size), feat_key, hf_repo=hf_repo, hf_path=test_hf_path,
        )

    # --- Build parameter grid ---
    if args.loss_fn == "IBProbit":
        param_grid = list(product(args.batch_size, args.num_update_iters))
        param_names = ("batch_size", "num_update_iters")
    elif args.loss_fn == "CrossEntropy":
        param_grid = list(product(args.batch_size, args.lr, args.weight_decay))
        param_names = ("batch_size", "lr", "weight_decay")
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    n_combos = len(param_grid)
    print(f"\nLoss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    print(f"Parameter combinations: {n_combos}")

    # --- MLflow experiment setup ---
    mlflow.set_experiment(args.experiment_name)

    # --- Sweep over parameter combinations ---
    for combo_idx, combo in enumerate(param_grid):
        params = dict(zip(param_names, combo))
        batch_size = params["batch_size"]

        tags = {}
        if args.run_id:
            tags["mlflow.parentRunId"] = args.run_id

        mlflow.start_run(tags=tags)
        mlflow.log_params({
            "model": args.model,
            "dataset": args.dataset,
            "loss_fn": args.loss_fn,
            "epochs": args.epochs,
            "label_smooth": args.label_smooth,
            "seed": args.seed,
            **params,
        })

        print(f"\n{'='*60}")
        print(f"[{combo_idx + 1}/{n_combos}] {params}")
        print(f"{'='*60}")

        # Initialize loss function fresh for each combo
        key, loss_key = jr.split(key)
        head = None
        opt_state = None
        optimizer = None

        if args.loss_fn == "IBProbit":
            loss_fn = IBProbit(embed_dim, num_classes, key=loss_key)
            num_update_iters = params["num_update_iters"]
        elif args.loss_fn == "CrossEntropy":
            loss_fn = CrossEntropy(args.label_smooth, num_classes)
            key, head_key = jr.split(key)
            head = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
            optimizer = optax.adamw(params["lr"], weight_decay=params["weight_decay"])
            opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
            num_update_iters = 0

        # Initial evaluation
        acc, ece, nll = evaluate_model(
            loss_fn, test_features, test_labels, batch_size,
            loss_type=3, head=head
        )
        print(f"Initial: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")

        # Training loop
        for epoch in range(args.epochs):
            key, train_key = jr.split(key)

            loss_fn, head, opt_state, train_loss = train_epoch(
                loss_fn,
                train_features,
                train_labels,
                batch_size,
                num_update_iters,
                loss_type=3,
                key=train_key,
                head=head,
                opt_state=opt_state,
                optimizer=optimizer,
            )

            acc, ece, nll = evaluate_model(
                loss_fn, test_features, test_labels, batch_size,
                loss_type=3, head=head
            )

            print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")

            mlflow.log_metrics({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_acc": float(acc),
                "test_ece": float(ece),
                "test_nll": float(nll),
            }, step=epoch + 1)

        mlflow.end_run()

    print("\nAll runs complete!")


def build_argparser():
    parser = argparse.ArgumentParser(description="ViT Classification with Bayesian Last Layer")

    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_small",
        choices=list(EQUIMO_MODELS.keys()),
        help="Pretrained ViT model from equimo"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use"
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="IBProbit",
        choices=["IBProbit", "CrossEntropy"],
        help="Loss function"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[64], help="Batch size(s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, nargs="+", default=[16], help="CAVI iterations per batch (IBProbit)")
    parser.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing (for CrossEntropy)")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--experiment-name", type=str, default="bllarse", help="MLflow experiment name")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="Learning rate(s) (CrossEntropy)")
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-4], help="Weight decay(s) (CrossEntropy)")
    parser.add_argument("--run-id", type=str, default=None, help="Parent MLflow run ID for nested grouping")
    parser.add_argument("--cache-dir", type=str, default=".cache/features", help="Directory for cached features")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation of features")
    parser.add_argument("--hf-repo", type=str, default="dimarkov/bllarse-features", help="HF dataset repo for feature cache")
    parser.add_argument("--no-hf-cache", action="store_true", help="Disable HF Hub caching (local-only)")

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
