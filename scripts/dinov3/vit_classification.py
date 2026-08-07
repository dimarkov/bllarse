"""
Vision Transformer Classification with Equimo

Fine-tunes ViT models from the equimo library on image classification tasks
using Bayesian last-layer methods (IBProbit, IBPolyaGamma).

Dependencies:
    pip install equimo>=1.1.0

Example usage:
    python scripts/dinov3/vit_classification.py \
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
from tqdm import tqdm

load_dotenv()
load_dotenv(".env.secrets", override=False) # loads secrets if present locally, no-op otherwise

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
try:
    import mlflow
except ImportError:
    mlflow = None
from huggingface_hub import hf_hub_download, HfApi

from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image

from jax import random as jr, config, vmap
from datasets import load_dataset

try:
    import equimo.models as equimo_models
    NO_EQUIMO = False
except ImportError:
    NO_EQUIMO = True
    warnings.warn("equimo not installed. Run: pip install equimo>=1.1.0")

from mlpox.load_models import load_model as mlpox_load_model

from bllarse.losses import IBProbit, CrossEntropy

from calibration import evaluate_classification


# Model configurations available in equimo
EQUIMO_MODELS = {
    # DINOv3 models (ViT with LayerNorm only) — expect CHW input
    "dinov3_small": {"img_size": 224, "embed_dim": 384, "channels_first": True},
    "dinov3_big": {"img_size": 224, "embed_dim": 768, "channels_first": True},
    "dinov3_large": {"img_size": 224, "embed_dim": 1024, "channels_first": True},
    "dinov3_huge": {"img_size": 224, "embed_dim": 1280, "channels_first": True},
    "dinov3_max": {"img_size": 224, "embed_dim": 4096, "channels_first": True},
    # DeepMLP models — expect HWC input
    "deepMLP_big": {"img_size": 64, "embed_dim": 512, "channels_first": False},
    "deepMLP_large": {"img_size": 64, "embed_dim": 1024, "channels_first": False},
}

# Dataset-specific input sizes for DINOv3 models (patch_size=16, so size must be divisible by 16).
# Chosen based on native dataset resolution; DeepMLP models ignore this table.
DINOV3_DATASET_IMG_SIZES = {
    "cifar10":       224,
    "cifar100":      224,
    "svhn":          224,
    "oxford_pets":   512,
    "food101":       512,
    "flowers102":    512,
    "stanford_cars": 512,
    "dtd":           512,
    "imagenet1k":    512,
}


def get_img_size(model_name: str, dataset_name: str) -> int:
    """Return the appropriate input image size for a model/dataset pair."""
    if "dinov3" in model_name:
        default = EQUIMO_MODELS[model_name]["img_size"]  # 224
        return DINOV3_DATASET_IMG_SIZES.get(dataset_name, default)
    return EQUIMO_MODELS[model_name]["img_size"]

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
    "svhn": {
        "hf_path": "ufldl-stanford/svhn",
        "hf_config": "cropped_digits",
        "num_classes": 10,
        "label_key": "label",
        "img_key": "image",
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
        raise ImportError("equimo not installed. Run: pip install equimo>=1.1.0")

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
        # equimo >=1.0 loads pretrained weights via registered factory functions
        model = getattr(equimo_models, model_id)(pretrained=True)
        return eqx.nn.inference_mode(model, True)
    else:
        model = mlpox_load_model(model_id)
        # Remove the classification head to get embeddings
        model = eqx.tree_at(lambda m: m.fc, model, eqx.nn.Identity())
        return eqx.nn.inference_mode(model, True)


NORM_STATS = {
    "cifar10": (np.array([0.49139968, 0.48215827, 0.44653124]),
                np.array([0.24703233, 0.24348505, 0.26158768])),
    "cifar100": (np.array([0.49139968, 0.48215827, 0.44653124]),
                 np.array([0.24703233, 0.24348505, 0.26158768])),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _load_and_process(dataset_split, img_key, img_size, idx):
    """Load, decode, convert and resize a single image by index."""
    img = dataset_split[int(idx)][img_key]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(img)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def iterate_batches(dataset_split, img_key, label_key, batch_size, img_size, mean, std, key=None, channels_first=True, num_workers=8, progress=True):
    """Yield (images, labels) batches from a HuggingFace dataset split.

    If key is provided, shuffles the dataset indices. Otherwise iterates sequentially.
    Images are loaded, converted, resized, and normalized per batch to avoid OOM.
    Uses a thread pool to parallelize image decode and resize across CPU cores.
    """
    n_samples = len(dataset_split)
    all_labels = np.array(dataset_split[label_key])

    if key is not None:
        indices = jr.permutation(key, n_samples)
        indices = np.array(indices)
    else:
        indices = np.arange(n_samples)

    worker_fn = partial(_load_and_process, dataset_split, img_key, img_size)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in tqdm(range(0, n_samples, batch_size), disable=not progress):
            batch_idx = indices[start:start + batch_size]

            images = list(pool.map(worker_fn, batch_idx))
            labels = jnp.asarray(all_labels[batch_idx])

            images = jnp.stack(images)
            # Normalize
            images = (images - mean) / std
            # Convert to channel-first (B, C, H, W) for models that expect it
            if channels_first:
                images = jnp.transpose(images, (0, 3, 1, 2))

            yield images, labels


@eqx.filter_jit
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


def prefetch_iterator(iterator, n_prefetch=2):
    """Wrap an iterator to prefetch items in a background thread."""
    q = Queue(maxsize=n_prefetch)
    sentinel = object()

    def fill():
        for item in iterator:
            q.put(item)
        q.put(sentinel)

    t = Thread(target=fill, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


def precompute_features(model, dataset_split, ds_config, img_size, mean, std, batch_size, key, channels_first=True):
    """Extract features for an entire split once; return CPU numpy arrays.

    Uses memory-mapped temporary files to avoid holding all features in RAM.
    """
    import tempfile

    n_samples = len(dataset_split)
    all_features = None
    all_labels = None
    feat_tmp = None
    label_tmp = None
    offset = 0

    batches = iterate_batches(
        dataset_split, ds_config["img_key"], ds_config["label_key"],
        batch_size=batch_size, img_size=img_size, mean=mean, std=std, key=None,
        channels_first=channels_first,
    )
    try:
        for images, labels in prefetch_iterator(batches, n_prefetch=2):
            features = np.asarray(extract_features(model, images, key))
            labels_np = np.asarray(labels)

            if all_features is None:
                feat_tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
                label_tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
                all_features = np.memmap(
                    feat_tmp.name, dtype=features.dtype, mode="w+",
                    shape=(n_samples, features.shape[-1]),
                )
                all_labels = np.memmap(
                    label_tmp.name, dtype=labels_np.dtype, mode="w+",
                    shape=(n_samples,),
                )

            n = len(features)
            all_features[offset:offset + n] = features
            all_labels[offset:offset + n] = labels_np
            offset += n

        # Copy to regular arrays for return (will be saved to npz immediately)
        result_features = np.array(all_features[:offset])
        result_labels = np.array(all_labels[:offset])
    finally:
        # Clean up temp files
        if feat_tmp is not None:
            del all_features
            os.unlink(feat_tmp.name)
        if label_tmp is not None:
            del all_labels
            os.unlink(label_tmp.name)

    return result_features, result_labels


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
    hf_repo=None, hf_path=None, channels_first=True,
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
        channels_first=channels_first,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, features=features, labels=labels)
    print(f"Cached features saved to {cache_path}")

    if hf_repo and hf_path:
        upload_features_to_hf(hf_repo, cache_path, hf_path)

    return features, labels


def iterate_feature_batches(features, labels, batch_size, key=None, progress=False):
    """Yield (features, labels) batches from cached CPU numpy arrays."""
    n = len(features)
    batch_size = min(batch_size, n)
    indices = np.array(jr.permutation(key, n)) if key is not None else np.arange(n)
    for start in tqdm(range(0, n, batch_size), disable=not progress):
        batch_idx = indices[start:start + batch_size]
        yield jnp.array(features[batch_idx]), jnp.array(labels[batch_idx])


def evaluate_model(loss_fn, test_features, test_labels, batch_size, loss_type, head=None):
    """Evaluate model on cached test features.

    loss_fn is either IBProbit or CrossEntropy (--loss-fn), so loss_type keeps being
    forwarded for the generic call signature; IBProbit ignores it (self.loss_type
    fixes its branch).
    """
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

    loss_type keeps being forwarded to current_loss_fn -- the generic call signature
    serves both classes -- even though IBProbit ignores it (self.loss_type fixes its
    branch).
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
        raise ImportError("equimo is required. Install with: pip install equimo>=1.1.0")

    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS.get(args.model, {"img_size": 224, "embed_dim": 384, "channels_first": True})
    img_size = get_img_size(args.model, args.dataset)
    embed_dim = model_config["embed_dim"]
    channels_first = model_config.get("channels_first", True)
    num_classes = ds_config["num_classes"]

    key = jr.PRNGKey(args.seed)

    # --- Load/cache features (shared across all param combos) ---
    test_split_name = ds_config["test_split"]
    train_cache = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_res{img_size}_train.npz")
    test_cache = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_res{img_size}_{test_split_name}.npz")

    hf_repo = None if args.no_hf_cache else args.hf_repo
    train_hf_path = f"{args.model}/{args.dataset}_res{img_size}_train.npz" if hf_repo else None
    test_hf_path = f"{args.model}/{args.dataset}_res{img_size}_{test_split_name}.npz" if hf_repo else None

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
        ds = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))
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
            channels_first=channels_first,
        )
        print("Precomputing test features...")
        test_features, test_labels = load_or_compute_features(
            test_cache, backbone, test_split, ds_config, img_size, mean, std,
            max(args.batch_size), feat_key, hf_repo=hf_repo, hf_path=test_hf_path,
            channels_first=channels_first,
        )

    # --- Validation split for early stopping ---
    # Cached features are stored in dataset order, so draw a seeded random
    # subset rather than a leading slice (which would be class-skewed for
    # datasets whose train split is grouped by class).
    val_features = val_labels = None
    if args.patience is not None:
        n_val = max(1, int(len(train_features) * args.val_fraction))
        key, split_key = jr.split(key)
        perm = np.array(jr.permutation(split_key, len(train_features)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_features, val_labels = train_features[val_idx], train_labels[val_idx]
        train_features, train_labels = train_features[train_idx], train_labels[train_idx]
        print(f"Early stopping enabled: patience={args.patience}, val_size={n_val}")

    # --- Build parameter grid ---
    if args.loss_fn == "IBProbit":
        param_grid = list(product(args.batch_size, args.num_update_iters))
        param_names = ("batch_size", "num_update_iters")
    elif args.loss_fn == "CrossEntropy":
        param_grid = list(product(args.batch_size, args.lr, args.weight_decay))
        param_names = ("batch_size", "lr", "weight_decay")
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    # Filter out batch sizes larger than the training set
    n_train = len(train_features)
    param_grid = [c for c in param_grid if c[0] <= n_train]

    n_combos = len(param_grid)
    print(f"\nLoss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    print(f"Parameter combinations: {n_combos}")
    
    log_mlfow = args.log_mlflow
    if log_mlfow:
        # --- MLflow experiment setup ---
        mlflow.set_experiment(args.experiment_name)

    # --- Sweep over parameter combinations ---
    for combo_idx, combo in enumerate(param_grid):
        params = dict(zip(param_names, combo))
        batch_size = params["batch_size"]



        if log_mlfow:
            # Nest under the parent run via parent_run_id + nested=True (matches
            # scripts/finetuning.py); the reserved ``mlflow.parentRunId`` tag set by hand
            # does not make the UI group the child under the parent (MLflow 3.x).
            mlflow.start_run(nested=bool(args.run_id), parent_run_id=args.run_id)
            log_params = {
                "model": args.model,
                "dataset": args.dataset,
                "loss_fn": args.loss_fn,
                "epochs": args.epochs,
                "label_smooth": args.label_smooth,
                "seed": args.seed,
                "patience": args.patience,
                "val_fraction": args.val_fraction if args.patience is not None else None,
                **params,
            }
            if args.loss_fn == "IBProbit":
                log_params.update(
                    px=args.px, learn_q=not args.no_learn_q, alpha=args.alpha,
                    q_init=args.q_init, kappa=args.kappa, warmup=args.warmup,
                    eps_init=args.eps_init,
                    eps_rho=max(0.0, 1.0 - batch_size / n_train),
                )
            mlflow.log_params(log_params)

        print(f"\n{'='*60}")
        print(f"[{combo_idx + 1}/{n_combos}] {params}")
        print(f"{'='*60}")

        # Initialize loss function fresh for each combo
        key, loss_key = jr.split(key)
        head = None
        opt_state = None
        optimizer = None

        if args.loss_fn == "IBProbit":
            # eps forgetting factor rho = 1 - B/N (one-pass Beta contamination memory).
            eps_rho = max(0.0, 1.0 - batch_size / n_train)
            loss_fn = IBProbit(
                embed_dim, num_classes, key=loss_key,
                q_init=args.q_init, alpha=args.alpha, batch_size=batch_size,
                kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
                warmup=args.warmup, eps_init=args.eps_init, rho=eps_rho,
            )
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
        best_selection_nll = float("inf")  # val NLL (or test NLL if no val set)
        best_test_metrics = None           # (acc, ece, nll) at the best epoch
        best_epoch = 0
        patience_counter = 0
        stopped_epoch = args.epochs

        pbar = tqdm(range(args.epochs), desc="epochs")
        for epoch in pbar:
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

            metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_acc": float(acc),
                "test_ece": float(ece),
                "test_nll": float(nll),
            }
            postfix = {
                "loss": f"{float(train_loss):.4f}",
                "acc": f"{float(acc):.4f}",
                "ece": f"{float(ece):.4f}",
                "nll": f"{float(nll):.4f}",
            }

            # Surface the self-tuned IBProbit scalars: the mean diagonal of Qhat, and
            # eps (always live -- the contamination mixture is structural now).
            if args.loss_fn == "IBProbit":
                q_mean = float(jnp.trace(loss_fn.Qhat) / loss_fn.Qhat.shape[0])
                metrics["q"] = q_mean
                postfix["q"] = f"{q_mean:.3g}"
                metrics["eps"] = float(loss_fn.eps)
                postfix["eps"] = f"{float(loss_fn.eps):.3g}"

            # Model selection by val NLL when early stopping, else by test NLL
            selection_nll = nll
            if args.patience is not None:
                _, _, val_nll = evaluate_model(
                    loss_fn, val_features, val_labels, batch_size,
                    loss_type=3, head=head
                )
                metrics["val_nll"] = float(val_nll)
                postfix["val_nll"] = f"{float(val_nll):.4f}"
                selection_nll = val_nll

            if selection_nll < best_selection_nll:
                best_selection_nll = float(selection_nll)
                best_test_metrics = (float(acc), float(ece), float(nll))
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if args.patience is not None:
                postfix["patience"] = f"{patience_counter}/{args.patience}"

            pbar.set_postfix(postfix)

            if log_mlfow:
                mlflow.log_metrics(metrics, step=epoch + 1)

            if args.patience is not None and patience_counter >= args.patience:
                stopped_epoch = epoch + 1
                pbar.close()
                print(f"Early stopping at epoch {stopped_epoch}")
                break

        if best_test_metrics is not None:
            b_acc, b_ece, b_nll = best_test_metrics
            print(f"Best @ epoch {best_epoch}: acc={b_acc:.4f}, ece={b_ece:.4f}, nll={b_nll:.4f}")

        if log_mlfow:
            if best_test_metrics is not None:
                b_acc, b_ece, b_nll = best_test_metrics
                mlflow.log_metrics({
                    "best_epoch": best_epoch,
                    "best_test_acc": b_acc,
                    "best_test_ece": b_ece,
                    "best_test_nll": b_nll,
                })
            if args.patience is not None:
                mlflow.log_params({"stopped_epoch": stopped_epoch})
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
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=None,
        help="Early stopping patience in epochs (None = disabled)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
        help="Fraction of training data held out as validation set for early stopping")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[64], help="Batch size(s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, nargs="+", default=[16], help="CAVI iterations per batch (IBProbit)")
    # --- IBProbit head knobs (mirror scripts/dinov3/ood_evaluation.py) ---
    parser.add_argument("--px", action="store_true",
                        help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    parser.add_argument("--no-learn-q", action="store_true",
                        help="hold Q at its isotropic prior q0*I (the ablation arm); default infers it")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="prior scale on the process noise: q0 = alpha^2 / batch_size "
                             "(q-update-spec 5.5e). At alpha=1e-3 the q recursion is inert "
                             "(learn_q on/off give identical mu); 0.1 puts q0 near the "
                             "pre-Wishart 1/D operating point.")
    parser.add_argument("--q-init", type=float, default=None,
                        help="override q0 directly; pass 1/D to reproduce the pre-Wishart regime")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Inverse-Wishart prior pseudo-count (default: 0.5, i.e. half a pseudo-observation against the K the likelihood adds per step)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="defer the gate and the q recursion for the first N steps (the mu=0 seed is uninformative)")
    parser.add_argument("--eps-init", type=float, default=0.01,
                        help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). Forgetting rho = 1 - B/N is set automatically.")
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
    parser.add_argument("--log-mlflow", action="store_true", help="Disable MLFlow logging")

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
