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
import argparse
import warnings

# Do not preallocate GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow

from jax import random as jr, config, vmap, image as jimg
from datasets import load_dataset

try:
    from equimo.io import load_model as equimo_load_model
    NO_EQUIMO = False
except ImportError:
    NO_EQUIMO = True
    warnings.warn("equimo not installed. Run: pip install equimo>=0.3.0")

from bllarse.losses import IBProbit, CrossEntropy

from calibration import evaluate_classification


# Model configurations available in equimo
EQUIMO_MODELS = {
    # DINOv2 models (ViT with LayerNorm only)
    "dinov3_small": {"img_size": 224, "embed_dim": 384},
    "dinov3_big": {"img_size": 224, "embed_dim": 768},
    "dinov3_large": {"img_size": 224, "embed_dim": 1024},
    "dinov3_huge": {"img_size": 224, "embed_dim": 1280},
    "dinov3_max": {"img_size": 224, "embed_dim": 4096}
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
    equimo_id_map = {
        "dinov3_small": ("vit", "dinov3_vits16plus_pretrain_lvd1689m"),
        "dinov3_big": ("vit", "dinov3_vitb16_pretrain_lvd1689m"),
        "dinov3_large": ("vit", "dinov3_vitl16_pretrain_lvd1689m"),
        "dinov3_huge": ("vit", "dinov3_vith16plus_pretrain_lvd1689m"),
        "dinov3_max": ("vit", "dinov3_vit7b16_pretrain_lvd1689m")
    }

    if model_name not in equimo_id_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(equimo_id_map.keys())}")

    arch_type, model_id = equimo_id_map[model_name]
    model = equimo_load_model(arch_type, model_id)
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
                img = np.array(img, dtype=np.float32) / 255.0
            else:
                img = np.array(img, dtype=np.float32)
                if img.max() > 1.0:
                    img /= 255.0
            # Ensure 3 channels (H, W, C)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.concatenate([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            images.append(img)

        images = jnp.array(np.stack(images))
        labels = jnp.array(subset[label_key])

        # Resize to target size
        images = jimg.resize(
            images,
            (images.shape[0], img_size, img_size, 3),
            method="bilinear",
            antialias=True,
        )

        # Normalize
        images = (images - mean) / std

        yield images, labels


def extract_features(model, images, key):
    """Extract features from ViT model (CLS token or global average)."""
    def single_forward(x):
        # equimo models typically return (class_token, patch_tokens)
        # or just features depending on architecture
        out = model(x, enable_dropout=False, key=key)
        if isinstance(out, tuple):
            return out[0]  # CLS token
        return out

    return vmap(single_forward)(images)


def evaluate_model(model, loss_fn, test_split, ds_config, img_size, mean, std, loss_type, key, head=None):
    """Evaluate model on test split using batched iteration."""
    all_logits = []
    all_labels = []

    for images, labels in iterate_batches(
        test_split, ds_config["img_key"], ds_config["label_key"],
        batch_size=64, img_size=img_size, mean=mean, std=std, key=None
    ):
        features = extract_features(model, images, key)
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
    model,
    loss_fn,
    train_split,
    ds_config,
    img_size,
    mean,
    std,
    batch_size,
    num_update_iters,
    loss_type,
    key,
    head=None,
    opt_state=None,
    optimizer=None,
):
    """Train for one epoch.

    For IBProbit: uses CAVI updates (head/opt_state/optimizer are None).
    For CrossEntropy: runs gradient-based updates on the linear head.
    """
    key, shuffle_key = jr.split(key)

    current_loss_fn = loss_fn
    total_loss = 0.0
    n_batches = 0

    for images, labels in iterate_batches(
        train_split, ds_config["img_key"], ds_config["label_key"],
        batch_size=batch_size, img_size=img_size, mean=mean, std=std, key=shuffle_key
    ):
        # Extract features
        features = extract_features(model, images, key)

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

    # MLflow run setup
    if args.run_id:
        # Parent run already belongs to an experiment; resuming it sets the context
        mlflow.start_run(run_id=args.run_id)
    else:
        mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(nested=bool(args.run_id))
    mlflow.log_params({
        "model": args.model,
        "dataset": args.dataset,
        "loss_fn": args.loss_fn,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_update_iters": args.num_update_iters,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smooth": args.label_smooth,
        "seed": args.seed,
    })

    key = jr.PRNGKey(args.seed)

    # Load dataset (keep as HF dataset, convert per batch)
    print(f"Loading dataset: {args.dataset}")
    hf_path = ds_config["hf_path"]
    ds = load_dataset(hf_path)

    train_split = ds["train"]
    test_split = ds[ds_config["test_split"]]

    print(f"Train samples: {len(train_split)}, Test samples: {len(test_split)}")

    # Normalization stats
    mean, std = NORM_STATS.get(args.dataset, (IMAGENET_MEAN, IMAGENET_STD))

    # Load pretrained backbone
    print(f"Loading pretrained model: {args.model}")
    key, model_key = jr.split(key)
    backbone = get_pretrained_backbone(args.model, model_key)

    # Initialize loss function
    key, loss_key = jr.split(key)
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]

    head = None
    opt_state = None
    optimizer = None

    if args.loss_fn == "IBProbit":
        loss_fn = IBProbit(embed_dim, num_classes, key=loss_key)
    elif args.loss_fn == "CrossEntropy":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
        key, head_key = jr.split(key)
        head = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
        optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    print(f"Loss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")

    # Initial evaluation
    key, eval_key = jr.split(key)
    acc, ece, nll = evaluate_model(
        backbone, loss_fn, test_split, ds_config, img_size, mean, std,
        loss_type=3, key=eval_key, head=head
    )
    print(f"Initial: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")

    # Training loop
    for epoch in range(args.epochs):
        key, train_key, eval_key = jr.split(key, 3)

        loss_fn, head, opt_state, train_loss = train_epoch(
            backbone,
            loss_fn,
            train_split,
            ds_config,
            img_size,
            mean,
            std,
            args.batch_size,
            args.num_update_iters,
            loss_type=3,
            key=train_key,
            head=head,
            opt_state=opt_state,
            optimizer=optimizer,
        )

        # Evaluate
        acc, ece, nll = evaluate_model(
            backbone, loss_fn, test_split, ds_config, img_size, mean, std,
            loss_type=3, key=eval_key, head=head
        )

        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")

        mlflow.log_metrics({
            "train_loss": float(train_loss),
            "test_acc": float(acc),
            "test_ece": float(ece),
            "test_nll": float(nll),
        }, step=epoch + 1)

    print("Training complete!")
    mlflow.end_run()  # end child (or standalone) run
    if args.run_id:
        mlflow.end_run()  # end parent run


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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, default=16, help="CAVI iterations per batch")
    parser.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing (for CrossEntropy)")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--experiment-name", type=str, default="bllarse", help="MLflow experiment name")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (for CrossEntropy)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (for CrossEntropy)")
    parser.add_argument("--run-id", type=str, default=None, help="Parent MLflow run ID for nested grouping")

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
