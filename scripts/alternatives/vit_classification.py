"""
Vision Transformer Classification with Equimo

Fine-tunes ViT models from the equimo library on image classification tasks
using Bayesian last-layer methods (IBProbit, IBPolyaGamma).

Dependencies:
    pip install equimo>=0.3.0

Example usage:
    python scripts/alternatives/vit_classification.py \
        --model vit_small_patch14_dinov2 \
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

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax

from functools import partial
from jax import random as jr, config, vmap, lax
from datasets import load_dataset

try:
    import wandb
    NO_WANDB = False
except ImportError:
    NO_WANDB = True

try:
    from equimo.io import load_model as equimo_load_model
    NO_EQUIMO = False
except ImportError:
    NO_EQUIMO = True
    warnings.warn("equimo not installed. Run: pip install equimo>=0.3.0")

from bllarse.losses import IBProbit, IBPolyaGamma, CrossEntropy
from bllarse.utils import augmentdata, MEAN_DICT, STD_DICT

from calibration import evaluate_classification

config.update("jax_default_matmul_precision", "highest")

# Model configurations available in equimo
EQUIMO_MODELS = {
    # DINOv2 models (ViT with LayerNorm only)
    "vit_small_patch14_dinov2": {"img_size": 224, "embed_dim": 384},
    "vit_base_patch14_dinov2": {"img_size": 224, "embed_dim": 768},
    "vit_large_patch14_dinov2": {"img_size": 224, "embed_dim": 1024},
    # FasterViT models
    "faster_vit_0_224": {"img_size": 224, "embed_dim": 512},
    "faster_vit_1_224": {"img_size": 224, "embed_dim": 640},
    # SHViT models
    "shvit_s4": {"img_size": 224, "embed_dim": 256},
}

# Dataset configurations
DATASET_CONFIGS = {
    "cifar10": {"num_classes": 10, "label_key": "label", "img_key": "img"},
    "cifar100": {"num_classes": 100, "label_key": "fine_label", "img_key": "img"},
}


def get_pretrained_backbone(model_name: str, key):
    """Load pretrained ViT backbone from equimo."""
    if NO_EQUIMO:
        raise ImportError("equimo not installed. Run: pip install equimo>=0.3.0")
    
    # Map to equimo model identifiers
    equimo_id_map = {
        "vit_small_patch14_dinov2": ("vit", "dinov2_vits14"),
        "vit_base_patch14_dinov2": ("vit", "dinov2_vitb14"),
        "vit_large_patch14_dinov2": ("vit", "dinov2_vitl14"),
        "faster_vit_0_224": ("fastervit", "faster_vit_0_224"),
        "faster_vit_1_224": ("fastervit", "faster_vit_1_224"),
        "shvit_s4": ("shvit", "shvit_s4"),
    }
    
    if model_name not in equimo_id_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(equimo_id_map.keys())}")
    
    arch_type, model_id = equimo_id_map[model_name]
    model = equimo_load_model(arch_type, model_id)
    return eqx.nn.inference_mode(model, True)


def resize_images(img, img_size):
    """Resize images to target size."""
    from jax import image as jimg
    
    if img.dtype == jnp.uint8:
        img = img.astype(jnp.float32) / 255.0
    elif jnp.issubdtype(img.dtype, jnp.floating) and jnp.max(img) > 1.0:
        img = img / 255.0
    
    return jimg.resize(
        img,
        (img.shape[0], img_size, img_size, img.shape[3]),
        method="bilinear",
        antialias=True,
    )


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


def evaluate_model(model, loss_fn, images, labels, data_aug_fn, loss_type, key):
    """Evaluate model and return accuracy, ECE, NLL."""
    aug_images = data_aug_fn(images, key=None)
    features = extract_features(model, aug_images, key)
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=loss_type)
    return evaluate_classification(logits, labels)


def train_epoch(
    model,
    loss_fn,
    train_images,
    train_labels,
    data_aug_fn,
    batch_size,
    num_update_iters,
    loss_type,
    key,
):
    """Train for one epoch using CAVI updates."""
    n_samples = train_images.shape[0]
    n_batches = n_samples // batch_size
    
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, n_samples)
    
    shuffled_images = train_images[perm]
    shuffled_labels = train_labels[perm]
    
    current_loss_fn = loss_fn
    total_loss = 0.0
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        
        batch_images = shuffled_images[batch_start:batch_end]
        batch_labels = shuffled_labels[batch_start:batch_end]
        
        key, aug_key = jr.split(key)
        aug_images = data_aug_fn(batch_images, key=aug_key)
        
        # Extract features
        features = extract_features(model, aug_images, key)
        
        # CAVI update for Bayesian loss
        if hasattr(current_loss_fn, 'update'):
            current_loss_fn = current_loss_fn.update(
                features, batch_labels, num_iters=num_update_iters
            )
        
        # Compute loss
        loss = current_loss_fn(features, batch_labels, loss_type=loss_type).mean()
        total_loss += loss
    
    avg_loss = total_loss / n_batches
    return current_loss_fn, avg_loss


def main(args):
    if NO_EQUIMO:
        raise ImportError("equimo is required. Install with: pip install equimo>=0.3.0")
    
    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS.get(args.model, {"img_size": 224, "embed_dim": 384})
    
    # Initialize wandb
    if args.enable_wandb and not NO_WANDB:
        wandb.init(
            project="bllarse_alternatives",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "loss_fn": args.loss_fn,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "num_update_iters": args.num_update_iters,
            },
        )
    
    key = jr.PRNGKey(args.seed)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset).with_format("jax")
    
    train_ds = {
        "image": ds["train"][ds_config["img_key"]][:].astype(jnp.float32),
        "label": ds["train"][ds_config["label_key"]][:],
    }
    test_ds = {
        "image": ds["test"][ds_config["img_key"]][:].astype(jnp.float32),
        "label": ds["test"][ds_config["label_key"]][:],
    }
    
    # Resize images
    print(f"Resizing images to {model_config['img_size']}x{model_config['img_size']}")
    train_ds["image"] = resize_images(train_ds["image"], model_config["img_size"])
    test_ds["image"] = resize_images(test_ds["image"], model_config["img_size"])
    
    # Data augmentation
    mean = MEAN_DICT.get(args.dataset, jnp.array([0.485, 0.456, 0.406]))
    std = STD_DICT.get(args.dataset, jnp.array([0.229, 0.224, 0.225]))
    data_aug_fn = partial(augmentdata, mean=mean, std=std)
    
    # Load pretrained backbone
    print(f"Loading pretrained model: {args.model}")
    key, model_key = jr.split(key)
    backbone = get_pretrained_backbone(args.model, model_key)
    
    # Initialize loss function
    key, loss_key = jr.split(key)
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]
    
    if args.loss_fn == "IBProbit":
        loss_fn = IBProbit(embed_dim, num_classes, key=loss_key)
    elif args.loss_fn == "IBPolyaGamma":
        loss_fn = IBPolyaGamma(embed_dim, num_classes, key=loss_key)
    elif args.loss_fn == "CrossEntropy":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
    
    print(f"Loss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    
    # Initial evaluation
    key, eval_key = jr.split(key)
    acc, ece, nll = evaluate_model(
        backbone, loss_fn, test_ds["image"], test_ds["label"],
        data_aug_fn, loss_type=3, key=eval_key
    )
    print(f"Initial: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        key, train_key, eval_key = jr.split(key, 3)
        
        loss_fn, train_loss = train_epoch(
            backbone,
            loss_fn,
            train_ds["image"],
            train_ds["label"],
            data_aug_fn,
            args.batch_size,
            args.num_update_iters,
            loss_type=3,
            key=train_key,
        )
        
        # Evaluate
        acc, ece, nll = evaluate_model(
            backbone, loss_fn, test_ds["image"], test_ds["label"],
            data_aug_fn, loss_type=3, key=eval_key
        )
        
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")
        
        if args.enable_wandb and not NO_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_acc": float(acc),
                "test_ece": float(ece),
                "test_nll": float(nll),
            })
    
    print("Training complete!")
    
    if args.enable_wandb and not NO_WANDB:
        wandb.finish()


def build_argparser():
    parser = argparse.ArgumentParser(description="ViT Classification with Bayesian Last Layer")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="vit_small_patch14_dinov2",
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
        choices=["IBProbit", "IBPolyaGamma", "CrossEntropy"],
        help="Loss function"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, default=16, help="CAVI iterations per batch")
    parser.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing (for CrossEntropy)")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
