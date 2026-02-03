"""
Semantic Segmentation with Transformer Backbones

Semantic segmentation using Vision Transformer backbones with Bayesian
per-pixel classification heads.

Note: This is a simplified implementation. Full segmentation pipelines
(like SegFormer) require additional decoder components.

Dependencies:
    pip install equimo>=0.3.0

Example usage:
    python scripts/alternatives/segmentation.py \
        --model vit_small_patch14_dinov2 \
        --dataset pascal_voc \
        --loss-fn IBProbit \
        --epochs 5
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
import numpy as np

from functools import partial
from jax import random as jr, config, vmap, nn
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

from bllarse.losses import IBProbit, IBPolyaGamma

from calibration import compute_ece_segmentation

config.update("jax_default_matmul_precision", "highest")

# Model configurations
EQUIMO_MODELS = {
    "vit_small_patch14_dinov2": {"img_size": 224, "embed_dim": 384, "patch_size": 14},
    "vit_base_patch14_dinov2": {"img_size": 224, "embed_dim": 768, "patch_size": 14},
}

# Dataset configurations
DATASET_CONFIGS = {
    "pascal_voc": {
        "num_classes": 21,  # 20 classes + background
        "ignore_index": 255,
        "hf_path": "scene_parse_150",  # Simplified - real VOC needs separate loading
    },
    "ade20k": {
        "num_classes": 150,
        "ignore_index": 0,
        "hf_path": "scene_parse_150",
    },
}


def get_pretrained_backbone(model_name: str, key):
    """Load pretrained ViT backbone from equimo."""
    if NO_EQUIMO:
        raise ImportError("equimo not installed. Run: pip install equimo>=0.3.0")
    
    equimo_id_map = {
        "vit_small_patch14_dinov2": ("vit", "dinov2_vits14"),
        "vit_base_patch14_dinov2": ("vit", "dinov2_vitb14"),
    }
    
    if model_name not in equimo_id_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    arch_type, model_id = equimo_id_map[model_name]
    model = equimo_load_model(arch_type, model_id)
    return eqx.nn.inference_mode(model, True)


def extract_patch_features(model, images, key, patch_size=14):
    """
    Extract patch-level features for segmentation.
    Returns features of shape (batch, h_patches, w_patches, embed_dim)
    """
    def single_forward(x):
        out = model(x, enable_dropout=False, key=key)
        if isinstance(out, tuple):
            # (cls_token, patch_tokens)
            return out[1]  # Return patch tokens
        return out
    
    patch_features = vmap(single_forward)(images)
    
    # Reshape from (batch, n_patches, embed_dim) to (batch, h, w, embed_dim)
    batch_size, n_patches, embed_dim = patch_features.shape
    h = w = int(np.sqrt(n_patches))
    
    return patch_features.reshape(batch_size, h, w, embed_dim)


def simple_upsample(features, target_size):
    """
    Simple bilinear upsampling of feature maps.
    features: (batch, h, w, c)
    target_size: (target_h, target_w)
    """
    from jax import image as jimg
    
    batch_size, h, w, c = features.shape
    target_h, target_w = target_size
    
    return jimg.resize(
        features,
        (batch_size, target_h, target_w, c),
        method="bilinear",
    )


def compute_segmentation_metrics(logits, labels, ignore_index=255):
    """
    Compute segmentation metrics: mIoU, pixel accuracy, ECE.
    
    logits: (batch, h, w, num_classes)
    labels: (batch, h, w)
    """
    predictions = jnp.argmax(logits, axis=-1)
    
    # Valid mask
    valid_mask = labels != ignore_index
    
    # Pixel accuracy
    correct = (predictions == labels) & valid_mask
    pixel_acc = jnp.sum(correct) / jnp.sum(valid_mask)
    
    # ECE
    ece = compute_ece_segmentation(logits, labels, ignore_index=ignore_index)
    
    # Simplified mIoU (for demonstration)
    num_classes = logits.shape[-1]
    ious = []
    for c in range(num_classes):
        pred_c = predictions == c
        label_c = labels == c
        intersection = jnp.sum((pred_c & label_c) & valid_mask)
        union = jnp.sum((pred_c | label_c) & valid_mask)
        iou = jnp.where(union > 0, intersection / union, 0.0)
        ious.append(iou)
    
    miou = jnp.mean(jnp.array(ious))
    
    return miou, pixel_acc, ece


def create_synthetic_segmentation_data(num_samples, img_size, num_classes, key):
    """
    Create synthetic segmentation data for testing.
    In practice, you would load real datasets like PASCAL VOC.
    """
    key1, key2 = jr.split(key)
    
    # Random images
    images = jr.uniform(key1, (num_samples, img_size, img_size, 3))
    
    # Random labels (simplified)
    labels = jr.randint(key2, (num_samples, img_size, img_size), 0, num_classes)
    
    return images, labels


def train_epoch(
    model,
    seg_head,
    images,
    labels,
    batch_size,
    num_update_iters,
    patch_size,
    key,
    ignore_index=255,
):
    """Train for one epoch on segmentation task."""
    n_samples = images.shape[0]
    n_batches = max(1, n_samples // batch_size)
    
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, n_samples)
    
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]
    
    current_seg_head = seg_head
    total_loss = 0.0
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        
        batch_images = shuffled_images[batch_start:batch_end]
        batch_labels = shuffled_labels[batch_start:batch_end]
        
        key, feat_key = jr.split(key)
        
        # Extract patch features
        patch_features = extract_patch_features(model, batch_images, feat_key, patch_size)
        
        # Upsample to full resolution
        target_size = (batch_labels.shape[1], batch_labels.shape[2])
        upsampled_features = simple_upsample(patch_features, target_size)
        
        # Flatten for Bayesian layer
        batch_size_actual, h, w, embed_dim = upsampled_features.shape
        flat_features = upsampled_features.reshape(-1, embed_dim)
        flat_labels = batch_labels.reshape(-1)
        
        # Filter valid pixels
        valid_mask = flat_labels != ignore_index
        valid_features = flat_features[valid_mask]
        valid_labels = flat_labels[valid_mask]
        
        if valid_features.shape[0] == 0:
            continue
        
        # CAVI update
        if hasattr(current_seg_head, 'update'):
            current_seg_head = current_seg_head.update(
                valid_features, valid_labels, num_iters=num_update_iters
            )
        
        # Compute loss
        loss = current_seg_head(valid_features, valid_labels, loss_type=3).mean()
        total_loss += loss
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return current_seg_head, avg_loss


def evaluate_model(model, seg_head, images, labels, patch_size, ignore_index=255, key=None):
    """Evaluate segmentation model."""
    if key is None:
        key = jr.PRNGKey(0)
    
    # Extract and upsample features
    patch_features = extract_patch_features(model, images, key, patch_size)
    target_size = (labels.shape[1], labels.shape[2])
    upsampled_features = simple_upsample(patch_features, target_size)
    
    # Get logits for all pixels
    batch_size, h, w, embed_dim = upsampled_features.shape
    flat_features = upsampled_features.reshape(-1, embed_dim)
    flat_labels = labels.reshape(-1)
    
    _, logits = seg_head(flat_features, flat_labels, with_logits=True, loss_type=3)
    
    # Reshape logits back to spatial
    num_classes = logits.shape[-1]
    logits_spatial = logits.reshape(batch_size, h, w, num_classes)
    
    return compute_segmentation_metrics(logits_spatial, labels, ignore_index)


def main(args):
    if NO_EQUIMO:
        raise ImportError("equimo is required. Install with: pip install equimo>=0.3.0")
    
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS.get(args.model, {"img_size": 224, "embed_dim": 384, "patch_size": 14})
    
    # Initialize wandb
    if args.enable_wandb and not NO_WANDB:
        wandb.init(
            project="bllarse_alternatives_seg",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "loss_fn": args.loss_fn,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
        )
    
    key = jr.PRNGKey(args.seed)
    
    # Load model
    print(f"Loading pretrained model: {args.model}")
    key, model_key = jr.split(key)
    backbone = get_pretrained_backbone(args.model, model_key)
    
    # Create synthetic data for demonstration
    # In practice, you would load PASCAL VOC or ADE20K
    print("Creating synthetic segmentation data (replace with real dataset loading)")
    key, data_key = jr.split(key)
    train_images, train_labels = create_synthetic_segmentation_data(
        args.num_samples, model_config["img_size"], ds_config["num_classes"], data_key
    )
    key, data_key = jr.split(key)
    test_images, test_labels = create_synthetic_segmentation_data(
        args.num_samples // 10, model_config["img_size"], ds_config["num_classes"], data_key
    )
    
    # Initialize segmentation head
    key, head_key = jr.split(key)
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]
    
    if args.loss_fn == "IBProbit":
        seg_head = IBProbit(embed_dim, num_classes, key=head_key)
    elif args.loss_fn == "IBPolyaGamma":
        seg_head = IBPolyaGamma(embed_dim, num_classes, key=head_key)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
    
    print(f"Loss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    
    # Initial evaluation
    key, eval_key = jr.split(key)
    miou, pixel_acc, ece = evaluate_model(
        backbone, seg_head, test_images, test_labels,
        model_config["patch_size"], ds_config["ignore_index"], eval_key
    )
    print(f"Initial: mIoU={miou:.4f}, pixel_acc={pixel_acc:.4f}, ece={ece:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        key, train_key, eval_key = jr.split(key, 3)
        
        seg_head, train_loss = train_epoch(
            backbone,
            seg_head,
            train_images,
            train_labels,
            args.batch_size,
            args.num_update_iters,
            model_config["patch_size"],
            train_key,
            ds_config["ignore_index"],
        )
        
        # Evaluate
        miou, pixel_acc, ece = evaluate_model(
            backbone, seg_head, test_images, test_labels,
            model_config["patch_size"], ds_config["ignore_index"], eval_key
        )
        
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, mIoU={miou:.4f}, acc={pixel_acc:.4f}, ece={ece:.4f}")
        
        if args.enable_wandb and not NO_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_miou": float(miou),
                "test_pixel_acc": float(pixel_acc),
                "test_ece": float(ece),
            })
    
    print("Training complete!")
    
    if args.enable_wandb and not NO_WANDB:
        wandb.finish()


def build_argparser():
    parser = argparse.ArgumentParser(description="Semantic Segmentation with Bayesian Head")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="vit_small_patch14_dinov2",
        choices=list(EQUIMO_MODELS.keys()),
        help="Pretrained ViT model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal_voc",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset (currently uses synthetic data)"
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="IBProbit",
        choices=["IBProbit", "IBPolyaGamma"],
        help="Bayesian loss function"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Synthetic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, default=8, help="CAVI iterations")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B")
    
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
