"""The framework-neutral half of the model and dataset registry.

Kept apart from `scripts/dinov3/vit_classification.py`, which imports jax at
module scope, because the VBLL fine-tuning arm needs the same constants from a
PyTorch environment that deliberately has no jax in it (jax[cuda] and torch
each ship a full CUDA stack and fight over the device).

Nothing here is JAX-specific: they are dataset coordinates, class counts,
embedding widths and normalisation constants. `vit_classification` re-exports
every name, so `from vit_classification import DATASET_CONFIGS` call sites
work and there is exactly ONE definition of each. Separate copies per arm
would drift silently -- and a `num_classes` or a `test_split` that disagrees
between arms is not a crash, it is a quietly incomparable result.
"""

import numpy as np

EQUIMO_MODELS = {
    # DINOv3 models (ViT with LayerNorm only) — expect CHW input
    "dinov3_small": {"img_size": 224, "embed_dim": 384, "channels_first": True},
    "dinov3_big": {"img_size": 224, "embed_dim": 768, "channels_first": True},
    "dinov3_large": {"img_size": 224, "embed_dim": 1024, "channels_first": True},
    "dinov3_huge": {"img_size": 224, "embed_dim": 1280, "channels_first": True},
    "dinov3_max": {"img_size": 224, "embed_dim": 4096, "channels_first": True},
}

# Dataset-specific input sizes for DINOv3 models (patch_size=16, so size must be divisible by 16).
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

# The ImageNet statistics every DINOv3 path normalises with. Both arms read
# them from here, so the JAX and PyTorch trunks cannot be fed differently
# scaled inputs -- which would show up as a trunk disagreement in
# scripts/finetune/trunk_parity.py and be blamed on the implementations.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
