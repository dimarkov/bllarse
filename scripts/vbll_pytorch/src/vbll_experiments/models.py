"""Model utilities for VBLL experiments with scaling MLPs."""
import os
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

import vbll

# Add vendored scaling_mlps to path
_SCALING_MLPS_PATH = Path(__file__).parent.parent.parent / "scaling_mlps"
if str(_SCALING_MLPS_PATH) not in sys.path:
    sys.path.insert(0, str(_SCALING_MLPS_PATH))

from models.networks import get_model as scaling_get_model, model_list


# Checkpoint download URLs (from scaling_mlps Google Drive)
CHECKPOINT_URLS = {
    "B_6-Wi_512": {
        "in21k": "https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW",
    },
    "B_12-Wi_1024": {
        "in21k": "https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW",
    },
}


class VBLLFinetunedMLP(nn.Module):
    """Pretrained scaling MLP backbone with VBLL classification head.
    
    The backbone is loaded from scaling_mlps checkpoints and the final
    classification layer is replaced with a VBLL head for Bayesian
    uncertainty estimation.
    
    Supports both discriminative (DiscClassification) and generative
    (GenClassification) VBLL heads.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        num_classes: int,
        regularization_weight: float = 1.0,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        parameterization: str = "lowrank",
        cov_rank: int = 2,
        vbll_type: str = "discriminative",
        return_ood: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.vbll_type = vbll_type
        
        # Select VBLL classification head type
        vbll_class = vbll.DiscClassification if vbll_type == "discriminative" else vbll.GenClassification
        
        # Create VBLL classification head
        if vbll_type == "generative":
            if parameterization != "diagonal":
                print(f"Warning: GenClassification currently only supports 'diagonal' parameterization. Forcing parameterization='diagonal' (was '{parameterization}').")
                parameterization = "diagonal"
            # GenClassification does not accept cov_rank
            self.vbll_head = vbll_class(
                in_features=input_dim,
                out_features=num_classes,
                regularization_weight=regularization_weight,
                parameterization=parameterization,
                prior_scale=prior_scale,
                wishart_scale=wishart_scale,
                return_ood=return_ood,
            )
        else:
            # DiscClassification accepts cov_rank
            self.vbll_head = vbll_class(
                in_features=input_dim,
                out_features=num_classes,
                regularization_weight=regularization_weight,
                parameterization=parameterization,
                prior_scale=prior_scale,
                wishart_scale=wishart_scale,
                return_ood=return_ood,
                cov_rank=cov_rank,
            )
    
    def forward(self, x: torch.Tensor) -> vbll.DiscClassification:
        """Forward pass returning VBLL output distribution."""
        features = self.get_features(x)
        return self.vbll_head(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone before final linear layer.
        
        The scaling_mlps BottleneckMLP has structure:
        linear_in -> [blocks with layernorms and residuals] -> linear_out
        We intercept before linear_out.
        """
        # Flatten input if needed (scaling_mlps expects flattened input)
        if x.dim() == 4:  # (B, C, H, W)
            x = x.flatten(1)
        
        # Run through linear_in and blocks (matching BottleneckMLP.forward)
        x = self.backbone.linear_in(x)
        for block, norm in zip(self.backbone.blocks, self.backbone.layernorms):
            x = x + block(norm(x))
        
        return x


def load_pretrained_backbone(
    architecture: str = "B_6-Wi_512",
    checkpoint: Literal["in21k", "in21k_cifar10", "in21k_cifar100"] = "in21k",
    resolution: int = 64,
    checkpoint_dir: str = None,
) -> tuple[nn.Module, int]:
    """Load pretrained scaling MLP backbone.
    
    Args:
        architecture: Model architecture (e.g., "B_6-Wi_512", "B_12-Wi_1024")
        checkpoint: Pretrained checkpoint to load
        resolution: Input image resolution
        checkpoint_dir: Directory containing checkpoints (passed to scaling_mlps)
    
    Returns:
        Tuple of (backbone model, embedding dimension)
    """
    # Determine num_classes based on checkpoint for model init
    if checkpoint == "in21k":
        num_classes = 11230  # ImageNet21k classes
    elif checkpoint == "in21k_cifar10":
        num_classes = 10
    elif checkpoint == "in21k_cifar100":
        num_classes = 100
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    
    # Parse embedding dimension from architecture string
    # Format: "B_{num_blocks}-Wi_{embed_dim}"
    embed_dim = int(architecture.split("-Wi_")[1])
    
    # scaling_mlps checkpoint naming:
    # - For ImageNet21k pretrained: "in21k"
    # - For finetuned checkpoints: "in21k_cifar10" (needs to pass checkpoint with in21k prefix)
    # The BottleneckMLP.load() method constructs: self.name + '_' + checkpoint
    # So for "B_6-Wi_512_res_64" with checkpoint "in21k_cifar10" it looks up:
    # "B_6-Wi_512_res_64_in21k_cifar10" in default_checkpoints
    
    if checkpoint_dir:
        # Use custom checkpoint path
        model = model_list[architecture](
            dim_in=resolution**2 * 3,
            dim_out=num_classes,
            checkpoint=None,  # Don't auto-load, we'll load manually
        )
        # Try to load checkpoint manually
        ckpt_pattern = f"{architecture}_res_{resolution}_{checkpoint}"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_pattern, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random weights")
    else:
        # Let scaling_mlps handle checkpoint downloading
        # Pass the checkpoint name as-is (e.g., "in21k" or "in21k_cifar10")
        # scaling_mlps will construct: {architecture}_res_{resolution}_{checkpoint}
        try:
            model = scaling_get_model(
                architecture=architecture,
                checkpoint=checkpoint,  # Pass full checkpoint name
                resolution=resolution,
                num_classes=num_classes,
            )
            print(f"Loaded scaling_mlps model with checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}), using random weights")
            model = model_list[architecture](
                dim_in=resolution**2 * 3,
                dim_out=num_classes,
                checkpoint=None,
            )
    
    return model, embed_dim


def create_vbll_model(
    architecture: str = "B_6-Wi_512",
    checkpoint: Literal["in21k", "in21k_cifar10", "in21k_cifar100"] = "in21k",
    num_classes: int = 10,
    resolution: int = 64,
    regularization_weight: float = 1.0,
    prior_scale: float = 1.0,
    wishart_scale: float = 0.1,
    parameterization: str = "lowrank",
    cov_rank: int = 2,
    vbll_type: str = "discriminative",
    return_ood: bool = True,
    freeze_backbone: bool = False,
    checkpoint_dir: str = None,
) -> VBLLFinetunedMLP:
    """Create a VBLL finetuned MLP model.
    
    Args:
        architecture: Model architecture
        checkpoint: Pretrained checkpoint
        num_classes: Number of output classes
        resolution: Input image resolution
        regularization_weight: VBLL regularization weight (KL weight)
        prior_scale: VBLL prior scale parameter
        wishart_scale: VBLL Wishart scale parameter
        parameterization: VBLL parameterization ('lowrank', 'dense', or 'diagonal')
        cov_rank: Rank for lowrank parameterization
        vbll_type: 'discriminative' or 'generative' classification head
        return_ood: If True, compute OOD scores during forward pass
        freeze_backbone: If True, freeze backbone weights
        checkpoint_dir: Directory containing pretrained checkpoints
    
    Returns:
        VBLLFinetunedMLP model
    """
    backbone, embed_dim = load_pretrained_backbone(
        architecture, checkpoint, resolution, checkpoint_dir
    )
    
    model = VBLLFinetunedMLP(
        backbone=backbone,
        input_dim=embed_dim,
        num_classes=num_classes,
        regularization_weight=regularization_weight,
        prior_scale=prior_scale,
        wishart_scale=wishart_scale,
        parameterization=parameterization,
        cov_rank=cov_rank,
        vbll_type=vbll_type,
        return_ood=return_ood,
    )
    
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    return model
