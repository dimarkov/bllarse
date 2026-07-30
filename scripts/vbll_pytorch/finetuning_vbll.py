"""VBLL finetuning script for scaling MLPs."""
import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import vbll
from tqdm import tqdm

try:
    import mlflow
    no_mlflow = False
except Exception:
    mlflow = None
    no_mlflow = True

# Allow running this script directly from repo root with `python ...`.
_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_SRC = _THIS_DIR / "src"
_REPO_SRC = _THIS_DIR.parent.parent / "src"
for _path in (_LOCAL_SRC, _REPO_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# --- Monkey Patch for VBLL LowRankNormal bug ---
# Fixes RuntimeError: Expected all tensors to be on the same device ... found cuda:0 and cpu!
from vbll.utils.distributions import LowRankNormal, tp

def patched_logdet_covariance(self):
    # Apply Matrix determinant lemma
    term1 = torch.log(self.cov_diag).sum(-1)
    arg1 = tp(self.cov_factor) @ (self.cov_factor/self.cov_diag.unsqueeze(-1))
    # FIX: ensure eye is on same device as arg1
    term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1], device=arg1.device)).log()
    return term1 + term2

# Apply the patch
LowRankNormal.logdet_covariance = property(patched_logdet_covariance)
# -----------------------------------------------

from vbll_experiments.data import get_dataloaders, get_num_classes
from vbll_experiments.models import create_vbll_model
from vbll_experiments.training import run_training


def build_argparser():
    parser = argparse.ArgumentParser(description="VBLL Finetuning script (PyTorch)")
    
    # Dataset and training
    parser.add_argument("-ds", "--dataset", default="cifar10", type=str,
                        choices=["cifar10", "cifar100"],
                        help="Dataset to use")
    parser.add_argument("--seed", default=137, type=int,
                        help="Random seed")
    parser.add_argument("-e", "--epochs", default=100, type=int,
                        help="Number of training epochs")
    parser.add_argument("-bs", "--batch-size", default=64, type=int,
                        help="Batch size")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="Number of PyTorch dataloader workers")
    
    # Optimizer
    parser.add_argument("-o", "--optimizer", default="adamw", type=str,
                        choices=["adamw", "lion"],
                        help="Optimizer to use")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument("-wd", "--weight-decay", default=1e-4, type=float,
                        help="Weight decay")
    
    # Model architecture
    parser.add_argument("--num-blocks", default=6, type=int,
                        choices=[6, 12],
                        help="Number of blocks/layers in MLP")
    parser.add_argument("--embed-dim", default=512, type=int,
                        choices=[512, 1024],
                        help="Embedding dimension")
    parser.add_argument("--pretrained", default="in21k_cifar", type=str,
                        choices=["in21k", "in21k_cifar"],
                        help="Pretrained checkpoint source")
    
    # Fine-tuning mode
    parser.add_argument("--tune-mode", default="last_layer", type=str,
                        choices=["last_layer", "full_network"],
                        help="Whether to tune only VBLL head or full network")
    
    # Data augmentation
    parser.add_argument("--nodataaug", action="store_true",
                        help="Disable data augmentation")
    
    # VBLL parameters
    parser.add_argument("--prior-scale", default=1.0, type=float,
                        help="VBLL prior scale parameter")
    parser.add_argument("--wishart-scale", default=0.1, type=float,
                        help="VBLL Wishart scale parameter")
    parser.add_argument("--regularization-weight", default=None, type=float,
                        help="VBLL regularization weight (KL weight). Default: 1/N_train")
    parser.add_argument("--parameterization", default="diagonal", type=str,
                        choices=["lowrank", "dense", "diagonal"],
                        help="VBLL covariance parameterization (diagonal=least memory, dense=most)")
    parser.add_argument("--cov-rank", default=2, type=int,
                        help="Rank for lowrank parameterization (higher = more expressive)")
    parser.add_argument("--vbll-type", default="discriminative", type=str,
                        choices=["discriminative", "generative"],
                        help="VBLL classification type (discriminative or generative)")
    parser.add_argument(
        "--return-ood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute OOD scores during forward pass",
    )
    
    # MLflow logging
    parser.add_argument("--enable-mlflow", "--enable_mlflow", action="store_true",
                        help="Enable MLflow logging")
    parser.add_argument("--enable-wandb", "--enable_wandb", action="store_true",
                        help="Deprecated alias for --enable-mlflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", ""),
                        help="MLflow tracking URI (defaults to env MLFLOW_TRACKING_URI)")
    parser.add_argument("--mlflow-experiment", type=str, default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "bllarse"),
                        help="MLflow experiment name")
    parser.add_argument("--experiment-name", default=None, type=str,
                        help="Deprecated alias for --mlflow-experiment")
    parser.add_argument("--group-id", type=str, default="vbll_finetuning",
                        help="Tag runs with a shared MLflow group id")
    parser.add_argument("--uid", type=str, default=None,
                        help="Optional MLflow run name (auto-generated if omitted)")
    
    # Device
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use (cuda, cpu)")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", default=None, type=str,
                        help="Directory containing pretrained scaling MLP checkpoints")
    
    return parser


def _build_run_config(args, regularization_weight: float) -> dict:
    return {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_blocks": args.num_blocks,
        "embed_dim": args.embed_dim,
        "pretrained": args.pretrained,
        "tune_mode": args.tune_mode,
        "prior_scale": args.prior_scale,
        "wishart_scale": args.wishart_scale,
        "regularization_weight": regularization_weight,
        "parameterization": args.parameterization,
        "cov_rank": args.cov_rank,
        "vbll_type": args.vbll_type,
        "return_ood": args.return_ood,
        "nodataaug": args.nodataaug,
    }


def main(args):
    enable_mlflow = args.enable_mlflow
    if args.enable_wandb:
        enable_mlflow = True
        print("[bllarse] --enable-wandb is deprecated; enabling MLflow logging instead.")
    if args.experiment_name:
        print("[bllarse] --experiment-name is deprecated; use --mlflow-experiment.")
        args.mlflow_experiment = args.experiment_name
    if enable_mlflow and no_mlflow:
        print("[bllarse] MLflow is not installed; disabling MLflow logging.")
        enable_mlflow = False

    mlflow_context = nullcontext()
    if enable_mlflow:
        from bllarse.mlflow_utils import load_mlflow_env_defaults

        load_mlflow_env_defaults()
        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        experiment = args.mlflow_experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "bllarse"
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment:
            mlflow.set_experiment(experiment)
        mlflow_context = mlflow.start_run(
            run_name=args.uid or None,
            tags={
                "group_id": args.group_id,
                "optimizer": args.optimizer,
                "tune_mode": args.tune_mode,
                "method": "vbll",
            },
            nested=bool(parent_run_id),
            parent_run_id=parent_run_id,
        )

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    
    # Determine checkpoint name based on pretrained source
    if args.pretrained == "in21k":
        checkpoint = "in21k"
    else:  # in21k_cifar
        checkpoint = f"in21k_{args.dataset}"
    
    # Build architecture string
    architecture = f"B_{args.num_blocks}-Wi_{args.embed_dim}"
    
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Architecture: {architecture}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Tune mode: {args.tune_mode}")
    print(f"Optimizer: {args.optimizer} (lr={args.learning_rate}, wd={args.weight_decay})")
    
    # Create data loaders first (needed for evaluation)
    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        augment=not args.nodataaug,
        num_workers=args.num_workers,
    )
    
    # Load original pretrained backbone for baseline evaluation
    from vbll_experiments.models import load_pretrained_backbone
    backbone, embed_dim = load_pretrained_backbone(
        architecture=architecture,
        checkpoint=checkpoint,
        resolution=64,
        checkpoint_dir=args.checkpoint_dir,
    )
    backbone = backbone.to(device)
    
    # Evaluate original pretrained model (with its trained linear layer)
    # Only if checkpoint matches dataset (e.g. not in21k on CIFAR)
    pre_acc = 0.0
    pre_nll = 0.0
    pre_ece = 0.0
    
    if checkpoint != "in21k":
        print("\nPre-finetuning evaluation (original model):")
        from vbll_experiments.training import compute_ece
        import numpy as np
        backbone.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_nll = 0.0
            all_probs = []
            all_labels = []
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                # Flatten for MLP input
                images_flat = images.flatten(1)
                logits = backbone(images_flat)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_nll += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            pre_acc = correct / total
            pre_nll = total_nll / total
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            pre_ece = compute_ece(all_probs, all_labels)
        print(f"  Accuracy: {pre_acc:.4f}")
        print(f"  NLL: {pre_nll:.4f}")
        print(f"  ECE: {pre_ece:.4f}")
    else:
        print("\nSkipping pre-finetuning evaluation (in21k checkpoint has incompatible head)")
    
    # Determine regularization weight
    if args.regularization_weight is None:
        num_train_samples = len(train_loader.dataset)
        regularization_weight = 1.0 / num_train_samples
        print(f"\nAuto-scaled regularization_weight: {regularization_weight:.2e} (1/{num_train_samples})")
    else:
        regularization_weight = args.regularization_weight
        print(f"\nUsing manual regularization_weight: {regularization_weight}")

    # Now create VBLL model by wrapping the backbone
    print(f"Creating VBLL model ({args.vbll_type}, {args.parameterization}, rank={args.cov_rank})...")
    freeze_backbone = (args.tune_mode == "last_layer")
    model = create_vbll_model(
        architecture=architecture,
        checkpoint=checkpoint,
        num_classes=num_classes,
        regularization_weight=regularization_weight,
        prior_scale=args.prior_scale,
        wishart_scale=args.wishart_scale,
        parameterization=args.parameterization,
        cov_rank=args.cov_rank,
        vbll_type=args.vbll_type,
        return_ood=args.return_ood,
        freeze_backbone=freeze_backbone,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)
    
    # Initialize centroids for Generative VBLL
    if args.vbll_type == "generative":
        print("Initializing generative model centroids (required for convergence)...")
        model.eval()
        num_classes_model = model.vbll_head.mu_mean.shape[0]
        embed_dim = model.vbll_head.mu_mean.shape[1]
        
        centroids = torch.zeros(num_classes_model, embed_dim, device=device)
        counts = torch.zeros(num_classes_model, device=device)
        
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Computing centroids"):
                images = images.to(device)
                labels = labels.to(device)
                features = model.get_features(images)
                for c in range(num_classes_model):
                    mask = (labels == c)
                    if mask.any():
                        centroids[c] += features[mask].sum(0)
                        counts[c] += mask.sum()
        
        centroids = centroids / (counts.unsqueeze(1) + 1e-8)
        model.vbll_head.mu_mean.data.copy_(centroids)
        print("Centroids initialized.")
        model.train() # Set back to train mode

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create parameter groups
    # Notebook recommendation: weight decay should be 0 for VBLL layer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    vbll_params = [p for p in model.vbll_head.parameters() if p.requires_grad]
    
    param_groups = [
        {'params': backbone_params, 'weight_decay': args.weight_decay},
        {'params': vbll_params, 'weight_decay': 0.0}
    ]

    # Create optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.learning_rate,
        )
    elif args.optimizer == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                param_groups,
                lr=args.learning_rate,
            )
        except ImportError:
            raise ImportError("Lion optimizer requires: pip install lion-pytorch")
    
    # Run training
    print(f"\nStarting training for {args.epochs} epochs...")
    with mlflow_context:
        if enable_mlflow:
            mlflow.log_params(_build_run_config(args, regularization_weight))
            if checkpoint != "in21k":
                mlflow.log_metrics({
                    "pre_finetune_accuracy": pre_acc,
                    "pre_finetune_nll": pre_nll,
                    "pre_finetune_ece": pre_ece,
                })

        mlflow_run = mlflow.active_run() if enable_mlflow else None
        history = run_training(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            tune_mode=args.tune_mode,
            mlflow_run=mlflow_run,
        )
        if enable_mlflow:
            try:
                mlflow.log_metrics(
                    {
                        "final_accuracy": history["test_accuracy"][-1],
                        "best_accuracy": max(history["test_accuracy"]),
                        "final_nll": history["test_nll"][-1],
                        "final_ece": history["test_ece"][-1],
                        "final_train_loss": history["train_loss"][-1],
                        "epochs_completed": len(history["train_loss"]),
                    }
                )
            except Exception as exc:
                print(f"[bllarse] WARNING: failed to log final metrics to MLflow: {exc}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best accuracy: {max(history['test_accuracy']):.4f}")
    print(f"Final accuracy: {history['test_accuracy'][-1]:.4f}")
    print(f"Final NLL: {history['test_nll'][-1]:.4f}")
    print(f"Final ECE: {history['test_ece'][-1]:.4f}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
