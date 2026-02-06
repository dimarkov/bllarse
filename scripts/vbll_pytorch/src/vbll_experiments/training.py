"""Training utilities for VBLL experiments."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities of shape (N, C)
        labels: True labels of shape (N,)
        n_bins: Number of calibration bins
    
    Returns:
        ECE value
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)
    
    return ece


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: VBLL model
        loader: Data loader
        device: Device to run evaluation on
    
    Returns:
        Dictionary with 'accuracy', 'nll', 'ece' metrics
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    total_nll = 0.0
    n_samples = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        
        # Get predictive distribution from VBLL
        # Use .probs property which handles differences between Disc/Gen models
        if hasattr(output.predictive, 'probs'):
            probs = output.predictive.probs
        else:
            # Fallback for older VBLL versions or unexpected return types
            logits = output.predictive.logits
            probs = F.softmax(logits, dim=-1)
            
        # Compute NLL directly from probabilities (consistent with Acc/ECE)
        # Add epsilon to avoid log(0) = -inf
        log_probs = torch.log(probs + 1e-12)
        nll = F.nll_loss(log_probs, labels, reduction='sum')
        total_nll += nll.item()
            
        n_samples += labels.size(0)
        
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    accuracy = (all_probs.argmax(axis=1) == all_labels).mean()
    nll = total_nll / n_samples
    ece = compute_ece(all_probs, all_labels)
    
    metrics = {"accuracy": accuracy, "nll": nll, "ece": ece}
    
    # Check if we have OOD scores (requires collection during loop if we implemented it)
    # Since we didn't collect them in the loop above, let's just return what we have
    # Implementing full OOD evaluation would require a separate OOD dataset
    
    return metrics


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tune_mode: str = "full_network",
) -> float:
    """Train model for one epoch.
    
    Args:
        model: VBLL model
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        tune_mode: 'last_layer' or 'full_network'
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    
    # Set backbone to eval mode if only tuning last layer
    if tune_mode == "last_layer":
        model.backbone.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(images)
        
        # VBLL loss = NLL + KL divergence
        # The output object has train_loss_fn() method that computes the ELBO
        loss = output.train_loss_fn(labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    tune_mode: str = "full_network",
    mlflow_run=None,
) -> dict[str, list]:
    """Run full training loop.
    
    Args:
        model: VBLL model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        tune_mode: 'last_layer' or 'full_network'
        mlflow_run: Optional MLflow run for logging
    
    Returns:
        Dictionary with training history
    """
    history = {
        "train_loss": [],
        "test_accuracy": [],
        "test_nll": [],
        "test_ece": [],
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, tune_mode)
        
        # Evaluate
        metrics = evaluate(model, test_loader, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["test_accuracy"].append(metrics["accuracy"])
        history["test_nll"].append(metrics["nll"])
        history["test_ece"].append(metrics["ece"])
        
        # Log to MLflow if available
        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "test_accuracy": metrics["accuracy"],
                "test_nll": metrics["nll"],
                "test_ece": metrics["ece"],
            }, step=epoch)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"NLL: {metrics['nll']:.4f} | "
            f"ECE: {metrics['ece']:.4f}"
        )
    
    return history
