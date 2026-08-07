"""
Unified calibration metrics for classification, segmentation, and NLP tasks.

All metrics work with JAX arrays and use the existing TensorFlow Probability
implementation from the bllarse.utils module as reference.
"""

import jax.numpy as jnp
from jax import vmap, nn
from jaxtyping import Array
from typing import Optional, Tuple
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as _tfp_ece,
)


def compute_ece_classification(
    logits: Array,
    labels: Array,
    num_bins: int = 20,
) -> Array:
    """
    Expected Calibration Error for standard classification.
    
    Args:
        logits: Model logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        num_bins: Number of bins for ECE computation
        
    Returns:
        Scalar ECE value
    """
    predictions = jnp.argmax(logits, axis=-1)
    return _tfp_ece(num_bins, logits=logits, labels_true=labels, labels_predicted=predictions)


def compute_ece_segmentation(
    logits: Array,
    labels: Array,
    num_bins: int = 20,
    ignore_index: int = 255,
) -> Array:
    """
    Expected Calibration Error for semantic segmentation (per-pixel classification).
    
    Flattens spatial dimensions and computes ECE across all valid pixels.
    
    Args:
        logits: Model logits of shape (batch_size, num_classes, height, width)
                or (batch_size, height, width, num_classes)
        labels: Ground truth labels of shape (batch_size, height, width)
        num_bins: Number of bins for ECE computation
        ignore_index: Label value to ignore (e.g., 255 for void class)
        
    Returns:
        Scalar ECE value averaged over valid pixels
    """
    # Ensure logits are in (B, H, W, C) format
    if logits.shape[1] < logits.shape[-1]:
        # (B, C, H, W) -> (B, H, W, C)
        logits = jnp.transpose(logits, (0, 2, 3, 1))
    
    batch_size, height, width, num_classes = logits.shape
    
    # Flatten spatial dimensions
    flat_logits = logits.reshape(-1, num_classes)  # (B*H*W, C)
    flat_labels = labels.reshape(-1)  # (B*H*W,)
    
    # Create mask for valid pixels
    valid_mask = flat_labels != ignore_index
    
    # Filter valid pixels
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    if valid_logits.shape[0] == 0:
        return jnp.array(0.0)
    
    predictions = jnp.argmax(valid_logits, axis=-1)
    return _tfp_ece(
        num_bins, 
        logits=valid_logits, 
        labels_true=valid_labels, 
        labels_predicted=predictions
    )


def compute_ece_token_level(
    logits: Array,
    labels: Array,
    attention_mask: Optional[Array] = None,
    num_bins: int = 20,
    ignore_index: int = -100,
) -> Array:
    """
    Expected Calibration Error for token-level NLP tasks (e.g., language modeling, NER).
    
    Flattens sequence dimensions and computes ECE across all valid tokens.
    
    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        labels: Ground truth labels of shape (batch_size, seq_len)
        attention_mask: Optional mask of shape (batch_size, seq_len), 1 for valid tokens
        num_bins: Number of bins for ECE computation
        ignore_index: Label value to ignore (typically -100 for padding)
        
    Returns:
        Scalar ECE value averaged over valid tokens
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten sequence dimensions
    flat_logits = logits.reshape(-1, vocab_size)  # (B*S, V)
    flat_labels = labels.reshape(-1)  # (B*S,)
    
    # Create mask for valid tokens
    valid_mask = flat_labels != ignore_index
    if attention_mask is not None:
        flat_attention = attention_mask.reshape(-1)
        valid_mask = valid_mask & (flat_attention == 1)
    
    # Filter valid tokens
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    if valid_logits.shape[0] == 0:
        return jnp.array(0.0)
    
    predictions = jnp.argmax(valid_logits, axis=-1)
    return _tfp_ece(
        num_bins, 
        logits=valid_logits, 
        labels_true=valid_labels, 
        labels_predicted=predictions
    )


def compute_nll(
    logits: Array,
    labels: Array,
    reduction: str = "mean",
) -> Array:
    """
    Negative log-likelihood (cross-entropy) loss.
    
    Args:
        logits: Model logits of shape (..., num_classes)
        labels: Ground truth labels of shape (...)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        NLL value (scalar if reduction != 'none')
    """
    log_probs = nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(
        log_probs, 
        labels[..., None], 
        axis=-1
    ).squeeze(-1)
    
    if reduction == "mean":
        return jnp.mean(nll)
    elif reduction == "sum":
        return jnp.sum(nll)
    else:
        return nll


def compute_accuracy(
    logits: Array,
    labels: Array,
) -> Array:
    """
    Classification accuracy.
    
    Args:
        logits: Model logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        
    Returns:
        Accuracy as a scalar in [0, 1]
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def evaluate_classification(
    logits: Array,
    labels: Array,
    num_bins: int = 20,
) -> Tuple[Array, Array, Array]:
    """
    Evaluate classification with accuracy, ECE, and NLL.
    
    Args:
        logits: Model logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        num_bins: Number of bins for ECE
        
    Returns:
        Tuple of (accuracy, ece, nll)
    """
    acc = compute_accuracy(logits, labels)
    ece = compute_ece_classification(logits, labels, num_bins)
    nll = compute_nll(logits, labels)
    return acc, ece, nll
