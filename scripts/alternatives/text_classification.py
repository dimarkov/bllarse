"""
NLP Text Classification with Flax Transformers

Fine-tunes BERT/DistilBERT models from HuggingFace on text classification tasks
using Bayesian last-layer methods (IBProbit).

Dependencies:
    pip install transformers[flax]>=4.35.0

Example usage:
    python scripts/alternatives/text_classification.py \
        --model distilbert-base-uncased \
        --dataset sst2 \
        --loss-fn IBProbit \
        --epochs 5 \
        --batch-size 32
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
from jax import random as jr, config, vmap
from datasets import load_dataset

try:
    import wandb
    NO_WANDB = False
except ImportError:
    NO_WANDB = True

try:
    from transformers import (
        FlaxAutoModel,
        AutoTokenizer,
        FlaxDistilBertModel,
        FlaxBertModel,
    )
    NO_TRANSFORMERS = False
except ImportError:
    NO_TRANSFORMERS = True
    warnings.warn("transformers[flax] not installed. Run: pip install transformers[flax]")

from bllarse.losses import IBProbit, CrossEntropy

from calibration import evaluate_classification

# Model configurations
MODEL_CONFIGS = {
    "distilbert-base-uncased": {"embed_dim": 768, "max_length": 128},
    "google/bert_uncased_L-4_H-256_A-4": {"embed_dim": 256, "max_length": 128},  # TinyBERT
    "google/bert_uncased_L-2_H-128_A-2": {"embed_dim": 128, "max_length": 128},  # Smaller TinyBERT
    "prajjwal1/bert-tiny": {"embed_dim": 128, "max_length": 128},
}

# Dataset configurations
DATASET_CONFIGS = {
    "sst2": {"num_classes": 2, "text_key": "sentence", "label_key": "label", "hf_path": "glue", "hf_subset": "sst2"},
    "imdb": {"num_classes": 2, "text_key": "text", "label_key": "label", "hf_path": "imdb", "hf_subset": None},
    "ag_news": {"num_classes": 4, "text_key": "text", "label_key": "label", "hf_path": "ag_news", "hf_subset": None},
}


class FlaxBertWrapper(eqx.Module):
    """Wrapper for HuggingFace Flax models to work with Equinox."""
    
    model: any
    tokenizer: any
    max_length: int
    
    def __init__(self, model_name: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FlaxAutoModel.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, texts):
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
    
    def __call__(self, input_ids, attention_mask):
        """Extract [CLS] token embeddings."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


def load_text_dataset(dataset_name: str, split: str):
    """Load a text classification dataset."""
    ds_config = DATASET_CONFIGS[dataset_name]
    
    if ds_config["hf_subset"]:
        ds = load_dataset(ds_config["hf_path"], ds_config["hf_subset"], split=split)
    else:
        ds = load_dataset(ds_config["hf_path"], split=split)
    
    # Convert to plain Python list[str] for tokenizer compatibility
    texts = list(ds[ds_config["text_key"]])
    labels = np.array(ds[ds_config["label_key"]])
    
    return texts, labels


def extract_features(model, input_ids, attention_mask):
    """Extract CLS embeddings from transformer model."""
    return model(input_ids, attention_mask)


def evaluate_model(model, loss_fn, input_ids, attention_mask, labels, loss_type):
    """Evaluate model and return accuracy, ECE, NLL."""
    features = extract_features(model, input_ids, attention_mask)
    features = jnp.array(features)  # Convert to JAX array
    
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=loss_type)
    return evaluate_classification(logits, labels)


def train_epoch(
    model,
    loss_fn,
    input_ids,
    attention_mask,
    labels,
    batch_size,
    num_update_iters,
    loss_type,
    key,
):
    """Train for one epoch using CAVI updates."""
    n_samples = labels.shape[0]
    n_batches = n_samples // batch_size
    
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, n_samples)
    perm_np = np.array(perm)  # Convert to numpy for indexing
    
    shuffled_ids = input_ids[perm_np]
    shuffled_mask = attention_mask[perm_np]
    shuffled_labels = labels[perm_np]
    
    current_loss_fn = loss_fn
    total_loss = 0.0
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        
        batch_ids = shuffled_ids[batch_start:batch_end]
        batch_mask = shuffled_mask[batch_start:batch_end]
        batch_labels = jnp.array(shuffled_labels[batch_start:batch_end])
        
        # Extract features
        features = extract_features(model, batch_ids, batch_mask)
        features = jnp.array(features)
        
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
    if NO_TRANSFORMERS:
        raise ImportError("transformers[flax] is required. Install with: pip install transformers[flax]")
    
    # Validate configs
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.model not in MODEL_CONFIGS:
        warnings.warn(f"Model {args.model} not in configs, using default embed_dim=768")
    
    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = MODEL_CONFIGS.get(args.model, {"embed_dim": 768, "max_length": 128})
    
    # Initialize wandb
    if args.enable_wandb and not NO_WANDB:
        wandb.init(
            project="bllarse_alternatives_nlp",
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
    
    # Load model
    print(f"Loading model: {args.model}")
    model = FlaxBertWrapper(args.model, max_length=model_config["max_length"])
    
    # Load and tokenize dataset
    print(f"Loading dataset: {args.dataset}")
    train_texts, train_labels = load_text_dataset(args.dataset, "train")
    
    # Handle different test split names
    try:
        test_texts, test_labels = load_text_dataset(args.dataset, "test")
    except ValueError:
        test_texts, test_labels = load_text_dataset(args.dataset, "validation")
    
    print(f"Tokenizing {len(train_texts)} train and {len(test_texts)} test samples")
    train_tokens = model.tokenize(train_texts)
    test_tokens = model.tokenize(test_texts)
    
    train_ids = np.array(train_tokens["input_ids"])
    train_mask = np.array(train_tokens["attention_mask"])
    test_ids = np.array(test_tokens["input_ids"])
    test_mask = np.array(test_tokens["attention_mask"])
    
    # Initialize loss function
    key, loss_key = jr.split(key)
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]
    
    if args.loss_fn == "IBProbit":
        loss_fn = IBProbit(embed_dim, num_classes, key=loss_key)
    elif args.loss_fn == "CrossEntropy":
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
    
    print(f"Loss function: {args.loss_fn}")
    print(f"Embedding dim: {embed_dim}, Classes: {num_classes}")
    
    # Initial evaluation
    test_labels_jax = jnp.array(test_labels)
    acc, ece, nll = evaluate_model(
        model, loss_fn, test_ids, test_mask, test_labels_jax, loss_type=3
    )
    print(f"Initial: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        key, train_key = jr.split(key)
        
        loss_fn, train_loss = train_epoch(
            model,
            loss_fn,
            train_ids,
            train_mask,
            train_labels,
            args.batch_size,
            args.num_update_iters,
            loss_type=3,
            key=train_key,
        )
        
        # Evaluate
        acc, ece, nll = evaluate_model(
            model, loss_fn, test_ids, test_mask, test_labels_jax, loss_type=3
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
    parser = argparse.ArgumentParser(description="Text Classification with Bayesian Last Layer")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="distilbert-base-uncased",
        help="Pretrained transformer model from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
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
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
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
