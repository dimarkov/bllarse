"""
Language Modeling with Bayesian Output Layer

Fine-tunes GPT-style language models with Bayesian output layers
for next-token prediction with calibrated uncertainty.

Dependencies:
    pip install transformers[flax]>=4.35.0

Example usage:
    python scripts/alternatives/language_modeling.py \
        --model gpt2 \
        --dataset wikitext \
        --loss-fn IBProbit \
        --epochs 3 \
        --batch-size 8
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
import math

from functools import partial
from jax import random as jr, config, vmap, nn
from datasets import load_dataset

try:
    import wandb
    NO_WANDB = False
except ImportError:
    NO_WANDB = True

try:
    from transformers import (
        FlaxGPT2LMHeadModel,
        FlaxGPT2Model,
        GPT2Tokenizer,
        AutoTokenizer,
    )
    NO_TRANSFORMERS = False
except ImportError:
    NO_TRANSFORMERS = True
    warnings.warn("transformers[flax] not installed. Run: pip install transformers[flax]")

from bllarse.losses import IBProbit, IBPolyaGamma

from calibration import compute_ece_token_level, compute_nll

config.update("jax_default_matmul_precision", "highest")

# Model configurations
MODEL_CONFIGS = {
    "gpt2": {"embed_dim": 768, "vocab_size": 50257, "max_length": 128},
    "distilgpt2": {"embed_dim": 768, "vocab_size": 50257, "max_length": 128},
}

# Dataset configurations
DATASET_CONFIGS = {
    "wikitext": {"hf_path": "wikitext", "hf_subset": "wikitext-2-v1", "text_key": "text"},
    "ptb": {"hf_path": "ptb_text_only", "hf_subset": None, "text_key": "sentence"},
}


class GPT2Wrapper(eqx.Module):
    """Wrapper for HuggingFace Flax GPT-2 to extract hidden states."""
    
    model: any
    tokenizer: any
    max_length: int
    
    def __init__(self, model_name: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = FlaxGPT2Model.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, texts):
        """Tokenize a batch of texts for language modeling."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
    
    def __call__(self, input_ids, attention_mask):
        """Extract hidden states (for all tokens)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def load_lm_dataset(dataset_name: str, split: str, max_samples: int = None):
    """Load a language modeling dataset."""
    ds_config = DATASET_CONFIGS[dataset_name]
    
    if ds_config["hf_subset"]:
        ds = load_dataset(ds_config["hf_path"], ds_config["hf_subset"], split=split)
    else:
        ds = load_dataset(ds_config["hf_path"], split=split)
    
    texts = ds[ds_config["text_key"]]
    
    # Filter empty texts
    texts = [t for t in texts if len(t.strip()) > 10]
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts


def create_lm_labels(input_ids, pad_token_id):
    """
    Create labels for language modeling (shift by 1).
    Labels are shifted right, and padding positions get -100 (ignored).
    """
    labels = np.roll(input_ids, -1, axis=1)
    labels[:, -1] = pad_token_id  # Last position can't predict next
    
    # Mask padding tokens
    labels = np.where(labels == pad_token_id, -100, labels)
    
    return labels


def compute_perplexity(logits, labels, ignore_index=-100):
    """Compute perplexity from logits and labels."""
    # Flatten
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    
    # Mask valid tokens
    valid_mask = flat_labels != ignore_index
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    if valid_logits.shape[0] == 0:
        return jnp.array(0.0)
    
    # Compute cross-entropy
    log_probs = nn.log_softmax(valid_logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, valid_labels[:, None], axis=-1).squeeze(-1)
    
    # Perplexity = exp(mean NLL)
    return jnp.exp(jnp.mean(nll))


def evaluate_model(model, lm_head, input_ids, attention_mask, labels):
    """Evaluate language model and return perplexity and ECE."""
    hidden_states = model(input_ids, attention_mask)
    hidden_states = jnp.array(hidden_states)
    
    # Apply LM head (simple linear projection)
    # For Bayesian head, this would be the IBProbit layer
    batch_size, seq_len, embed_dim = hidden_states.shape
    
    # Reshape for loss function
    flat_hidden = hidden_states.reshape(-1, embed_dim)
    flat_labels = jnp.array(labels.reshape(-1))
    
    # Valid token mask
    valid_mask = flat_labels != -100
    valid_hidden = flat_hidden[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    if valid_hidden.shape[0] == 0:
        return jnp.array(float('inf')), jnp.array(0.0), jnp.array(0.0)
    
    # Get logits
    _, logits = lm_head(valid_hidden, valid_labels, with_logits=True, loss_type=3)
    
    # Compute metrics
    vocab_size = logits.shape[-1]
    
    # Perplexity
    log_probs = nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, valid_labels[:, None], axis=-1).squeeze(-1)
    ppl = jnp.exp(jnp.mean(nll))
    
    # ECE (using logits directly since they're already for valid tokens)
    predictions = jnp.argmax(logits, axis=-1)
    from tensorflow_probability.substrates.jax.stats import expected_calibration_error as tfp_ece
    ece = tfp_ece(20, logits=logits, labels_true=valid_labels, labels_predicted=predictions)
    
    return ppl, ece, jnp.mean(nll)


def train_epoch(
    model,
    lm_head,
    input_ids,
    attention_mask,
    labels,
    batch_size,
    num_update_iters,
    key,
):
    """Train for one epoch."""
    n_samples = labels.shape[0]
    n_batches = max(1, n_samples // batch_size)
    
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, n_samples)
    perm_np = np.array(perm)
    
    shuffled_ids = input_ids[perm_np]
    shuffled_mask = attention_mask[perm_np]
    shuffled_labels = labels[perm_np]
    
    current_lm_head = lm_head
    total_loss = 0.0
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        
        batch_ids = shuffled_ids[batch_start:batch_end]
        batch_mask = shuffled_mask[batch_start:batch_end]
        batch_labels = shuffled_labels[batch_start:batch_end]
        
        # Extract hidden states
        hidden_states = model(batch_ids, batch_mask)
        hidden_states = jnp.array(hidden_states)
        
        batch_size_actual, seq_len, embed_dim = hidden_states.shape
        
        # Flatten for Bayesian layer
        flat_hidden = hidden_states.reshape(-1, embed_dim)
        flat_labels = jnp.array(batch_labels.reshape(-1))
        
        # Valid token mask
        valid_mask = flat_labels != -100
        valid_hidden = flat_hidden[valid_mask]
        valid_labels = flat_labels[valid_mask]
        
        if valid_hidden.shape[0] == 0:
            continue
        
        # CAVI update
        if hasattr(current_lm_head, 'update'):
            current_lm_head = current_lm_head.update(
                valid_hidden, valid_labels, num_iters=num_update_iters
            )
        
        # Compute loss
        loss = current_lm_head(valid_hidden, valid_labels, loss_type=3).mean()
        total_loss += loss
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return current_lm_head, avg_loss


def main(args):
    if NO_TRANSFORMERS:
        raise ImportError("transformers[flax] is required. Install with: pip install transformers[flax]")
    
    # Validate configs
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.model not in MODEL_CONFIGS:
        warnings.warn(f"Model {args.model} not in configs, using default")
    
    model_config = MODEL_CONFIGS.get(args.model, {"embed_dim": 768, "vocab_size": 50257, "max_length": 128})
    
    # Initialize wandb
    if args.enable_wandb and not NO_WANDB:
        wandb.init(
            project="bllarse_alternatives_lm",
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
    model = GPT2Wrapper(args.model, max_length=model_config["max_length"])
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_texts = load_lm_dataset(args.dataset, "train", max_samples=args.max_samples)
    test_texts = load_lm_dataset(args.dataset, "validation" if args.dataset == "wikitext" else "test", 
                                  max_samples=args.max_samples // 10 if args.max_samples else 1000)
    
    print(f"Tokenizing {len(train_texts)} train and {len(test_texts)} test samples")
    train_tokens = model.tokenize(train_texts)
    test_tokens = model.tokenize(test_texts)
    
    train_ids = np.array(train_tokens["input_ids"])
    train_mask = np.array(train_tokens["attention_mask"])
    test_ids = np.array(test_tokens["input_ids"])
    test_mask = np.array(test_tokens["attention_mask"])
    
    # Create labels
    pad_token_id = model.tokenizer.pad_token_id
    train_labels = create_lm_labels(train_ids, pad_token_id)
    test_labels = create_lm_labels(test_ids, pad_token_id)
    
    # Initialize Bayesian LM head
    key, lm_key = jr.split(key)
    embed_dim = model_config["embed_dim"]
    vocab_size = model_config["vocab_size"]
    
    print(f"Initializing Bayesian LM head: {embed_dim} -> {vocab_size}")
    
    if args.loss_fn == "IBProbit":
        lm_head = IBProbit(embed_dim, vocab_size, key=lm_key)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")
    
    print(f"Loss function: {args.loss_fn}")
    
    # Initial evaluation
    ppl, ece, nll = evaluate_model(model, lm_head, test_ids, test_mask, test_labels)
    print(f"Initial: ppl={ppl:.2f}, ece={ece:.4f}, nll={nll:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        key, train_key = jr.split(key)
        
        lm_head, train_loss = train_epoch(
            model,
            lm_head,
            train_ids,
            train_mask,
            train_labels,
            args.batch_size,
            args.num_update_iters,
            key=train_key,
        )
        
        # Evaluate
        ppl, ece, nll = evaluate_model(model, lm_head, test_ids, test_mask, test_labels)
        
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, ppl={ppl:.2f}, ece={ece:.4f}, nll={nll:.4f}")
        
        if args.enable_wandb and not NO_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_ppl": float(ppl),
                "test_ece": float(ece),
                "test_nll": float(nll),
            })
    
    print("Training complete!")
    
    if args.enable_wandb and not NO_WANDB:
        wandb.finish()


def build_argparser():
    parser = argparse.ArgumentParser(description="Language Modeling with Bayesian Output Layer")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        help="Pretrained GPT model from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use"
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="IBProbit",
        choices=["IBProbit", "IBPolyaGamma"],
        help="Bayesian loss function"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples (for quick testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-update-iters", type=int, default=8, help="CAVI iterations per batch")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
