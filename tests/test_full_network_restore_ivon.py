import argparse
import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from blrax.optim import ivon
from bllarse.losses import IBProbit
from bllarse.utils import load_checkpoint_bundle
from mlpox.load_models import load_model


# Known dataset metadata needed to reconstruct training setup.
_DATASET_SIZES = {
    "cifar10": 50_000,
    "cifar100": 50_000,
}

_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}


def _coerce_value(value: str) -> Any:
    """Best-effort conversion of W&B config scalar values."""
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_wandb_config(config_path: Path) -> Dict[str, Any]:
    """Lightweight parser for W&B offline config.yaml files."""
    cfg: Dict[str, Any] = {}
    current_key: Optional[str] = None

    with config_path.open("r") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line:
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0]
                current_key = key if key and not key.startswith("_") else None
                continue
            if current_key is None:
                continue
            stripped = line.lstrip()
            if stripped.startswith("value:"):
                value = stripped.split(":", 1)[1]
                cfg[current_key] = _coerce_value(value)
                current_key = None
    return cfg


def _resolve_run_dir(run_dir: str) -> Path:
    """Accept absolute/relative run_dir or just the W&B run name."""
    path = Path(run_dir)
    if path.exists():
        return path.resolve()
    wandb_path = Path("wandb") / run_dir
    if wandb_path.exists():
        return wandb_path.resolve()
    raise FileNotFoundError(f"Could not find run directory at '{run_dir}' or '{wandb_path}'.")


def _select_checkpoint(run_path: Path, checkpoint: Optional[str]) -> Path:
    files_dir = run_path / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"Expected '{files_dir}' to exist inside the run directory.")

    if checkpoint:
        ckpt_path = files_dir / checkpoint
        if ckpt_path.exists():
            return ckpt_path
        # allow passing bare epoch number or epoch suffix
        pattern = f"ivon_epoch_{checkpoint}.eqx"
        ckpt_path = files_dir / pattern
        if ckpt_path.exists():
            return ckpt_path
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' not found inside '{files_dir}'.")

    matches = sorted(glob.glob(str(files_dir / "ivon_epoch_*.eqx")))
    if not matches:
        raise FileNotFoundError(f"No 'ivon_epoch_*.eqx' checkpoints found in '{files_dir}'.")
    return Path(matches[-1])


def _build_backbone(config: Dict[str, Any]) -> Tuple[eqx.Module, str, int]:
    dataset = str(config.get("dataset"))
    embed_dim = int(config.get("embed_dim"))
    num_blocks = int(config.get("num_blocks"))
    pretrained_variant = str(config.get("pretrained", "in21k"))

    if dataset not in _NUM_CLASSES:
        raise KeyError(f"Unsupported dataset '{dataset}'. Update dataset metadata mappings.")

    model_name = f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k"
    if pretrained_variant == "in21k_cifar":
        model_name = f"{model_name}_{dataset}"

    model = load_model(model_name)
    model = eqx.nn.inference_mode(model, True)
    # Training removes the classification head and replaces it with an identity module.
    backbone = eqx.tree_at(lambda m: m.fc, model, eqx.nn.Identity())
    return backbone, dataset, embed_dim


def _build_ibprobit_head(embed_dim: int, dataset: str) -> IBProbit:
    num_classes = _NUM_CLASSES[dataset]
    key = jr.PRNGKey(0)
    return IBProbit(embed_dim, num_classes, key=key)


def _rebuild_optimizer(config: Dict[str, Any], dataset: str):
    optimizer_name = str(config.get("optimizer"))
    if optimizer_name.lower() != "ivon":
        raise NotImplementedError("This helper currently supports checkpoints from the IVON optimizer only.")

    datasize = _DATASET_SIZES.get(dataset)
    if datasize is None:
        raise KeyError(f"Dataset size for '{dataset}' unknown. Please extend '_DATASET_SIZES'.")

    num_epochs = int(config.get("epochs"))
    batch_size = int(config.get("batch_size"))
    num_iters = max(1, (num_epochs * datasize) // max(1, batch_size))

    lr_conf = {
        "init_value": 1e-3,
        "peak_value": 2e-2,
        "end_value": 1e-4,
        "decay_steps": num_iters,
        "warmup_steps": max(1, num_iters // 10),
    }
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(**lr_conf)

    ivon_conf = {
        "weight_decay": float(config.get("ivon_weight_decay", 1e-6)),
        "hess_init": float(config.get("ivon_hess_init", 1.0)),
        "clip_radius": 1e3,
        "ess": datasize,
    }
    return ivon(lr_schedule, **ivon_conf)


def restore_full_network_checkpoint(run_dir: str, checkpoint: Optional[str] = None):
    """Load a full-network IVON checkpoint and run a dummy forward pass."""
    run_path = _resolve_run_dir(run_dir)
    cfg_path = run_path / "files" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"W&B config file not found at '{cfg_path}'.")

    config = _parse_wandb_config(cfg_path)
    model_like, dataset, embed_dim = _build_backbone(config)
    bayes_head_like = _build_ibprobit_head(embed_dim, dataset)

    ckpt_path = _select_checkpoint(run_path, checkpoint)
    optim = _rebuild_optimizer(config, dataset)

    restored_models, restored_opt_state = load_checkpoint_bundle(
        ckpt_path,
        model_likes={"backbone": model_like, "bayes_head": bayes_head_like},
        optim=optim,
        opt_target="backbone",
    )
    restored_backbone = restored_models["backbone"]
    restored_head = restored_models["bayes_head"]

    img_size = 64  # Full-network training uses 64x64 inputs.
    dummy_batch = jnp.zeros((2, img_size, img_size, 3), dtype=jnp.float32)
    feats = jax.vmap(restored_backbone)(dummy_batch)

    dummy_labels = jnp.zeros((feats.shape[0],), dtype=jnp.int32)
    _, logits = restored_head(feats, dummy_labels, with_logits=True)

    num_classes = _NUM_CLASSES[dataset]
    if logits.ndim == 0:
        raise ValueError(f"Model output is scalar; expected classification logits with dim {num_classes}.")
    assert logits.shape[-1] == num_classes, (
        f"Expected logits dimension {num_classes}, got {logits.shape}."
    )

    return logits, restored_opt_state, ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Restore a full-network IVON checkpoint and run a forward pass."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the W&B run directory (e.g. 'wandb/run-YYYYMMDD_HHMMSS-uid').",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint filename or epoch suffix (defaults to the latest).",
    )
    args = parser.parse_args()

    logits, _, ckpt_path = restore_full_network_checkpoint(args.run_dir, args.checkpoint)
    print(f"Loaded checkpoint '{ckpt_path}'.")
    print(f"Forward-pass logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
