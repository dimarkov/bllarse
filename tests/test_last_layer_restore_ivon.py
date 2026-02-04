import argparse
import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

from blrax.optim import ivon
from bllarse.layers import LastLayer
from bllarse.utils import load_ivon_checkpoint
from mlpox.load_models import load_model


_DATASET_SIZES = {
    "cifar10": 50_000,
    "cifar100": 50_000,
}

_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}


def _coerce_value(value: str) -> Any:
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
    """Legacy parser for W&B offline config.yaml files."""
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


def _get_mlflow_client(tracking_uri: Optional[str]) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _parse_mlflow_params(client: MlflowClient, run_id: str) -> Dict[str, Any]:
    run = client.get_run(run_id)
    return {k: _coerce_value(v) for k, v in run.data.params.items()}


def _normalize_checkpoint_name(checkpoint: str) -> str:
    name = checkpoint.strip()
    if name.endswith(".eqx"):
        return name
    if name.startswith("ivon_epoch_"):
        return f"{name}.eqx"
    if name.isdigit():
        return f"ivon_epoch_{name}.eqx"
    return name


def _list_mlflow_checkpoints(client: MlflowClient, run_id: str, artifact_root: str) -> list[str]:
    infos = client.list_artifacts(run_id, artifact_root)
    paths = [info.path for info in infos if info.path.endswith(".eqx")]
    return paths


def _select_latest_checkpoint(paths: list[str]) -> str:
    def _epoch_from_path(path: str) -> int:
        name = Path(path).name
        if name.startswith("ivon_epoch_") and name.endswith(".eqx"):
            suffix = name[len("ivon_epoch_"):-4]
            try:
                return int(suffix)
            except ValueError:
                return -1
        return -1

    scored = [(path, _epoch_from_path(path)) for path in paths]
    scored.sort(key=lambda item: item[1])
    return scored[-1][0]


def _select_mlflow_checkpoint(
    client: MlflowClient,
    run_id: str,
    artifact_root: str,
    checkpoint: Optional[str],
) -> str:
    if checkpoint:
        name = _normalize_checkpoint_name(checkpoint)
        if artifact_root:
            return f"{artifact_root}/{name}" if not name.startswith(f"{artifact_root}/") else name
        return name

    paths = _list_mlflow_checkpoints(client, run_id, artifact_root)
    if not paths:
        raise FileNotFoundError(f"No 'ivon_epoch_*.eqx' checkpoints found in MLflow under '{artifact_root}'.")
    return _select_latest_checkpoint(paths)


def _download_mlflow_checkpoint(run_id: str, artifact_path: str) -> Path:
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    return Path(local_path)


def _resolve_run_dir(run_dir: str) -> Path:
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
        epoch_pattern = files_dir / f"ivon_epoch_{checkpoint}.eqx"
        if epoch_pattern.exists():
            return epoch_pattern
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' not found inside '{files_dir}'.")
    matches = sorted(glob.glob(str(files_dir / "ivon_epoch_*.eqx")))
    if not matches:
        raise FileNotFoundError(f"No 'ivon_epoch_*.eqx' checkpoints found in '{files_dir}'.")
    return Path(matches[-1])


def _build_backbone(config: Dict[str, Any]) -> Tuple[eqx.Module, str, int]:
    dataset = str(config.get("dataset"))
    embed_dim = int(config.get("embed_dim"))
    num_blocks = int(config.get("num_blocks"))
    pretrained_variant = str(config.get("pretrained", "in21k_cifar"))

    model_name = f"B_{num_blocks}-Wi_{embed_dim}_res_64_in21k"
    if pretrained_variant == "in21k_cifar":
        model_name = f"{model_name}_{dataset}"

    backbone = load_model(model_name)
    backbone = eqx.nn.inference_mode(backbone, True)
    return backbone, dataset, embed_dim


def _build_last_layer(embed_dim: int, dataset: str) -> LastLayer:
    if dataset not in _NUM_CLASSES:
        raise KeyError(f"Unknown dataset '{dataset}'. Please extend _NUM_CLASSES.")
    num_classes = _NUM_CLASSES[dataset]
    key = jr.PRNGKey(0)
    linear = eqx.nn.Linear(embed_dim, num_classes, key=key)
    return LastLayer(linear)


def _rebuild_optimizer(config: Dict[str, Any], dataset: str):
    if str(config.get("optimizer", "")).lower() != "ivon":
        raise NotImplementedError("This restore helper currently supports IVON checkpoints only.")

    datasize = _DATASET_SIZES.get(dataset)
    if datasize is None:
        raise KeyError(f"Dataset size for '{dataset}' unknown. Please extend _DATASET_SIZES.")

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


def restore_last_layer_checkpoint(
    *,
    run_id: Optional[str] = None,
    run_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    artifact_root: str = "checkpoints",
):
    if run_id:
        client = _get_mlflow_client(tracking_uri)
        config = _parse_mlflow_params(client, run_id)
        backbone, dataset, embed_dim = _build_backbone(config)
        last_layer_like = _build_last_layer(embed_dim, dataset)

        artifact_path = _select_mlflow_checkpoint(client, run_id, artifact_root, checkpoint)
        ckpt_path = _download_mlflow_checkpoint(run_id, artifact_path)
    else:
        if not run_dir:
            raise ValueError("Either run_id (MLflow) or run_dir (legacy) must be provided.")
        run_path = _resolve_run_dir(run_dir)
        cfg_path = run_path / "files" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Legacy W&B config file not found at '{cfg_path}'.")

        config = _parse_wandb_config(cfg_path)
        backbone, dataset, embed_dim = _build_backbone(config)
        last_layer_like = _build_last_layer(embed_dim, dataset)

        ckpt_path = _select_checkpoint(run_path, checkpoint)

    optim = _rebuild_optimizer(config, dataset)

    restored_last_layer, restored_opt_state = load_ivon_checkpoint(
        ckpt_path, last_layer_like, optim
    )

    dummy_image = jnp.zeros((64, 64, 3), dtype=jnp.float32)
    logits = restored_last_layer(backbone, dummy_image)

    num_classes = _NUM_CLASSES[dataset]
    if logits.shape[-1] != num_classes:
        raise ValueError(f"Expected logits dimension {num_classes}, got {logits.shape}.")

    return logits, restored_opt_state, ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Restore a last-layer IVON checkpoint and run a forward pass."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-id",
        help="MLflow run ID to restore from.",
    )
    group.add_argument(
        "--run-dir",
        help="Legacy W&B run directory (e.g. 'wandb/run-YYYYMMDD_HHMMSS-uid').",
    )
    parser.add_argument(
        "--tracking-uri",
        default="",
        help="Optional MLflow tracking URI (defaults to env MLFLOW_TRACKING_URI).",
    )
    parser.add_argument(
        "--artifact-root",
        default="checkpoints",
        help="Artifact subdirectory containing checkpoints in MLflow.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint filename or epoch suffix (defaults to the latest).",
    )
    args = parser.parse_args()

    logits, _, ckpt_path = restore_last_layer_checkpoint(
        run_id=args.run_id,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        tracking_uri=args.tracking_uri or None,
        artifact_root=args.artifact_root,
    )
    print(f"Loaded checkpoint '{ckpt_path}'.")
    print(f"Forward-pass logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
