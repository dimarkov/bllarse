from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Tuple

# do not preallocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import nn, random as jr
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as compute_ece,
)

from bllarse.losses import IBProbit

try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

_SPLITS = ("train", "validation_matched", "validation_mismatched")
_CACHE_FILES = {
    "X_train": "X_train.npz",
    "y_train": "y_train.npz",
    "X_val_m": "X_val_m.npz",
    "y_val_m": "y_val_m.npz",
    "X_val_mm": "X_val_mm.npz",
    "y_val_mm": "y_val_mm.npz",
    "metadata": "metadata.json",
}


def make_cache_key(
    backbone: str,
    max_length: int,
    cache_dtype: str,
    *,
    split_schema: str = "train|validation_matched|validation_mismatched",
    version: str = "v2",
) -> str:
    payload = f"{version}|{backbone}|{max_length}|{cache_dtype}|{split_schema}"
    digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_backbone = backbone.replace("/", "__").replace("-", "_")
    return f"{safe_backbone}_len{max_length}_{cache_dtype}_{digest}"


def build_hf_cache_prefix(hf_subdir_prefix: str, cache_key: str) -> str:
    clean_prefix = hf_subdir_prefix.strip("/")
    return f"{clean_prefix}/{cache_key}"


def hf_sync_flags(mode: str) -> Tuple[bool, bool]:
    mapping = {
        "none": (False, False),
        "pull": (True, False),
        "push": (False, True),
        "pull_push": (True, True),
    }
    if mode not in mapping:
        raise ValueError(f"Unknown hf_sync mode '{mode}'.")
    return mapping[mode]


def compute_metrics_from_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    num_bins: int = 20,
) -> Dict[str, float]:
    logits = jnp.asarray(logits, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.int32)
    preds = jnp.argmax(logits, axis=-1)
    one_hot = nn.one_hot(labels, logits.shape[-1])
    nll = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(preds == labels)
    ece = compute_ece(
        num_bins,
        logits=logits,
        labels_true=labels,
        labels_predicted=preds,
    )
    return {
        "acc": float(acc),
        "nll": float(nll),
        "ece": float(ece),
    }


def _cache_paths(cache_root: Path) -> Dict[str, Path]:
    return {name: cache_root / filename for name, filename in _CACHE_FILES.items()}


def _cache_complete(cache_root: Path) -> bool:
    paths = _cache_paths(cache_root)
    return all(path.exists() for path in paths.values())


def _load_metadata(cache_root: Path) -> Dict[str, Any]:
    metadata_path = _cache_paths(cache_root)["metadata"]
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cache_arrays(cache_root: Path) -> Dict[str, np.ndarray]:
    paths = _cache_paths(cache_root)
    arrays = {
        "X_train": np.load(paths["X_train"])["data"],
        "y_train": np.load(paths["y_train"])["data"],
        "X_val_m": np.load(paths["X_val_m"])["data"],
        "y_val_m": np.load(paths["y_val_m"])["data"],
        "X_val_mm": np.load(paths["X_val_mm"])["data"],
        "y_val_mm": np.load(paths["y_val_mm"])["data"],
    }
    return arrays


def _save_cache(cache_root: Path, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_root)
    np.savez_compressed(paths["X_train"], data=arrays["X_train"])
    np.savez_compressed(paths["y_train"], data=arrays["y_train"])
    np.savez_compressed(paths["X_val_m"], data=arrays["X_val_m"])
    np.savez_compressed(paths["y_val_m"], data=arrays["y_val_m"])
    np.savez_compressed(paths["X_val_mm"], data=arrays["X_val_mm"])
    np.savez_compressed(paths["y_val_mm"], data=arrays["y_val_mm"])
    with paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _require_nlp_stack():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, FlaxAutoModel
    except Exception as exc:
        raise ImportError(
            "Missing NLP dependencies. Install with `uv sync --extra nlp`."
        ) from exc
    return load_dataset, AutoTokenizer, FlaxAutoModel


def _from_pretrained_with_token(load_fn, model_name: str, token: str | None, **kwargs):
    if token:
        try:
            return load_fn(model_name, token=token, **kwargs)
        except TypeError:
            return load_fn(model_name, use_auth_token=token, **kwargs)
    return load_fn(model_name, **kwargs)


def _load_mnli_dataset(token: str | None):
    load_dataset, _, _ = _require_nlp_stack()
    if token:
        try:
            return load_dataset("glue", "mnli", token=token)
        except TypeError:
            return load_dataset("glue", "mnli", use_auth_token=token)
    return load_dataset("glue", "mnli")


def _extract_split_features(
    split,
    *,
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    cache_dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = len(split)
    hidden_size = int(model.config.hidden_size)
    output_dtype = np.float16 if cache_dtype == "float16" else np.float32

    features = np.empty((num_rows, hidden_size), dtype=output_dtype)
    labels_out = np.empty((num_rows,), dtype=np.int32)
    cursor = 0

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = split[start:end]
        labels = np.asarray(batch["label"], dtype=np.int32)
        valid = labels >= 0
        if not np.any(valid):
            continue

        premises = batch["premise"]
        hypotheses = batch["hypothesis"]
        if not np.all(valid):
            idxs = np.nonzero(valid)[0]
            premises = [premises[i] for i in idxs]
            hypotheses = [hypotheses[i] for i in idxs]
            labels = labels[valid]

        encoded = tokenizer(
            premises,
            hypotheses,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        outputs = model(**encoded, train=False)
        cls_features = np.asarray(outputs.last_hidden_state[:, 0, :], dtype=np.float32)
        if output_dtype is not np.float32:
            cls_features = cls_features.astype(output_dtype)

        n = cls_features.shape[0]
        features[cursor : cursor + n] = cls_features
        labels_out[cursor : cursor + n] = labels
        cursor += n

    return features[:cursor], labels_out[:cursor]


def _extract_features_to_cache(
    *,
    cache_root: Path,
    cache_key: str,
    backbone: str,
    max_length: int,
    extract_batch_size: int,
    cache_dtype: str,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> Dict[str, Any]:
    _, AutoTokenizer, FlaxAutoModel = _require_nlp_stack()
    token = os.environ.get("HF_TOKEN", "").strip() or None

    dataset = _load_mnli_dataset(token)
    tokenizer = _from_pretrained_with_token(
        AutoTokenizer.from_pretrained,
        backbone,
        token,
    )
    model = _from_pretrained_with_token(
        FlaxAutoModel.from_pretrained,
        backbone,
        token,
        dtype=jnp.float32,
    )

    split_samples = {
        "train": max_train_samples,
        "validation_matched": max_val_samples,
        "validation_mismatched": max_val_samples,
    }

    arrays: Dict[str, np.ndarray] = {}
    split_rows: Dict[str, int] = {}

    for split_name in _SPLITS:
        split = dataset[split_name]
        limit = split_samples[split_name]
        if limit is not None:
            limit = min(limit, len(split))
            split = split.select(range(limit))

        x_split, y_split = _extract_split_features(
            split,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            batch_size=extract_batch_size,
            cache_dtype=cache_dtype,
        )
        split_rows[split_name] = int(y_split.shape[0])

        if split_name == "train":
            arrays["X_train"] = x_split
            arrays["y_train"] = y_split
        elif split_name == "validation_matched":
            arrays["X_val_m"] = x_split
            arrays["y_val_m"] = y_split
        else:
            arrays["X_val_mm"] = x_split
            arrays["y_val_mm"] = y_split

    metadata = {
        "cache_key": cache_key,
        "dataset": "glue/mnli",
        "backbone": backbone,
        "max_length": max_length,
        "cache_dtype": cache_dtype,
        "split_schema": list(_SPLITS),
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": split_rows,
    }
    _save_cache(cache_root, arrays, metadata)
    return metadata


def _pull_cache_from_hf(*, repo_id: str, prefix: str, cache_root: Path, token: str | None) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required for HF sync; install dependencies first."
        ) from exc

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    remote_prefix = f"{prefix.strip('/')}/"
    selected_files = [path for path in repo_files if path.startswith(remote_prefix)]

    downloaded = 0
    for remote_path in selected_files:
        rel_path = remote_path[len(remote_prefix) :]
        if not rel_path:
            continue
        local_path = cache_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_path,
            token=token,
        )
        shutil.copy2(downloaded_path, local_path)
        downloaded += 1

    return downloaded


def _push_cache_to_hf(
    *,
    repo_id: str,
    prefix: str,
    cache_root: Path,
    token: str | None,
    private: bool,
) -> None:
    if not token:
        raise RuntimeError("HF_TOKEN is required when hf_sync includes push.")

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required for HF sync; install dependencies first."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(cache_root),
        path_in_repo=prefix,
        token=token,
    )


def _truncate_samples(
    arrays: Dict[str, np.ndarray],
    *,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> Dict[str, np.ndarray]:
    out = dict(arrays)
    if max_train_samples is not None:
        n = min(max_train_samples, out["y_train"].shape[0])
        out["X_train"] = out["X_train"][:n]
        out["y_train"] = out["y_train"][:n]

    if max_val_samples is not None:
        n_m = min(max_val_samples, out["y_val_m"].shape[0])
        n_mm = min(max_val_samples, out["y_val_mm"].shape[0])
        out["X_val_m"] = out["X_val_m"][:n_m]
        out["y_val_m"] = out["y_val_m"][:n_m]
        out["X_val_mm"] = out["X_val_mm"][:n_mm]
        out["y_val_mm"] = out["y_val_mm"][:n_mm]

    return out


def _linear_logits(
    params: Dict[str, jnp.ndarray],
    features: np.ndarray,
    *,
    batch_size: int,
) -> jnp.ndarray:
    logits_chunks = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        x = jnp.asarray(features[start:end], dtype=jnp.float32)
        logits_chunks.append(x @ params["w"] + params["b"])
    return jnp.concatenate(logits_chunks, axis=0)


def _ibprobit_logits(
    model: IBProbit,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> jnp.ndarray:
    @jax.jit
    def _predict(m: IBProbit, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        _, logits = m(x, y, with_logits=True, loss_type=3)
        return logits

    logits_chunks = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        x = jnp.asarray(features[start:end], dtype=jnp.float32)
        y = jnp.asarray(labels[start:end], dtype=jnp.int32)
        logits_chunks.append(_predict(model, x, y))
    return jnp.concatenate(logits_chunks, axis=0)


def _train_linear_head(
    *,
    optimizer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    num_features = x_train.shape[1]
    num_classes = int(y_train.max()) + 1

    key = jr.PRNGKey(seed)
    key, w_key = jr.split(key)
    params = {
        "w": 1e-2 * jr.normal(w_key, (num_features, num_classes), dtype=jnp.float32),
        "b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }

    if optimizer_name == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "lion":
        optimizer = optax.lion(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}' for linear head.")

    @jax.jit
    def _step(
        p: Dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        def _loss_fn(pp: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            logits = x @ pp["w"] + pp["b"]
            targets = nn.one_hot(y, num_classes)
            return optax.softmax_cross_entropy(logits, targets).mean()

        loss, grads = jax.value_and_grad(_loss_fn)(p)
        updates, next_opt_state = optimizer.update(grads, opt_state, p)
        next_params = optax.apply_updates(p, updates)
        return next_params, next_opt_state, loss

    rng = np.random.default_rng(seed)
    opt_state = optimizer.init(params)
    epoch_losses: list[float] = []
    n = y_train.shape[0]

    for _ in range(epochs):
        perm = rng.permutation(n)
        batch_losses: list[float] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x_batch = jnp.asarray(x_train[idx], dtype=jnp.float32)
            y_batch = jnp.asarray(y_train[idx], dtype=jnp.int32)
            params, opt_state, batch_loss = _step(params, opt_state, x_batch, y_batch)
            batch_losses.append(float(batch_loss))
        epoch_losses.append(float(np.mean(batch_losses)))

    training_info = {
        "epoch_losses": epoch_losses,
    }
    return params, training_info


def _train_ibprobit_head(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    epochs: int,
    num_update_iters: int,
    seed: int,
) -> Tuple[IBProbit, Dict[str, Any]]:
    if num_update_iters <= 0:
        raise ValueError("num_update_iters must be > 0 when optimizer=cavi.")

    num_features = x_train.shape[1]
    num_classes = int(y_train.max()) + 1

    key = jr.PRNGKey(seed)
    key, init_key = jr.split(key)
    model = IBProbit(num_features, num_classes, key=init_key)

    @jax.jit
    def _cavi_step(m: IBProbit, x: jnp.ndarray, y: jnp.ndarray) -> IBProbit:
        return m.update(x, y, num_iters=num_update_iters)

    rng = np.random.default_rng(seed)
    n = y_train.shape[0]
    epoch_losses: list[float] = []

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x_batch = jnp.asarray(x_train[idx], dtype=jnp.float32)
            y_batch = jnp.asarray(y_train[idx], dtype=jnp.int32)
            model = _cavi_step(model, x_batch, y_batch)

        probe_idx = perm[: min(batch_size, n)]
        x_probe = jnp.asarray(x_train[probe_idx], dtype=jnp.float32)
        y_probe = jnp.asarray(y_train[probe_idx], dtype=jnp.int32)
        epoch_loss = jnp.mean(model(x_probe, y_probe, loss_type=3))
        epoch_losses.append(float(epoch_loss))

    training_info = {
        "epoch_losses": epoch_losses,
    }
    return model, training_info


def _train_and_evaluate(
    *,
    optimizer_name: str,
    arrays: Dict[str, np.ndarray],
    train_batch_size: int,
    epochs: int,
    num_update_iters: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Dict[str, Any]:
    x_train = np.asarray(arrays["X_train"])
    y_train = np.asarray(arrays["y_train"])
    x_val_m = np.asarray(arrays["X_val_m"])
    y_val_m = np.asarray(arrays["y_val_m"])
    x_val_mm = np.asarray(arrays["X_val_mm"])
    y_val_mm = np.asarray(arrays["y_val_mm"])

    if optimizer_name in {"adamw", "lion"}:
        params, training_info = _train_linear_head(
            optimizer_name=optimizer_name,
            x_train=x_train,
            y_train=y_train,
            batch_size=train_batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
        logits_m = _linear_logits(params, x_val_m, batch_size=train_batch_size)
        logits_mm = _linear_logits(params, x_val_mm, batch_size=train_batch_size)
    elif optimizer_name == "cavi":
        model, training_info = _train_ibprobit_head(
            x_train=x_train,
            y_train=y_train,
            batch_size=train_batch_size,
            epochs=epochs,
            num_update_iters=num_update_iters,
            seed=seed,
        )
        logits_m = _ibprobit_logits(
            model,
            x_val_m,
            y_val_m,
            batch_size=train_batch_size,
        )
        logits_mm = _ibprobit_logits(
            model,
            x_val_mm,
            y_val_mm,
            batch_size=train_batch_size,
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

    metrics_m = compute_metrics_from_logits(logits_m, jnp.asarray(y_val_m))
    metrics_mm = compute_metrics_from_logits(logits_mm, jnp.asarray(y_val_mm))
    return {
        "matched": metrics_m,
        "mismatched": metrics_mm,
        "training": training_info,
    }


def _setup_mlflow(args) -> Any:
    if not args.enable_mlflow:
        return nullcontext(), False

    if not _HAS_MLFLOW:
        print("[bllarse] MLflow is not installed; disabling MLflow logging.")
        return nullcontext(), False

    from bllarse.mlflow_utils import load_mlflow_env_defaults

    load_mlflow_env_defaults()
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    experiment = (
        args.mlflow_experiment
        or os.environ.get("MLFLOW_EXPERIMENT_NAME", "")
        or "bllarse"
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    tags = {
        "group_id": args.group_id,
        "optimizer": args.optimizer,
        "task": "mnli_roberta_llf",
        "stage": args.stage,
    }
    context = mlflow.start_run(
        run_name=args.uid or None,
        tags=tags,
        nested=bool(parent_run_id),
        parent_run_id=parent_run_id,
    )
    return context, True


def _resolve_cache_root(args) -> Path:
    cache_key = make_cache_key(args.backbone, args.max_length, args.cache_dtype)
    cache_root = Path(args.cache_dir).expanduser().resolve() / args.hf_subdir_prefix / cache_key
    return cache_root


def _ensure_cache(args, cache_root: Path) -> Dict[str, Any]:
    pull_enabled, push_enabled = hf_sync_flags(args.hf_sync)
    token = os.environ.get("HF_TOKEN", "").strip() or None
    cache_key = make_cache_key(args.backbone, args.max_length, args.cache_dtype)
    prefix = build_hf_cache_prefix(args.hf_subdir_prefix, cache_key)

    metadata: Dict[str, Any] = {}
    extracted = False

    if args.stage in {"extract", "all"}:
        if args.reuse_cache and _cache_complete(cache_root):
            print(f"[bllarse] Reusing existing local cache at {cache_root}")
        else:
            if args.reuse_cache and pull_enabled:
                try:
                    downloaded = _pull_cache_from_hf(
                        repo_id=args.hf_repo_id,
                        prefix=prefix,
                        cache_root=cache_root,
                        token=token,
                    )
                    if downloaded:
                        print(f"[bllarse] Pulled {downloaded} file(s) from HF cache.")
                except Exception as exc:
                    print(f"[bllarse] HF pull failed before extraction: {exc}")

            if not _cache_complete(cache_root):
                print("[bllarse] Extracting RoBERTa features for MNLI...")
                metadata = _extract_features_to_cache(
                    cache_root=cache_root,
                    cache_key=cache_key,
                    backbone=args.backbone,
                    max_length=args.max_length,
                    extract_batch_size=args.extract_batch_size,
                    cache_dtype=args.cache_dtype,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_val_samples=args.max_val_samples,
                )
                extracted = True
                print(f"[bllarse] Saved feature cache to {cache_root}")
        if push_enabled and _cache_complete(cache_root):
            _push_cache_to_hf(
                repo_id=args.hf_repo_id,
                prefix=prefix,
                cache_root=cache_root,
                token=token,
                private=args.hf_private,
            )
            print(f"[bllarse] Uploaded cache to HF dataset {args.hf_repo_id}:{prefix}")

    if args.stage in {"train_eval", "all"} and not _cache_complete(cache_root):
        if pull_enabled:
            try:
                downloaded = _pull_cache_from_hf(
                    repo_id=args.hf_repo_id,
                    prefix=prefix,
                    cache_root=cache_root,
                    token=token,
                )
                if downloaded:
                    print(f"[bllarse] Pulled {downloaded} file(s) from HF cache.")
            except Exception as exc:
                print(f"[bllarse] HF pull failed before train/eval: {exc}")

    if _cache_complete(cache_root):
        metadata = _load_metadata(cache_root)

    return metadata


def _validate_args(args) -> None:
    if args.stage == "extract":
        return
    if args.optimizer in {"adamw", "lion"} and args.num_update_iters != 0:
        print(
            "[bllarse] Ignoring --num-update-iters for non-CAVI optimizer "
            f"({args.optimizer}); expected 0."
        )
    if args.optimizer == "cavi" and args.num_update_iters <= 0:
        raise ValueError("--num-update-iters must be > 0 when --optimizer cavi.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoBERTa + MNLI feature-cached LLF experiments."
    )
    parser.add_argument("--stage", choices=["extract", "train_eval", "all"], default="all")
    parser.add_argument(
        "--backbone",
        choices=["FacebookAI/roberta-base", "FacebookAI/roberta-large"],
        default="FacebookAI/roberta-base",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache")
    parser.add_argument("--reuse-cache", action="store_true")

    parser.add_argument("--optimizer", choices=["adamw", "lion", "cavi"], default="cavi")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--num-update-iters", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--enable-mlflow", "--enable_mlflow", action="store_true")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "bllarse"),
    )
    parser.add_argument("--group-id", type=str, default="mnli_roberta_llf")
    parser.add_argument("--uid", type=str, default=None)

    parser.add_argument("--hf-repo-id", type=str, default="dimarkov/bllarse-features")
    parser.add_argument(
        "--hf-subdir-prefix",
        type=str,
        default="roberta_activations/mnli_roberta_cls",
    )
    parser.add_argument(
        "--hf-sync",
        choices=["none", "pull", "push", "pull_push"],
        default="pull_push",
    )
    parser.add_argument("--hf-private", action="store_true")

    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser


def main(args) -> None:
    _validate_args(args)

    cache_key = make_cache_key(args.backbone, args.max_length, args.cache_dtype)
    cache_root = _resolve_cache_root(args)
    mlflow_context, mlflow_enabled = _setup_mlflow(args)

    with mlflow_context:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "stage": args.stage,
                    "backbone": args.backbone,
                    "max_length": args.max_length,
                    "cache_dir": str(cache_root),
                    "cache_key": cache_key,
                    "optimizer": args.optimizer,
                    "epochs": args.epochs,
                    "train_batch_size": args.train_batch_size,
                    "num_update_iters": args.num_update_iters,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "hf_repo_id": args.hf_repo_id,
                    "hf_subdir_prefix": args.hf_subdir_prefix,
                    "hf_sync": args.hf_sync,
                    "cache_dtype": args.cache_dtype,
                    "max_train_samples": -1 if args.max_train_samples is None else args.max_train_samples,
                    "max_val_samples": -1 if args.max_val_samples is None else args.max_val_samples,
                }
            )

        metadata = _ensure_cache(args, cache_root)
        if args.stage == "extract":
            print("[bllarse] Extraction stage complete.")
            return

        if not _cache_complete(cache_root):
            raise FileNotFoundError(
                "Feature cache is missing required files after sync/extract. "
                f"Expected cache root: {cache_root}"
            )

        arrays = _load_cache_arrays(cache_root)
        arrays = _truncate_samples(
            arrays,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )

        results = _train_and_evaluate(
            optimizer_name=args.optimizer,
            arrays=arrays,
            train_batch_size=args.train_batch_size,
            epochs=args.epochs,
            num_update_iters=args.num_update_iters,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

        metrics_m = results["matched"]
        metrics_mm = results["mismatched"]
        print(
            "[bllarse] val_matched "
            f"acc={metrics_m['acc']:.4f} nll={metrics_m['nll']:.4f} ece={metrics_m['ece']:.4f}"
        )
        print(
            "[bllarse] val_mismatched "
            f"acc={metrics_mm['acc']:.4f} nll={metrics_mm['nll']:.4f} ece={metrics_mm['ece']:.4f}"
        )

        if metadata:
            print(f"[bllarse] cache metadata rows={metadata.get('rows', {})}")

        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "val_matched_acc": metrics_m["acc"],
                    "val_matched_nll": metrics_m["nll"],
                    "val_matched_ece": metrics_m["ece"],
                    "val_mismatched_acc": metrics_mm["acc"],
                    "val_mismatched_nll": metrics_mm["nll"],
                    "val_mismatched_ece": metrics_mm["ece"],
                }
            )
            for epoch_idx, epoch_loss in enumerate(results["training"]["epoch_losses"], start=1):
                mlflow.log_metrics({"train_epoch_loss": float(epoch_loss)}, step=epoch_idx)


if __name__ == "__main__":
    parser = build_argparser()
    cli_args = parser.parse_args()
    main(cli_args)
