from __future__ import annotations

"""Feature-cached RoBERTa/MNLI experiments.

Stages:
- `extract`: build or refresh cached CLS features and optionally push them to HF.
  Extraction is treated as one-off data preparation and does not log to MLflow.
- `train_eval`: load cached features and train/evaluate a last-layer baseline.
- `all`: extract if needed, then train/evaluate in one invocation.

Implementation notes:
- feature extraction uses `FlaxAutoModel` and a jitted fixed-shape forward pass
  for each tokenized batch
- deterministic last-layer training keeps cached arrays on device and runs the
  minibatch epoch loop through `lax.scan`
"""

import argparse
import json
import os
import shutil
import time
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
import equinox as eqx
from jax import config, lax, nn, random as jr
from tensorflow_probability.substrates.jax.stats import (
    expected_calibration_error as compute_ece,
)

from bllarse.losses import CrossEntropy, IBProbit

config.update("jax_default_matmul_precision", "highest")

try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

_SPLITS = ("train", "validation_matched", "validation_mismatched")
_FULL_SPLIT_ROWS = {
    "train": 392702,
    "validation_matched": 9815,
    "validation_mismatched": 9832,
}
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
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    split_schema: str = "train|validation_matched|validation_mismatched",
    version: str = "v2",
) -> str:
    payload = f"{version}|{backbone}|{max_length}|{cache_dtype}|{split_schema}"
    if max_train_samples is not None or max_val_samples is not None:
        train_tag = "all" if max_train_samples is None else str(max_train_samples)
        val_tag = "all" if max_val_samples is None else str(max_val_samples)
        payload = f"{payload}|train{train_tag}|val{val_tag}"
    digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_backbone = backbone.replace("/", "__").replace("-", "_")
    key_parts = [f"len{max_length}", cache_dtype]
    if max_train_samples is not None or max_val_samples is not None:
        train_tag = "all" if max_train_samples is None else str(max_train_samples)
        val_tag = "all" if max_val_samples is None else str(max_val_samples)
        key_parts.extend([f"tr{train_tag}", f"val{val_tag}"])
    return f"{safe_backbone}_{'_'.join(key_parts)}_{digest}"


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
    losses: jnp.ndarray | None = None,
    num_bins: int = 20,
) -> Dict[str, float]:
    logits = jnp.asarray(logits, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.int32)
    preds = jnp.argmax(logits, axis=-1)
    if losses is None:
        one_hot = nn.one_hot(labels, logits.shape[-1])
        nll = optax.softmax_cross_entropy(logits, one_hot).mean()
    else:
        nll = jnp.asarray(losses, dtype=jnp.float32).mean()
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


@jax.jit
def _linear_logits_and_losses(
    params: Dict[str, jnp.ndarray],
    features: jnp.ndarray,
    labels: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(features, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.int32)
    logits = x @ params["w"] + params["b"]
    targets = nn.one_hot(y, logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, targets)
    return logits, losses


def checkpoint_is_better(
    candidate: Dict[str, float],
    best: Dict[str, float] | None,
) -> bool:
    if best is None:
        return True

    cand_acc = float(candidate["acc"])
    best_acc = float(best["acc"])
    if cand_acc > best_acc + 1e-12:
        return True
    if cand_acc < best_acc - 1e-12:
        return False

    cand_nll = float(candidate["nll"])
    best_nll = float(best["nll"])
    if cand_nll < best_nll - 1e-12:
        return True
    if cand_nll > best_nll + 1e-12:
        return False

    return float(candidate["ece"]) < float(best["ece"]) - 1e-12


def _copy_linear_params(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return {name: jnp.array(value) for name, value in params.items()}


def _apply_feature_dropout(
    x: jnp.ndarray,
    *,
    key: jax.Array,
    dropout_rate: float,
) -> jnp.ndarray:
    if dropout_rate <= 0.0:
        return x

    keep_prob = 1.0 - dropout_rate
    mask = jr.bernoulli(key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)


def _cache_paths(cache_root: Path) -> Dict[str, Path]:
    return {name: cache_root / filename for name, filename in _CACHE_FILES.items()}


def _cache_complete(cache_root: Path) -> bool:
    paths = _cache_paths(cache_root)
    return all(path.exists() for path in paths.values())


def _requested_row_counts(
    *,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> Dict[str, int]:
    requested = {
        "train": _FULL_SPLIT_ROWS["train"],
        "validation_matched": _FULL_SPLIT_ROWS["validation_matched"],
        "validation_mismatched": _FULL_SPLIT_ROWS["validation_mismatched"],
    }
    if max_train_samples is not None:
        requested["train"] = min(max_train_samples, requested["train"])
    if max_val_samples is not None:
        requested["validation_matched"] = min(
            max_val_samples,
            requested["validation_matched"],
        )
        requested["validation_mismatched"] = min(
            max_val_samples,
            requested["validation_mismatched"],
        )
    return requested


def _cache_satisfies_request(
    cache_root: Path,
    *,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> bool:
    if not _cache_complete(cache_root):
        return False

    metadata = _load_metadata(cache_root)
    rows = metadata.get("rows", {})
    requested = _requested_row_counts(
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    return all(int(rows.get(split_name, -1)) >= required for split_name, required in requested.items())


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
    pad_token_id = tokenizer.pad_token_id or 0

    @jax.jit
    def _forward(batch_inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        outputs = model(**batch_inputs, params=model.params, train=False)
        return outputs.last_hidden_state[:, 0, :]

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

        actual_n = int(labels.shape[0])
        if actual_n < batch_size:
            pad_rows = batch_size - actual_n
            encoded = {
                key: np.pad(
                    value,
                    ((0, pad_rows), (0, 0)),
                    constant_values=(pad_token_id if key == "input_ids" else 0),
                )
                for key, value in encoded.items()
            }

        encoded_jax = {key: jnp.asarray(value) for key, value in encoded.items()}
        cls_features = np.asarray(_forward(encoded_jax), dtype=np.float32)[:actual_n]
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


def _evaluate_linear_head(
    params: Dict[str, jnp.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> Dict[str, float]:
    del batch_size

    labels_jnp = jnp.asarray(labels, dtype=jnp.int32)
    logits_all, losses_all = _linear_logits_and_losses(
        params,
        jnp.asarray(features),
        labels_jnp,
    )
    return compute_metrics_from_logits(logits_all, labels_jnp, losses=losses_all)


def _evaluate_ibprobit_head(
    model: IBProbit,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> Dict[str, float]:
    @eqx.filter_jit
    def _eval_batch(m: IBProbit, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return m(x, y, with_logits=True, loss_type=3)

    logits_chunks: list[jnp.ndarray] = []
    loss_chunks: list[jnp.ndarray] = []

    for start in range(0, labels.shape[0], batch_size):
        end = min(start + batch_size, labels.shape[0])
        x = jnp.asarray(features[start:end], dtype=jnp.float32)
        y = jnp.asarray(labels[start:end], dtype=jnp.int32)
        loss, logits = _eval_batch(model, x, y)
        logits_chunks.append(logits)
        loss_chunks.append(loss)

    logits_all = jnp.concatenate(logits_chunks, axis=0)
    losses_all = jnp.concatenate(loss_chunks, axis=0)
    return compute_metrics_from_logits(logits_all, jnp.asarray(labels), losses=losses_all)


def _train_linear_head(
    *,
    optimizer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_m: np.ndarray,
    y_val_m: np.ndarray,
    x_val_mm: np.ndarray,
    y_val_mm: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
    seed: int,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    num_features = x_train.shape[1]
    num_classes = int(y_train.max()) + 1
    n = int(y_train.shape[0])
    n_batches = (n + batch_size - 1) // batch_size
    pad = n_batches * batch_size - n

    key = jr.PRNGKey(seed)
    key, w_key = jr.split(key)
    params = {
        "w": 1e-2 * jr.normal(w_key, (num_features, num_classes), dtype=jnp.float32),
        "b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }

    weight_decay_mask = {"w": True, "b": False}
    if optimizer_name == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask=weight_decay_mask,
        )
    elif optimizer_name == "lion":
        optimizer = optax.lion(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask=weight_decay_mask,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}' for linear head.")

    @jax.jit
    def _step(
        p: Dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        x: jnp.ndarray,
        y: jnp.ndarray,
        dropout_key: jax.Array,
    ) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        def _loss_fn(pp: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            x_in = _apply_feature_dropout(x, key=dropout_key, dropout_rate=dropout_rate)
            logits = x_in @ pp["w"] + pp["b"]
            targets = nn.one_hot(y, num_classes)
            return optax.softmax_cross_entropy(logits, targets).mean()

        loss, grads = jax.value_and_grad(_loss_fn)(p)
        updates, next_opt_state = optimizer.update(grads, opt_state, p)
        next_params = optax.apply_updates(p, updates)
        return next_params, next_opt_state, loss

    @jax.jit
    def _run_epoch(
        p: Dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        epoch_key: jax.Array,
    ) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        perm_key, batch_key = jr.split(epoch_key)
        perm = jr.permutation(perm_key, n)
        if pad > 0:
            perm = jnp.concatenate([perm, perm[:pad]], axis=0)
        batch_indices = perm.reshape((n_batches, batch_size))
        batch_keys = jr.split(batch_key, n_batches)

        def _batch_body(
            carry: Tuple[Dict[str, jnp.ndarray], optax.OptState],
            scans: Tuple[jnp.ndarray, jax.Array],
        ) -> Tuple[Tuple[Dict[str, jnp.ndarray], optax.OptState], jnp.ndarray]:
            params, state = carry
            idx, dropout_key = scans
            x_batch = jnp.asarray(x_data[idx], dtype=jnp.float32)
            y_batch = y_data[idx]
            next_params, next_state, batch_loss = _step(
                params,
                state,
                x_batch,
                y_batch,
                dropout_key,
            )
            return (next_params, next_state), batch_loss

        (next_params, next_opt_state), batch_losses = lax.scan(
            _batch_body,
            (p, opt_state),
            (batch_indices, batch_keys),
        )
        return next_params, next_opt_state, jnp.mean(batch_losses)

    x_train_device = jax.device_put(jnp.asarray(x_train))
    y_train_device = jax.device_put(jnp.asarray(y_train, dtype=jnp.int32))
    x_val_m_device = jax.device_put(jnp.asarray(x_val_m))
    y_val_m_device = jax.device_put(jnp.asarray(y_val_m, dtype=jnp.int32))
    x_val_mm_device = jax.device_put(jnp.asarray(x_val_mm))
    y_val_mm_device = jax.device_put(jnp.asarray(y_val_mm, dtype=jnp.int32))

    opt_state = optimizer.init(params)
    epoch_losses: list[float] = []
    history: list[Dict[str, float]] = []
    best_params: Dict[str, jnp.ndarray] | None = None
    best_epoch = 0
    best_metrics_m: Dict[str, float] | None = None

    for epoch_idx in range(epochs):
        epoch_start = time.perf_counter()
        key, epoch_key = jr.split(key)
        params, opt_state, epoch_loss = _run_epoch(
            params,
            opt_state,
            x_train_device,
            y_train_device,
            epoch_key,
        )
        epoch_train_loss = float(epoch_loss)
        epoch_losses.append(epoch_train_loss)

        metrics_m = _evaluate_linear_head(
            params,
            x_val_m_device,
            y_val_m_device,
            batch_size=batch_size,
        )
        metrics_mm = _evaluate_linear_head(
            params,
            x_val_mm_device,
            y_val_mm_device,
            batch_size=batch_size,
        )
        epoch_seconds = float(time.perf_counter() - epoch_start)
        history.append(
            {
                "loss": epoch_train_loss,
                "epoch_seconds": epoch_seconds,
                "val_matched_acc": metrics_m["acc"],
                "val_matched_nll": metrics_m["nll"],
                "val_matched_ece": metrics_m["ece"],
                "val_mismatched_acc": metrics_mm["acc"],
                "val_mismatched_nll": metrics_mm["nll"],
                "val_mismatched_ece": metrics_mm["ece"],
            }
        )
        print(
            "[bllarse] "
            f"epoch={epoch_idx + 1}/{epochs} "
            f"epoch_seconds={epoch_seconds:.2f} "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_matched_acc={metrics_m['acc']:.4f} "
            f"val_matched_nll={metrics_m['nll']:.4f} "
            f"val_matched_ece={metrics_m['ece']:.4f} "
            f"val_mismatched_acc={metrics_mm['acc']:.4f} "
            f"val_mismatched_nll={metrics_mm['nll']:.4f} "
            f"val_mismatched_ece={metrics_mm['ece']:.4f}"
        )

        if checkpoint_is_better(metrics_m, best_metrics_m):
            best_params = _copy_linear_params(params)
            best_epoch = epoch_idx + 1
            best_metrics_m = dict(metrics_m)

    training_info = {
        "epoch_losses": epoch_losses,
        "history": history,
        "best_epoch": best_epoch,
        "selected_split": "validation_matched",
    }
    if best_params is None:
        raise RuntimeError("Linear-head training did not produce a checkpoint.")
    return best_params, training_info


def _train_ibprobit_head(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_m: np.ndarray,
    y_val_m: np.ndarray,
    x_val_mm: np.ndarray,
    y_val_mm: np.ndarray,
    batch_size: int,
    epochs: int,
    num_update_iters: int,
    ibprobit_alpha: float,
    reset_loss_per_epoch: bool,
    seed: int,
) -> Tuple[IBProbit, Dict[str, Any]]:
    if num_update_iters <= 0:
        raise ValueError("num_update_iters must be > 0 when optimizer=cavi.")
    if ibprobit_alpha <= 0.0:
        raise ValueError("ibprobit_alpha must be > 0 when optimizer=cavi.")

    num_features = x_train.shape[1]
    num_classes = int(y_train.max()) + 1
    n = int(y_train.shape[0])
    n_batches = n // batch_size
    if n_batches <= 0:
        raise ValueError(
            "IBProbit training requires train_batch_size <= number of training samples "
            f"(got batch_size={batch_size}, n_train={n})."
        )

    key = jr.PRNGKey(seed)
    key, init_key = jr.split(key)
    model = IBProbit(num_features, num_classes, key=init_key)
    loss_params, loss_static = eqx.partition(model, eqx.is_array)

    @eqx.filter_jit
    def _run_epoch(
        curr_loss_params,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        batch_indices: jnp.ndarray,
    ):
        def _batch_body(loss_params, idx):
            current_loss = eqx.combine(loss_params, loss_static)
            x_batch = jnp.asarray(x_data[idx], dtype=jnp.float32)
            y_batch = y_data[idx]
            updated_loss = current_loss.update(
                x_batch,
                y_batch,
                num_iters=num_update_iters,
                alpha=ibprobit_alpha,
            )
            updated_loss_params = eqx.filter(updated_loss, eqx.is_array)
            batch_loss = updated_loss(x_batch, y_batch, loss_type=3).mean()
            return updated_loss_params, batch_loss

        return lax.scan(_batch_body, curr_loss_params, batch_indices)

    x_train_device = jax.device_put(jnp.asarray(x_train))
    y_train_device = jax.device_put(jnp.asarray(y_train, dtype=jnp.int32))
    x_val_m_device = jax.device_put(jnp.asarray(x_val_m))
    y_val_m_device = jax.device_put(jnp.asarray(y_val_m, dtype=jnp.int32))
    x_val_mm_device = jax.device_put(jnp.asarray(x_val_mm))
    y_val_mm_device = jax.device_put(jnp.asarray(y_val_mm, dtype=jnp.int32))
    epoch_losses: list[float] = []
    history: list[Dict[str, float]] = []
    best_model: IBProbit | None = None
    best_epoch = 0
    best_metrics_m: Dict[str, float] | None = None

    for epoch_idx in range(epochs):
        epoch_start = time.perf_counter()
        key, epoch_key = jr.split(key)
        reset_key, perm_key = jr.split(epoch_key)
        if reset_loss_per_epoch:
            current_model = eqx.combine(loss_params, loss_static)
            reset_model = current_model.reset(reset_key)
            loss_params = eqx.filter(reset_model, eqx.is_array)

        perm = jr.permutation(perm_key, n)
        batch_indices = perm[: n_batches * batch_size].reshape(n_batches, batch_size)
        loss_params, batch_losses = _run_epoch(
            loss_params,
            x_train_device,
            y_train_device,
            jax.device_put(jnp.asarray(batch_indices, dtype=jnp.int32)),
        )
        model = eqx.combine(loss_params, loss_static)
        epoch_train_loss = float(jnp.mean(batch_losses))
        epoch_losses.append(epoch_train_loss)

        metrics_m = _evaluate_ibprobit_head(
            model,
            x_val_m_device,
            y_val_m_device,
            batch_size=batch_size,
        )
        metrics_mm = _evaluate_ibprobit_head(
            model,
            x_val_mm_device,
            y_val_mm_device,
            batch_size=batch_size,
        )
        epoch_seconds = float(time.perf_counter() - epoch_start)
        history.append(
            {
                "loss": epoch_train_loss,
                "epoch_seconds": epoch_seconds,
                "val_matched_acc": metrics_m["acc"],
                "val_matched_nll": metrics_m["nll"],
                "val_matched_ece": metrics_m["ece"],
                "val_mismatched_acc": metrics_mm["acc"],
                "val_mismatched_nll": metrics_mm["nll"],
                "val_mismatched_ece": metrics_mm["ece"],
            }
        )
        print(
            "[bllarse] "
            f"epoch={epoch_idx + 1}/{epochs} "
            f"epoch_seconds={epoch_seconds:.2f} "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_matched_acc={metrics_m['acc']:.4f} "
            f"val_matched_nll={metrics_m['nll']:.4f} "
            f"val_matched_ece={metrics_m['ece']:.4f} "
            f"val_mismatched_acc={metrics_mm['acc']:.4f} "
            f"val_mismatched_nll={metrics_mm['nll']:.4f} "
            f"val_mismatched_ece={metrics_mm['ece']:.4f}"
        )

        if checkpoint_is_better(metrics_m, best_metrics_m):
            best_model = eqx.combine(
                jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), loss_params),
                loss_static,
            )
            best_epoch = epoch_idx + 1
            best_metrics_m = dict(metrics_m)

        epoch_is_finite = np.isfinite(epoch_train_loss) and all(
            np.isfinite(v)
            for v in (
                metrics_m["acc"],
                metrics_m["nll"],
                metrics_m["ece"],
                metrics_mm["acc"],
                metrics_mm["nll"],
                metrics_mm["ece"],
            )
        )
        if not epoch_is_finite:
            print(
                "[bllarse] Non-finite IBProbit metrics detected; "
                "stopping early and keeping the best finite checkpoint."
            )
            break

    training_info = {
        "epoch_losses": epoch_losses,
        "history": history,
        "best_epoch": best_epoch,
        "selected_split": "validation_matched",
    }
    if best_model is None:
        raise RuntimeError("IBProbit training did not produce a checkpoint.")
    return best_model, training_info


def _train_and_evaluate(
    *,
    optimizer_name: str,
    arrays: Dict[str, np.ndarray],
    train_batch_size: int,
    epochs: int,
    num_update_iters: int,
    ibprobit_alpha: float,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
    reset_loss_per_epoch: bool,
    seed: int,
) -> Dict[str, Any]:
    x_train = np.asarray(arrays["X_train"])
    y_train = np.asarray(arrays["y_train"])
    x_val_m = np.asarray(arrays["X_val_m"])
    y_val_m = np.asarray(arrays["y_val_m"])
    x_val_mm = np.asarray(arrays["X_val_mm"])
    y_val_mm = np.asarray(arrays["y_val_mm"])

    if optimizer_name in {"adam", "adamw", "lion"}:
        params, training_info = _train_linear_head(
            optimizer_name=optimizer_name,
            x_train=x_train,
            y_train=y_train,
            x_val_m=x_val_m,
            y_val_m=y_val_m,
            x_val_mm=x_val_mm,
            y_val_mm=y_val_mm,
            batch_size=train_batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            seed=seed,
        )
        metrics_m = _evaluate_linear_head(params, x_val_m, y_val_m, batch_size=train_batch_size)
        metrics_mm = _evaluate_linear_head(params, x_val_mm, y_val_mm, batch_size=train_batch_size)
    elif optimizer_name == "cavi":
        model, training_info = _train_ibprobit_head(
            x_train=x_train,
            y_train=y_train,
            x_val_m=x_val_m,
            y_val_m=y_val_m,
            x_val_mm=x_val_mm,
            y_val_mm=y_val_mm,
            batch_size=train_batch_size,
            epochs=epochs,
            num_update_iters=num_update_iters,
            ibprobit_alpha=ibprobit_alpha,
            reset_loss_per_epoch=reset_loss_per_epoch,
            seed=seed,
        )
        metrics_m = _evaluate_ibprobit_head(model, x_val_m, y_val_m, batch_size=train_batch_size)
        metrics_mm = _evaluate_ibprobit_head(model, x_val_mm, y_val_mm, batch_size=train_batch_size)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

    return {
        "matched": metrics_m,
        "mismatched": metrics_mm,
        "training": training_info,
    }


def _setup_mlflow(args) -> Any:
    if args.stage == "extract":
        return nullcontext(), False

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
    cache_key = make_cache_key(
        args.backbone,
        args.max_length,
        args.cache_dtype,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    cache_root = Path(args.cache_dir).expanduser().resolve() / args.hf_subdir_prefix / cache_key
    return cache_root


def _ensure_cache(args, cache_root: Path) -> Tuple[Dict[str, Any], bool]:
    pull_enabled, push_enabled = hf_sync_flags(args.hf_sync)
    token = os.environ.get("HF_TOKEN", "").strip() or None
    cache_key = make_cache_key(
        args.backbone,
        args.max_length,
        args.cache_dtype,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    prefix = build_hf_cache_prefix(args.hf_subdir_prefix, cache_key)

    metadata: Dict[str, Any] = {}
    extracted = False

    if args.stage in {"extract", "all"}:
        cache_ready = _cache_satisfies_request(
            cache_root,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
        if args.reuse_cache and cache_ready:
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

            if not _cache_satisfies_request(
                cache_root,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
            ):
                if _cache_complete(cache_root):
                    print(
                        "[bllarse] Existing cache does not satisfy the requested sample "
                        "counts; extracting a matching cache."
                    )
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
        if push_enabled and _cache_satisfies_request(
            cache_root,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        ):
            _push_cache_to_hf(
                repo_id=args.hf_repo_id,
                prefix=prefix,
                cache_root=cache_root,
                token=token,
                private=args.hf_private,
            )
            print(f"[bllarse] Uploaded cache to HF dataset {args.hf_repo_id}:{prefix}")

    if args.stage in {"train_eval", "all"} and not _cache_satisfies_request(
        cache_root,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    ):
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

    return metadata, extracted


def _validate_args(args) -> None:
    if args.stage == "extract":
        return
    if args.optimizer in {"adam", "adamw", "lion"} and args.num_update_iters != 0:
        print(
            "[bllarse] Ignoring --num-update-iters for non-CAVI optimizer "
            f"({args.optimizer}); expected 0."
        )
    if args.optimizer != "cavi" and not 0.0 <= args.dropout_rate < 1.0:
        raise ValueError("--dropout-rate must be in [0, 1) for linear-head training.")
    if args.optimizer == "cavi" and args.dropout_rate != 0.0:
        print("[bllarse] Ignoring --dropout-rate for optimizer cavi.")
    if args.stage != "extract" and args.epochs <= 0:
        raise ValueError("--epochs must be > 0 for training.")
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

    parser.add_argument("--optimizer", choices=["adam", "adamw", "lion", "cavi"], default="cavi")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--num-update-iters", type=int, default=16)
    parser.add_argument("--ibprobit-alpha", type=float, default=1e-3)
    parser.add_argument("--reset-loss-per-epoch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
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

    cache_key = make_cache_key(
        args.backbone,
        args.max_length,
        args.cache_dtype,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
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
                    "ibprobit_alpha": args.ibprobit_alpha,
                    "reset_loss_per_epoch": int(args.reset_loss_per_epoch),
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout_rate": args.dropout_rate,
                    "seed": args.seed,
                    "hf_repo_id": args.hf_repo_id,
                    "hf_subdir_prefix": args.hf_subdir_prefix,
                    "hf_sync": args.hf_sync,
                    "cache_dtype": args.cache_dtype,
                    "max_train_samples": -1 if args.max_train_samples is None else args.max_train_samples,
                    "max_val_samples": -1 if args.max_val_samples is None else args.max_val_samples,
                }
            )

        cache_started = time.perf_counter()
        metadata, extracted = _ensure_cache(args, cache_root)
        cache_seconds = time.perf_counter() - cache_started
        if args.stage == "extract":
            rows = metadata.get("rows", {}) if metadata else {}
            total_rows = int(sum(int(v) for v in rows.values())) if rows else 0
            print(
                "[bllarse] extraction "
                f"cache_seconds={cache_seconds:.2f} "
                f"cache_extracted={int(extracted)} "
                f"cache_total_rows={total_rows}"
            )
            if mlflow_enabled:
                mlflow.log_metric("cache_seconds", cache_seconds)
                mlflow.log_metric("cache_extracted", int(extracted))
                if total_rows > 0:
                    mlflow.log_metric("cache_total_rows", total_rows)
                    if extracted and cache_seconds > 0.0:
                        mlflow.log_metric(
                            "extract_rows_per_second",
                            total_rows / cache_seconds,
                        )
            print("[bllarse] Extraction stage complete.")
            return

        if not _cache_satisfies_request(
            cache_root,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        ):
            raise FileNotFoundError(
                "Feature cache is missing required files or sufficient rows after sync/extract. "
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
            ibprobit_alpha=args.ibprobit_alpha,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout_rate,
            reset_loss_per_epoch=args.reset_loss_per_epoch,
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
            best_epoch = int(results["training"].get("best_epoch", 0))
            if best_epoch > 0:
                mlflow.log_metric("best_epoch", best_epoch)
            history = results["training"].get("history", [])
            for epoch_idx, row in enumerate(history, start=1):
                mlflow.log_metrics(
                    {
                        # Keep canonical keys aligned with finetuning/run_training for primary split.
                        "loss": float(row["loss"]),
                        "acc": float(row["val_matched_acc"]),
                        "nll": float(row["val_matched_nll"]),
                        "ece": float(row["val_matched_ece"]),
                        # Also log mismatched split explicitly.
                        "val_matched_acc": float(row["val_matched_acc"]),
                        "val_matched_nll": float(row["val_matched_nll"]),
                        "val_matched_ece": float(row["val_matched_ece"]),
                        "val_mismatched_acc": float(row["val_mismatched_acc"]),
                        "val_mismatched_nll": float(row["val_mismatched_nll"]),
                        "val_mismatched_ece": float(row["val_mismatched_ece"]),
                    },
                    step=epoch_idx,
                )


if __name__ == "__main__":
    parser = build_argparser()
    cli_args = parser.parse_args()
    main(cli_args)
