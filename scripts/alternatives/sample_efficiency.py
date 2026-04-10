"""
Sample Efficiency Comparison: IBProbit (sequential Bayesian) vs CrossEntropy (linear probing)

For IBProbit:
  - Process training data in successive batches of `batch_size` (default 16384)
  - After each new batch, evaluate on the accumulated training data and full test set
  - Posterior state is carried forward — each update refines rather than restarts

For CrossEntropy + AdamW:
  - At each step k, retrain a freshly-initialised linear head from scratch on the
    first k*batch_size training samples
  - Uses a smaller mini-batch size (default 512) and a fixed number of epochs (default 100)
  - Evaluate on the same accumulated training data and full test set

Reference targets from sweep3a (dinov3_big, imagenet1k, full dataset):
  acc=0.72, ece=0.18, nll=1.42

Example usage:
    python scripts/alternatives/sample_efficiency.py \\
        --model dinov3_big --dataset imagenet1k \\
        --max-batches 10 --log-mlflow
"""
import os
import argparse
import warnings

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow

from tqdm import tqdm
from functools import partial
from jax import random as jr, config, vmap, lax
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread


from equimo.io import load_model as equimo_load_model

from mlpox.load_models import load_model as mlpox_load_model
from bllarse.losses import IBProbit, CrossEntropy
from calibration import evaluate_classification

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Model / dataset metadata (mirrors vit_classification.py) ─────────────────

EQUIMO_MODELS = {
    "dinov3_small": {"img_size": 224, "embed_dim": 384,  "channels_first": True},
    "dinov3_big":   {"img_size": 224, "embed_dim": 768,  "channels_first": True},
    "dinov3_large": {"img_size": 224, "embed_dim": 1024, "channels_first": True},
    "dinov3_huge":  {"img_size": 224, "embed_dim": 1280, "channels_first": True},
    "dinov3_max":   {"img_size": 224, "embed_dim": 4096, "channels_first": True},
    "deepMLP_big":  {"img_size": 64,  "embed_dim": 512,  "channels_first": False},
    "deepMLP_large":{"img_size": 64,  "embed_dim": 1024, "channels_first": False},
}

DINOV3_DATASET_IMG_SIZES = {
    "cifar10":       224, "cifar100":      224,
    "oxford_pets":   512, "food101":       512,
    "flowers102":    512, "stanford_cars": 512,
    "dtd":           512, "imagenet1k":    512,
}

DATASET_CONFIGS = {
    "cifar10":       {"hf_path": "cifar10",                 "num_classes": 10,   "label_key": "label",      "img_key": "img",   "test_split": "test"},
    "cifar100":      {"hf_path": "cifar100",                "num_classes": 100,  "label_key": "fine_label", "img_key": "img",   "test_split": "test"},
    "oxford_pets":   {"hf_path": "timm/oxford-iiit-pet",   "num_classes": 37,   "label_key": "label",      "img_key": "image", "test_split": "test"},
    "food101":       {"hf_path": "ethz/food101",           "num_classes": 101,  "label_key": "label",      "img_key": "image", "test_split": "validation"},
    "flowers102":    {"hf_path": "nelorth/oxford-flowers", "num_classes": 102,  "label_key": "label",      "img_key": "image", "test_split": "test"},
    "stanford_cars": {"hf_path": "tanganke/stanford_cars", "num_classes": 196,  "label_key": "label",      "img_key": "image", "test_split": "test"},
    "dtd":           {"hf_path": "tanganke/dtd",           "num_classes": 47,   "label_key": "label",      "img_key": "image", "test_split": "test"},
    "imagenet1k":    {"hf_path": "ILSVRC/imagenet-1k",     "num_classes": 1000, "label_key": "label",      "img_key": "image", "test_split": "validation"},
}

NORM_STATS = {
    "cifar10":  (np.array([0.49139968, 0.48215827, 0.44653124]),
                 np.array([0.24703233, 0.24348505, 0.26158768])),
    "cifar100": (np.array([0.49139968, 0.48215827, 0.44653124]),
                 np.array([0.24703233, 0.24348505, 0.26158768])),
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def get_img_size(model_name, dataset_name):
    if "dinov3" in model_name:
        return DINOV3_DATASET_IMG_SIZES.get(dataset_name, EQUIMO_MODELS[model_name]["img_size"])
    return EQUIMO_MODELS[model_name]["img_size"]


# ── Feature extraction (mirrors vit_classification.py) ───────────────────────

def get_pretrained_backbone(model_name, key):
    model_id_map = {
        "dinov3_small": ("vit", "dinov3_vits16plus_pretrain_lvd1689m"),
        "dinov3_big":   ("vit", "dinov3_vitb16_pretrain_lvd1689m"),
        "dinov3_large": ("vit", "dinov3_vitl16_pretrain_lvd1689m"),
        "dinov3_huge":  ("vit", "dinov3_vith16plus_pretrain_lvd1689m"),
        "dinov3_max":   ("vit", "dinov3_vit7b16_pretrain_lvd1689m"),
        "deepMLP_big":  ("mlp", "B_12-Wi_512_res_64_in21k"),
        "deepMLP_large":("mlp", "B_12-Wi_1024_res_64_in21k"),
    }
    arch_type, model_id = model_id_map[model_name]
    if "dinov3" in model_name:
        model = equimo_load_model(arch_type, model_id)
        return eqx.nn.inference_mode(model, True)
    else:
        model = mlpox_load_model(model_id)
        model = eqx.tree_at(lambda m: m.fc, model, eqx.nn.Identity())
        return eqx.nn.inference_mode(model, True)


def _load_and_process(dataset_split, img_key, img_size, idx):
    img = dataset_split[int(idx)][img_key]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(img)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def prefetch_iterator(iterator, n_prefetch=2):
    q = Queue(maxsize=n_prefetch)
    sentinel = object()
    def fill():
        for item in iterator:
            q.put(item)
        q.put(sentinel)
    t = Thread(target=fill, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


@eqx.filter_jit
def _extract_features_batch(model, images, key):
    def fwd(x):
        out = model(x, key=key)
        return out[0] if isinstance(out, tuple) else out
    return vmap(fwd)(images)


def precompute_features(model, dataset_split, ds_config, img_size, mean, std, batch_size, key, channels_first=True):
    import tempfile
    n = len(dataset_split)
    all_features = all_labels = feat_tmp = label_tmp = None
    offset = 0

    def iterate():
        all_labels_arr = np.array(dataset_split[ds_config["label_key"]])
        worker_fn = partial(_load_and_process, dataset_split, ds_config["img_key"], img_size)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in tqdm(range(0, n, batch_size), desc="Extracting"):
                idx = np.arange(start, min(start + batch_size, n))
                images = list(pool.map(worker_fn, idx))
                labels = all_labels_arr[idx]
                images = jnp.stack(images)
                images = (images - mean) / std
                if channels_first:
                    images = jnp.transpose(images, (0, 3, 1, 2))
                yield images, labels

    try:
        for images, labels in prefetch_iterator(iterate(), n_prefetch=2):
            feats = np.asarray(_extract_features_batch(model, images, key))
            labels_np = np.asarray(labels)
            if all_features is None:
                feat_tmp  = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
                label_tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
                all_features = np.memmap(feat_tmp.name,  dtype=feats.dtype,     mode="w+", shape=(n, feats.shape[-1]))
                all_labels   = np.memmap(label_tmp.name, dtype=labels_np.dtype, mode="w+", shape=(n,))
            nb = len(feats)
            all_features[offset:offset + nb] = feats
            all_labels  [offset:offset + nb] = labels_np
            offset += nb
        result = np.array(all_features[:offset]), np.array(all_labels[:offset])
    finally:
        if feat_tmp  is not None: del all_features; os.unlink(feat_tmp.name)
        if label_tmp is not None: del all_labels;   os.unlink(label_tmp.name)
    return result


def load_or_compute_features(cache_path, model, dataset_split, ds_config, img_size, mean, std, batch_size, key, channels_first=True):
    if os.path.exists(cache_path):
        print(f"Loading cached features: {cache_path}")
        d = np.load(cache_path)
        return d["features"], d["labels"]
    features, labels = precompute_features(model, dataset_split, ds_config, img_size, mean, std, batch_size, key, channels_first)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, features=features, labels=labels)
    print(f"Cached to {cache_path}")
    return features, labels


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval_batches(loss_fn, features, labels, eval_batch_size, loss_type, head=None):
    """Return (acc, ece, nll) over the given feature/label arrays."""
    n = len(features)
    all_logits, all_labels = [], []
    for start in range(0, n, eval_batch_size):
        f = jnp.array(features[start:start + eval_batch_size])
        l = jnp.array(labels  [start:start + eval_batch_size])
        if head is not None:
            logits = vmap(head)(f)
            _, logits = loss_fn(logits, l, with_logits=True, loss_type=loss_type)
        else:
            _, logits = loss_fn(f, l, with_logits=True, loss_type=loss_type)
        all_logits.append(logits)
        all_labels.append(l)
    return evaluate_classification(
        jnp.concatenate(all_logits, axis=0),
        jnp.concatenate(all_labels, axis=0),
    )


# ── IBProbit helpers ─────────────────────────────────────────────────────────

@eqx.filter_jit
def _ibprobit_update_scan(loss_fn, features, labels, num_update_iters, alpha, update_method="update"):
    """Run sequential IBProbit updates over stacked sub-batches using lax.scan.

    Args:
        features:      (n_sub_batches, sub_batch_size, embed_dim)
        labels:        (n_sub_batches, sub_batch_size)
        update_method: name of the IBProbit update method to call

    Returns updated loss_fn after all sub-batch updates.
    """
    dynamic, static = eqx.partition(loss_fn, eqx.is_array)

    def step(dynamic, batch):
        f, l = batch
        loss_fn = eqx.combine(dynamic, static)
        new_loss_fn = getattr(loss_fn, update_method)(f, l, num_iters=num_update_iters, alpha=alpha)
        new_dynamic, _ = eqx.partition(new_loss_fn, eqx.is_array)
        return new_dynamic, None

    dynamic, _ = lax.scan(step, dynamic, (features, labels))
    return eqx.combine(dynamic, static)


# ── IBProbit sequential experiment ───────────────────────────────────────────

def run_ibprobit_sequential(
    train_features, train_labels, test_features, test_labels,
    indices, embed_dim, num_classes,
    step_size, ibprobit_batch_size, num_update_iters,
    num_calibration_iters, eval_batch_size,
    max_batches, loss_type, key, update_method="update",
):
    """Retrain from scratch at each step on all accumulated data.

    alpha is set automatically to ibprobit_batch_size / n_samples so that
    each mini-batch contributes proportionally to the posterior update.

    Returns list of dicts with keys: n_samples, train_acc/ece/nll, test_acc/ece/nll, beta.
    """
    label = {"update": "IBProbit", "update_cavi": "IBProbit (CAVI-EMA)", "update_bmr": "IBProbit (BMR)"}.get(update_method, update_method)
    print("\n" + "="*60)
    print(f"{label} — retrain from scratch each step")
    print(f"  step_size={step_size:,}, ibprobit_batch_size={ibprobit_batch_size:,}")
    print("="*60)

    key, init_key = jr.split(key)
    results = []

    for step in range(max_batches):
        step_end  = (step + 1) * step_size
        n_samples = step_end

        alpha = 1e-2 * ibprobit_batch_size / n_samples

        # Reset from scratch and train on all accumulated data
        key, step_key = jr.split(key)
        loss_fn = IBProbit(embed_dim, num_classes, key=step_key)
        acc_indices = indices[:step_end]
        n_sub = n_samples // ibprobit_batch_size
        used  = n_sub * ibprobit_batch_size
        acc_f = jnp.array(train_features[acc_indices[:used]]).reshape(n_sub, ibprobit_batch_size, -1)
        acc_l = jnp.array(train_labels  [acc_indices[:used]]).reshape(n_sub, ibprobit_batch_size)

        print(f"\nStep {step + 1}/{max_batches} | n_samples={n_samples:,} | alpha={alpha:.4f} | updating...")
        loss_fn = _ibprobit_update_scan(loss_fn, acc_f, acc_l, num_update_iters, alpha, update_method)

        # Temperature scaling on a subsample of training data, sized to match the test set
        val_nll_before = val_nll_after = float("nan")
        if num_calibration_iters > 0:
            key, cal_key = jr.split(key)
            n_cal = min(n_samples, len(test_features))
            perm  = np.array(jr.permutation(cal_key, n_samples))
            cal_f = train_features[indices[:step_end]][perm[:n_cal]]
            cal_l = train_labels  [indices[:step_end]][perm[:n_cal]]
            _, _, val_nll_before = _eval_batches(loss_fn, cal_f, cal_l, eval_batch_size, loss_type)
            loss_fn = loss_fn.calibrate(jnp.array(cal_f), jnp.array(cal_l), num_iters=num_calibration_iters, loss_type=loss_type)
            _, _, val_nll_after = _eval_batches(loss_fn, cal_f, cal_l, eval_batch_size, loss_type)
            print(f"  Calibration (n_cal={n_cal:,}): nll {val_nll_before:.4f} → {val_nll_after:.4f}  beta={float(loss_fn.beta):.4f}")

        # Evaluate on accumulated training data
        acc_tr, ece_tr, nll_tr = _eval_batches(
            loss_fn,
            train_features[indices[:step_end]],
            train_labels  [indices[:step_end]],
            eval_batch_size, loss_type,
        )
        # Evaluate on full test set
        acc_te, ece_te, nll_te = _eval_batches(
            loss_fn, test_features, test_labels, eval_batch_size, loss_type,
        )

        beta = float(loss_fn.beta)
        print(f"  Train: acc={acc_tr:.4f} ece={ece_tr:.4f} nll={nll_tr:.4f}")
        print(f"  Test:  acc={acc_te:.4f} ece={ece_te:.4f} nll={nll_te:.4f}")
        print(f"  Beta:  {beta:.4f}")

        results.append({
            "n_samples":      n_samples,
            "train_acc":      float(acc_tr), "train_ece":  float(ece_tr), "train_nll":  float(nll_tr),
            "test_acc":       float(acc_te), "test_ece":   float(ece_te), "test_nll":   float(nll_te),
            "beta":           beta,
            "val_nll_before": val_nll_before,
            "val_nll_after":  val_nll_after,
        })

    return results


# ── CrossEntropy refit experiment ─────────────────────────────────────────────

@eqx.filter_jit
def _ce_epoch_scan(head, opt_state, features, labels, key, optimizer, loss_fn, batch_size, loss_type):
    """One epoch: lax.scan over mini-batches. Returns (head, opt_state).

    All batch gradient steps run inside a single compiled kernel — no Python
    dispatch overhead between steps.
    """
    n = features.shape[0]
    n_batches = n // batch_size
    used = n_batches * batch_size
    perm = jr.permutation(key, n)
    f = features[perm[:used]].reshape(n_batches, batch_size, -1)
    l = labels  [perm[:used]].reshape(n_batches, batch_size)

    def step(carry, batch):
        h, os = carry
        f_b, l_b = batch
        loss, g = eqx.filter_value_and_grad(
            lambda h: loss_fn(vmap(h)(f_b), l_b, loss_type=loss_type).mean()
        )(h)
        upd, nos = optimizer.update(g, os, h)
        return (eqx.apply_updates(h, upd), nos), None

    (head, opt_state), _ = lax.scan(step, (head, opt_state), (f, l))
    return head, opt_state


@eqx.filter_jit
def _ce_train_scan(
    head, opt_state, train_f, train_l, val_f, val_l, key,
    optimizer, loss_fn, batch_size, loss_type,
    max_epochs, patience, min_delta,
):
    """Nested lax.scan (epochs × batches) with masked early stopping on val NLL.

    Epochs after patience is exhausted are skipped via lax.cond — no gradient
    computation is wasted on those iterations.

    Returns (best_head, best_epoch, best_val_nll).
    """
    n = train_f.shape[0]
    n_batches = n // batch_size
    used = n_batches * batch_size
    keys = jr.split(key, max_epochs)

    def epoch_fn(carry, key_e):
        head, opt_state, best_nll, best_head, wait, best_epoch, epoch_idx = carry
        stopped = wait >= patience

        def run(_):
            perm = jr.permutation(key_e, n)
            f = train_f[perm[:used]].reshape(n_batches, batch_size, -1)
            l = train_l[perm[:used]].reshape(n_batches, batch_size)

            def step(c, batch):
                h, os = c
                f_b, l_b = batch
                loss, g = eqx.filter_value_and_grad(
                    lambda h: loss_fn(vmap(h)(f_b), l_b, loss_type=loss_type).mean()
                )(h)
                upd, nos = optimizer.update(g, os, h)
                return (eqx.apply_updates(h, upd), nos), None

            (nh, nos), _ = lax.scan(step, (head, opt_state), (f, l))
            return nh, nos

        new_head, new_opt_state = lax.cond(stopped, lambda _: (head, opt_state), run, None)

        val_nll  = loss_fn(vmap(new_head)(val_f), val_l, loss_type=loss_type).mean()
        improved = jnp.logical_and(jnp.logical_not(stopped), val_nll < best_nll - min_delta)

        best_nll   = jnp.where(improved, val_nll, best_nll)
        best_head  = lax.cond(improved, lambda _: new_head, lambda _: best_head, None)
        best_epoch = jnp.where(improved, epoch_idx + 1, best_epoch)
        wait       = jnp.where(stopped, wait, jnp.where(improved, 0, wait + 1))

        return (new_head, new_opt_state, best_nll, best_head, wait, best_epoch, epoch_idx + 1), None

    init = (
        head, opt_state,
        jnp.array(jnp.inf),          # best_nll
        head,                          # best_head
        jnp.zeros((), jnp.int32),     # wait
        jnp.zeros((), jnp.int32),     # best_epoch
        jnp.zeros((), jnp.int32),     # epoch_idx
    )
    (_, _, best_nll, best_head, _, best_epoch, _), _ = lax.scan(epoch_fn, init, keys)
    return best_head, best_epoch, best_nll


def _run_ce_with_early_stopping(
    loss_fn, embed_dim, num_classes,
    train_f, train_l, val_f, val_l,
    lr, wd, ce_batch_size,
    max_epochs, patience, min_delta, loss_type, key,
):
    """Build optimizer, call _ce_train_scan, return (best_head, best_epoch_int, best_val_nll)."""
    n_train = len(train_f)
    steps_per_epoch = max(1, n_train // ce_batch_size)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=steps_per_epoch,
        decay_steps=max_epochs * steps_per_epoch,
        end_value=lr * 1e-2,
    )
    optimizer = optax.adamw(schedule, weight_decay=wd)

    key, head_key, train_key = jr.split(key, 3)
    head      = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
    opt_state = optimizer.init(eqx.filter(head, eqx.is_array))

    best_head, best_epoch, best_val_nll = _ce_train_scan(
        head, opt_state,
        jnp.array(train_f), jnp.array(train_l),
        jnp.array(val_f),   jnp.array(val_l),
        train_key, optimizer, loss_fn, ce_batch_size, loss_type,
        max_epochs, patience, min_delta,
    )
    return best_head, max(1, int(best_epoch)), float(best_val_nll)


def run_crossentropy_refit(
    train_features, train_labels, test_features, test_labels,
    indices, embed_dim, num_classes,
    step_size, ce_batch_size,
    lr, wd, val_frac, patience, min_delta, max_epochs,
    eval_batch_size, max_batches, loss_type, key,
):
    """Early stopping on val NLL, then refit on full subset.

    For each cumulative subset:
      1. Hold out val_frac for early stopping.
      2. Train with cosine schedule + early stopping on val NLL to find best_epoch.
      3. Refit on the full subset for best_epoch epochs.

    Returns list of dicts: n_samples, train/test acc/ece/nll, best_epoch.
    """
    print("\n" + "="*60)
    print("CrossEntropy + AdamW — early stopping + refit")
    print(f"  lr={lr}, wd={wd}, val_frac={val_frac}, patience={patience}, max_epochs={max_epochs}")
    print("="*60)

    loss_fn = CrossEntropy(0.0, num_classes)
    results = []

    for step in range(max_batches):
        n_samples = (step + 1) * step_size
        all_f = train_features[indices[:n_samples]]
        all_l = train_labels  [indices[:n_samples]]

        # Val split: first val_frac of the shuffled subset
        n_val = max(1, int(n_samples * val_frac))
        val_f, val_l = all_f[:n_val], all_l[:n_val]
        sub_f, sub_l = all_f[n_val:], all_l[n_val:]

        print(f"\nStep {step + 1}/{max_batches} | n_samples={n_samples:,} | early stopping (val={n_val})...")

        key, ck = jr.split(key)
        _, best_epoch, best_val_nll = _run_ce_with_early_stopping(
            loss_fn, embed_dim, num_classes,
            sub_f, sub_l, val_f, val_l,
            lr, wd, ce_batch_size,
            max_epochs, patience, min_delta, loss_type, ck,
        )

        print(f"  Early stop at epoch={best_epoch}, val_nll={best_val_nll:.4f}")

        # Refit on full subset for best_epoch epochs (no val split, no early stopping)
        steps_per_epoch = max(1, n_samples // ce_batch_size)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=steps_per_epoch,
            decay_steps=best_epoch * steps_per_epoch,
            end_value=lr * 1e-2,
        )
        optimizer = optax.adamw(schedule, weight_decay=wd)
        key, hk, rk = jr.split(key, 3)
        head      = eqx.nn.Linear(embed_dim, num_classes, key=hk)
        opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
        all_f_jnp, all_l_jnp = jnp.array(all_f), jnp.array(all_l)

        for epoch in tqdm(range(best_epoch), desc="  Refit", leave=False):
            rk, ek = jr.split(rk)
            head, opt_state = _ce_epoch_scan(
                head, opt_state, all_f_jnp, all_l_jnp, ek,
                optimizer, loss_fn, ce_batch_size, loss_type,
            )

        # Evaluate
        acc_tr, ece_tr, nll_tr = _eval_batches(
            loss_fn, all_f, all_l, eval_batch_size, loss_type, head=head,
        )
        acc_te, ece_te, nll_te = _eval_batches(
            loss_fn, test_features, test_labels, eval_batch_size, loss_type, head=head,
        )

        print(f"  Train: acc={acc_tr:.4f} ece={ece_tr:.4f} nll={nll_tr:.4f}")
        print(f"  Test:  acc={acc_te:.4f} ece={ece_te:.4f} nll={nll_te:.4f}")

        results.append({
            "n_samples":  n_samples,
            "train_acc":  float(acc_tr), "train_ece":  float(ece_tr), "train_nll":  float(nll_tr),
            "test_acc":   float(acc_te), "test_ece":   float(ece_te), "test_nll":   float(nll_te),
            "best_epoch": best_epoch,
        })

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

# Reference values from sweep3a_table.tex (batch_size=16384, update_iters=256, full dataset)
SWEEP3A_REFS = {
    "dinov3_big": {
        "imagenet1k": {"test_acc": 0.72, "test_ece": 0.18, "test_nll": 1.42},
    },
    "dinov3_small": {
        "imagenet1k": {"test_acc": 0.62, "test_ece": 0.25, "test_nll": 1.99},
    },
    "dinov3_large": {
        "imagenet1k": {"test_acc": 0.79, "test_ece": 0.14, "test_nll": 1.08},
    },
    "dinov3_huge": {
        "imagenet1k": {"test_acc": 0.82, "test_ece": 0.13, "test_nll": 0.99},
    },
}

METRIC_DISPLAY = {
    "acc": "Accuracy ↑",
    "ece": "ECE ↓",
    "nll": "NLL ↓",
}


def plot_results(ibprobit_results, ce_results, output_dir, model, dataset, prefix="sample_efficiency", bmr_results=None):
    """3-row × 3-col figure + beta panel + val NLL calibration panel."""
    os.makedirs(output_dir, exist_ok=True)

    metrics    = ["acc", "ece", "nll"]
    split_rows = [("test", "Test set"), ("train", "Train set (accumulated)")]
    n_metric_rows = len(split_rows)
    n_cols = len(metrics)

    has_cal = (ibprobit_results and "val_nll_before" in ibprobit_results[0]
               and not np.isnan(ibprobit_results[0]["val_nll_before"]))
    bmr_has_cal = (bmr_results and "val_nll_before" in bmr_results[0]
                   and not np.isnan(bmr_results[0]["val_nll_before"]))
    n_bottom_rows = 1 + int(has_cal or bmr_has_cal)  # beta + optional cal panel
    n_total_rows  = n_metric_rows + n_bottom_rows

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_total_rows))
    gs  = fig.add_gridspec(n_total_rows, n_cols, hspace=0.55, wspace=0.35)

    ib_x = [r["n_samples"] for r in ibprobit_results]
    ce_x = [r["n_samples"] for r in ce_results]
    ref  = SWEEP3A_REFS.get(model, {}).get(dataset, {})

    # Metric rows
    for row_idx, (split, split_label) in enumerate(split_rows):
        for col_idx, metric in enumerate(metrics):
            ax  = fig.add_subplot(gs[row_idx, col_idx])
            key = f"{split}_{metric}"

            ax.plot(ib_x, [r[key] for r in ibprobit_results],
                    marker="o", ms=5, color="tab:blue", label="IBProbit (update)")
            ax.plot(ce_x, [r[key] for r in ce_results],
                    marker="s", ms=5, color="tab:orange", label="CrossEntropy + AdamW (refit)")
            if bmr_results:
                bmr_x = [r["n_samples"] for r in bmr_results]
                ax.plot(bmr_x, [r[key] for r in bmr_results],
                        marker="P", ms=4, color="tab:red", label="IBProbit (update_bmr)")
            ref_key = f"test_{metric}"
            if split == "test" and ref_key in ref:
                ax.axhline(ref[ref_key], ls="--", lw=1.2, color="tab:blue", alpha=0.5,
                           label=f"IBProbit full-data ref ({ref[ref_key]:.2f})")

            ax.set_xlabel("Training samples seen")
            ax.set_ylabel(METRIC_DISPLAY[metric])
            ax.set_title(f"{split_label} — {METRIC_DISPLAY[metric]}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Beta panel (spans all columns)
    ax_beta = fig.add_subplot(gs[n_metric_rows, :])
    if ibprobit_results and "beta" in ibprobit_results[0]:
        ax_beta.plot(ib_x, [r["beta"] for r in ibprobit_results],
                     marker="o", ms=5, color="tab:blue", label="IBProbit (update) β")
    if bmr_results and "beta" in bmr_results[0]:
        bmr_x = [r["n_samples"] for r in bmr_results]
        ax_beta.plot(bmr_x, [r["beta"] for r in bmr_results],
                     marker="P", ms=4, color="tab:red", label="IBProbit (update_bmr) β")
    ax_beta.axhline(1.0, ls="--", lw=1.0, color="gray", alpha=0.5, label="β=1 (no scaling)")
    ax_beta.set_xlabel("Training samples seen")
    ax_beta.set_ylabel("Inverse temperature β")
    ax_beta.set_title("IBProbit temperature parameter β over sequential steps")
    ax_beta.legend(fontsize=8)
    ax_beta.grid(True, alpha=0.3)
    ax_beta.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.setp(ax_beta.get_xticklabels(), rotation=30, ha="right")

    # Val NLL calibration panel (optional, spans all columns)
    if has_cal or bmr_has_cal:
        ax_cal = fig.add_subplot(gs[n_metric_rows + 1, :])
        if has_cal:
            ax_cal.plot(ib_x, [r["val_nll_before"] for r in ibprobit_results],
                        marker="o", ms=4, ls="--", color="tab:blue", alpha=0.5,
                        label="IBProbit val NLL (before cal)")
            ax_cal.plot(ib_x, [r["val_nll_after"] for r in ibprobit_results],
                        marker="o", ms=4, color="tab:blue",
                        label="IBProbit val NLL (after cal)")
        if bmr_has_cal:
            bmr_x = [r["n_samples"] for r in bmr_results]
            ax_cal.plot(bmr_x, [r["val_nll_before"] for r in bmr_results],
                        marker="P", ms=4, ls="--", color="tab:red", alpha=0.5,
                        label="IBProbit (BMR) val NLL (before cal)")
            ax_cal.plot(bmr_x, [r["val_nll_after"] for r in bmr_results],
                        marker="P", ms=4, color="tab:red",
                        label="IBProbit (BMR) val NLL (after cal)")
        ax_cal.set_xlabel("Training samples seen")
        ax_cal.set_ylabel("Val NLL")
        ax_cal.set_title("Temperature scaling: val NLL before/after calibration")
        ax_cal.legend(fontsize=8)
        ax_cal.grid(True, alpha=0.3)
        ax_cal.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        plt.setp(ax_cal.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(f"Sample efficiency: {model} on {dataset}", fontsize=13)
    fig.subplots_adjust(top=0.97)

    for ext in ("pdf", "png"):
        fname = os.path.join(output_dir, f"{prefix}_{model}_{dataset}.{ext}")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved {fname}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    ds_config    = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS[args.model]
    img_size     = get_img_size(args.model, args.dataset)
    embed_dim    = model_config["embed_dim"]
    channels_first = model_config["channels_first"]
    num_classes  = ds_config["num_classes"]
    loss_type    = 2

    key = jr.PRNGKey(args.seed)

    # ── Feature cache paths ──
    test_split_name = ds_config["test_split"]
    train_cache = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_res{img_size}_train.npz")
    test_cache  = os.path.join(args.cache_dir, f"{args.model}_{args.dataset}_res{img_size}_{test_split_name}.npz")

    cache_exists = os.path.exists(train_cache) and os.path.exists(test_cache)

    if cache_exists:
        print("Loading cached features...")
        train_data = np.load(train_cache)
        train_features, train_labels = train_data["features"], train_data["labels"]
        test_data   = np.load(test_cache)
        test_features,  test_labels  = test_data ["features"], test_data ["labels"]
    else:
        from datasets import load_dataset
        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset(ds_config["hf_path"])
        mean, std = NORM_STATS.get(args.dataset, (IMAGENET_MEAN, IMAGENET_STD))

        print(f"Loading pretrained backbone: {args.model}")
        key, model_key = jr.split(key)
        backbone = get_pretrained_backbone(args.model, model_key)

        key, feat_key = jr.split(key)
        print("Precomputing train features...")
        train_features, train_labels = load_or_compute_features(
            train_cache, backbone, ds["train"], ds_config, img_size, mean, std,
            args.ibprobit_batch_size, feat_key, channels_first=channels_first,
        )
        print("Precomputing test features...")
        test_features, test_labels = load_or_compute_features(
            test_cache, backbone, ds[test_split_name], ds_config, img_size, mean, std,
            args.ibprobit_batch_size, feat_key, channels_first=channels_first,
        )

    print(f"\nTrain features: {train_features.shape}, Test features: {test_features.shape}")

    # ── Fixed shuffled order for training data ──
    n_train = len(train_features)
    key, perm_key = jr.split(key)
    indices = np.array(jr.permutation(perm_key, n_train))

    # Cap max_batches to what fits
    max_possible = n_train // args.step_size
    max_batches  = min(args.max_batches, max_possible)
    if max_batches < args.max_batches:
        print(f"Warning: only {max_possible} full steps of {args.step_size} available; capping at {max_batches}.")

    eval_batch_size = args.eval_batch_size

    # ── MLflow setup ──
    if args.log_mlflow:
        mlflow.set_experiment(args.experiment_name)

    # ── IBProbit sequential (update) ──
    key, ib_key = jr.split(key)
    ibprobit_results = run_ibprobit_sequential(
        train_features, train_labels, test_features, test_labels,
        indices, embed_dim, num_classes,
        args.step_size, args.ibprobit_batch_size, args.num_update_iters,
        args.num_calibration_iters, eval_batch_size,
        max_batches, loss_type, ib_key, update_method="update",
    )

    # ── IBProbit sequential (update_bmr) ──
    key, bmr_key = jr.split(key)
    bmr_results = run_ibprobit_sequential(
        train_features, train_labels, test_features, test_labels,
        indices, embed_dim, num_classes,
        args.step_size, args.ibprobit_batch_size, args.num_update_iters,
        args.num_calibration_iters, eval_batch_size,
        max_batches, loss_type, bmr_key, update_method="update_bmr",
    )

    if args.log_mlflow:
        for method_label, results in [("IBProbit", ibprobit_results), ("IBProbit-BMR", bmr_results)]:
            with mlflow.start_run(run_name=f"sample_eff_{method_label.lower()}_{args.model}_{args.dataset}"):
                mlflow.log_params({
                    "model": args.model, "dataset": args.dataset,
                    "step_size": args.step_size,
                    "ibprobit_batch_size": args.ibprobit_batch_size,
                    "num_update_iters": args.num_update_iters,
                    "ibprobit_batch_size": args.ibprobit_batch_size,
                    "max_batches": max_batches, "seed": args.seed,
                    "method": method_label,
                })
                for r in results:
                    step = r["n_samples"]
                    mlflow.log_metrics({k: v for k, v in r.items() if k != "n_samples"}, step=step)

    # ── CrossEntropy refit ──
    key, ce_key = jr.split(key)
    ce_results = run_crossentropy_refit(
        train_features, train_labels, test_features, test_labels,
        indices, embed_dim, num_classes,
        args.step_size, args.ce_batch_size,
        args.ce_lr, args.ce_wd,
        args.ce_val_frac, args.ce_patience, args.ce_min_delta, args.ce_max_epochs,
        eval_batch_size, max_batches, loss_type, ce_key,
    )

    if args.log_mlflow:
        with mlflow.start_run(run_name=f"sample_eff_ce_{args.model}_{args.dataset}"):
            mlflow.log_params({
                "model": args.model, "dataset": args.dataset,
                "step_size": args.step_size,
                "ce_batch_size": args.ce_batch_size,
                "ce_lr": args.ce_lr,
                "ce_wd": args.ce_wd,
                "ce_val_frac": args.ce_val_frac,
                "ce_patience": args.ce_patience,
                "ce_max_epochs": args.ce_max_epochs,
                "max_batches": max_batches, "seed": args.seed,
                "method": "CrossEntropy",
            })
            for r in ce_results:
                step = r["n_samples"]
                mlflow.log_metrics({k: v for k, v in r.items() if k != "n_samples"}, step=step)

    # ── Print summary table ──
    print("\n" + "="*80)
    print(f"{'Samples':>10}  {'IB train acc':>12}  {'IB test acc':>12}  {'CE train acc':>12}  {'CE test acc':>12}")
    print("-"*80)
    for ib, ce in zip(ibprobit_results, ce_results):
        print(f"{ib['n_samples']:>10,}  {ib['train_acc']:>12.4f}  {ib['test_acc']:>12.4f}  "
              f"{ce['train_acc']:>12.4f}  {ce['test_acc']:>12.4f}")

    # ── Plot ──
    plot_results(
        ibprobit_results, ce_results,
        output_dir=args.output_dir,
        model=args.model, dataset=args.dataset,
        bmr_results=bmr_results,
    )

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_argparser():
    p = argparse.ArgumentParser(description="Sample efficiency: IBProbit vs CrossEntropy")
    p.add_argument("--model",   type=str, default="dinov3_large", choices=list(EQUIMO_MODELS.keys()))
    p.add_argument("--dataset", type=str, default="imagenet1k", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--step-size", type=int, default=16384,
                   help="Number of new training samples added at each sequential step")
    p.add_argument("--ibprobit-batch-size", type=int, default=16384,
                   help="Batch size for each IBProbit CAVI update call (must be <= step-size)")
    p.add_argument("--num-update-iters", type=int, default=512,
                   help="CAVI iterations per IBProbit batch update")
    p.add_argument("--num-calibration-iters", type=int, default=8,
                   help="Newton steps for temperature scaling on val data (0 = skip calibration)")
    p.add_argument("--ce-batch-size",  type=int, default=512,   help="Mini-batch size for CrossEntropy AdamW")
    p.add_argument("--ce-max-epochs", type=int,   default=200,  help="Max epochs (early stopping may fire sooner)")
    p.add_argument("--ce-lr",         type=float, default=1e-3, help="Learning rate for CrossEntropy AdamW")
    p.add_argument("--ce-wd",         type=float, default=1e-4, help="Weight decay for CrossEntropy AdamW")
    p.add_argument("--ce-val-frac",   type=float, default=0.2, help="Fraction of accumulated subset held out for val NLL selection")
    p.add_argument("--ce-patience",   type=int,   default=10,  help="Early stopping patience (epochs without val NLL improvement)")
    p.add_argument("--ce-min-delta",  type=float, default=1e-4, help="Minimum val NLL improvement to reset patience counter")
    p.add_argument("--max-batches",    type=int, default=10,    help="Maximum number of sequential batches")
    p.add_argument("--eval-batch-size",type=int, default=2048,  help="Batch size for evaluation passes")
    p.add_argument("--seed",           type=int, default=137)
    p.add_argument("--cache-dir",      type=str, default=".cache/features")
    p.add_argument("--output-dir",     type=str, default="scripts/alternatives/figures")
    p.add_argument("--device",         type=str, default="gpu")
    p.add_argument("--experiment-name",type=str, default="bllarse")
    p.add_argument("--log-mlflow",     action="store_true", help="Log runs to MLflow")
    # SVI
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args   = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)