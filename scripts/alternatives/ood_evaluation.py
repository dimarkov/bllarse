"""OOD calibration evaluation: IBProbit vs CrossEntropy last-layer heads.

Trains a single head on ImageNet-1k train features (with fixed best
hyperparameters) and evaluates it — without retraining — on ImageNet-1k
validation (in-distribution) and a suite of OOD ImageNet variants:
  - ImageNet-C severity 5 (8 corruptions)
  - ImageNet-R, ImageNet-A (200-class subsets; logits masked to the subset)
  - ImageNet-V2 (matched-frequency)

Features are loaded from the local cache or HF Hub (produced by
precompute_features.py / precompute_ood_features.py).

The CrossEntropy head is trained by AdamW with early stopping. The IBProbit
posterior is built by a streaming recursive-Bayesian CAVI pass: per minibatch, a
single CAVI update assimilates the batch and -- when the head is built with
learn_q=True (the default; disable with --no-learn-q) -- infers the process
noise Q as a conjugate Inverse-Wishart posterior (prior scale q0 = alpha^2 /
batch_size, or --q-init to override it directly). --warmup defers the gate and
the Q recursion for the first N steps.

One MLflow run per seed, nested under a parent run per method.

Example:
    python scripts/alternatives/ood_evaluation.py \
        --model dinov3_huge --method IBProbit --seed 137 --batch-size 16384 \
        --num-update-iters 128 --epochs 25 --log-mlflow
"""

import os
import time
import argparse
import logging

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random as jr, config, vmap
from tqdm import tqdm

try:
    import mlflow
except ImportError:
    mlflow = None

from bllarse.losses import IBProbit, CrossEntropy
from calibration import evaluate_classification
from vit_classification import (
    EQUIMO_MODELS,
    get_img_size,
    iterate_feature_batches,
    download_features_from_hf,
)

# OOD features were extracted at imagenet1k's resolution (see precompute_ood_features.py)
OOD_IMG_SIZE = 512

# OOD evaluation sets, in report order. SUBSET keys are 200-class subsets of
# ImageNet-1k whose logits are masked to the present classes before scoring.
IMAGENET_C_KEYS = [
    f"imagenet_c_{c}" for c in
    ["brightness", "fog", "gaussian_noise", "jpeg_compression",
     "motion_blur", "pixelate", "shot_noise", "zoom_blur"]
]
OOD_KEYS = IMAGENET_C_KEYS + ["imagenet_r", "imagenet_a", "imagenet_v2"]
SUBSET_KEYS = {"imagenet_r", "imagenet_a"}

MASK_NEG = -1e9  # additive logit mask for out-of-subset classes


def load_cached_features(cache_path, hf_repo=None, hf_path=None):
    """Load features from local cache, else HF Hub. Raises if unavailable."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["features"], data["labels"]
    if hf_repo and hf_path and download_features_from_hf(hf_repo, hf_path, cache_path):
        data = np.load(cache_path)
        return data["features"], data["labels"]
    raise FileNotFoundError(
        f"Features not found locally ({cache_path}) or on HF ({hf_repo}/{hf_path})"
    )


def collect_logits(loss_fn, features, labels, batch_size, loss_type=3, head=None,
                   subset=None, progress=False, desc="eval"):
    """Forward pass over cached features -> (logits, labels) concatenated arrays.

    Logits are the ``with_logits=True`` predictive (CBC-transformed moderated score
    for IBProbit; raw head logits scored through the loss for CE). For 200-class
    subsets (ImageNet-R/A), out-of-subset logits are pushed to MASK_NEG so the
    softmax/argmax is restricted to the subset (the standard protocol).

    loss_fn is either IBProbit or CrossEntropy (--method), so loss_type keeps being
    forwarded for the generic call signature; IBProbit ignores it (self.loss_type
    fixes its branch).
    """
    subset_arr = np.asarray(subset) if subset is not None else None
    keep = None  # built lazily from the logits' class dimension

    n_batches = (len(features) + batch_size - 1) // batch_size
    all_logits, all_labels = [], []
    batches = iterate_feature_batches(features, labels, batch_size=batch_size, key=None)
    for feats, ys in tqdm(batches, total=n_batches, desc=desc, disable=not progress, leave=False):
        if head is not None:
            logits = vmap(head)(feats)
            _, logits = loss_fn(logits, ys, with_logits=True, loss_type=loss_type)
        else:
            _, logits = loss_fn(feats, ys, with_logits=True, loss_type=loss_type)
        if subset_arr is not None:
            if keep is None:
                keep = jnp.zeros(logits.shape[-1], dtype=bool).at[subset_arr].set(True)
            logits = jnp.where(keep, logits, MASK_NEG)
        all_logits.append(logits)
        all_labels.append(ys)

    return jnp.concatenate(all_logits, axis=0), jnp.concatenate(all_labels, axis=0)


def evaluate_features(loss_fn, features, labels, batch_size, loss_type=3, head=None,
                      subset=None, progress=False, desc="eval"):
    """Evaluate a head/loss on cached features -> (acc, ece, nll).

    Thin wrapper around collect_logits; see its docstring for why loss_type is
    still forwarded even though IBProbit ignores it.
    """
    logits, labels_c = collect_logits(loss_fn, features, labels, batch_size,
                                      loss_type=loss_type, head=head, subset=subset,
                                      progress=progress, desc=desc)
    return evaluate_classification(logits, labels_c)


def compute_brier(logits, labels):
    """Multiclass Brier score: mean_i sum_k (p_ik - 1[k=y_i])^2, in [0, 2]. A proper,
    *bounded* score -- less tail-sensitive than NLL, so disagreement between the two
    flags a tail-driven (few-point) NLL gap rather than a sharpness one."""
    p = jax.nn.softmax(logits, axis=-1)
    oh = jax.nn.one_hot(labels, p.shape[-1])
    return jnp.mean(jnp.sum((p - oh) ** 2, axis=-1))


def fit_temperature(logits, labels, steps=200, lr=0.05):
    """Fit one scalar T>0 on (frozen) ID-val logits by minimizing NLL of softmax(z/T).
    NLL is convex in 1/T, so a few gradient steps reach the global optimum. T>1
    softens, T<1 sharpens. The fit sees ONLY in-distribution data; the same T is then
    applied to the OOD logits to measure how the ID calibration transfers."""
    logits, labels = jnp.asarray(logits), jnp.asarray(labels)
    opt = optax.adam(lr)

    def nll(log_T):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits / jnp.exp(log_T), labels).mean()

    @jax.jit
    def step(log_T, state):
        g = jax.grad(nll)(log_T)
        upd, state = opt.update(g, state)
        return optax.apply_updates(log_T, upd), state

    log_T = jnp.zeros(())
    state = opt.init(log_T)
    for _ in range(steps):
        log_T, state = step(log_T, state)
    return float(jnp.exp(log_T))


def metrics_from_logits(logits, labels, temperature=1.0):
    """Metrics at T=1 and at the ID-val temperature T*. acc is T-invariant; the
    ``_ts`` keys are on softmax(z / T*), the plain keys at T=1. T is fit to minimize
    NLL, so ece_ts/brier_ts are the post-calibration values under that same T."""
    z_ts = logits / temperature
    acc, ece, nll = evaluate_classification(logits, labels)
    _, ece_ts, nll_ts = evaluate_classification(z_ts, labels)
    return dict(acc=float(acc),
                ece=float(ece), nll=float(nll), brier=float(compute_brier(logits, labels)),
                ece_ts=float(ece_ts), nll_ts=float(nll_ts), brier_ts=float(compute_brier(z_ts, labels)))


def _ce_epoch(loss_fn, head, opt_state, optimizer, feats, labels, batch_size, key, desc):
    """One CE gradient epoch over the features, with a batch progress bar."""
    key, shuffle_key = jr.split(key)
    n_batches = (len(feats) + batch_size - 1) // batch_size
    batches = iterate_feature_batches(feats, labels, batch_size=batch_size, key=shuffle_key)
    total, n = 0.0, 0
    bar = tqdm(batches, total=n_batches, desc=desc, leave=False)
    for f, y in bar:
        @eqx.filter_value_and_grad
        def ce_loss(h):
            return loss_fn(vmap(h)(f), y, loss_type=3).mean()
        loss, grads = ce_loss(head)
        updates, opt_state = optimizer.update(grads, opt_state, head)
        head = eqx.apply_updates(head, updates)
        total += float(loss); n += 1
        bar.set_postfix(loss=f"{total / n:.4f}")
    return head, opt_state, total / max(n, 1)


def _scalars_str(loss_fn):
    """Learned-scalar readout: mean diagonal of Qhat, and eps (always live -- the
    contamination mixture is structural now)."""
    q_mean = float(jnp.trace(loss_fn.Qhat) / loss_fn.Qhat.shape[0])
    return f"q={q_mean:.4g} eps={float(loss_fn.eps):.4g}"


def stream_train_ibprobit(loss_fn, train_f, train_l, val_f, val_l, *, batch_size,
                          num_iters, epochs, patience, eval_bs, key, px=False):
    """Streaming recursive-Bayesian IBProbit trainer. Per minibatch, a single CAVI
    ``update`` of (mu, Sigma); when the head was built with ``learn_q=True`` that
    same update also advances the conjugate Inverse-Wishart posterior on the process
    noise Q. The streamed posterior is returned directly -- no separate optimizer or
    posterior pass. Early-stops on val NLL; returns (best_loss_fn, best_epoch)."""
    loss_fn = loss_fn.reset_posterior()
    n_batches = (len(train_f) + batch_size - 1) // batch_size
    best_val, best_loss_fn, best_epoch, patience_ctr = float("inf"), loss_fn, 0, 0
    for epoch in range(epochs):
        key, shuffle_key = jr.split(key)
        batches = iterate_feature_batches(train_f, train_l, batch_size=batch_size, key=shuffle_key)
        bar = tqdm(batches, total=n_batches, desc=f"IBProbit stream e{epoch + 1}/{epochs}", leave=False)
        for f, y in bar:
            loss_fn = loss_fn.update(f, y, num_iters=num_iters, px=px)
            q_mean = float(jnp.trace(loss_fn.Qhat) / loss_fn.Qhat.shape[0])
            post = {"q": f"{q_mean:.3g}", "eps": f"{float(loss_fn.eps):.3g}"}
            bar.set_postfix(**post)

        if val_f is None:
            best_loss_fn, best_epoch = loss_fn, epoch + 1
            print(f"  epoch {epoch + 1}/{epochs}: {_scalars_str(loss_fn)}")
            continue
        _, _, val_nll = evaluate_features(loss_fn, val_f, val_l, eval_bs, head=None)
        val_nll = float(val_nll)
        improved = val_nll < best_val
        if improved:
            best_val, best_loss_fn, best_epoch, patience_ctr = val_nll, loss_fn, epoch + 1, 0
        else:
            patience_ctr += 1
        print(f"  epoch {epoch + 1}/{epochs}: val_nll={val_nll:.4f}{' *best' if improved else ''} "
              f"{_scalars_str(loss_fn)} "
              f"(patience {patience_ctr}/{patience})")
        if patience is not None and patience_ctr >= patience:
            break
    return best_loss_fn, best_epoch


def train_ce_head(embed_dim, num_classes, train_f, train_l, val_f, val_l,
                  batch_size, lr, weight_decay, label_smooth, epochs, patience, eval_bs, key):
    """Train a linear CE head. With a val set, select the lowest-val-NLL epoch
    and early-stop on patience; without one, run all epochs and keep the final."""
    key, head_key = jr.split(key)
    head = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
    loss_fn = CrossEntropy(label_smooth, num_classes)

    best_val, best_head, best_epoch, patience_ctr = float("inf"), head, 0, 0
    for epoch in range(epochs):
        key, train_key = jr.split(key)
        head, opt_state, train_loss = _ce_epoch(
            loss_fn, head, opt_state, optimizer, train_f, train_l, batch_size,
            train_key, desc=f"CE epoch {epoch + 1}/{epochs}",
        )
        if val_f is None:
            best_head, best_epoch = head, epoch + 1
            print(f"  epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}")
            continue
        _, _, val_nll = evaluate_features(loss_fn, val_f, val_l, eval_bs, head=head)
        val_nll = float(val_nll)
        improved = val_nll < best_val
        if improved:
            best_val, best_head, best_epoch, patience_ctr = val_nll, head, epoch + 1, 0
        else:
            patience_ctr += 1
        print(f"  epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} "
              f"val_nll={val_nll:.4f}{' *best' if improved else ''} "
              f"(patience {patience_ctr}/{patience})")
        if patience is not None and patience_ctr >= patience:
            break
    return loss_fn, best_head, best_epoch


def main(args):
    if args.model not in EQUIMO_MODELS:
        raise ValueError(f"Unknown model: {args.model}")
    embed_dim = EQUIMO_MODELS[args.model]["embed_dim"]
    num_classes = 1000  # ImageNet-1k label space
    img_size = get_img_size(args.model, "imagenet1k")
    hf_repo = None if args.no_hf_cache else args.hf_repo
    key = jr.PRNGKey(args.seed)

    def cache_and_hf(key_name, split):
        cache = os.path.join(args.cache_dir, f"{args.model}_{key_name}_res{img_size}_{split}.npz")
        hf = f"{args.model}/{key_name}_res{img_size}_{split}.npz" if hf_repo else None
        return cache, hf

    # --- ImageNet-1k train (head training) + validation (in-distribution eval) ---
    print(f"Loading ImageNet-1k train/val features for {args.model} (res{img_size})...")
    tr_cache, tr_hf = cache_and_hf("imagenet1k", "train")
    val_cache, val_hf = cache_and_hf("imagenet1k", "validation")
    train_f, train_l = load_cached_features(tr_cache, hf_repo, tr_hf)
    id_val_f, id_val_l = load_cached_features(val_cache, hf_repo, val_hf)
    print(f"  train: {train_f.shape}, imagenet1k_val: {id_val_f.shape}")

    # The OOD sets are the held-out test data, so early stopping selects on the
    # ImageNet-1k validation split directly and the head trains on the FULL train
    # set (no train/val holdout). Engages only for multi-epoch runs with patience;
    # the default single-pass IBProbit (epochs=1) does no selection.
    use_early_stop = args.patience is not None and args.epochs > 1
    es_val_f = es_val_l = None
    if use_early_stop:
        es_val_f, es_val_l = id_val_f, id_val_l
        print(f"  early stopping on imagenet1k_val ({len(es_val_f)} samples); "
              f"training on full train set ({len(train_f)})")

    # --- Train head ---
    print(f"\nTraining {args.method} head...")
    t_train = time.perf_counter()
    if args.method == "IBProbit":
        eps_rho = max(0.0, 1.0 - args.batch_size / len(train_f))  # 1 - B/N: one-pass eps memory
        base = IBProbit(embed_dim, num_classes, key=jr.split(key)[1],
                        q_init=args.q_init, alpha=args.alpha, batch_size=args.batch_size,
                        kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
                        warmup=args.warmup, eps_init=args.eps_init, rho=eps_rho)
        loss_fn, selected_epoch = stream_train_ibprobit(
            base, train_f, train_l, es_val_f, es_val_l,
            batch_size=args.batch_size, num_iters=args.num_update_iters,
            epochs=args.epochs, patience=args.patience,
            eval_bs=args.eval_batch_size, key=key, px=args.px,
        )
        head = None
    elif args.method == "CrossEntropy":
        loss_fn, head, selected_epoch = train_ce_head(
            embed_dim, num_classes, train_f, train_l, es_val_f, es_val_l,
            args.batch_size, args.lr, args.weight_decay, args.label_smooth,
            args.epochs, args.patience, args.eval_batch_size, key,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    # Block on the trained artifact so the timer captures the full (async) compute.
    jax.block_until_ready(eqx.filter(head if head is not None else loss_fn, eqx.is_array))
    train_time_s = time.perf_counter() - t_train
    print(f"  trained {args.method}, selected epoch {selected_epoch}"
          + (f" (early stopping, patience {args.patience})" if use_early_stop else " (single pass)")
          + f"; train_time={train_time_s:.1f}s")

    # (The streaming IBProbit trainer already built the posterior in the same pass.)

    # --- MLflow run ---
    log = args.log_mlflow and mlflow is not None
    if log:
        mlflow.set_experiment(args.experiment_name)
        # Nest under the parent run the SAME way scripts/finetuning.py does: pass
        # parent_run_id + nested=True. Setting the reserved ``mlflow.parentRunId`` tag by
        # hand does not make the UI group the child under the parent (MLflow 3.x).
        mlflow.start_run(nested=bool(args.run_id), parent_run_id=args.run_id)
        params = {
            "model": args.model, "method": args.method, "seed": args.seed,
            "img_size": img_size, "batch_size": args.batch_size,
            "epochs": args.epochs, "selected_epoch": selected_epoch,
        }
        if use_early_stop:
            params["patience"] = args.patience
        if args.method == "IBProbit":
            params.update(num_update_iters=args.num_update_iters, px=args.px,
                          learn_q=not args.no_learn_q, alpha=args.alpha,
                          q_init=args.q_init, kappa=args.kappa, warmup=args.warmup,
                          eps_init=args.eps_init,
                          eps_rho=max(0.0, 1.0 - args.batch_size / len(train_f)))
            mlflow.log_metric("eps", float(loss_fn.eps))
        else:
            params.update(lr=args.lr, weight_decay=args.weight_decay,
                          label_smooth=args.label_smooth)
        mlflow.log_params(params)
        mlflow.log_metric("train_time_s", train_time_s)

    # --- Evaluate: in-distribution + all OOD ---
    METRIC_KEYS = ["acc", "ece", "nll", "brier", "ece_ts", "nll_ts", "brier_ts"]

    def report(name, logits, lab):
        m = metrics_from_logits(logits, lab, temperature=T)
        print(f"  {name:28s} acc={m['acc']:.4f}  "
              f"ece={m['ece']:.4f}/{m['ece_ts']:.4f}  "
              f"nll={m['nll']:.4f}/{m['nll_ts']:.4f}  "
              f"brier={m['brier']:.4f}/{m['brier_ts']:.4f}  (T=1/T*)")
        if log:
            mlflow.log_metrics({f"{name}_{k}": m[k] for k in METRIC_KEYS})
        return m

    print("\nEvaluating...")
    # ID-val logits drive both the temperature fit and the in-distribution report.
    idval_logits, idval_lab = collect_logits(
        loss_fn, id_val_f, id_val_l, args.eval_batch_size, head=head,
        progress=True, desc="eval imagenet1k_val")
    T = fit_temperature(idval_logits, idval_lab)
    print(f"  ID-val temperature T*={T:.3f}  (z/T softmax; >1 softens, <1 sharpens)")
    if log:
        mlflow.log_metric("temp_T", T)
    report("imagenet1k_val", idval_logits, idval_lab)

    c_metrics = []
    for ood_key in OOD_KEYS:
        cache, hf = cache_and_hf(ood_key, "test")
        try:
            feats, labels = load_cached_features(cache, hf_repo, hf)
        except FileNotFoundError:
            print(f"  {ood_key:28s} SKIPPED (features not found)")
            continue
        subset = np.unique(labels) if ood_key in SUBSET_KEYS else None
        logits, lab = collect_logits(loss_fn, feats, labels, args.eval_batch_size,
                                     head=head, subset=subset, progress=True, desc=f"eval {ood_key}")
        m = report(ood_key, logits, lab)
        if ood_key in IMAGENET_C_KEYS:
            c_metrics.append(m)

    # ImageNet-C aggregate over corruptions.
    if c_metrics:
        means = {f"imagenet_c_mean_{k}": float(np.mean([m[k] for m in c_metrics]))
                 for k in METRIC_KEYS}
        print("  imagenet_c_mean              "
              + "  ".join(f"{k}={means['imagenet_c_mean_' + k]:.4f}" for k in METRIC_KEYS))
        if log:
            mlflow.log_metrics(means)

    if log:
        mlflow.end_run()
    print("\nDone.")


def build_argparser():
    p = argparse.ArgumentParser(description="OOD calibration evaluation (IBProbit vs CrossEntropy)")
    p.add_argument("--model", default="dinov3_huge", choices=list(EQUIMO_MODELS.keys()))
    p.add_argument("--method", required=True, choices=["IBProbit", "CrossEntropy"])
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--batch-size", type=int, required=True, help="Training batch size")
    p.add_argument("--eval-batch-size", type=int, default=8192, help="Evaluation batch size")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (default: 25 for IBProbit, 1000 for CrossEntropy)")
    # Gradient-training hyperparameters (CE head; IBProbit prior)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smooth", type=float, default=0.0, help="CrossEntropy only")
    p.add_argument("--patience", type=int, default=5)
    # IBProbit streaming trainer (process-noise CAVI; q self-tuned in-update, sigma_z^2 gauge-fixed at 1)
    p.add_argument("--num-update-iters", type=int, default=128, help="CAVI mean iters per batch")
    p.add_argument("--px", action="store_true",
                   help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    p.add_argument("--no-learn-q", action="store_true",
                   help="hold Q at its isotropic prior q0*I (the ablation arm); default infers it")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="prior scale on the process noise: q0 = alpha^2 / batch_size "
                        "(q-update-spec 5.5e). At alpha=1e-3 the q recursion is inert "
                        "(learn_q on/off give identical mu); 0.1 puts q0 near the "
                        "pre-Wishart 1/D operating point.")
    p.add_argument("--q-init", type=float, default=None,
                   help="override q0 directly; pass 1/D to reproduce the pre-Wishart regime")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart prior pseudo-count (default: 0.5, i.e. half a pseudo-observation against the K the likelihood adds per step)")
    p.add_argument("--warmup", type=int, default=1,
                   help="defer the gate and the q recursion for the first N steps (the mu=0 seed is uninformative)")
    p.add_argument("--eps-init", type=float, default=0.01,
                   help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). Forgetting rho = 1 - B/N set automatically.")
    # Infra
    p.add_argument("--run-id", default=None, help="Parent MLflow run ID")
    p.add_argument("--experiment-name", default="bllarse")
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--hf-repo", default="dimarkov/bllarse-features")
    p.add_argument("--no-hf-cache", action="store_true")
    p.add_argument("--device", default="gpu")
    p.add_argument("--log-mlflow", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.epochs is None:
        # Each IBProbit streaming epoch is a full CAVI pass; CE epochs are cheap.
        args.epochs = 25 if args.method == "IBProbit" else 1000
    config.update("jax_platform_name", args.device)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    main(args)
