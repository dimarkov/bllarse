"""
Full-Network Finetuning Comparison: IBProbit vs CrossEntropy

Finetunes the full dinov3 backbone on imagenet1k, comparing an IBProbit
Bayesian last layer (CAVI head update + AdamW backbone gradients through the
IBProbit loss) against a plain CrossEntropy linear head (AdamW on backbone +
head). Test ACC/ECE/NLL are computed on a fixed validation subset after every
optimizer step, for a fixed number of steps.

The backbone gradient is accumulated over micro-batches to reach a target
(effective) batch size that need not fit in memory: --batch-size sets the
target and each optimizer step accumulates grads over round(target / micro)
micro-batches of the largest size that fits (--micro-batch-size, else
auto-probed). For the IBProbit arm each micro-batch is forwarded ONCE: the trunk
gradient is taken against the current head and the backbone features are stored;
after the accumulated trunk gradient is applied, those stored features are
assimilated into the head in a SINGLE CAVI update over the whole target batch (so
the grad forward pass is reused -- no second forward). Trunk gradients are w.r.t.
the backbone only; the head scalars q and eps are self-tuned solely inside that
one update() call.

Both arms consume the identical shuffled data stream (same seed, same batch
order) and share AdamW hyperparameters for the backbone.

Example usage:
    python scripts/alternatives/full_finetune_comparison.py \
        --model dinov3_small --dataset imagenet1k \
        --batch-size 4096 --num-steps 500 --log-mlflow
"""

import os
import argparse
import gc
import time

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.secrets", override=False)

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
try:
    import mlflow
except ImportError:
    mlflow = None

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from jax import random as jr, config, vmap
from datasets import load_dataset

from blrax.states import ScaleByEvonState, MatrixEvonLeaf, _is_evon_leaf
from blrax.utils import (
    _evon_sample_leaves,
    _project,
    _project_back,
    precision as bayes_precision,
    rademacher_like,
    sample_posterior,
)

from bllarse.losses import IBProbit, CrossEntropy
from bllarse.utils import bayes_optimizer

from calibration import evaluate_classification
from vit_classification import (
    EQUIMO_MODELS,
    DATASET_CONFIGS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NORM_STATS,
    get_pretrained_backbone,
    iterate_batches,
    prefetch_iterator,
    _load_and_process,
)
from precompute_ood_features import build_ood_registry, load_wnid_to_idx


def _to_default_device(module):
    """Copy a module's arrays to the default (accelerator) device.

    Used to materialize per-arm device copies of the pristine backbone, which
    main() parks on host so it does not occupy GPU memory during training."""
    params, static = eqx.partition(module, eqx.is_array)
    return eqx.combine(jax.device_put(params, jax.devices()[0]), static)


def forward_features(backbone, images, key, bf16=False):
    """Batched backbone forward returning (batch, embed_dim) features.

    With ``bf16=True`` the backbone params and inputs are cast to bfloat16 for the
    forward pass (halving activation memory) while the output features are cast back
    to float32. Because the cast lives inside the differentiable region, JAX returns
    float32 gradients w.r.t. the original float32 master weights — the optimizer and
    grad accumulation stay in float32, only the saved activations are bf16."""
    if bf16:
        backbone = jax.tree.map(
            lambda a: a.astype(jnp.bfloat16) if eqx.is_array(a) else a, backbone)
        images = images.astype(jnp.bfloat16)

    def single(x):
        out = backbone(x, key=key)
        if isinstance(out, tuple):
            out = out[0]  # CLS token
        return out.astype(jnp.float32) if bf16 else out

    return vmap(single)(images)


def build_ibprobit(args, embed_dim, num_classes, key, rho=1.0, batch_size=None):
    """Construct the IBProbit head from the exposed CLI knobs (Inverse-Wishart process
    noise Q, PX scan, eps-contamination mixture). Mirrors the construction in
    scripts/alternatives/ood_evaluation.py so all three scripts share the same
    operating point. ``rho`` is the eps forgetting factor (callers pass 1 - B/N; the
    default 1.0 is fine for the throwaway batch-size probe)."""
    return IBProbit(
        embed_dim, num_classes, key=key,
        q_init=args.q_init, alpha=args.alpha, batch_size=batch_size,
        kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
        warmup=args.warmup, eps_init=args.eps_init, rho=rho,
    )


# --- Jitted micro-batch gradients + shared accumulate/apply (gradient accumulation) ---

@eqx.filter_jit
def ibprobit_micro_forward(backbone, loss_fn, images, labels, key, bf16=False):
    """One IBProbit micro-batch: single backbone forward, grad w.r.t. backbone only.

    The head is *not* updated here. The backbone features are returned so they can
    be stored and, after the accumulated backbone gradient is applied, assimilated
    into the head in a SINGLE CAVI update over the whole target batch. q and eps are
    tuned only inside that update()."""
    def loss_value_fn(bb):
        fts = forward_features(bb, images, key, bf16=bf16)
        loss = loss_fn(fts, labels, loss_type=2).mean()
        return loss, fts

    (loss, fts), grads = eqx.filter_value_and_grad(loss_value_fn, has_aux=True)(backbone)
    return grads, fts, loss


@eqx.filter_jit
def ibprobit_micro_forward_nograd(backbone, loss_fn, images, labels, key, bf16=False):
    """Forward-only micro-batch for the very first macro batch.

    With the seed posterior (mu=0) the backbone gradient is zero, so we skip
    gradient computation and only store features/labels for the head update."""
    fts = forward_features(backbone, images, key, bf16=bf16)
    loss = loss_fn(fts, labels, loss_type=2).mean()
    return fts, loss


@eqx.filter_jit
def ce_micro_grad(model, loss_fn, images, labels, key, bf16=False):
    """One CrossEntropy micro-batch: (backbone, head) grads + loss, one forward."""

    def loss_value_fn(m):
        bb, head = m
        fts = forward_features(bb, images, key, bf16=bf16)
        return loss_fn(vmap(head)(fts), labels).mean()

    loss, grads = eqx.filter_value_and_grad(loss_value_fn)(model)
    return eqx.filter(grads, eqx.is_array), loss


@partial(jax.jit, donate_argnums=0)
def _tree_add(grads, acc):
    """Sum two grad pytrees leaf-wise, tolerating None leaves.

    ``grads`` (the fresh micro-batch tree) is donated: the sum is written into
    its buffers, so accumulation holds two param-sized trees instead of three
    (~3.1 GiB saved off the recurring peak for dinov3_huge). Argument order
    matters -- the caller rebinds ``acc`` immediately (freeing its old buffers),
    while the stale ``grads`` name lives until the next micro-batch; donating
    ``grads`` means that stale reference holds no memory."""
    return jax.tree.map(
        lambda g, a: g if a is None else (a if g is None else g + a),
        grads, acc,
    )


@eqx.filter_jit(donate="all-except-first")
def apply_grads(model, grads, opt_state, optimizer, num_accum):
    """Average the accumulated grads over the window and take one optimizer step.

    ``grads`` and ``opt_state`` are donated: XLA writes the new params/optimizer
    state into their buffers instead of double-buffering ~3x the param size at the
    step boundary. The model is NOT donated -- stale Python references to its
    arrays remain in scope at call time (fwd_backbone/model aliases), and the
    once-per-step transient copy is cheap next to the per-micro-batch peak."""
    grads = jax.tree.map(lambda g: g / num_accum, grads)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(model, updates), opt_state


# --- Bayesian (IVON/EVON) optimizer support ---
#
# blrax's noisy_value_and_grad estimates gradient and Hessian in one call, which
# does not compose with gradient accumulation. Instead we draw ONE posterior
# sample (estimator='sampling') or ONE Rademacher probe (estimator='hutchinson')
# per optimizer step, shared across the whole accumulation window, and average
# grads (and HVPs) over the micro-batches. Because the macro-batch loss is the
# mean of the micro losses, this reproduces exactly a single-sample IVON/EVON
# step on the full effective batch.


def make_bayes_optimizer(args, ess, param_labels=None):
    """IVON/EVON over the optimized params; ess = training-set size.

    With ``param_labels`` (CE arm) and --bayes-head-lr, the head gets its own
    learning rate: bayes_optimizer builds the chain flat so the blrax state
    stays at opt_state[0] and posterior sampling keeps working."""
    head_lr = args.bayes_head_lr if param_labels is not None else None
    common = dict(ess=float(ess), hess_init=args.ivon_hess_init,
                  weight_decay=args.ivon_weight_decay, clip_radius=args.clip_radius,
                  head_learning_rate=head_lr, param_labels=param_labels)
    if args.optimizer == "ivon":
        return bayes_optimizer(
            "ivon", args.bayes_lr, b1=args.ivon_b1, b2=args.ivon_b2,
            hess_every=args.hess_every if args.hess_every is not None else 1,
            **common)
    return bayes_optimizer(
        "evon", args.bayes_lr, b1=args.evon_b1, b2=args.evon_b2, b3=args.evon_b3,
        precond_every=args.precond_every, hess_every=args.hess_every,
        hess_clip_ratio=args.hess_clip_ratio,
        max_precond_dim=args.max_precond_dim,
        one_sided=args.one_sided, **common)


@eqx.filter_jit
def draw_bayes_sample(key, params, opt_state):
    """One posterior draw around the mean ``params`` for this optimizer step.

    Returns (sampled_params, noise_aux): for IVON the param-space noise pytree,
    for EVON the list of eigenbasis draws (leaf order) to stash before update."""
    st = opt_state[0]
    if isinstance(st, ScaleByEvonState):
        return _evon_sample_leaves(key, params, st)
    return sample_posterior(key, params, st)


@eqx.filter_jit
def draw_hutchinson_probe(key, params, opt_state):
    """One Rademacher probe for this optimizer step.

    Returns (tangent_tree, probe_aux): the param-space tangent for the HVP and
    the raw probe needed to assemble the Hessian estimate (for EVON the probe is
    drawn in eigenbasis coordinates and projected back, mirroring
    blrax.utils.evon_hutchinson_and_grad)."""
    st = opt_state[0]
    if isinstance(st, ScaleByEvonState):
        p_leaves, treedef = jax.tree.flatten(params)
        s_leaves = jax.tree.leaves(st.leaves, is_leaf=_is_evon_leaf)
        probes, tangents = [], []
        k = key
        for p, s in zip(p_leaves, s_leaves):
            k, rk = jr.split(k)
            E = jr.rademacher(rk, shape=s.H.shape, dtype=p.dtype)
            if isinstance(s, MatrixEvonLeaf):
                Vp = _project_back(s.QL, s.QR, E).reshape(p.shape)
            else:
                Vp = E.reshape(p.shape)
            probes.append(E)
            tangents.append(Vp)
        return jax.tree.unflatten(treedef, tangents), probes
    u = rademacher_like(key, params)
    return u, u


def _stash_bayes_hessian(opt_state, grads_avg, estimator, noise_aux, hvp_avg):
    """Attach this step's Hessian estimate to the blrax state before update().

    ``hvp_avg is None`` on hutchinson skip-steps (hess_every gating): the previous
    Hessian is handed back so the EMA is a no-op, matching blrax's reuse path."""
    st = opt_state[0]
    if isinstance(st, ScaleByEvonState):
        s_leaves = jax.tree.leaves(st.leaves, is_leaf=_is_evon_leaf)
        if estimator == "hutchinson":
            if hvp_avg is None:
                new = [s._replace(h_hat=s.H) for s in s_leaves]
            else:
                hvp_leaves = jax.tree.leaves(hvp_avg)
                new = []
                for s, E, hv in zip(s_leaves, noise_aux, hvp_leaves):
                    if isinstance(s, MatrixEvonLeaf):
                        d, o = s.H.shape
                        new.append(s._replace(
                            h_hat=E * _project(s.QL, s.QR, hv.reshape(d, o))))
                    else:
                        new.append(s._replace(h_hat=E * hv))
        else:
            # update() computes Hhat = noise * (Q^T g_avg Q) * precision internally.
            new = [s._replace(noise=n) for s, n in zip(s_leaves, noise_aux)]
        _, treedef = jax.tree.flatten(grads_avg)
        st = st._replace(leaves=jax.tree.unflatten(treedef, new))
    else:  # ScaleByIvonState
        if estimator == "hutchinson":
            if hvp_avg is None:
                h_bar = st.hess
            else:
                h_bar = jax.tree.map(lambda u, hv: u * hv, noise_aux, hvp_avg)
        else:
            h_bar = jax.tree.map(
                lambda g, n, h: g * n * bayes_precision(h, st.ess, st.weight_decay),
                grads_avg, noise_aux, st.hess)
            # blrax freezes the hessian EMA outside estimation steps.
            estimate = (st.count % st.hess_every) == 0
            h_bar = jax.tree.map(
                lambda hb, h: jnp.where(estimate, hb, h), h_bar, st.hess)
        st = st._replace(h_bar=h_bar)
    return (st,) + tuple(opt_state[1:])


@eqx.filter_jit
def ibprobit_micro_hvp(backbone, tangent, loss_fn, images, labels, key, bf16=False):
    """IBProbit micro-batch for the Hutchinson estimator: gradient at the mean plus
    one Hessian-vector product with the shared per-step probe (forward-over-reverse).
    Features come from the mean backbone, so the head CAVI update sees clean features."""
    params, static = eqx.partition(backbone, eqx.is_array)

    def loss_value_fn(p):
        bb = eqx.combine(p, static)
        fts = forward_features(bb, images, key, bf16=bf16)
        loss = loss_fn(fts, labels, loss_type=2).mean()
        return loss, fts

    ((loss, fts), grads), (_, hvp) = jax.jvp(
        jax.value_and_grad(loss_value_fn, has_aux=True), (params,), (tangent,))
    return grads, hvp, fts, loss


@eqx.filter_jit
def ce_micro_hvp(model, tangent, loss_fn, images, labels, key, bf16=False):
    """CrossEntropy micro-batch for the Hutchinson estimator ((backbone, head))."""
    params, static = eqx.partition(model, eqx.is_array)

    def loss_value_fn(p):
        bb, head = eqx.combine(p, static)
        fts = forward_features(bb, images, key, bf16=bf16)
        return loss_fn(vmap(head)(fts), labels).mean()

    (loss, grads), (_, hvp) = jax.jvp(
        jax.value_and_grad(loss_value_fn), (params,), (tangent,))
    return grads, hvp, loss


@eqx.filter_jit(donate="all-except-first")
def apply_grads_bayes(model, grads, opt_state, optimizer, num_accum,
                      estimator, noise_aux, hvp_acc):
    """apply_grads for IVON/EVON: average grads (and HVPs), stash the per-step
    Hessian estimate into the blrax state, then take one optimizer step at the
    mean ``model``. Same donation contract as apply_grads: grads/opt_state (and
    the per-step noise/hvp trees) are donated, the model is not."""
    grads = jax.tree.map(lambda g: g / num_accum, grads)
    hvp_avg = (jax.tree.map(lambda h: h / num_accum, hvp_acc)
               if hvp_acc is not None else None)
    params = eqx.filter(model, eqx.is_array)
    opt_state = _stash_bayes_hessian(opt_state, grads, estimator, noise_aux, hvp_avg)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(model, updates), opt_state


# --- Evaluation ---

@eqx.filter_jit
def _eval_logits_ibprobit(backbone, loss_fn, images, labels, key, bf16=False):
    features = forward_features(backbone, images, key, bf16=bf16)
    _, logits = loss_fn(features, labels, with_logits=True, loss_type=2)
    return logits


@eqx.filter_jit
def _eval_logits_ce(backbone, head, images, key, bf16=False):
    features = forward_features(backbone, images, key, bf16=bf16)
    return vmap(head)(features)


def evaluate(arm, backbone, head_or_loss, eval_images, eval_labels, chunk_size, key, bf16=False):
    """Chunked eval on preloaded host arrays. Returns (acc, ece, nll)."""
    inference_bb = eqx.nn.inference_mode(backbone, True)
    all_logits = []
    n = len(eval_images)
    for start in range(0, n, chunk_size):
        imgs = jnp.asarray(eval_images[start:start + chunk_size])
        labels = jnp.asarray(eval_labels[start:start + chunk_size])
        if arm == "ibprobit":
            logits = _eval_logits_ibprobit(inference_bb, head_or_loss, imgs, labels, key, bf16=bf16)
        else:
            logits = _eval_logits_ce(inference_bb, head_or_loss, imgs, key, bf16=bf16)
        all_logits.append(logits)

    all_logits = jnp.concatenate(all_logits, axis=0)
    acc, ece, nll = evaluate_classification(all_logits, jnp.asarray(eval_labels))
    return float(acc), float(ece), float(nll)


def load_eval_subset(split, ds_config, img_size, mean, std, n_subset, key, num_workers=8):
    """Decode/resize a fixed random validation subset once into host arrays (CHW)."""
    n_total = len(split)
    if n_subset >= n_total:
        indices = np.arange(n_total)
    else:
        indices = np.array(jr.choice(key, n_total, shape=(n_subset,), replace=False))

    all_labels = np.array(split[ds_config["label_key"]])[indices]
    worker_fn = partial(_load_and_process, split, ds_config["img_key"], img_size)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        images = np.stack(list(pool.map(worker_fn, indices)))
    images = (images - mean) / std
    images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
    return images, all_labels


# --- OOD evaluation (ImageNet-C / R / A / V2), ported from ood_evaluation.py ---
#
# The backbone is being finetuned, so OOD features cannot be precomputed: each set
# is loaded as RAW images (relabeled to canonical ImageNet-1k indices) and
# forwarded through the CURRENT backbone/head at the finetuning resolution. We
# report acc/ece/nll/brier per set with NO temperature scaling -- the network is
# finetuned, so a post-hoc ID temperature is not applied. Only valid when the ID
# task is imagenet1k (canonical 1000-class label space).

IMAGENET_C_KEYS = [
    f"imagenet_c_{c}" for c in
    ["brightness", "fog", "gaussian_noise", "jpeg_compression",
     "motion_blur", "pixelate", "shot_noise", "zoom_blur"]
]
OOD_KEYS = IMAGENET_C_KEYS + ["imagenet_r", "imagenet_a", "imagenet_v2"]
SUBSET_KEYS = {"imagenet_r", "imagenet_a"}  # 200-class subsets: mask out-of-subset logits
MASK_NEG = -1e9  # additive logit mask for out-of-subset classes

# Default OOD set = the two cheapest datasets. All 8 ImageNet-C corruptions are
# 50k (tied -> jpeg_compression by choice); imagenet_a (7.5k) is the shortest of
# the R/A/V2 size-variants. Override with --ood-keys (or --ood-keys all).
DEFAULT_OOD_KEYS = ["imagenet_c_jpeg_compression", "imagenet_a"]


def resolve_ood_keys(keys):
    """CLI --ood-keys -> concrete OOD dataset list. None -> the two cheapest;
    ['all'] -> every dataset; else validated against OOD_KEYS."""
    if keys is None:
        return list(DEFAULT_OOD_KEYS)
    if list(keys) == ["all"]:
        return list(OOD_KEYS)
    unknown = [k for k in keys if k not in OOD_KEYS]
    if unknown:
        raise ValueError(f"unknown OOD key(s) {unknown}; choose from {OOD_KEYS} or 'all'")
    return list(keys)


def compute_brier(logits, labels):
    """Multiclass Brier score: mean_i sum_k (p_ik - 1[k=y_i])^2, in [0, 2]."""
    p = jax.nn.softmax(logits, axis=-1)
    oh = jax.nn.one_hot(labels, p.shape[-1])
    return jnp.mean(jnp.sum((p - oh) ** 2, axis=-1))


def _ood_metrics_from_logits(logits, labels, subset=None):
    """acc/ece/nll/brier on (optionally subset-masked) logits. No temperature scaling.

    For 200-class subsets (ImageNet-R/A) out-of-subset logits are pushed to
    MASK_NEG so the softmax/argmax is restricted to the present classes."""
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    if subset is not None:
        keep = jnp.zeros(logits.shape[-1], dtype=bool).at[jnp.asarray(subset)].set(True)
        logits = jnp.where(keep, logits, MASK_NEG)
    acc, ece, nll = evaluate_classification(logits, labels)
    return {"acc": float(acc), "ece": float(ece), "nll": float(nll),
            "brier": float(compute_brier(logits, labels))}


def load_ood_subset(name, prepare_fn, wnid_to_idx, img_size, mean, std, n_subset, key):
    """Load a fixed random subset of one OOD split into host arrays (CHW), once.

    Subset classes for ImageNet-R/A are taken from the FULL split labels so the
    logit mask covers every present class regardless of which samples were drawn."""
    split, img_key = prepare_fn(wnid_to_idx)
    subset = np.unique(np.asarray(split["label"])) if name in SUBSET_KEYS else None
    ds_config = {"img_key": img_key, "label_key": "label"}
    images, labels = load_eval_subset(split, ds_config, img_size, mean, std, n_subset, key)
    return images, labels, subset


def evaluate_ood(arm, backbone, head_or_loss, images, labels, subset, chunk_size, key, bf16=False):
    """Forward preloaded OOD host arrays through the CURRENT backbone/head and
    return the 4 OOD metrics (subset-masked for ImageNet-R/A). Mirrors evaluate()
    but keeps the logits so the mask/Brier can be applied."""
    inference_bb = eqx.nn.inference_mode(backbone, True)
    all_logits = []
    for start in range(0, len(images), chunk_size):
        imgs = jnp.asarray(images[start:start + chunk_size])
        labs = jnp.asarray(labels[start:start + chunk_size])
        if arm == "ibprobit":
            logits = _eval_logits_ibprobit(inference_bb, head_or_loss, imgs, labs, key, bf16=bf16)
        else:
            logits = _eval_logits_ce(inference_bb, head_or_loss, imgs, key, bf16=bf16)
        all_logits.append(logits)
    return _ood_metrics_from_logits(jnp.concatenate(all_logits, axis=0), labels, subset)


# --- Micro-batch size probe ---

def probe_micro_batch_size(backbone, args, embed_dim, num_classes, img_size,
                           init_bs, max_bs, key):
    """Double the micro-batch size until the (heavier) IBProbit micro-step OOMs;
    return the last size that fits. This is the per-forward micro-batch -- the target
    (effective) batch is reached by gradient accumulation over several of these."""
    bf16 = args.bf16
    bs = init_bs
    last_ok = None
    while bs <= max_bs:
        try:
            loss_fn = build_ibprobit(args, embed_dim, num_classes, key, batch_size=bs)
            images = jnp.zeros((bs, 3, img_size, img_size), dtype=jnp.float32)
            labels = jnp.zeros((bs,), dtype=jnp.int32)
            if args.optimizer in ("ivon", "evon") and args.estimator == "hutchinson":
                # The jvp-of-grad pass roughly doubles activation memory; probe
                # with it so the chosen micro-batch also fits on HVP steps.
                params = eqx.filter(backbone, eqx.is_array)
                tangent = jax.tree.map(jnp.zeros_like, params)
                out = ibprobit_micro_hvp(backbone, tangent, loss_fn, images, labels, key, bf16=bf16)
            else:
                out = ibprobit_micro_forward(backbone, loss_fn, images, labels, key, bf16=bf16)
            jax.block_until_ready(out[-1])
            last_ok = bs
            print(f"[probe] micro_batch_size={bs} OK")
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() or "OOM" in msg:
                print(f"[probe] micro_batch_size={bs} OOM, backing off")
                break
            raise
        finally:
            del loss_fn
            jax.clear_caches()
            gc.collect()
        bs *= 2

    if last_ok is None:
        raise RuntimeError(f"Even micro_batch_size={init_bs} does not fit in memory.")
    return last_ok


# --- Training loops ---

def stream_batches(train_split, ds_config, batch_size, img_size, mean, std, num_steps, seed):
    """Yield exactly num_steps batches, reshuffling and continuing if a pass ends."""
    epoch = 0
    steps = 0
    while steps < num_steps:
        shuffle_key = jr.fold_in(jr.PRNGKey(seed), epoch)
        batches = iterate_batches(
            train_split, ds_config["img_key"], ds_config["label_key"],
            batch_size=batch_size, img_size=img_size, mean=mean, std=std,
            key=shuffle_key, channels_first=True, progress=False,
        )
        for images, labels in prefetch_iterator(batches, n_prefetch=10):
            if len(labels) < batch_size:
                continue  # keep batch shape constant for jit
            yield images, labels
            steps += 1
            if steps >= num_steps:
                return
        epoch += 1


def run_arm(arm, args, backbone_init, train_split, ds_config, eval_images, eval_labels,
            micro, num_accum, embed_dim, num_classes, img_size, mean, std, key,
            ood_data=None):
    print(f"\n{'='*60}\nArm: {arm}\n{'='*60}")

    effective_batch = micro * num_accum
    # Keep the backbone in inference mode during finetuning: the pretrained DINOv3
    # ships with all DropPath/Dropout rates at 0.0, so training mode adds no
    # regularization and only differs by consuming the forward key. Inference mode
    # makes the forward deterministic and independent of the per-step key.
    # backbone_init lives on host; make this arm's device copy here.
    backbone = eqx.nn.inference_mode(_to_default_device(backbone_init), True)
    key, init_key, fwd_key = jr.split(key, 3)
    sample_key = jr.fold_in(key, 0x5A)  # per-step posterior draws / probes
    bayes_opt = args.optimizer in ("ivon", "evon")

    if arm == "ibprobit":
        # eps forgetting rho = 1 - B/N: the head assimilates the whole effective batch each
        # step, so B = effective_batch, N = train-set size (one-pass contamination memory).
        eps_rho = max(0.0, 1.0 - effective_batch / len(train_split))
        loss_fn = build_ibprobit(args, embed_dim, num_classes, init_key, rho=eps_rho,
                                  batch_size=effective_batch)
        head = None
        model = None
        if bayes_opt:
            optimizer = make_bayes_optimizer(args, ess=len(train_split))
        else:
            optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    else:
        loss_fn = CrossEntropy(args.label_smooth, num_classes)
        head = eqx.nn.Linear(embed_dim, num_classes, key=init_key)
        model = (backbone, head)
        # The fresh head needs a much higher lr than the pretrained backbone:
        # AdamW moves weights by ~lr/step, and the head's random init scale
        # (~0.015) would take thousands of 1e-4 steps to overwrite.
        param_labels = (
            jax.tree.map(lambda _: "backbone", eqx.filter(backbone, eqx.is_array)),
            jax.tree.map(lambda _: "head", eqx.filter(head, eqx.is_array)),
        )
        if bayes_opt:
            # Separate head lr (--bayes-head-lr) via the flat bayes_optimizer
            # chain; the blrax state stays at opt_state[0] so sampling works.
            optimizer = make_bayes_optimizer(
                args, ess=len(train_split), param_labels=param_labels)
        else:
            optimizer = optax.multi_transform(
                {
                    "backbone": optax.adamw(args.lr, weight_decay=args.weight_decay),
                    "head": optax.adamw(args.head_lr, weight_decay=args.weight_decay),
                },
                param_labels,
            )
    opt_state = optimizer.init(eqx.filter(model if model is not None else backbone, eqx.is_array))
    # Drop the (backbone, head) tuple: it is only needed for optimizer.init, and
    # keeping it in scope would pin the arm's initial backbone buffers for the
    # whole run after apply_grads rebinds `backbone` to updated arrays.
    del model

    if args.log_mlflow:
        # Nest under the parent run the SAME way scripts/finetuning.py does: pass
        # parent_run_id + nested=True. Setting the reserved ``mlflow.parentRunId`` tag by
        # hand does not make the UI group the child under the parent (MLflow 3.x).
        mlflow.start_run(run_name=f"full_finetune_{arm}", tags={"arm": arm},
                         nested=bool(args.run_id), parent_run_id=args.run_id)
        run_params = {
            "model": args.model,
            "dataset": args.dataset,
            "arm": arm,
            "img_size": img_size,
            "batch_size": effective_batch,
            "micro_batch_size": micro,
            "num_accum_steps": num_accum,
            "num_steps": args.num_steps,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "num_update_iters": args.num_update_iters,
            "label_smooth": args.label_smooth,
            "bf16": args.bf16,
            "eval_subset": len(eval_labels),
            "seed": args.seed,
        }
        if args.optimizer in ("ivon", "evon"):
            run_params.update(
                estimator=args.estimator, bayes_lr=args.bayes_lr,
                bayes_head_lr=args.bayes_head_lr,
                ivon_weight_decay=args.ivon_weight_decay,
                ivon_hess_init=args.ivon_hess_init, hess_every=args.hess_every,
            )
            if args.optimizer == "ivon":
                run_params.update(ivon_b2=args.ivon_b2)
            else:
                run_params.update(
                    evon_b2=args.evon_b2, evon_b3=args.evon_b3,
                    precond_every=args.precond_every,
                    hess_clip_ratio=args.hess_clip_ratio,
                    max_precond_dim=args.max_precond_dim,
                    one_sided=args.one_sided,
                )
        if arm == "ibprobit":
            run_params.update(
                px=args.px, learn_q=not args.no_learn_q, alpha=args.alpha,
                q_init=args.q_init, kappa=args.kappa, warmup=args.warmup,
                eps_init=args.eps_init,
                eps_rho=max(0.0, 1.0 - effective_batch / len(train_split)),
            )
        if ood_data:
            run_params.update(
                ood_eval=True, ood_every=args.ood_every, ood_subset=args.ood_subset,
                ood_keys=",".join(name for name, *_ in ood_data),
            )
        mlflow.log_params(run_params)

    # Initial evaluation (step 0)
    head_or_loss = loss_fn if arm == "ibprobit" else head
    acc, ece, nll = evaluate(arm, backbone, head_or_loss, eval_images, eval_labels,
                             args.eval_chunk, fwd_key, bf16=args.bf16)
    print(f"step 0: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")
    if args.log_mlflow:
        mlflow.log_metrics({"test_acc": acc, "test_ece": ece, "test_nll": nll}, step=0)

    total_micro = args.num_steps * num_accum
    micro_iter = stream_batches(train_split, ds_config, micro, img_size,
                                mean, std, total_micro, args.seed)
    pbar = tqdm(range(1, args.num_steps + 1), total=args.num_steps, desc=arm, unit="step")
    train_time = 0.0
    # IBProbit step 1 uses the no-grad path (compiles ibprobit_micro_forward_nograd);
    # step 2 first uses the grad path (compiles ibprobit_micro_forward + _tree_add +
    # apply_grads). CE step 1 compiles ce_micro_grad + _tree_add + apply_grads. Exclude
    # all compilation steps from the steady-state timing.
    warmup_steps = 2 if arm == "ibprobit" else 1
    for step in pbar:
        t0 = time.perf_counter()
        # Per-step training forward key so stochastic-depth / dropout masks differ
        # across steps (eval keeps the fixed fwd_key for determinism).
        train_fwd_key = jr.fold_in(fwd_key, step)
        # Accumulate backbone grads over `num_accum` micro-batches -> one optimizer step.
        # For IBProbit we forward with the *current* head, accumulate backbone grads,
        # store the features, then update the backbone, and finally assimilate the whole
        # macro-batch of stored features into the head in a single CAVI update. q and eps
        # are tuned only inside that one update(). For CE the (backbone, head) grads simply
        # sum. Forward-pass activations free between micro-batches; stored features add
        # effective_batch x embed_dim of device memory (small relative to activations).
        # Bayesian optimizers: one posterior sample (sampling) or one Rademacher
        # probe (hutchinson) per optimizer step, shared across the accumulation
        # window (see the blrax support section above). Sampling forwards every
        # micro-batch at the SAMPLED params (so the stored IBProbit features are
        # one posterior draw of the trunk); hutchinson forwards at the mean.
        noise_aux, tangent, hvp_acc = None, None, None
        hutch_step = False
        fwd_backbone, fwd_head = backbone, head
        if bayes_opt and not (arm == "ibprobit" and step == 1):
            step_module = backbone if arm == "ibprobit" else (backbone, head)
            step_params, step_static = eqx.partition(step_module, eqx.is_array)
            draw_key = jr.fold_in(sample_key, step)
            st = opt_state[0]
            if args.estimator == "hutchinson":
                # HVP cadence: IVON estimates on count % k == 0, EVON on
                # (count + 1) % k == 0 (aligned with its eigenbasis refresh).
                c, k_every = int(st.count), max(int(st.hess_every), 1)
                hutch_step = (((c + 1) % k_every == 0)
                              if isinstance(st, ScaleByEvonState)
                              else (c % k_every == 0))
                if hutch_step:
                    tangent, noise_aux = draw_hutchinson_probe(draw_key, step_params, opt_state)
            else:
                sampled, noise_aux = draw_bayes_sample(draw_key, step_params, opt_state)
                sampled_module = eqx.combine(sampled, step_static)
                if arm == "ibprobit":
                    fwd_backbone = sampled_module
                else:
                    fwd_backbone, fwd_head = sampled_module

        grads_acc, loss_sum = None, 0.0
        all_fts, all_labels = [], []
        for _ in range(num_accum):
            images, labels = next(micro_iter)
            if arm == "ibprobit":
                if step == 1:
                    # Seed posterior (mu=0) gives zero backbone gradient; skip it.
                    fts, mloss = ibprobit_micro_forward_nograd(
                        backbone, loss_fn, images, labels, train_fwd_key, bf16=args.bf16)
                elif hutch_step:
                    grads, hvp, fts, mloss = ibprobit_micro_hvp(
                        fwd_backbone, tangent, loss_fn, images, labels, train_fwd_key, bf16=args.bf16)
                    grads_acc = grads if grads_acc is None else _tree_add(grads, grads_acc)
                    hvp_acc = hvp if hvp_acc is None else _tree_add(hvp, hvp_acc)
                else:
                    grads, fts, mloss = ibprobit_micro_forward(
                        fwd_backbone, loss_fn, images, labels, train_fwd_key, bf16=args.bf16)
                    grads_acc = grads if grads_acc is None else _tree_add(grads, grads_acc)
                all_fts.append(fts)
                all_labels.append(labels)
            else:
                if hutch_step:
                    grads, hvp, mloss = ce_micro_hvp(
                        (fwd_backbone, fwd_head), tangent, loss_fn, images, labels,
                        train_fwd_key, bf16=args.bf16)
                    hvp_acc = hvp if hvp_acc is None else _tree_add(hvp, hvp_acc)
                else:
                    grads, mloss = ce_micro_grad(
                        (fwd_backbone, fwd_head), loss_fn, images, labels, train_fwd_key, bf16=args.bf16)
                grads_acc = grads if grads_acc is None else _tree_add(grads, grads_acc)
            loss_sum = loss_sum + mloss
        if arm == "ibprobit":
            if step > 1:
                if bayes_opt:
                    backbone, opt_state = apply_grads_bayes(
                        backbone, grads_acc, opt_state, optimizer, num_accum,
                        args.estimator, noise_aux, hvp_acc)
                else:
                    backbone, opt_state = apply_grads(
                        backbone, grads_acc, opt_state, optimizer, num_accum)
            macro_fts = jnp.concatenate(all_fts, axis=0)
            macro_labels = jnp.concatenate(all_labels, axis=0)
            loss_fn = loss_fn.update(macro_fts, macro_labels, num_iters=args.num_update_iters)
            head_or_loss = loss_fn
        else:
            if bayes_opt:
                (backbone, head), opt_state = apply_grads_bayes(
                    (backbone, head), grads_acc, opt_state, optimizer, num_accum,
                    args.estimator, noise_aux, hvp_acc)
            else:
                (backbone, head), opt_state = apply_grads(
                    (backbone, head), grads_acc, opt_state, optimizer, num_accum)
            head_or_loss = head
        loss = float(jax.block_until_ready(loss_sum)) / num_accum
        step_time = time.perf_counter() - t0
        if step > warmup_steps:
            train_time += step_time

        metrics = {"train_loss": loss, "step_time": step_time, "train_time": train_time}
        if arm == "ibprobit":
            metrics["ibprobit_q"] = float(jnp.trace(loss_fn.Qhat) / loss_fn.Qhat.shape[0])
            metrics["ibprobit_eps"] = float(loss_fn.eps)
        if step % args.eval_every == 0 or step == args.num_steps:
            acc, ece, nll = evaluate(arm, backbone, head_or_loss, eval_images,
                                     eval_labels, args.eval_chunk, fwd_key, bf16=args.bf16)
            metrics.update({"test_acc": acc, "test_ece": ece, "test_nll": nll})
            pbar.set_postfix(acc=f"{acc:.4f}", ece=f"{ece:.4f}", nll=f"{nll:.4f}")
        if args.log_mlflow:
            mlflow.log_metrics(metrics, step=step)

        # OOD evaluation -- forward the preloaded OOD subsets through the current
        # backbone/head every ood_every steps (and at the final step) and log the
        # 4 metrics per dataset. Runs after step_time is banked above, so it is
        # excluded from the reported train_time.
        if ood_data and (step % args.ood_every == 0 or step == args.num_steps):
            ood_log, c_metrics = {}, []
            for name, oimgs, olabels, osubset in ood_data:
                m = evaluate_ood(arm, backbone, head_or_loss, oimgs, olabels,
                                 osubset, args.eval_chunk, fwd_key, bf16=args.bf16)
                for mk, mv in m.items():
                    ood_log[f"ood_{name}_{mk}"] = mv
                if name in IMAGENET_C_KEYS:
                    c_metrics.append(m)
            if c_metrics:
                for mk in ("acc", "ece", "nll", "brier"):
                    ood_log[f"ood_imagenet_c_mean_{mk}"] = float(
                        np.mean([m[mk] for m in c_metrics]))
            tqdm.write("  [ood step {}] ".format(step) + "  ".join(
                "{}: acc={:.3f} nll={:.3f}".format(
                    name, ood_log[f"ood_{name}_acc"], ood_log[f"ood_{name}_nll"])
                for name, *_ in ood_data))
            if args.log_mlflow:
                mlflow.log_metrics(ood_log, step=step)
    pbar.close()

    n_timed = max(args.num_steps - warmup_steps, 1)
    print(f"[{arm}] train time: {train_time:.1f}s total, "
          f"{train_time / n_timed * 1000:.0f}ms/step (excl. first {warmup_steps} compile step(s))")
    if args.log_mlflow:
        mlflow.log_metrics({
            "total_train_time": train_time,
            "mean_step_time": train_time / n_timed,
        })

    return backbone, head_or_loss


def final_full_eval(arm, backbone, head_or_loss, test_split, ds_config, img_size,
                    mean, std, args, key):
    """Stream the full test split through the eval path."""
    inference_bb = eqx.nn.inference_mode(backbone, True)
    all_logits, all_labels = [], []
    batches = iterate_batches(
        test_split, ds_config["img_key"], ds_config["label_key"],
        batch_size=args.eval_chunk, img_size=img_size, mean=mean, std=std,
        key=None, channels_first=True,
    )
    for images, labels in prefetch_iterator(batches, n_prefetch=2):
        if arm == "ibprobit":
            logits = _eval_logits_ibprobit(inference_bb, head_or_loss, images, labels, key, bf16=args.bf16)
        else:
            logits = _eval_logits_ce(inference_bb, head_or_loss, images, key, bf16=args.bf16)
        all_logits.append(np.asarray(logits))
        all_labels.append(np.asarray(labels))

    logits = jnp.asarray(np.concatenate(all_logits))
    labels = jnp.asarray(np.concatenate(all_labels))
    acc, ece, nll = evaluate_classification(logits, labels)
    return float(acc), float(ece), float(nll)


def main(args):
    ds_config = DATASET_CONFIGS[args.dataset]
    model_config = EQUIMO_MODELS[args.model]
    embed_dim = model_config["embed_dim"]
    num_classes = ds_config["num_classes"]
    img_size = args.img_size

    key = jr.PRNGKey(args.seed)
    mean, std = NORM_STATS.get(args.dataset, (IMAGENET_MEAN, IMAGENET_STD))

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(ds_config["hf_path"], name=ds_config.get("hf_config"))
    train_split = ds["train"]
    test_split = ds[ds_config["test_split"]]
    print(f"Train samples: {len(train_split)}, Test samples: {len(test_split)}")

    print(f"Loading pretrained model: {args.model}")
    key, model_key = jr.split(key)
    backbone_init = get_pretrained_backbone(args.model, model_key)
    # Park the pristine weights on host: each consumer makes its own device copy
    # via _to_default_device, so during training the GPU holds ONE param tree
    # (the current weights) instead of two (current + pristine-for-the-next-arm).
    # For dinov3_huge that second fp32 tree is ~3.1 GiB.
    _params, _static = eqx.partition(backbone_init, eqx.is_array)
    backbone_init = eqx.combine(jax.device_put(_params, jax.devices("cpu")[0]), _static)
    del _params, _static

    key, subset_key = jr.split(key)
    print(f"Preloading evaluation subset ({args.eval_subset} images)...")
    eval_images, eval_labels = load_eval_subset(
        test_split, ds_config, img_size, mean, std, args.eval_subset, subset_key,
    )

    # Preload the OOD eval subsets once (shared across arms). ImageNet-only, since
    # the OOD sets live in the canonical 1000-class ImageNet-1k label space.
    ood_data = None
    if args.ood_eval:
        if args.dataset != "imagenet1k":
            print(f"[ood] --ood-eval requires --dataset imagenet1k "
                  f"(got {args.dataset}); skipping OOD evaluation.")
        else:
            ood_keys = resolve_ood_keys(args.ood_keys)
            print(f"Preloading OOD subsets ({args.ood_subset}/set) at {img_size}px: {ood_keys}")
            wnid_to_idx = load_wnid_to_idx()
            registry = build_ood_registry()
            ood_data = []
            for name in ood_keys:
                key, ood_key_rng = jr.split(key)
                oimgs, olabels, osubset = load_ood_subset(
                    name, registry[name], wnid_to_idx, img_size, mean, std,
                    args.ood_subset, ood_key_rng,
                )
                ood_data.append((name, oimgs, olabels, osubset))
                extra = f", {len(osubset)}-class subset" if osubset is not None else ""
                print(f"  {name}: {len(olabels)} images{extra}")

    # Determine micro-batch (per-forward) and target (effective, via accumulation) sizes.
    if args.micro_batch_size is not None:
        micro = args.micro_batch_size
        print(f"Using fixed micro-batch size: {micro}")
    else:
        print("Probing maximal micro-batch size...")
        key, probe_key = jr.split(key)
        micro = probe_micro_batch_size(
            eqx.nn.inference_mode(_to_default_device(backbone_init), True),
            args, embed_dim, num_classes,
            img_size, args.batch_size_init, args.batch_size_max, probe_key,
        )
        print(f"Probed micro-batch size: {micro}")

    target = args.batch_size if args.batch_size is not None else micro
    if target < micro:
        micro = target  # target smaller than what fits -> just use one micro-batch
    num_accum = max(1, round(target / micro))
    effective_batch = micro * num_accum
    if effective_batch != target:
        print(f"Note: target batch {target} adjusted to {effective_batch} "
              f"(= {num_accum} x micro {micro})")
    print(f"Batch config: effective~{effective_batch} = {num_accum} accum-step(s) x micro {micro}")

    if args.log_mlflow:
        if mlflow is None:
            raise ImportError("--log-mlflow requires mlflow: pip install mlflow")
        mlflow.set_experiment(args.experiment_name)

    arms = ["ibprobit", "ce"] if args.arm == "both" else [args.arm]
    for arm in arms:
        key, arm_key = jr.split(jr.PRNGKey(args.seed))  # identical key per arm
        backbone, head_or_loss = run_arm(
            arm, args, backbone_init, train_split, ds_config,
            eval_images, eval_labels, micro, num_accum, embed_dim, num_classes,
            img_size, mean, std, arm_key, ood_data=ood_data,
        )

        if args.final_full_eval:
            acc, ece, nll = final_full_eval(
                arm, backbone, head_or_loss, test_split, ds_config,
                img_size, mean, std, args, arm_key,
            )
            print(f"[{arm}] final full eval: acc={acc:.4f}, ece={ece:.4f}, nll={nll:.4f}")
            if args.log_mlflow:
                mlflow.log_metrics({
                    "final_full_acc": acc, "final_full_ece": ece, "final_full_nll": nll,
                })

        if args.log_mlflow:
            mlflow.end_run()

        del backbone, head_or_loss
        jax.clear_caches()
        gc.collect()

    print("\nAll arms complete!")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Full-network finetuning comparison: IBProbit vs CrossEntropy"
    )
    parser.add_argument("--model", type=str, default="dinov3_small",
                        choices=list(EQUIMO_MODELS.keys()))
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--arm", type=str, default="both",
                        choices=["both", "ibprobit", "ce"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-subset", type=int, default=5000)
    parser.add_argument("--eval-chunk", type=int, default=256)
    parser.add_argument("--final-full-eval", action="store_true")
    parser.add_argument("--ood-eval", action="store_true",
                        help="Evaluate on ImageNet OOD variants (ImageNet-C/R/A/V2) every "
                             "--ood-every steps. Requires --dataset imagenet1k. Reports "
                             "acc/ece/nll/brier per set (no temperature scaling); the OOD "
                             "images are forwarded through the current finetuned backbone "
                             "and the eval is excluded from the reported train time.")
    parser.add_argument("--ood-every", type=int, default=25,
                        help="Run OOD evaluation every N optimizer steps (and at the final step)")
    parser.add_argument("--ood-subset", type=int, default=2000,
                        help="Random sample cap per OOD dataset (forwarded through the finetuned "
                             "backbone each OOD eval)")
    parser.add_argument("--ood-keys", nargs="+", default=None,
                        help="OOD datasets to evaluate. Default: the two cheapest "
                             "(imagenet_c_jpeg_compression + imagenet_a). Pass 'all' for the "
                             "full 8x ImageNet-C + R/A/V2, or an explicit space-separated list.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Target (effective) batch size reached by gradient accumulation; "
                             "if omitted, equals the micro-batch size (no accumulation)")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Per-forward micro-batch size (largest that fits); if omitted, auto-probe")
    parser.add_argument("--batch-size-init", type=int, default=32,
                        help="Smallest micro-batch tried by the auto-probe")
    parser.add_argument("--batch-size-max", type=int, default=8192,
                        help="Largest micro-batch tried by the auto-probe")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "ivon", "evon"],
                        help="Backbone optimizer. ivon/evon are the blrax bayesian "
                             "optimizers: one posterior sample (--estimator sampling) or "
                             "one Rademacher probe (--estimator hutchinson) per optimizer "
                             "step, shared across the grad-accumulation window. With "
                             "ivon/evon the CE arm uses a single lr (--bayes-lr) for "
                             "backbone+head (--head-lr is adamw-only).")
    parser.add_argument("--estimator", type=str, default="sampling",
                        choices=["sampling", "hutchinson"],
                        help="IVON/EVON gradient/Hessian estimator: reparameterized "
                             "posterior sampling (default) or Hutchinson (gradient at "
                             "the mean + one HVP)")
    parser.add_argument("--bayes-lr", type=float, default=1e-2,
                        help="Learning rate for IVON/EVON (constant, like --lr)")
    parser.add_argument("--bayes-head-lr", type=float, default=None,
                        help="Optional separate head lr for IVON/EVON on the CE arm "
                             "(default: share --bayes-lr). Applied as a per-part lr "
                             "stage after the bayesian transform; the posterior is "
                             "lr-independent and unaffected.")
    parser.add_argument("--clip-radius", type=float, default=1e-2,
                        help="Gradient clip radius for IVON/EVON")
    parser.add_argument("--ivon-weight-decay", type=float, default=1e-6,
                        help="Weight decay for IVON/EVON")
    parser.add_argument("--ivon-hess-init", type=float, default=0.01,
                        help="Hessian initialisation scale for IVON/EVON")
    parser.add_argument("--ivon-b1", type=float, default=0.95,
                        help="Beta1 (momentum EMA) for IVON")
    parser.add_argument("--ivon-b2", type=float, default=0.9999,
                        help="Beta2 (hessian EMA) for IVON")
    parser.add_argument("--evon-b1", type=float, default=0.9,
                        help="Beta1 (momentum EMA) for EVON")
    parser.add_argument("--evon-b2", type=float, default=0.95,
                        help="Beta2 (hessian EMA) for EVON")
    parser.add_argument("--evon-b3", type=float, default=0.95,
                        help="Beta3 (preconditioner EMA) for EVON")
    parser.add_argument("--precond-every", type=int, default=10,
                        help="EVON: refresh the eigenbasis every k steps")
    parser.add_argument("--hess-every", type=int, default=None,
                        help="Re-estimate the IVON/EVON hessian every k steps "
                             "(default: 1 for IVON, precond-every for EVON)")
    parser.add_argument("--hess-clip-ratio", type=float, default=10.0,
                        help="EVON: adaptive element-wise hessian clip ratio")
    parser.add_argument("--max-precond-dim", type=int, default=10000,
                        help="EVON: axes larger than this fall back to diagonal")
    parser.add_argument("--one-sided", action="store_true",
                        help="EVON: precondition only the smaller axis per matrix")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Backbone learning rate (both arms)")
    parser.add_argument("--head-lr", type=float, default=1e-2,
                        help="Learning rate for the CE arm's linear head")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smooth", type=float, default=0.0)
    parser.add_argument("--num-update-iters", type=int, default=16)
    parser.add_argument("--bf16", action="store_true",
                        help="Run the backbone forward in bfloat16 (mixed precision: master "
                             "weights and optimizer stay float32, only saved activations "
                             "are bf16). Halves activation memory and speeds up matmuls.")
    # --- IBProbit head knobs (mirror scripts/alternatives/ood_evaluation.py) ---
    parser.add_argument("--px", action="store_true",
                        help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    parser.add_argument("--no-learn-q", action="store_true",
                        help="hold Q at its isotropic prior q0*I (the ablation arm); default infers it")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="prior scale on the process noise: q0 = alpha^2 / batch_size (q-update-spec 5.5e)")
    parser.add_argument("--q-init", type=float, default=None,
                        help="override q0 directly; pass 1/D to reproduce the pre-Wishart regime")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Inverse-Wishart prior pseudo-count (default: num_classes, i.e. one batch's weight)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="defer the gate and the q recursion for the first N steps (the mu=0 seed is uninformative)")
    parser.add_argument("--eps-init", type=float, default=0.01,
                        help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). Forgetting rho = 1 - B/N set automatically.")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="bllarse")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Parent MLflow run ID for nested grouping")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)
    main(args)
