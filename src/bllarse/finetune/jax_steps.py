"""Jitted micro-batch kernels and the gradient-accumulation step.

Lifted out of scripts/alternatives/full_finetune_comparison.py so that script
and scripts/finetune/finetune_jax.py share ONE copy. These carry buffer-donation
contracts tuned against a measured peak (commit cb81798, ~6 GiB off dinov3_huge)
and the argument order of `_tree_add` is load-bearing -- a second copy that
drifted would regress memory silently, and only on the largest backbone.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random as jr, vmap

from blrax.states import ScaleByEvonState, MatrixEvonLeaf, _is_evon_leaf
from blrax.utils import (
    _evon_sample_leaves,
    _project,
    _project_back,
    precision as bayes_precision,
    rademacher_like,
    sample_posterior,
)
from bllarse.utils import bayes_optimizer


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
