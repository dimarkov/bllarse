"""The grad-accumulation glue in full_finetune_comparison must reproduce a
blrax noisy_value_and_grad step EXACTLY: one posterior sample / Rademacher probe
per optimizer step, gradients (and HVPs) averaged over equal-size micro-batches,
Hessian estimate stashed before optimizer.update.

Because the macro loss is the mean of the micro losses, the accumulated step
must match the single full-batch step bit-for-bit (up to float tolerance) when
both use the same draw. The reference draw keys are derived from blrax's
internal key-splitting so the sample/probe coincide.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "alternatives"))

import full_finetune_comparison as ffc  # noqa: E402
from blrax.utils import noisy_value_and_grad  # noqa: E402

N, MICRO, D_IN, D_OUT = 12, 4, 6, 4
NUM_ACCUM = N // MICRO


def _loss(params, x, y):
    pred = jnp.tanh(x @ params["w"] + params["b"])
    return jnp.mean((pred - y) ** 2)


def _data(key):
    kx, ky, kw, kb = jr.split(key, 4)
    X = jr.normal(kx, (N, D_IN))
    Y = jr.normal(ky, (N, D_OUT))
    params = {"w": 0.3 * jr.normal(kw, (D_IN, D_OUT)), "b": 0.1 * jr.normal(kb, (D_OUT,))}
    return X, Y, params


def _args(optimizer, estimator, hess_every=1, precond_every=1):
    import argparse

    return argparse.Namespace(
        optimizer=optimizer, estimator=estimator, bayes_lr=1e-2,
        ivon_weight_decay=1e-4, ivon_hess_init=1.0, ivon_b2=0.999,
        evon_b1=0.9, evon_b2=0.95, evon_b3=0.95, ivon_b1=0.95,
        bayes_head_lr=None, precond_every=precond_every,
        hess_every=hess_every, hess_clip_ratio=10.0,
        max_precond_dim=10000, one_sided=False,
        # Added by ibprobit-process-noise's tip commit (1630974 "add clip
        # radius"), which introduced args.clip_radius in
        # full_finetune_comparison.make_bayes_optimizer without updating this
        # Namespace. Value matches the argparse default.
        clip_radius=1e-2,
    )


def _glue_step(optimizer, opt_state, params, X, Y, estimator, draw_key, hutch_step):
    """The comparison-script step: shared per-step draw + micro accumulation."""
    noise_aux, tangent, hvp_acc, grads_acc = None, None, None, None
    if estimator == "hutchinson" and hutch_step:
        tangent, noise_aux = ffc.draw_hutchinson_probe(draw_key, params, opt_state)
        fwd_params = params
    elif estimator == "sampling":
        fwd_params, noise_aux = ffc.draw_bayes_sample(draw_key, params, opt_state)
    else:  # hutchinson skip-step: plain gradient at the mean
        fwd_params = params

    for i in range(NUM_ACCUM):
        xs, ys = X[i * MICRO:(i + 1) * MICRO], Y[i * MICRO:(i + 1) * MICRO]
        if estimator == "hutchinson" and hutch_step:
            (_, g), (_, hv) = jax.jvp(
                jax.value_and_grad(lambda p: _loss(p, xs, ys)), (params,), (tangent,))
            hvp_acc = hv if hvp_acc is None else jax.tree.map(jnp.add, hvp_acc, hv)
        else:
            g = jax.grad(lambda p: _loss(p, xs, ys))(fwd_params)
        grads_acc = g if grads_acc is None else jax.tree.map(jnp.add, grads_acc, g)

    return ffc.apply_grads_bayes(
        params, grads_acc, opt_state, optimizer, NUM_ACCUM, estimator, noise_aux, hvp_acc)


def _reference_step(optimizer, opt_state, params, X, Y, estimator, ref_key):
    """blrax's own full-batch step via noisy_value_and_grad."""
    loss_fn = lambda p, x, y, k: _loss(p, x, y)  # noqa: E731 (key unused)
    _, grads, opt_state = noisy_value_and_grad(
        loss_fn, opt_state, params, ref_key, X, Y, mc_samples=1, estimator=estimator)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state


def _draw_key_for(optimizer_name, estimator, ref_key):
    """Reproduce blrax's internal key-splitting so the draws coincide."""
    if estimator == "sampling" and optimizer_name == "ivon":
        return jr.split(ref_key, 3)[1]  # sequential_sampling: rn, k1, k2
    return jr.split(ref_key)[0]  # k_sample / k_probe


@pytest.mark.parametrize("optimizer_name", ["ivon", "evon"])
@pytest.mark.parametrize("estimator", ["sampling", "hutchinson"])
def test_accumulated_step_matches_full_batch_blrax_step(optimizer_name, estimator):
    X, Y, params = _data(jr.PRNGKey(0))
    args = _args(optimizer_name, estimator)
    optimizer = ffc.make_bayes_optimizer(args, ess=N)
    opt_state = optimizer.init(params)

    ref_key = jr.PRNGKey(42)
    ref_params, ref_state = _reference_step(
        optimizer, opt_state, params, X, Y, estimator, ref_key)

    draw_key = _draw_key_for(optimizer_name, estimator, ref_key)
    new_params, new_state = _glue_step(
        optimizer, opt_state, params, X, Y, estimator, draw_key, hutch_step=True)

    jax.tree.map(
        lambda a, b: None if a is None else _assert_close(a, b),
        ref_params, new_params)
    # the hessian state must also agree (EMA fed by the same estimate)
    if optimizer_name == "ivon":
        jax.tree.map(_assert_close, ref_state[0].hess, new_state[0].hess)
    else:
        for s_ref, s_new in zip(
            jax.tree.leaves(ref_state[0].leaves, is_leaf=ffc._is_evon_leaf),
            jax.tree.leaves(new_state[0].leaves, is_leaf=ffc._is_evon_leaf),
        ):
            _assert_close_hessian(s_ref.H, s_new.H)


def test_hutchinson_skip_step_matches_blrax_reuse():
    """With hess_every > 1 at count=0, EVON skips the HVP ((count+1) % k != 0);
    the glue's reuse path must match blrax's frozen-hessian step."""
    X, Y, params = _data(jr.PRNGKey(1))
    args = _args("evon", "hutchinson", hess_every=5, precond_every=5)
    optimizer = ffc.make_bayes_optimizer(args, ess=N)
    opt_state = optimizer.init(params)

    ref_key = jr.PRNGKey(7)
    ref_params, ref_state = _reference_step(
        optimizer, opt_state, params, X, Y, "hutchinson", ref_key)

    new_params, new_state = _glue_step(
        optimizer, opt_state, params, X, Y, "hutchinson",
        _draw_key_for("evon", "hutchinson", ref_key), hutch_step=False)

    jax.tree.map(_assert_close, ref_params, new_params)
    for s_ref, s_new in zip(
        jax.tree.leaves(ref_state[0].leaves, is_leaf=ffc._is_evon_leaf),
        jax.tree.leaves(new_state[0].leaves, is_leaf=ffc._is_evon_leaf),
    ):
        _assert_close_hessian(s_ref.H, s_new.H)


def _assert_close(a, b):
    assert jnp.allclose(a, b, atol=1e-5), f"max diff {jnp.max(jnp.abs(a - b))}"


def _assert_close_hessian(a, b):
    # EVON's QR-projected Hessian estimate involves more float32 ops than IVON's
    # plain diagonal one, so full-batch vs. micro-batch-accumulated summation
    # order picks up more rounding drift here; ivon's h_bar still matches to 1e-5.
    assert jnp.allclose(a, b, atol=2e-4), f"max diff {jnp.max(jnp.abs(a - b))}"
