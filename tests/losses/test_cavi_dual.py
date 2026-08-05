"""The CAVI scan has two algebraically identical forms; they must agree.

`_cavi_scan` carries eta (D, K) and costs 4NDK per iteration. Since eta is
always eta_old + X^T E_q[z], the prediction it feeds back is

    x @ eta = x @ eta_old + (x @ X^T) E_q[z] = c + S E_q[z],

so carrying pred (N, K) against the precomputed hat matrix S = X (I + X^T X)^-1 X^T
costs 2N^2K per iteration instead. Cheaper whenever N is small relative to D.
These tests pin the equivalence and the dispatch rule.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import lax, nn
from jax.scipy.stats import norm

from bllarse.losses.bayesian import IBProbit, mm_update


def reference_cavi_scan(fts, x, y_one_hot, eta_init, eta_old, num_iters):
    """The pre-optimisation primal scan, kept verbatim as the oracle."""

    def step_fn(eta_carry, *args):
        pred = mm_update(x, eta_carry)

        log_pdf = norm.logpdf(pred)
        logcdf1 = norm.logcdf(pred)
        logcdf2 = norm.logcdf(-pred)
        mills_pos = jnp.exp(log_pdf - logcdf1)
        mills_neg = jnp.exp(log_pdf - logcdf2)
        correction = jnp.where(y_one_hot, mills_pos, -mills_neg)
        E_q_z = pred + correction

        return eta_old + mm_update(fts.T, E_q_z), None

    eta_new, _ = lax.scan(step_fn, eta_init, jnp.arange(num_iters))
    return eta_new


def assert_agrees(got, want):
    """Agreement at the scale fp32 can actually hold.

    Both forms are the same algebra, so they differ only in rounding: mm_update
    pins F32_F32_F32, and a per-iteration eps of 1.2e-7 accumulates to ~1e-6
    relative over tens of iterations. Element-wise allclose would instead police
    the few near-zero components of an array whose typical element is ~24, which
    fp32 cannot support. For reference, the iteration's own step-to-step drift is
    ~6e-2 at T=128 -- four orders of magnitude above this bound.
    """
    assert jnp.isfinite(got).all()
    rel = jnp.linalg.norm(got - want) / jnp.linalg.norm(want)
    assert rel < 1e-5, f"relative gap {rel:.3e} exceeds fp32 accumulation"


def _inputs(N, D, K, seed=0):
    """fts / x as update() builds them, so S really is the hat matrix."""
    keys = jr.split(jr.key(seed), 4)
    features = jr.normal(keys[0], (N, D))
    y = jr.randint(keys[1], (N,), 0, K)

    lf = IBProbit(D, K, key=keys[2], use_bias=True)

    fts = features.astype(jnp.float32)
    fts = jnp.pad(fts, [(0, 0), (0, 1)], constant_values=1.0)

    from jax.scipy.linalg import cho_factor, cho_solve

    Lambda_accum = jnp.eye(D + 1) + mm_update(fts.T, fts)
    x = cho_solve(cho_factor(Lambda_accum), fts.T).T

    eta_init = jr.normal(keys[3], (D + 1, K)) * 0.01
    return lf, fts, x, nn.one_hot(y, K), eta_init


@pytest.mark.parametrize("num_iters", [14, 32, 128])
def test_dual_form_matches_primal(num_iters):
    """Iteration counts that actually dispatch to the dual must match the oracle."""
    N, D, K = 64, 64, 5
    lf, fts, x, y_one_hot, eta_init = _inputs(N, D, K)
    eta_old = jnp.zeros_like(eta_init)

    assert IBProbit._prefers_dual(N, D + 1, K, num_iters), "test no longer exercises the dual"
    got = lf._cavi_scan(fts, x, y_one_hot, eta_init, eta_old, num_iters, jnp.float32(1.0))
    want = reference_cavi_scan(fts, x, y_one_hot, eta_init, eta_old, num_iters)

    assert_agrees(got, want)


@pytest.mark.parametrize("num_iters", [1, 2, 5])
def test_primal_form_retained_when_iterations_are_too_few(num_iters):
    """Below the amortisation point the hat matrix is not worth building."""
    N, D, K = 64, 64, 5
    lf, fts, x, y_one_hot, eta_init = _inputs(N, D, K)
    eta_old = jnp.zeros_like(eta_init)

    assert not IBProbit._prefers_dual(N, D + 1, K, num_iters)
    got = lf._cavi_scan(fts, x, y_one_hot, eta_init, eta_old, num_iters, jnp.float32(1.0))
    want = reference_cavi_scan(fts, x, y_one_hot, eta_init, eta_old, num_iters)

    assert_agrees(got, want)


def test_dual_form_matches_primal_with_nonzero_eta_old():
    """The signature admits a nonzero prior term; the offset must survive it."""
    N, D, K = 64, 64, 5
    lf, fts, x, y_one_hot, eta_init = _inputs(N, D, K, seed=1)
    eta_old = jr.normal(jr.key(9), eta_init.shape) * 0.1

    assert IBProbit._prefers_dual(N, D + 1, K, 16)
    got = lf._cavi_scan(fts, x, y_one_hot, eta_init, eta_old, 16, jnp.float32(1.0))
    want = reference_cavi_scan(fts, x, y_one_hot, eta_init, eta_old, 16)

    assert_agrees(got, want)


def test_wide_batch_keeps_primal_form():
    """A batch far wider than the features must not build an N x N matrix."""
    N, D, K = 512, 64, 5
    lf, fts, x, y_one_hot, eta_init = _inputs(N, D, K, seed=2)
    eta_old = jnp.zeros_like(eta_init)

    assert not IBProbit._prefers_dual(N, D + 1, K, 32)
    got = lf._cavi_scan(fts, x, y_one_hot, eta_init, eta_old, 32, jnp.float32(1.0))
    want = reference_cavi_scan(fts, x, y_one_hot, eta_init, eta_old, 32)

    assert_agrees(got, want)


def test_dispatch_rejects_the_huge_batch_regime():
    """bs=16384 at D=1280 would allocate a 1 GB hat matrix; it must stay primal."""
    assert not IBProbit._prefers_dual(16384, 1281, 102, 32)
    assert IBProbit._prefers_dual(512, 1281, 102, 32)


def test_update_matches_across_cavi_forms(monkeypatch):
    """update() must reach the same posterior whichever CAVI form the dispatch picks.

    The two forms must be compared at the SAME num_iters -- varying num_iters to flip
    the dispatch would change the answer as well as the form, so the comparison would
    prove nothing. Force the primal by patching the dispatch predicate instead.
    """
    D, K, N, iters = 40, 3, 12, 8            # narrow batch, enough iters -> dual
    keys = jr.split(jr.key(7), 3)
    lf = IBProbit(D, K, key=keys[0], use_bias=True, q_init=1e-2)
    fts = jr.normal(keys[1], (N, D))
    y = jr.randint(keys[2], (N,), 0, K)

    assert IBProbit._prefers_dual(N, D + 1, K, iters), "shapes must select the dual"
    dual = lf.update(fts, y, num_iters=iters)

    monkeypatch.setattr(IBProbit, "_prefers_dual", staticmethod(lambda *a, **k: False))
    primal = lf.update(fts, y, num_iters=iters)

    assert jnp.allclose(dual.mu, primal.mu, atol=1e-4), jnp.abs(dual.mu - primal.mu).max()
    assert jnp.allclose(dual.Sigma, primal.Sigma, atol=1e-5)


def test_dual_matches_primal_under_process_noise_and_gate():
    """The dual re-association must survive the eta_pred anchor and the BMR gate."""
    import jax.numpy as jnp
    import jax.random as jr
    from jax.scipy.linalg import cho_factor, cho_solve
    from jax import nn
    from bllarse.losses.bayesian import IBProbit, mm_update

    D, K, N, iters = 40, 3, 12, 6           # N < 2D and enough iters -> dual is chosen
    keys = jr.split(jr.key(11), 4)
    lf = IBProbit(D, K, key=keys[0], use_bias=True, q_init=1e-2)
    d = D + 1

    fts = jnp.pad(jr.normal(keys[1], (N, D)), [(0, 0), (0, 1)], constant_values=1.0)
    y = jr.randint(keys[2], (N,), 0, K)
    y_one_hot = nn.one_hot(y, K)
    g = jr.uniform(keys[3], (N, 1), minval=0.2, maxval=1.0)

    Sigma_pred = lf.Sigma.at[jnp.diag_indices(d)].add(lf.q)
    tmp = cho_solve(cho_factor(Sigma_pred), jnp.concatenate([lf.mu, jnp.eye(d)], axis=-1))
    eta_pred, Lambda_pred = tmp[:, :-d], tmp[:, -d:]

    Lambda_new = Lambda_pred + mm_update((g * fts).T, fts)
    L = cho_factor(0.5 * (Lambda_new + Lambda_new.T))
    x = cho_solve(L, fts.T).T
    sigma_z = jnp.sqrt(lf.sigma_z2)

    assert IBProbit._prefers_dual(N, d, K, iters), "shapes must select the dual path"
    dual = lf._cavi_scan(fts, x, y_one_hot, eta_pred, eta_pred, iters, sigma_z, g)

    # primal, written out longhand so the test does not depend on the dispatch
    eta = eta_pred
    for _ in range(iters):
        pred = mm_update(x, eta)
        E_z = IBProbit._expected_z(pred / sigma_z, y_one_hot) * sigma_z
        eta = eta_pred + mm_update(fts.T, g * E_z) / lf.sigma_z2

    assert jnp.allclose(dual, eta, atol=1e-4), jnp.abs(dual - eta).max()
