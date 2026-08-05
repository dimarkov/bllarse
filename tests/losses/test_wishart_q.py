"""The Inverse-Wishart posterior on the process-noise matrix Q.

Design: docs/superpowers/specs/2026-08-05-ibprobit-merge-and-wishart-q-design.md
Target algorithm: ~/projects/papers/bllarse/q-update-spec.md sec 5.5, generalized
from d independent Gammas to one Inverse-Wishart on the full matrix.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

from bllarse.losses.bayesian import IBProbit


def test_prior_expectation_is_isotropic_at_q0():
    """Psi_0 = q0 * kappa * I and nu_0 = d + 1 + kappa give E_0[Q] = q0 * I exactly."""
    D, K, B, alpha = 12, 5, 32, 1e-3
    lf = IBProbit(D, K, key=jr.key(0), alpha=alpha, batch_size=B)
    d = D + 1
    q0 = alpha ** 2 / B
    assert jnp.allclose(lf.Qhat, q0 * jnp.eye(d), rtol=1e-5)
    assert float(lf.nu) > d + 1


def test_q_init_overrides_the_derived_scale():
    """The old 1/D regime must stay reachable for the spec sec 12.5 A/B."""
    D, K = 12, 5
    lf = IBProbit(D, K, key=jr.key(0), q_init=1.0 / D)
    assert jnp.allclose(jnp.diagonal(lf.Qhat), 1.0 / D, rtol=1e-5)


def test_batch_size_required_when_q_init_absent():
    with pytest.raises(ValueError, match="q_init or batch_size"):
        IBProbit(12, 5, key=jr.key(0))


def test_kappa_defaults_to_num_classes():
    D, K, B = 12, 5, 32
    lf = IBProbit(D, K, key=jr.key(0), alpha=1e-3, batch_size=B)
    assert lf.kappa == float(K)
    assert float(lf.nu) == pytest.approx(D + 1 + 1 + K)   # d = D + 1 with bias


def test_pc_prior_mean_is_one_minus_rho():
    lf = IBProbit(12, 5, key=jr.key(0), q_init=1e-3, rho=0.9)
    assert float(lf.p_a / (lf.p_a + lf.p_b)) == pytest.approx(0.1, abs=1e-6)


def test_learn_q_off_freezes_the_hyper_state():
    """learn_q=False must hold Qhat at q0*I and never touch Psi/nu/p_a/p_b."""
    D, K, N = 12, 5, 24
    lf = IBProbit(D, K, key=jr.key(0), q_init=1e-3, learn_q=False, warmup=0)
    keys = jr.split(jr.key(1), 2)
    fts, y = jr.normal(keys[0], (N, D)), jr.randint(keys[1], (N,), 0, K)

    Qhat0, Psi0, nu0, pa0, pb0 = lf.Qhat, lf.Psi, lf.nu, lf.p_a, lf.p_b
    for _ in range(3):
        lf = lf.update(fts, y, num_iters=4)

    assert jnp.array_equal(lf.Psi, Psi0)
    assert jnp.array_equal(lf.nu, nu0)
    assert jnp.array_equal(lf.p_a, pa0) and jnp.array_equal(lf.p_b, pb0)
    assert jnp.allclose(lf.Qhat, Qhat0)
    assert jnp.allclose(lf.Qhat, 1e-3 * jnp.eye(D + 1), rtol=1e-5)


def test_sigma_z2_is_gone():
    lf = IBProbit(12, 5, key=jr.key(0), q_init=1e-3)
    assert not hasattr(lf, "log_sigma_z2")
    assert not hasattr(lf, "sigma_z2")
    with pytest.raises(TypeError):
        IBProbit(12, 5, key=jr.key(0), q_init=1e-3, sigma_z2_init=2.0)


def test_key_is_accepted_but_not_carried():
    """key stays in the signature for parity with the sibling loss classes and with
    reset(key), but nothing in IBProbit draws a random number -- mu seeds to zeros and
    the scan starts at eta_pred -- so storing it would carry a dead leaf through every
    jit and scan."""
    import equinox as eqx
    import jax

    lf = IBProbit(12, 5, key=jr.key(0), q_init=1e-3)
    assert not hasattr(lf, "key")

    leaves = jax.tree_util.tree_leaves(eqx.filter(lf, eqx.is_array))
    assert not any(jnp.issubdtype(l.dtype, jax.dtypes.prng_key) for l in leaves)

    # reset(key) keeps its signature: src/bllarse/utils.py calls it on every loss class
    assert lf.reset(jr.key(1)).mu.shape == lf.mu.shape


def test_moderated_predictive_is_the_exact_marginal():
    """q-update-spec sec 5.8: probit composed with a Gaussian conjugates EXACTLY, so
        E_{w ~ N(mu, Sigma + Qhat)}[Phi(w'f)] = Phi(mu'f / sqrt(1 + f'(Sigma + Qhat)f)).
    This is the paper's central claim -- 'exact, not an approximation' -- and the
    reason the moderation denominator can be read as an input-dependent temperature.
    Checked by Monte Carlo against the closed form.

    (The gauge invariance of sec 5.7 is deliberately NOT tested: deleting sigma_z2 is
    what fixes the gauge, so there is no free scale left to vary. It is an argument
    about the model, discharged by the deletion itself.)"""
    import equinox as eqx
    from jax.scipy.stats import norm as jnorm

    D, K = 6, 3
    d = D + 1
    lf = IBProbit(D, K, key=jr.key(0), q_init=1e-2, use_approx_cdf=False)
    keys = jr.split(jr.key(21), 4)

    A = jr.normal(keys[0], (d, d)) * 0.3
    lf = eqx.tree_at(lambda m: (m.mu, m.Sigma), lf,
                     (jr.normal(keys[1], (d, K)) * 0.5, A @ A.T + 0.2 * jnp.eye(d)))

    f_raw = jr.normal(keys[2], (4, D))
    f = jnp.pad(f_raw, [(0, 0), (0, 1)], constant_values=1.0)
    V = lf.Sigma + lf.Qhat

    closed = jnorm.cdf(lf._predictive_logits(f_raw))

    # 400k samples put the per-entry MC standard error near 8e-4; 4e-3 is ~5 sigma on
    # the max over 12 entries, which keeps this from flaking.
    L = jnp.linalg.cholesky(V)
    w = lf.mu[None, :, :] + jnp.einsum("ij,sjk->sik", L, jr.normal(keys[3], (400_000, d, K)))
    mc = jnorm.cdf(jnp.einsum("nd,sdk->snk", f, w)).mean(axis=0)

    assert jnp.abs(closed - mc).max() < 4e-3, jnp.abs(closed - mc).max()
