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


@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_nonpositive_kappa_is_rejected(bad):
    with pytest.raises(ValueError, match="kappa must be positive"):
        IBProbit(12, 5, key=jr.key(0), q_init=1e-3, kappa=bad)


def _random_state(key, d, K, q_scale=1e-2):
    """A plausible (Qhat, Lambda_pred, Sigma_new, Delta) tuple."""
    k1, k2, k3 = jr.split(key, 3)
    A = jr.normal(k1, (d, d))
    Sigma_prev = A @ A.T / d + jnp.eye(d)
    Qhat = q_scale * jnp.eye(d)
    Lambda_pred = jnp.linalg.inv(Sigma_prev + Qhat)
    Bm = jr.normal(k2, (d, d))
    Sigma_new = Bm @ Bm.T / d + 0.5 * jnp.eye(d)
    Delta = jr.normal(k3, (d, K)) * 0.1
    return Qhat, Lambda_pred, Sigma_prev, Sigma_new, Delta


def test_increment_statistic_vanishes_at_zero_process_noise():
    """S_t = 0 when Qhat = 0. The naive mean-field statistic returns 2K*Sigma_jj > 0
    here, which is the level error q-update-spec sec 5.5(a) exists to fix."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    _, _, Sigma_prev, Sigma_new, Delta = _random_state(jr.key(1), d, K)
    Qhat = jnp.zeros((d, d))
    Lambda_pred = jnp.linalg.inv(Sigma_prev)
    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    assert jnp.allclose(S, 0.0, atol=1e-9), jnp.abs(S).max()


def test_increment_statistic_is_symmetric_psd():
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    Qhat, Lambda_pred, _, Sigma_new, Delta = _random_state(jr.key(2), d, K)
    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    assert jnp.allclose(S, S.T, atol=1e-6)
    assert (jnp.linalg.eigvalsh(S) > -1e-6).all()


def test_increment_statistic_diagonal_matches_the_scalar_spec():
    """diag(S_t) must equal q-update-spec sec 5.5(a)'s per-component S_{t,j}:
        S_j = ||(G Delta)_j.||^2 + K [ (G Sigma_{t-1})_jj + (G Sigma_t G')_jj ]"""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    Qhat, Lambda_pred, Sigma_prev, Sigma_new, Delta = _random_state(jr.key(3), d, K)
    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)

    # precision="highest" because a bare @ runs at TF32 on this GPU while the
    # implementation pins true F32 -- without it the REFERENCE is the inaccurate side.
    def hp(a, b):
        return jnp.matmul(a, b, precision="highest")

    G = hp(Qhat, Lambda_pred)
    expected = (jnp.sum(hp(G, Delta) ** 2, axis=-1)
                + K * (jnp.diagonal(hp(G, Sigma_prev))
                       + jnp.diagonal(hp(hp(G, Sigma_new), G.T))))
    assert jnp.allclose(jnp.diagonal(S), expected, rtol=1e-4), (
        jnp.abs(jnp.diagonal(S) - expected).max())


def test_increment_statistic_grows_with_real_drift():
    """A genuine mean shift must enter S_t and not shrink away with Qhat."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    Qhat, Lambda_pred, _, Sigma_new, Delta = _random_state(jr.key(4), d, K)
    small = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    large = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, 10.0 * Delta)
    assert jnp.trace(large) > jnp.trace(small)


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


def test_log_bayes_factor_is_zero_at_the_prior():
    """The sec 5.5(c) anchor: at t=0 the belief IS the prior, so log S_BF = 0 exactly
    and gamma reduces to E[p_c] whatever S happens to be."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    Psi0, nu0 = lf._q_seed()
    S = jnp.eye(d) * 3.7 + 0.1
    S = 0.5 * (S + S.T)
    assert float(lf._log_bayes_factor(S, Psi0, nu0, K)) == pytest.approx(0.0, abs=1e-4)


def test_surprise_rate_at_prior_equals_expected_pc():
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    S = jnp.eye(d) * 2.0
    gamma = lf._surprise_rate(S, Psi0, nu0, lf.p_a, lf.p_b, K)
    assert float(gamma) == pytest.approx(0.1, abs=1e-3)   # E_0[p_c] = 1 - rho


def test_surprise_rate_rises_with_an_unexpected_increment():
    """S typical under the prior but atypical under a sharpened belief must raise
    gamma above E[p_c]: the node resets toward the prior when it is surprised."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    # a belief that has concentrated on a much smaller Q than the prior
    Psi_t, nu_t = Psi0 * 1e-3, nu0 + 50.0
    S_big = jnp.eye(d) * float(jnp.diagonal(Psi0)[0]) * 10.0

    g_prior = lf._surprise_rate(S_big, Psi0, nu0, lf.p_a, lf.p_b, K)
    g_sharp = lf._surprise_rate(S_big, Psi_t, nu_t, lf.p_a, lf.p_b, K)
    assert float(g_sharp) > float(g_prior)


def test_surprise_rate_stays_in_the_unit_interval():
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    for scale in (1e-8, 1e-3, 1.0, 1e3, 1e8):
        g = lf._surprise_rate(jnp.eye(d) * scale, Psi0 * 1e-2, nu0 + 10.0,
                              lf.p_a, lf.p_b, K)
        assert 0.0 <= float(g) <= 1.0 and jnp.isfinite(g), (scale, g)
