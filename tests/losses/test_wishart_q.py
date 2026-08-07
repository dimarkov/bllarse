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


def test_alpha_default_is_off_the_inert_operating_point():
    """The alpha default is load-bearing, so pin it. At alpha=1e-3 the recursion is
    provably inert -- q0 = 1e-6/B puts the drift signal at ~1e-8 of S_t, and
    learn_q=True vs False give bit-identical mu over 40 steps. The default was raised
    to 0.1 (q0 = 0.01/B, near the pre-Wishart 1/D operating point) for that reason.
    Every other test passes alpha explicitly, so nothing else would catch a revert."""
    D, K, B = 12, 5, 32
    lf = IBProbit(D, K, key=jr.key(0), batch_size=B)
    assert lf.q0 == pytest.approx(0.01 / B), (
        f"default alpha appears to have changed: q0={lf.q0}, expected {0.01 / B}")
    # explicitly passing the old value must still be reachable for the q0 A/B
    old = IBProbit(D, K, key=jr.key(0), alpha=1e-3, batch_size=B)
    assert old.q0 == pytest.approx(1e-6 / B)


def test_kappa_default_leaves_the_prior_outweighed_by_the_data():
    """The kappa default is load-bearing, so pin it. At kappa = K the prior contributes
    as much transition evidence per step as the whole batch does, which holds Q at its
    q0 seed on top of the covariance floor that already does so. The default is 0.5 --
    half a pseudo-observation against K real ones -- so the likelihood dominates."""
    D, K, B = 12, 5, 32
    lf = IBProbit(D, K, key=jr.key(0), alpha=1e-3, batch_size=B)
    assert lf.kappa == 0.5, f"default kappa appears to have changed: {lf.kappa}"
    assert lf.kappa < K, "the prior must stay lighter than one batch of evidence"
    assert float(lf.nu) == pytest.approx(D + 1 + 1 + 0.5)   # d = D + 1 with bias
    # kappa scales Psi_0 and nu_0 together, so the prior mean is q0 * I regardless
    d, q0 = D + 1, 1e-6 / B
    assert jnp.allclose(lf.Qhat, q0 * jnp.eye(d), rtol=1e-5)


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


def test_log_bayes_factor_reads_its_arguments_not_self():
    """Guards two bugs the zero-at-prior anchor cannot see: a swapped block() operand
    order, and reading self.Psi/self.nu instead of the passed belief."""
    import equinox as eqx
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    S = jnp.eye(d) * float(jnp.diagonal(Psi0)[0]) * 1.5

    # move self.Psi away from the prior; passing the prior explicitly must still give 0
    moved = eqx.tree_at(lambda m: (m.Psi, m.nu), lf, (Psi0 * 0.25, nu0 + 7.0))
    assert float(moved._log_bayes_factor(S, Psi0, nu0, K)) == pytest.approx(0.0, abs=1e-4)

    # and a sharpened belief must give a NONZERO, positively-signed surprise
    sharp = float(moved._log_bayes_factor(S, Psi0 * 0.25, nu0 + 7.0, K))
    assert abs(sharp) > 1e-3
    assert sharp > 0.0, "S atypical under the sharpened belief must be positive surprise"


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
    # 1e20 is past where the naive Sbar itself would still fit float32; 1e30 and 1e35
    # are past the ~1e30 point where exp(log_Sbar) overflows float32 and the naive
    # m*Sbar/(1+m*Sbar) form would divide inf by inf and return NaN. The sigmoid form
    # must stay finite and in [0, 1] straight through.
    for scale in (1e-8, 1e-3, 1.0, 1e3, 1e8, 1e20, 1e30, 1e35):
        g = lf._surprise_rate(jnp.eye(d) * scale, Psi0 * 1e-2, nu0 + 10.0,
                              lf.p_a, lf.p_b, K)
        assert 0.0 <= float(g) <= 1.0 and jnp.isfinite(g), (scale, g)


def test_surprise_rate_normalizes_by_d_times_K():
    """One matrix node absorbs d*K scalar observations, so the rate divides log S_BF
    by d*K rather than the scalar spec's K. Pinned as an exact identity: a regression
    to /K cannot pass this, whereas a behavioural check at small d cannot see the
    difference (both divisors saturate the sigmoid)."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    # calibrated so gamma lands strictly inside (0,1): Psi0 * 0.5, nu0 + 3.0, S scaled
    # 1.5x the prior's diagonal give gamma ~= 0.251 (settled by direct search, not
    # tuned to a target -- the requirement was only "not saturated").
    Psi_t, nu_t = Psi0 * 0.5, nu0 + 3.0
    S = jnp.eye(d) * float(jnp.diagonal(Psi0)[0]) * 1.5

    gamma = lf._surprise_rate(S, Psi_t, nu_t, lf.p_a, lf.p_b, K)
    assert 1e-6 < float(gamma) < 1.0 - 1e-6, (
        f"gamma={float(gamma)} is saturated; the logit below is uninformative there. "
        "Soften the perturbation until it is not.")

    log_SBF = lf._log_bayes_factor(S, Psi_t, nu_t, K)
    logit = jnp.log(gamma) - jnp.log1p(-gamma)
    recovered = float(logit - (jnp.log(lf.p_a) - jnp.log(lf.p_b)))

    assert recovered == pytest.approx(float(log_SBF) / (d * K), rel=1e-4)
    assert abs(recovered - float(log_SBF) / K) > 1e-3, "must not be the /K normalization"


def test_q_recursion_at_the_prior_is_the_plain_conjugate_update():
    """With gamma = E[p_c] at t=0, the update is the convex combination of
    sec 5.5(c). Check it against the formula written out longhand."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, rho=0.9)
    Qhat, Lambda_pred, _, Sigma_new, Delta = _random_state(jr.key(5), d, K)
    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    Psi0, nu0 = lf._q_seed()

    Psi_n, nu_n, _, _, gamma = lf._q_recursion(Qhat, Lambda_pred, Sigma_new, Delta, K)

    assert jnp.allclose(nu_n, (1 - gamma) * lf.nu + gamma * nu0 + K, rtol=1e-5)
    assert jnp.allclose(Psi_n, (1 - gamma) * lf.Psi + gamma * Psi0 + S, rtol=1e-4)


def test_q_recursion_convex_combination_off_the_prior():
    """The t=0 test is degenerate: with self.Psi == Psi0 the convex combination
    collapses for ANY gamma. Repeat the assertions from a moved state, where a
    gamma/(1-gamma) swap or a Psi0/self.Psi swap would change the answer."""
    import equinox as eqx
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    lf = eqx.tree_at(lambda m: (m.Psi, m.nu), lf, (Psi0 * 0.25, nu0 + 7.0))

    Qhat, Lambda_pred, _, Sigma_new, Delta = _random_state(jr.key(31), d, K)
    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    Psi_n, nu_n, _, _, gamma = lf._q_recursion(Qhat, Lambda_pred, Sigma_new, Delta, K)

    assert not jnp.allclose(lf.Psi, Psi0), "state must be off the prior for this to bite"
    assert jnp.allclose(nu_n, (1 - gamma) * lf.nu + gamma * nu0 + K, rtol=1e-5)
    assert jnp.allclose(Psi_n, (1 - gamma) * lf.Psi + gamma * Psi0 + S, rtol=1e-4)
    # a gamma/(1-gamma) swap must be distinguishable at this state
    swapped = gamma * lf.nu + (1 - gamma) * nu0 + K
    assert abs(float(nu_n - swapped)) > 1e-3


def test_q_recursion_keeps_psi_pd_and_nu_above_d_plus_one():
    d, K, N = 13, 4, 32
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9, warmup=0)
    keys = jr.split(jr.key(6), 2)
    fts, y = jr.normal(keys[0], (N, d - 1)), jr.randint(keys[1], (N,), 0, K)
    for _ in range(8):
        lf = lf.update(fts, y, num_iters=4)
        assert float(lf.nu) > d + 1, float(lf.nu)
        assert (jnp.linalg.eigvalsh(lf.Psi) > 0).all()
        assert jnp.isfinite(lf.Qhat).all()


def _with_sigma(lf, Sigma):
    import equinox as eqx
    return eqx.tree_at(lambda m: m.Sigma, lf, Sigma)


def test_isotropic_reduction_matches_the_scalar_spec():
    """With isotropic Psi and Sigma the matrix recursion must reproduce
    q-update-spec sec 5.5's scalar one coordinate-wise. This pins the
    generalization against the spec it generalizes."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9)
    lf = _with_sigma(lf, jnp.eye(d) * 0.5)

    Qhat = lf.Qhat
    Lambda_pred = jnp.linalg.inv(lf.Sigma + Qhat)
    Sigma_new = jnp.eye(d) * 0.4
    Delta = jnp.zeros((d, K))                     # no drift -> S is purely the covariance part

    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    assert jnp.allclose(S, jnp.diag(jnp.diagonal(S)), atol=1e-6), "isotropic in => diagonal out"

    g = float(jnp.diagonal(Qhat)[0]) * float(jnp.diagonal(Lambda_pred)[0])
    s_scalar = K * (g * 0.5 + g * 0.4 * g)
    assert float(jnp.diagonal(S)[0]) == pytest.approx(s_scalar, rel=1e-4)


def test_qhat_contracts_under_a_stationary_head():
    """q-update-spec sec 6: the recursion contracts toward q = 0 when the head is not
    moving -- the estimator does not mistake its own tracking noise for drift. The
    contraction is CUBIC in the Kalman gain, so it is invisible at realistic drift
    rates; this constructs a large-gain state deliberately rather than running a long
    realistic loop and hoping."""
    d, K, N = 9, 4, 64
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=0.5, rho=0.99, warmup=0)
    keys = jr.split(jr.key(7), 2)
    fts, y = jr.normal(keys[0], (N, d - 1)), jr.randint(keys[1], (N,), 0, K)

    lf = lf.update(fts, y, num_iters=8)          # settle off the mu=0 seed
    first = float(jnp.trace(lf.Qhat))
    for _ in range(6):
        lf = lf.update(fts, y, num_iters=8)      # same batch every step: no real drift
    assert float(jnp.trace(lf.Qhat)) < first


def test_warmup_defers_the_q_recursion():
    d, K, N = 9, 4, 24
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9, warmup=2)
    keys = jr.split(jr.key(8), 2)
    fts, y = jr.normal(keys[0], (N, d - 1)), jr.randint(keys[1], (N,), 0, K)

    Psi0 = lf.Psi
    lf = lf.update(fts, y, num_iters=4)
    assert jnp.array_equal(lf.Psi, Psi0), "step 0 is inside warmup"
    lf = lf.update(fts, y, num_iters=4)
    assert jnp.array_equal(lf.Psi, Psi0), "step 1 is inside warmup"
    lf = lf.update(fts, y, num_iters=4)
    assert not jnp.array_equal(lf.Psi, Psi0), "step 2 must engage the recursion"


def test_pc_posterior_moves_with_gamma():
    d, K, N = 9, 4, 24
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9, warmup=0)
    keys = jr.split(jr.key(9), 2)
    fts, y = jr.normal(keys[0], (N, d - 1)), jr.randint(keys[1], (N,), 0, K)
    before = float(lf.p_a + lf.p_b)
    lf = lf.update(fts, y, num_iters=4)
    assert float(lf.p_a + lf.p_b) > before, "one soft count per step must accumulate"


def test_update_runs_under_jit_and_scan():
    import equinox as eqx
    from jax import lax
    d, K, N = 9, 4, 24
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-2, rho=0.9, warmup=1)
    keys = jr.split(jr.key(10), 2)
    fts = jr.normal(keys[0], (3, N, d - 1))
    ys = jr.randint(keys[1], (3, N), 0, K)
    params, static = eqx.partition(lf, eqx.is_array)

    @eqx.filter_jit
    def run(params, fts, ys):
        def body(p, batch):
            m = eqx.combine(p, static).update(batch[0], batch[1], num_iters=4)
            return eqx.filter(m, eqx.is_array), None
        out, _ = lax.scan(body, params, (fts, ys))
        return out

    out = run(params, fts, ys)
    final = eqx.combine(out, static)
    assert jnp.isfinite(final.mu).all() and jnp.isfinite(final.Psi).all()


def test_q_recursion_has_a_fixed_point_at_the_covariance_floor():
    """When Qhat << Sigma the statistic is dominated by K*Qhat, and S_t = K*Qhat is an
    EXACT fixed point of the gamma=0 update: Psi(1 + K/(nu-d-1))/(nu-d-1+K) = Psi/(nu-d-1).
    So the estimator sits at its seed and the only forcing is the O(Qhat^2) drift term.
    Pinned deliberately: this is a real, measured property of the recursion, not an
    aspiration, and a future change to the statistic or the recursion should have to
    confront it.

    Exercised through the real `_increment_statistic` and `_q_recursion`, not
    hand-derived arithmetic. `Delta=0` (mu did not move) and
    `Sigma_new = Sigma_prev + Qhat` (the M-step added no precision, e.g. every point
    in the batch got gated out) make `_increment_statistic` GENUINELY return `K*Qhat`
    -- checked directly below before trusting the rest. `(Psi, nu)` are moved off the
    prior first, because at t=0 (Psi == Psi0, nu == nu0) the convex combination in
    `_q_recursion` collapses to the fixed point for ANY gamma, which would let this
    pass without actually isolating the gamma=0 case (see
    test_q_recursion_convex_combination_off_the_prior). `(p_a, p_b)` are set to an
    extreme ratio so the surprise rate gamma computed by the real code underflows
    toward 0, rather than being asserted or hardcoded."""
    import equinox as eqx
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, rho=0.9)
    Psi0, nu0 = lf._q_seed()
    lf = eqx.tree_at(lambda m: (m.Psi, m.nu, m.p_a, m.p_b), lf,
                     (Psi0 * 0.7, nu0 + 5.0, jnp.float32(1e-30), jnp.float32(1.0)))

    Qhat = lf.Qhat
    Sigma_prev = lf.Sigma
    Lambda_pred = jnp.linalg.inv(Sigma_prev + Qhat)
    Sigma_new = Sigma_prev + Qhat                 # M-step added no precision
    Delta = jnp.zeros((d, K))                     # -> mu did not move

    S = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    assert jnp.allclose(S, K * Qhat, rtol=1e-5), (
        "construction failed to make the statistic genuinely equal K*Qhat")

    Psi_n, nu_n, _, _, gamma = lf._q_recursion(Qhat, Lambda_pred, Sigma_new, Delta, K)
    assert float(gamma) < 1e-4, "gamma must be ~0 to isolate the gamma=0 fixed point"
    Qhat_new = Psi_n / (nu_n - d - 1.0)
    assert jnp.allclose(Qhat_new, Qhat, rtol=1e-4), (
        "S = K*Qhat must reproduce Qhat through the real recursion; if this fails the "
        "statistic or the recursion changed")


def test_increment_statistic_drift_term_responds_to_delta():
    """The drift term is the only part of S_t carrying a change signal. It is swamped by
    the covariance floor in practice (~1e-5 of the total), but it must at least respond."""
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3)
    Qhat, Lambda_pred, _, Sigma_new, Delta = _random_state(jr.key(32), d, K)
    base = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
    moved = lf._increment_statistic(Qhat, Lambda_pred, Sigma_new, 5.0 * Delta)
    assert float(jnp.trace(moved)) > float(jnp.trace(base))


def test_eps_responsibility_uses_the_psi_form():
    """sec 5.2: r_i = sigmoid(E[log(1-eps)] - E[log eps] + log phat + log K), with the
    expectations under the Beta -- NOT its mean. Makes the update an exact stationary
    point of the free energy."""
    from jax.scipy.special import digamma
    import equinox as eqx

    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, eps_init=0.05)
    lf = eqx.tree_at(lambda m: (m.eps_alpha, m.eps_beta), lf,
                     (jnp.float32(3.0), jnp.float32(97.0)))
    loss_vec = jnp.array([0.1, 2.0, 8.0], dtype=jnp.float32)

    dig_ab = digamma(lf.eps_alpha + lf.eps_beta)
    E_log_eps = digamma(lf.eps_alpha) - dig_ab
    E_log_1meps = digamma(lf.eps_beta) - dig_ab
    from jax import nn as jnn
    expected = jnn.sigmoid(E_log_1meps - E_log_eps - loss_vec + jnp.log(K))

    got = lf._eps_responsibility(loss_vec)
    assert got.shape == (3, 1)
    assert jnp.allclose(got[:, 0], expected, atol=1e-6)


def test_eps_responsibility_orders_by_fit():
    d, K = 9, 4
    lf = IBProbit(d - 1, K, key=jr.key(0), q_init=1e-3, eps_init=0.05)
    r = lf._eps_responsibility(jnp.array([0.01, 1.0, 20.0], dtype=jnp.float32))[:, 0]
    assert r[0] > r[1] > r[2]
    assert (r > 0).all() and (r < 1).all()


def test_dead_helpers_are_gone():
    lf = IBProbit(9, 4, key=jr.key(0), q_init=1e-3)
    assert not hasattr(lf, "_gate_from_logits")
    assert not hasattr(lf, "_preq_objective")
