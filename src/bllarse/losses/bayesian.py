import equinox as eqx
import optax
import jax.numpy as jnp
from jax import nn, vmap, random as jr, lax, grad
from jax.scipy.stats import norm
from jax.scipy.special import gammaln, digamma
from jax.scipy.linalg import solve, lu_solve, lu_factor, cho_factor, cho_solve
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from typing import Optional, Tuple, Callable
from functools import partial

const = jnp.sqrt(2 / jnp.pi)

def approx_logcdf(x):
    z = 2 * const * (x + 0.044715 * x ** 3)
    return z - nn.softplus(z)

def mm(a, b, *, preset, dot_dtype):
    """matmul with explicit cast to dot_dtype and DotAlgorithmPreset."""
    return jnp.matmul(a.astype(dot_dtype), b.astype(dot_dtype), precision=preset)

mm_update = partial(mm, preset='F32_F32_F32', dot_dtype=jnp.float32)
mm_loss = partial(mm, preset='TF32_TF32_F32', dot_dtype=jnp.float32)

def _logdet_pd(M):
    """log|M| for symmetric positive-definite M, via Cholesky. Cheaper and better
    conditioned than slogdet, and every matrix it is called on here is PD by
    construction (see IBProbit._increment_statistic)."""
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(jnp.linalg.cholesky(M))))

class IBProbit(eqx.Module):
    """
    Bayesian final layer for a multinomial probit model with a recursive
    (streaming) CAVI posterior and a learnable predictive temperature.

    Generative model (per observation i, class k):
        z_ik ~ N(w_k^T f_i, 1),   sign-pattern (pseudo-)likelihood
        w_k  ~ carried Gaussian posterior, inflated by process noise Q each step.

    The posterior ``q(W) = N(mu, Sigma)`` is built by ``update`` as a recursive
    Bayesian filter: each minibatch's posterior is the next minibatch's prior,
    inflated by the process noise ``Q`` (the PREDICT step). There is no persistent
    fixed prior -- ``mu_0`` and the seed ``Sigma = I`` are just the initial state
    and wash out with data.

    The latent scale is GAUGE-FIXED at sigma_z^2 = 1 (q-update-spec sec 5.7):
    rescaling (z, W, Sigma, Q) by a common factor leaves the sign-pattern
    likelihood and the moderated predictive invariant, so sigma_z^2 carries no
    information and is not a free parameter.

    The process noise is a full matrix ``Q`` carried as a conjugate
    Inverse-Wishart posterior ``(Psi, nu)``; ``Qhat = Psi / (nu - d - 1)`` is its
    mean, the object the PREDICT step needs. Seeded isotropically at
    ``E_0[Q] = q0 I``. See ``_q_recursion``'s docstring for a MEASURED LIMITATION:
    at the operating points measured so far this recursion barely moves ``Qhat``.

    The predictive (used at eval and by every internal gate) is the moderated probit
        logit_ik = mu_k^T f / sqrt( 1 + f^T (Qhat + Sigma) f ),
    which is EXACT, not an approximation: a probit composed with a Gaussian
    conjugates in closed form. The process-noise floor ``f^T Qhat f`` keeps the
    predictive variance from collapsing as ``Sigma`` does, so moderation survives
    abundant data.
    """
    # --- recursive posterior state (set by `update`; not gradient-trained) ---
    mu: Array              # (d, K) posterior mean
    Sigma: Array           # (d, d) posterior covariance
    step: Array            # int32 scalar; number of update() steps (drives warmup)
    eps_alpha: Array       # scalar; Beta count a (contamination mass)
    eps_beta: Array        # scalar; Beta count b (model mass); eps = a/(a+b)
    Psi: Array             # (d, d) Inverse-Wishart scale on the process noise Q
    nu: Array              # scalar; Inverse-Wishart degrees of freedom, > d + 1
    p_a: Array             # scalar; Beta count on the change rate p_c
    p_b: Array             # scalar
    beta: Array            # scalar temperature (calibrate baseline only)
    # --- static-ish (non-array leaves) ---
    use_bias: bool
    logcdf: Callable
    posterior_ready: bool
    # --- static config ---
    learn_q: bool = eqx.field(static=True)
    q0: float = eqx.field(static=True)       # prior E[Q] scale: alpha^2 / B, or q_init
    kappa: float = eqx.field(static=True)    # Inverse-Wishart prior pseudo-count
    warmup: int = eqx.field(static=True)
    loss_type: int = eqx.field(static=True)
    px: bool = eqx.field(static=True)
    eps_init: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        key: PRNGKey,
        use_bias: bool = True,
        use_approx_cdf: bool = True,
        q_init: Optional[float] = None,
        alpha: float = 0.1,
        batch_size: Optional[int] = None,
        kappa: Optional[float] = None,
        learn_q: bool = True,
        loss_type: int = 2,
        px: bool = False,
        warmup: int = 1,
        eps_init: float = 0.01,
        rho: float = 1.0,
    ):
        d = input_dim + int(use_bias)
        # Prior scale on the process noise. q-update-spec sec 5.5(e) derives
        # q0 = alpha^2 / B by equating the covariance floor of the submitted paper's
        # alpha-forgetting with the Kalman steady state, so the paper's tuned
        # forgetting factor survives as the prior on Q. q_init overrides it, which is
        # how the old q = 1/D regime stays reachable for the sec 12.5 comparison.
        #
        # alpha defaults to 0.1, NOT the 1e-3 this class first shipped with. At 1e-3
        # and B = 32, q0 = alpha^2/B = 3.1e-8 and the whole Inverse-Wishart recursion
        # is provably inert -- learn_q=True and learn_q=False produce bit-identical mu
        # over 40 steps, because S_t is dominated by its K*Qhat covariance floor (an
        # exact fixed point of the update) and the O(Qhat^2) drift term that carries
        # the actual change signal is ~1e-8 of the total. alpha = 0.1 gives
        # q0 = 3.1e-4, near the pre-Wishart 1/D = 7.8e-4 operating point, and lifts
        # the drift fraction of S_t three orders to ~1.8e-5. The q0 = alpha^2/B
        # relation itself is unchanged; only where alpha sits on it.
        if q_init is not None:
            q0 = float(q_init)
        elif batch_size is not None:
            q0 = float(alpha) ** 2 / float(batch_size)
        else:
            raise ValueError("pass q_init or batch_size (q0 = alpha^2 / batch_size)")

        # `key` is accepted for interface parity with MultinomialPolyaGamma and
        # IBPolyaGamma (and with reset(key)) but deliberately NOT stored: mu seeds to
        # zeros and the CAVI scan starts from eta_pred, so nothing in this class draws
        # a random number. Keeping it would carry a dead pytree leaf through every
        # jit and scan. The siblings accept key and consume it immediately without
        # storing it either.
        del key
        self.q0 = q0
        # kappa defaults to 0.5, NOT the num_classes this class first shipped with. At
        # kappa = K the prior carries a full batch's worth of transition evidence every
        # step, which pins Q at its q0 seed: the recursion is already dominated by the
        # K*Qhat covariance floor (an exact fixed point), and a K-weight prior on top
        # removes what little room the O(Qhat^2) drift term had. At kappa = 0.5 the
        # prior carries half a pseudo-observation against K real ones, so the
        # likelihood dominates and Q is free to move. The prior MEAN is unchanged --
        # Psi_0/kappa = q0 * I for any kappa -- so this shifts only the prior's weight,
        # not the operating point it starts from.
        self.kappa = float(0.5 if kappa is None else kappa)
        if self.kappa <= 0.0:
            raise ValueError(
                f"kappa must be positive (got {self.kappa}): nu_0 = d + 1 + kappa and the "
                "Inverse-Wishart mean Psi/(nu - d - 1) requires nu > d + 1. At kappa == 0 "
                "the mean is 0/0; at kappa < 0 the scale factor cancels and returns a "
                "plausible q0 * I for a distribution whose mean does not exist."
            )
        self.learn_q = bool(learn_q)
        self.warmup = int(warmup)
        self.loss_type = int(loss_type)
        if self.loss_type not in (1, 2, 3):
            raise ValueError(
                f"loss_type must be 1 (CBM), 2 (CBC) or 3 (mixed), got {self.loss_type}. "
                "The contamination responsibility reads exp(-loss) as a normalised "
                "p_model(y), which holds only for the softmax losses; at loss_type=0 "
                "(per-class BCE sum) the gate collapses and the head stops assimilating."
            )
        self.px = bool(px)
        self.eps_init = float(eps_init)
        self.rho = float(rho)

        self.mu = jnp.zeros((d, num_classes), dtype=jnp.float32)
        self.Sigma = jnp.eye(d, dtype=jnp.float32)
        self.step = jnp.zeros((), dtype=jnp.int32)
        self.beta = jnp.ones((), dtype=jnp.float32)
        self.use_bias = use_bias
        self.logcdf = approx_logcdf if use_approx_cdf else norm.logcdf
        self.posterior_ready = False

        Psi0, nu0 = self._q_seed(d)
        self.Psi, self.nu = Psi0, nu0
        self.p_a, self.p_b = self._pc_seed()
        a0, b0 = self._eps_seed()
        self.eps_alpha = jnp.asarray(a0, dtype=jnp.float32)
        self.eps_beta = jnp.asarray(b0, dtype=jnp.float32)

    def _q_seed(self, d: Optional[int] = None) -> Tuple[Array, Array]:
        """Isotropic Inverse-Wishart prior on Q with E_0[Q] = q0 * I.

        With nu_0 = d + 1 + kappa the mean is Psi_0 / (nu_0 - d - 1) = Psi_0 / kappa,
        so Psi_0 = q0 * kappa * I. kappa is the prior pseudo-count, defaulting to 0.5 --
        half a pseudo-observation against the K the likelihood contributes per step, so
        the data dominates. The mean is q0 * I for every kappa; only the weight moves."""
        d = self.mu.shape[0] if d is None else d
        Psi0 = (self.q0 * self.kappa) * jnp.eye(d, dtype=jnp.float32)
        nu0 = jnp.asarray(d + 1.0 + self.kappa, dtype=jnp.float32)
        return Psi0, nu0

    def _pc_seed(self) -> Tuple[Array, Array]:
        """Unit-mass Beta seed with mean E_0[p_c] = 1 - rho.

        rho = 1 - B/N is the one-pass memory convention already used for eps, and
        q-update-spec sec 5.5(d) reads it as 'expect one change per epoch', so the
        change rate inherits it rather than adding a hyperparameter. rho = 1
        (unbounded memory) would give a degenerate zero-mass seed, so it floors."""
        pc0 = max(1.0 - self.rho, 1e-4)
        return (jnp.asarray(pc0, dtype=jnp.float32),
                jnp.asarray(1.0 - pc0, dtype=jnp.float32))

    @property
    def Qhat(self) -> Array:
        """Working estimate of the process noise: the Inverse-Wishart mean.

        This is the correct object for PREDICT -- Cov(W_t | data_{1..t-1}) =
        Sigma_{t-1} + E_q(Q)[Q] by the law of total covariance, since q(Q) is
        independent of q(W_{t-1}) in the variational family. (d, d)."""
        d = self.mu.shape[0]
        return self.Psi / (self.nu - d - 1.0)

    @property
    def eps(self) -> Array:
        """Mixture contamination weight eps = eps_alpha / (eps_alpha + eps_beta) in (0, 1),
        the mean of the conjugate Beta posterior. eps is always inferred conjugately
        (there is no learn_eps switch); eps_init only seeds the Beta prior's mean."""
        return self.eps_alpha / (self.eps_alpha + self.eps_beta)

    def _eps_seed(self) -> Tuple[float, float]:
        """Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init)."""
        return 1.0, (1.0 / self.eps_init if self.eps_init > 0.0 else 100.0)

    def reset_posterior(self) -> "IBProbit":
        """Reset the recursive posterior to the seed state (mu = 0, Sigma = I, Beta counts
        back to the prior), used before a fresh online pass over data."""
        d, nc = self.mu.shape
        a0, b0 = self._eps_seed()
        Psi0, nu0 = self._q_seed()
        pa0, pb0 = self._pc_seed()
        return eqx.tree_at(
            lambda m: (m.mu, m.Sigma, m.step, m.posterior_ready, m.eps_alpha, m.eps_beta,
                       m.Psi, m.nu, m.p_a, m.p_b),
            self,
            (jnp.zeros((d, nc), dtype=jnp.float32),
             jnp.eye(d, dtype=jnp.float32),
             jnp.zeros((), dtype=jnp.int32),
             False,
             jnp.asarray(a0, dtype=jnp.float32),
             jnp.asarray(b0, dtype=jnp.float32),
             Psi0, nu0, pa0, pb0)
        )

    def reset(self, key: PRNGKey) -> "IBProbit":
        """Re-initialize to defaults (mu = 0, Sigma = I seed, the Inverse-Wishart prior on
        Q, and both Beta count pairs back to their priors) and clear the posterior.
        Signature kept for `bllarse.utils` / tests; ``key`` unused."""
        d, num_classes = self.mu.shape
        a0, b0 = self._eps_seed()
        Psi0, nu0 = self._q_seed()
        pa0, pb0 = self._pc_seed()
        return eqx.tree_at(
            lambda m: (m.Psi, m.nu, m.p_a, m.p_b, m.eps_alpha, m.eps_beta, m.beta,
                       m.mu, m.Sigma, m.step, m.posterior_ready),
            self,
            (Psi0, nu0, pa0, pb0,
             jnp.asarray(a0, dtype=jnp.float32),
             jnp.asarray(b0, dtype=jnp.float32),
             jnp.ones((), dtype=jnp.float32),
             jnp.zeros((d, num_classes), dtype=jnp.float32),
             jnp.eye(d, dtype=jnp.float32),
             jnp.zeros((), dtype=jnp.int32), False),
        )

    @staticmethod
    def _prefers_dual(N: int, D: int, K: int, num_iters: int) -> bool:
        """Is the pred-carrying CAVI form cheaper than the eta-carrying one?

        Shapes are static at trace time, so this is a plain Python branch. The
        dual pays 2N^2D up front to save 2NK(2D - N) per iteration, so it needs
        both a narrow batch and enough iterations to amortise the hat matrix.
        """
        primal = 4 * num_iters * N * D * K
        dual = 2 * N * N * D + 4 * N * D * K + 2 * (num_iters - 1) * N * N * K
        return num_iters >= 1 and dual < primal

    @staticmethod
    def _expected_z(pred: Array, y_one_hot: Array) -> Array:
        """Sign-truncated latent mean E_q[z] = m + Mills(m) at the gauge-fixed
        sigma_z^2 = 1, via log-space Mills ratios. ``pred`` is the raw score m = mu' f."""
        log_pdf = norm.logpdf(pred)
        mills_pos = jnp.exp(log_pdf - norm.logcdf(pred))
        mills_neg = jnp.exp(log_pdf - norm.logcdf(-pred))
        return pred + jnp.where(y_one_hot, mills_pos, -mills_neg)

    def _cavi_scan(self, fts: Array, x: Array, y_one_hot: Array, eta_init: Array, eta_anchor: Array, num_iters: int, g: Optional[Array] = None) -> Array:
        """CAVI fixed-point scan for the posterior mean. Returns the natural
        parameter eta = eta_anchor + fts' (g .* E_q[z]), where E_q[z] is the
        sign-truncated latent mean E[z] = m + Mills(m), m = mu' f (the latent scale is
        gauge-fixed at sigma_z^2 = 1). ``x = fts * Lambda_post^-1`` so ``x * eta``
        recovers m. ``g`` is an optional per-point (n,1) weight (None = unweighted).

        Two algebraically identical forms, picked by whichever is cheaper at these
        shapes. The primal carries eta (d,K) at 4NdK per iteration. Since the carry is
        always eta_anchor + fts'(g .* E[z]), the prediction it feeds back satisfies

            x * eta = x * eta_anchor + (x * fts')(g .* E[z]) = c + S(g .* E[z]),

        so the dual carries pred (N,K) against the precomputed hat matrix S = x * fts'
        at 2N^2K per iteration. Both x and S are built from UNGATED fts, matching the
        primal -- the gate enters only through the contribution. The dual wins on wide
        features and modest batches (roughly N < 2D) and loses on large batches, where
        the N x N matrix is both slower and far bigger than eta. PX (``_cavi_scan_px``)
        has no dual form: it needs eta_hat in d-space for its scalar quadratic.
        """
        if self._prefers_dual(x.shape[0], x.shape[1], y_one_hot.shape[-1], num_iters):
            S = mm_update(x, fts.T)
            c = mm_update(x, eta_anchor)

            def step_fn(pred, *args):
                E_q_z = self._expected_z(pred, y_one_hot)
                contrib = E_q_z if g is None else g * E_q_z
                return c + mm_update(S, contrib), None

            # one prediction is consumed by the final projection below, so the
            # scan is one iteration shorter than the primal loop
            pred, _ = lax.scan(step_fn, mm_update(x, eta_init), jnp.arange(num_iters - 1))
            E_q_z = self._expected_z(pred, y_one_hot)
            contrib = E_q_z if g is None else g * E_q_z
            return eta_anchor + mm_update(fts.T, contrib)

        def step_fn(eta_carry, *args):
            E_q_z = self._expected_z(mm_update(x, eta_carry), y_one_hot)
            contrib = E_q_z if g is None else g * E_q_z
            return eta_anchor + mm_update(fts.T, contrib), None

        eta_new, _ = lax.scan(step_fn, eta_init, jnp.arange(num_iters))
        return eta_new

    def _cavi_scan_px(self, fts: Array, x: Array, y_one_hot: Array, eta_pred: Array,
                      L_post, Lambda_pred: Array, num_iters: int,
                      g: Optional[Array] = None) -> Array:
        """Parameter-expanded (PX) CAVI scan -- accelerates the slow scale mode.

        We expand the prior+latent by a single working scale ``a`` (latent variance
        and prior covariance by a^2, prior mean by a), which is a flat redundancy of
        the expanded free energy. Optimizing ``a`` in closed form and reducing back to
        the canonical gauge (a=1) every iteration turns the eta-space update into

            eta_new = eta_pred + b / a,     b = X^T E[z],

        i.e. PX just rescales the *data increment* by 1/a. ``a`` solves the scalar
        quadratic  nK a^2 + B a - C = 0  with  eta_hat = eta_pred + b  (the plain
        M-step natural parameter, a=1) and m_hat = X Lambda_post^{-1} eta_hat = x eta_hat:
            B = sum(p * eta_hat),                        p   = Lambda_post^{-1} eta_pred,
            C = S_z + sum(eta_hat * M_c eta_hat),        M_c = Lambda_post^{-1} Lambda_pred Lambda_post^{-1},
            S_z = sum_ik (E[z^2] - 2 E[z] m_hat + m_hat^2).
        Both B and C are written in terms of eta_hat (= eta_pred + b) and the two
        scan-constant matrices p, M_c, so the per-iteration mu_hat / cho_solve are not
        needed -- the loop is pure matmuls. At the plain-CAVI fixed point nK = C - B,
        so a=1 and the reduction is the identity -> PX shares the plain fixed point,
        only converging faster."""
        n, K = fts.shape[0], y_one_hot.shape[-1]
        # latent-normalizer count: n*K ungated; the effective sample size K*sum(g) gated.
        nK = float(n * K) if g is None else K * jnp.sum(g)
        # scan-constant matrices: p = Lambda_post^{-1} eta_pred,
        #   M_c = Lambda_post^{-1} Lambda_pred Lambda_post^{-1}.
        p = cho_solve(L_post, eta_pred)                          # (d,K)
        M_c = cho_solve(L_post, cho_solve(L_post, Lambda_pred).T)  # (d,d)

        def step_fn(eta_carry, *args):
            m = mm_update(x, eta_carry)              # canonical mean scores  (n,K)

            E_q_z = self._expected_z(m, y_one_hot)
            contrib = E_q_z if g is None else g * E_q_z

            b = mm_update(fts.T, contrib)            # data increment  (d,K)
            eta_hat = eta_pred + b                   # plain M-step nat-param (a=1)
            m_hat = mm_update(x, eta_hat)            # = fts @ Lambda_post^{-1} eta_hat  (n,K)

            E_q_z2 = m * E_q_z + 1.0                  # truncated 2nd moment (sigma_z^2 = 1)
            resid = E_q_z2 - 2.0 * E_q_z * m_hat + m_hat * m_hat
            S_z = jnp.sum(resid if g is None else g * resid)   # per-point g-weight (m_hat NOT re-gated)
            B = jnp.sum(p * eta_hat)
            Cq = S_z + jnp.sum(eta_hat * mm_update(M_c, eta_hat))
            a = (-B + jnp.sqrt(B * B + 4.0 * nK * Cq)) / (2.0 * nK)

            return eta_pred + b / a, None

        eta_new, _ = lax.scan(step_fn, eta_pred, jnp.arange(num_iters))
        return eta_new

    def _increment_statistic(self, Qhat: Array, Lambda_pred: Array,
                             Sigma_new: Array, Delta: Array) -> Array:
        """Expected squared increment S_t = sum_k E[(W_t - W_{t-1})_k (W_t - W_{t-1})_k'].

        q-update-spec sec 5.5(a), kept in matrix form rather than reduced to its
        diagonal. The mean-field family of sec 3 has no joint over (W_{t-1}, W_t), and
        taking it literally gives ||Delta||^2 + K[Sigma_t + Sigma_{t-1}]_jj, which
        stays positive at Q = 0: a model that cannot represent a stationary head will
        always report that it is moving. The fix is exact and free -- conditional on
        W_t the increment is independent of the batch (the emission touches only W_t),
        so its posterior follows from PREDICT alone. With G = Qhat * Lambda_pred:

            E[delta_k]  = G Delta_k
            Cov(delta)  = G Sigma_{t-1} + G Sigma_t G'
            S_t = (G Delta)(G Delta)' + K [ G Sigma_{t-1} + G Sigma_t G' ]

        G Sigma_{t-1} is computed as Qhat - Qhat Lambda_pred Qhat, which it equals
        because Sigma_{t-1} = Lambda_pred^-1 - Qhat. That form is symmetric by
        inspection and avoids differencing near-equal matrices.

        At Qhat = 0, G = 0 and S_t = 0. S_t is PSD: the first term is a rank-<=K outer
        product, and Qhat - Qhat Lambda_pred Qhat factors as
        Qhat^(1/2)(I - Qhat^(1/2) Lambda_pred Qhat^(1/2))Qhat^(1/2), PSD because
        Lambda_pred = (Sigma_{t-1} + Qhat)^-1 <= Qhat^-1. So Psi stays PD with no
        jitter. Returns (d, d)."""
        K = Delta.shape[-1]
        G = mm_update(Qhat, Lambda_pred)                    # (d,d), not symmetric
        GD = mm_update(G, Delta)                            # (d,K)
        A = Qhat - mm_update(G, Qhat)                       # = G Sigma_{t-1}, symmetric
        Cv = mm_update(mm_update(G, Sigma_new), G.T)        # = G Sigma_t G'
        S = mm_update(GD, GD.T) + K * (A + Cv)
        return 0.5 * (S + S.T)

    def _log_bayes_factor(self, S: Array, Psi: Array, nu: Array, K: int) -> Array:
        """log of the Bayes Factor Surprise for the process-noise node.

        q-update-spec sec 5.5(b), generalized from the Gamma-Gamma compound to the
        Wishart / Inverse-Wishart one. The observation arriving at the node is S_t,
        and under (G1) it is Wishart: S | Q ~ W_d(Q, K). Compounding with an
        Inverse-Wishart(Psi, nu) belief gives the closed-form marginal

            log p(S | Psi, nu) = [ (K-d-1)/2 log|S| - logGamma_d(K/2) ]   <- free of (Psi,nu)
                               + block(Psi, nu)

            block(Psi, nu) = (nu/2) log|Psi|
                           + sum_j [ lgamma((nu+K)/2 + (1-j)/2) - lgamma(nu/2 + (1-j)/2) ]
                           - ((nu+K)/2) log|Psi + S|

        The (Psi,nu)-free part cancels in the ratio, and so does the
        d(d-1)/4 log(pi) constant inside the multivariate gamma -- hence the bare
        lgamma sum. Evaluated against the PRE-update belief, otherwise the surprise
        is measured against a belief already updated to explain S.

        Cost: two d x d Choleskys (|Psi + S| at each of the two beliefs) plus 2d
        lgamma. That is the same order as the Cholesky update() already does, not the
        handful of lgamma the diagonal formulation advertised. Scalar."""
        d = self.mu.shape[0]
        j = jnp.arange(1, d + 1, dtype=jnp.float32)
        Psi0, nu0 = self._q_seed(d)

        def block(Psi_, nu_):
            half, half_K = 0.5 * nu_, 0.5 * (nu_ + K)
            lg = jnp.sum(gammaln(half_K + 0.5 * (1.0 - j)) - gammaln(half + 0.5 * (1.0 - j)))
            return half * _logdet_pd(Psi_) + lg - half_K * _logdet_pd(Psi_ + S)

        return block(Psi0, nu0) - block(Psi, nu)

    def _surprise_rate(self, S: Array, Psi: Array, nu: Array,
                       p_a: Array, p_b: Array, K: int) -> Array:
        """VarSMiLe's adaptive forgetting rate gamma for the process-noise node.

        "Adaptive" describes the derivation, not necessarily observed behaviour: gamma
        is computed correctly at every operating point measured so far, but the update
        it feeds (``_q_recursion``) is close to inert there -- see that docstring's
        MEASURED LIMITATION paragraph.

        q-update-spec sec 5.5(c): gamma = m Sbar / (1 + m Sbar) with m = E[p_c]/(1-E[p_c])
        = p_a/p_b, computed as a sigmoid of log(m) + log(Sbar) so it cannot overflow.

        NORMALIZATION. The spec divides log S_BF by K because each scalar node saw K
        observations. One Wishart node absorbs d*K scalar observations, and log S_BF
        grows with d through both log-determinants and the d lgamma pairs, so
        dividing by K alone saturates gamma at 1 for any realistic d -- the same trap
        sec 5.5(b) warns about one level down. 1/(d*K) keeps gamma free of both K and
        d and numerically comparable to the diagonal formulation. This is a choice
        this codebase makes; the spec only covers the scalar case. Scalar in (0, 1)."""
        d = self.mu.shape[0]
        log_Sbar = self._log_bayes_factor(S, Psi, nu, K) / (d * K)
        return nn.sigmoid(jnp.log(p_a) - jnp.log(p_b) + log_Sbar)

    def _q_recursion(self, Qhat: Array, Lambda_pred: Array, Sigma_new: Array,
                     Delta: Array, K: int) -> Tuple[Array, Array, Array, Array, Array]:
        """Steps 5-8 of q-update-spec sec 8: the conjugate hyper-updates.

        MEASURED LIMITATION. In practice this node barely moves. When Qhat << Sigma the
        covariance term of S_t is K*Qhat (first order in Qhat) while the drift signal is
        O(Qhat^2) (second order), and S_t = K*Qhat is an EXACT fixed point of the gamma=0
        update -- so Qhat sits at its seed and the only forcing is quadratically suppressed.
        At alpha=1e-3 (q0 = alpha^2/B = 3.1e-8) this made learn_q=True and learn_q=False
        give bit-identical mu over 40 steps; the default was raised to alpha=0.1, which
        lifts the drift fraction of S_t from ~1.2e-8 to ~1.8e-5 but does not make the
        mechanism responsive. Read "adaptive" and "inferred" below as descriptions of the
        derivation, not of observed behaviour at these operating points.

        S_t | Q ~ Wishart_d(Q, K), whose conjugate is the Inverse-Wishart, so absent
        forgetting the update is nu += K, Psi += S_t. VarSMiLe replaces the fixed EMA
        discount with the inferred rate gamma:

            nu_t  = (1 - gamma) nu_{t-1}  + gamma nu_0  + K
            Psi_t = (1 - gamma) Psi_{t-1} + gamma Psi_0 + S_t

        At large gamma this lands on (Psi_0 + S_t, nu_0 + K) -- the prior plus this
        batch -- so a detected change discards the stale drift estimate, WHEN gamma
        actually reaches a large value; per the limitation above, at the measured
        operating points S_t itself carries almost no drift signal to detect. gamma is
        the variational responsibility q(c_t = 1), so it doubles as the soft count for
        the Beta on the change rate p_c.

        Note the Beta on p_c gets ONE count per step here where the diagonal
        formulation got d, so p_c identifies roughly d times slower and the prior mean
        1 - rho carries more of the weight for longer. rho bounds the memory either
        way; this is a rate-of-identification change, not a stability one. Worth
        logging p_c's trajectory as a diagnostic.

        Returns (Psi_new, nu_new, p_a_new, p_b_new, gamma)."""
        Psi0, nu0 = self._q_seed()
        pa0, pb0 = self._pc_seed()

        S = self._increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta)
        gamma = self._surprise_rate(S, self.Psi, self.nu, self.p_a, self.p_b, K)

        nu_new = (1.0 - gamma) * self.nu + gamma * nu0 + K
        Psi_new = (1.0 - gamma) * self.Psi + gamma * Psi0 + S
        Psi_new = 0.5 * (Psi_new + Psi_new.T)

        p_a_new = pa0 + self.rho * (self.p_a - pa0) + gamma
        p_b_new = pb0 + self.rho * (self.p_b - pb0) + (1.0 - gamma)
        return Psi_new, nu_new, p_a_new, p_b_new, gamma

    def calibrate(self, features: Array, y: Array, *, num_iters: int = 8) -> "IBProbit":
        """Temperature scaling via Newton steps on beta, using held-out (val) data.

        No ``loss_type`` parameter: unlike ``__call__`` (kept generic for interface
        parity with the classical/PolyaGamma losses), this method is IBProbit-only,
        so a dead per-call override had no reason to exist here -- ``self.loss_type``
        already fixes the branch ``lf(...)`` scores.
        """
        def nll(b):
            lf = eqx.tree_at(lambda m: m.beta, self, b)
            return lf(features, y).mean()

        def newton_step(b, _):
            g = grad(nll)(b)
            h = grad(grad(nll))(b)
            return jnp.clip(b - 0.25 * g / (h + 1e-8), 0.01, 100.0), None

        beta_new, _ = lax.scan(newton_step, self.beta, jnp.arange(num_iters))
        return eqx.tree_at(lambda m: m.beta, self, beta_new)

    def update(self, features: Array, y: Array, *, num_iters: int = 32,
               px: Optional[bool] = None) -> "IBProbit":
        """One recursive-Bayesian / CAVI minibatch update, in the order of
        q-update-spec sec 8:

            0. PREDICTIVE  ungated, pre-update: the one-step-ahead prequential predictive
            1. q(s)        responsibilities r_i from E[log eps] and that predictive
            2. q(eps)      Beta discount-update with counts (1-r_i), r_i
            3. PREDICT     Sigma_pred = Sigma + Qhat; Lambda_pred, eta_pred
            4. UPDATE      Lambda_t, eta_t with r_i gating both moments; PX or plain CAVI
            5-8.           S_t, the surprise, q(Q) and q(p_c)          [learn_q only]

        Step 0 must be ungated: the gate is derived from it, so gating first would
        suppress the signal that forms it. Steps 5-8 follow step 4 because S_t needs
        (mu_t, Sigma_t), and Qhat_t is consumed at t+1. They are deferred for the
        first ``warmup`` steps -- at t=0 the predictive is uniform and Delta = mu_1 is
        a full-size jump off the mu=0 seed, so the first batch reports a huge S and
        would drive gamma to 1 on every coordinate.

        r_i weights BOTH moments because a contaminated point contributes neither.
        There is no gradient step anywhere in this method."""
        use_px = self.px if px is None else px
        d, K = self.mu.shape[0], self.mu.shape[-1]
        y_one_hot = nn.one_hot(y, K)
        Qhat = self.Qhat

        # 0. PREDICTIVE -- ungated, pre-update, and the only predictive pass per batch.
        raw = self._predictive_logits(features)
        loss_vec = self._loss_from_logits(raw, y_one_hot)[0]

        # 1-2. q(s) then q(eps): the psi-form responsibility (sec 5.2).
        r = self._eps_responsibility(loss_vec)                       # (n,1)
        a0, b0 = self._eps_seed()
        r_flat = r[:, 0]
        eps_alpha_new = a0 + self.rho * (self.eps_alpha - a0) + jnp.sum(1.0 - r_flat)
        eps_beta_new = b0 + self.rho * (self.eps_beta - b0) + jnp.sum(r_flat)

        # The gate IS the responsibility (sec 5.2); held at 1 through warmup.
        g = jnp.where(self.step >= self.warmup, r, jnp.float32(1.0))

        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)

        # 3. PREDICT -- previous posterior inflated by the process noise.
        Sigma_pred = self.Sigma + Qhat
        Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T)
        L = cho_factor(Sigma_pred)
        tmp = cho_solve(L, jnp.concatenate([self.mu, jnp.eye(d)], axis=-1))
        eta_pred, Lambda_pred = tmp[:, :-d], tmp[:, -d:]

        # 4. UPDATE -- precision closed-form, mean by CAVI. sigma_z^2 === 1 (sec 5.7).
        Lambda_new = Lambda_pred + mm_update((g * fts).T, fts)
        Lambda_new = 0.5 * (Lambda_new + Lambda_new.T)
        L = cho_factor(Lambda_new)
        x = cho_solve(L, fts.T).T
        if use_px:
            eta_post = self._cavi_scan_px(fts, x, y_one_hot, eta_pred, L,
                                          Lambda_pred, num_iters, g)
        else:
            eta_post = self._cavi_scan(fts, x, y_one_hot, eta_pred, eta_pred, num_iters, g)
        tmp = cho_solve(L, jnp.concatenate([eta_post, jnp.eye(d)], axis=-1))
        mu_new, Sigma_new = tmp[:, :-d], tmp[:, -d:]

        # 5-8. The conjugate hyper-updates on Q and p_c.
        Psi_new, nu_new, p_a_new, p_b_new = self.Psi, self.nu, self.p_a, self.p_b
        if self.learn_q:
            Psi_r, nu_r, pa_r, pb_r, _ = self._q_recursion(
                Qhat, Lambda_pred, Sigma_new, mu_new - self.mu, K)
            engaged = self.step >= self.warmup
            Psi_new = jnp.where(engaged, Psi_r, Psi_new)
            nu_new = jnp.where(engaged, nu_r, nu_new)
            p_a_new = jnp.where(engaged, pa_r, p_a_new)
            p_b_new = jnp.where(engaged, pb_r, p_b_new)

        return eqx.tree_at(
            lambda m: (m.mu, m.Sigma, m.step, m.posterior_ready,
                       m.eps_alpha, m.eps_beta, m.Psi, m.nu, m.p_a, m.p_b),
            self,
            (mu_new, Sigma_new, self.step + 1, True,
             eps_alpha_new, eps_beta_new, Psi_new, nu_new, p_a_new, p_b_new),
        )

    @property
    def params(self) -> Tuple[Array, Array]:
        params = self.mu
        return (params[:-1], params[-1]) if self.use_bias else (params, None)

    def predictive_variance(self, features: Array) -> Array:
        """Per-input predictive variance of the latent score w^T f:
            v(f) = 1 + f^T (Qhat + Sigma) f.
        This is the (class-shared) denominator squared of the moderated-probit predictive
        (both Sigma and Qhat are class-independent, so v is one scalar per input, not per
        class). It sums epistemic (f^T Sigma f), process-noise-floor (f^T Qhat f) and
        aleatoric (the gauge-fixed sigma_z^2 = 1) terms; large for inputs in feature
        directions the posterior does not constrain, so it doubles as an epistemic OOD
        score (higher => more OOD). (n,)."""
        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        var = jnp.sum(mm_update(fts, self.Qhat + self.Sigma) * fts, axis=-1)
        return 1.0 + var

    def _predictive_logits(self, features: Array) -> Array:
        """Raw moderated-probit predictive logit (BEFORE the loss_type transform):
            logit = mu^T f / sqrt( 1 + f^T (Qhat + Sigma) f ).
        The process-noise floor f^T Qhat f keeps the denominator from collapsing as the
        posterior covariance Sigma does. Shared by ``__call__``, the responsibility and
        the gate so they all consume the SAME predictive. (n, K)."""
        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        scores = mm_loss(fts, self.mu)
        var = jnp.sum(mm_update(fts, self.Qhat + self.Sigma) * fts, axis=-1)
        return scores / jnp.sqrt(1.0 + var)[:, None]

    def _loss_from_logits(self, logits: Array, y_one_hot: Array) -> Tuple[Array, Array]:
        """Map raw moderated logits -> (loss, beta * transformed logits) under
        ``self.loss_type``. The choice of loss is relevant for gradients over params."""
        if self.loss_type == 0:
            # binary loss
            loss = optax.sigmoid_binary_cross_entropy(self.beta * logits, y_one_hot).sum(-1)

        elif self.loss_type == 1:
            # CBM Loss
            logits = self.logcdf(logits)
            loss = optax.safe_softmax_cross_entropy(self.beta * logits, y_one_hot)

        elif self.loss_type == 2:
            # CBC Loss
            logits = self.logcdf(logits) - self.logcdf(-logits)
            loss = optax.safe_softmax_cross_entropy(self.beta * logits, y_one_hot)

        elif self.loss_type == 3:
            # mixed loss
            logits1 = self.logcdf(logits)
            logits = logits1 - self.logcdf(-logits)

            loss_cbm = optax.safe_softmax_cross_entropy(self.beta * logits1, y_one_hot)
            loss_cbc = optax.safe_softmax_cross_entropy(self.beta * logits, y_one_hot)

            loss = loss_cbm - nn.softplus(loss_cbm - loss_cbc) + jnp.log(2)

        return loss, self.beta * logits

    def _eps_responsibility(self, loss_vec: Array) -> Array:
        """Per-point responsibility r_i = q(s_i = 0) = P(point i came from the model).

        q-update-spec sec 5.2. The CAVI stationary point wants the Beta's EXPECTED LOG,
        not its mean:

            r_i = sigmoid( E[log(1-eps)] - E[log eps] + log phat(y_i | f_i) + log K )
            E[log eps] = digamma(a) - digamma(a + b)

        with eps ~ Beta(eps_alpha, eps_beta), eps_alpha the contamination mass. phat is
        the pre-update, marginalised, one-step-ahead predictive, so log phat = -loss_vec.
        Strict CAVI would recompute r_i in the inner loop; using the prequential
        predictive instead makes r_i an out-of-sample judgement and stops a point
        certifying its own inlier status. Deliberate.

        In (0,1): well-predicted -> ~1, badly-predicted -> ~0. Returns (n, 1)."""
        dig_ab = digamma(self.eps_alpha + self.eps_beta)
        E_log_eps = digamma(self.eps_alpha) - dig_ab
        E_log_1meps = digamma(self.eps_beta) - dig_ab
        log_K = jnp.log(self.mu.shape[-1])
        return nn.sigmoid(E_log_1meps - E_log_eps - loss_vec + log_K)[:, None]

    def _mixture_logp(self, raw: Array) -> Array:
        """(n, K) log mixture predictive log[ (1-eps) softmax(beta * transform(raw)) + eps/K ].
        Assumes a softmax loss_type (``self.loss_type`` in {1,2,3}); used for eval scoring so
        NLL/Brier/ECE see the floored predictive. softmax(log_p_eps) == p_eps (a no-op), so
        downstream code that softmaxes the returned logits recovers the mixture."""
        dummy = jnp.zeros((raw.shape[0], raw.shape[1]), dtype=raw.dtype)
        t_logits = self._loss_from_logits(raw, dummy)[1]             # beta * transformed (y-independent)
        log_p_model = nn.log_softmax(t_logits, axis=-1)
        eps, log_K = self.eps, jnp.log(raw.shape[-1])
        return jnp.logaddexp(jnp.log1p(-eps) + log_p_model, jnp.log(eps) - log_K)

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: Optional[int] = None) -> Tuple[Array, Optional[Array]]:
        # `loss_type` is accepted for interface compatibility (like the classical
        # losses) but IGNORED: the branch is fixed by the instance field
        # `self.loss_type`, so the convergence target cannot drift across calls.
        logits = self._predictive_logits(features)
        # Score the floored mixture: return log p_eps as the "logits" (softmax is then a
        # no-op), so loss = -log p_eps(y) and any downstream NLL/Brier/ECE see the mixture.
        # Structural now (sec 5.2): under the contamination model the gate IS r_i, always.
        log_p_eps = self._mixture_logp(logits)
        loss = -jnp.take_along_axis(log_p_eps, y[:, None], axis=1)[:, 0]
        return (loss, log_p_eps) if with_logits else loss

class MultinomialPolyaGamma(eqx.Module):

    mu: Array
    Sigma: Array
    use_bias: bool

    def __init__(
        self,
        input_dim: int,
        num_classes:int,
        *,
        use_bias: bool = False,
        key: PRNGKey
    ):

        self.use_bias = use_bias
        d = input_dim + int(use_bias)
        k = num_classes - 1
        self.mu = jr.normal(key, (d, k)) * 1e-3
        self.Sigma = jnp.eye(d)[None].repeat(k, axis=0)

    def reset(self, key: PRNGKey) -> "MultinomialPolyaGamma":
        d, k = self.mu.shape
        mu = jr.normal(key, (d, k)) * 1e-3
        Sigma = jnp.eye(d)[None].repeat(k, axis=0)

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu, Sigma))

    def __get_b_ky(self):
        nc = self.mu.shape[-1] + 1
        return 1 - jnp.pad(jnp.eye(nc)[..., :-2], [(0, 0), (1, 0)]).cumsum(-1)

    def get_b_kappa(self, y: Array):
        y_onehot = nn.one_hot(y, self.mu.shape[-1] + 1)
        b = y_onehot @ self.__get_b_ky()
        kappa = y_onehot[..., :-1] - b / 2

        return b, kappa

    def update(self, features: Array, y: Array, *, num_iters: int = 32) -> "MultinomialPolyaGamma":

        x = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
        b, kappa = self.get_b_kappa(y)

        # get first nat parameter for beta posterior
        lam = jnp.expand_dims(jnp.sum(x[..., None, :] * kappa[..., None], axis=0), -1)

        def step_fn(carry, _):

            mu, Sigma = carry

            E_betabetaT = Sigma + mu[..., None, :] * mu[..., None]
            psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))

            rho = 0.5 * b * jnp.tanh(psi / 2) / (psi + 1e-8)
            F = jnp.sum(rho[..., None, None] * xxT, axis=0)
            tmp = jnp.eye(x.shape[-1]) + self.Sigma @ F 
            Sigma_new = lu_solve(lu_factor(tmp), self.Sigma)
            mu_new = (Sigma_new @ lam).squeeze(-1)

            return (mu_new, Sigma_new), None

        init = (self.mu.mT, self.Sigma)
        (mu_trans, Sigma), _ = lax.scan(step_fn, init, jnp.arange(num_iters))

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu_trans.mT, Sigma))

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: int = 0):

        x = jnp.pad(features, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        logits = x @ self.mu
        b_ky = self.__get_b_ky().mT
        
        if loss_type == 0:
            logits = jnp.pad(logits, [(0, 0), (0, 1)]) - nn.softplus(logits) @ b_ky
            loss = - jnp.sum(logits * nn.one_hot(y, self.mu.shape[-1] + 1), -1)
            return (loss, logits) if with_logits else loss

        elif loss_type == 1:
            params = self.mu.mT
            E_betabetaT = self.Sigma + params[:, None, :] * params[..., None]
            
            xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
            psi = jnp.sqrt(
                jnp.sum(E_betabetaT * xxT, axis=(-1, -2))
            )

            if with_logits:
                logits = jnp.pad(logits, [(0, 0), (0, 1)]) + ((psi - logits) / 2  - nn.softplus(psi)) @ b_ky
                loss = - jnp.sum(logits * nn.one_hot(y, self.mu.shape[-1] + 1), -1)
                return loss, logits
            else:
                b, kappa = self.get_b_kappa(y)
                loss = - jnp.sum(logits * kappa + b * (psi / 2 - nn.softplus(psi) ), -1)
                return loss


class IBPolyaGamma(eqx.Module):

    mu: Array
    Sigma: Array
    use_bias: bool

    def __init__(
        self,
        input_dim: int,
        num_classes:int,
        *,
        use_bias: bool = False,
        key: PRNGKey
    ):

        self.use_bias = use_bias
        d = input_dim + int(use_bias)
        self.mu = jr.normal(key, (d, num_classes)) * 1e-3
        self.Sigma = jnp.eye(d)[None].repeat(num_classes, axis=0)

    def reset(self, key: PRNGKey) -> "IBPolyaGamma":
        d, k = self.mu.shape
        mu = jr.normal(key, (d, k)) * 1e-3
        Sigma = jnp.eye(d)[None].repeat(k, axis=0)

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (mu, Sigma))

    def get_kappa(self, y: Array):
        return nn.one_hot(y, self.mu.shape[-1]) - 0.5

    def update(self, features: Array, y: Array, *, num_iters: int = 32) -> "IBPolyaGamma":

        x = jnp.pad(lax.stop_gradient(features), [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
        # b is always 1 for binary case
        kappa = self.get_kappa(y)

        # get first nat parameter for beta posterior
        eta = solve(self.Sigma, self.mu.T[..., None], assume_a='pos')
        lam = jnp.expand_dims(jnp.sum(x[..., None, :] * kappa[..., None], axis=0), -1) + eta

        def step_fn(carry, _):

            mu, Sigma = carry

            E_betabetaT = Sigma + mu[..., None, :] * mu[..., None]
            psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))

            # as b is one always we do not need it here
            rho = 0.5 * jnp.tanh(psi / 2) / (psi + 1e-8)
            F = jnp.sum(rho[..., None, None] * xxT, axis=0)
            tmp = jnp.eye(x.shape[-1]) + self.Sigma @ F 
            Sigma_new = lu_solve(lu_factor(tmp), self.Sigma)
            mu_new = (Sigma_new @ lam).squeeze(-1)

            return (mu_new, Sigma_new), None

        init = (self.mu.mT, self.Sigma)
        (final_mu_trans, final_Sigma), _ = lax.scan(step_fn, init, jnp.arange(num_iters))

        return eqx.tree_at(lambda x: (x.mu, x.Sigma), self, (final_mu_trans.T, final_Sigma))

    @property
    def params(self):
        return (self.mu[:-1], self.mu[-1]) if self.use_bias else self.mu

    def __call__(self, features: Array, y: Array, *, with_logits: bool = False, loss_type: int = 3) -> Tuple[Array, Optional[Array]]:
        x = jnp.pad(features, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        y_one_hot = nn.one_hot(y, self.mu.shape[-1])

        # compute loss
        pred = x @ self.mu

        # note: computation of loss is relevant for gradients over model parameters
        if loss_type == 0:
            # binary loss
            logits = pred
            loss = optax.sigmoid_binary_cross_entropy(logits, y_one_hot).sum(-1)

        elif loss_type == 1:
            # CBM Loss
            # compute log p(y=1|params, x)
            logits = pred - nn.softplus(pred)
            # alternative but slower to compute
            # params = self.mu.mT
            # E_betabetaT = self.Sigma + params[:, None, :] * params[..., None]
            # xxT = jnp.expand_dims(x[..., None, :] * x[..., None], -3)
            # psi = jnp.sqrt(jnp.sum(E_betabetaT * xxT, axis=(-1, -2)))
            # logits = (psi - pred) / 2  - nn.softplus(psi)

            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 2:
            # CBC Loss
            logits = pred
            loss = optax.safe_softmax_cross_entropy(logits, y_one_hot)

        elif loss_type == 3:
            # mixed loss
            logits = pred
            loss_cbm = optax.safe_softmax_cross_entropy(logits - nn.softplus(pred), y_one_hot)
            loss_cbc = optax.safe_softmax_cross_entropy(logits, y_one_hot)

            loss = loss_cbm - nn.softplus(loss_cbm - loss_cbc) + jnp.log(2)

        if with_logits:
            return loss, logits
        else:
            return loss
