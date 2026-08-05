import equinox as eqx
import optax
import jax.numpy as jnp
from jax import nn, vmap, random as jr, lax, grad
from jax.scipy.stats import norm
from jax.scipy.linalg import solve, lu_solve, lu_factor, cho_factor, cho_solve
from jaxtyping import Array, PRNGKeyArray as PRNGKey
from typing import Optional, Tuple, Callable
from functools import partial

const = jnp.sqrt(2 / jnp.pi)

def approx_cdf(x):
    return nn.sigmoid(2 * const * (x + 0.044715 * x ** 3) )

def approx_logcdf(x):
    z = 2 * const * (x + 0.044715 * x ** 3)
    return z - nn.softplus(z)

def mm(a, b, *, preset, dot_dtype):
    """matmul with explicit cast to dot_dtype and DotAlgorithmPreset."""
    return jnp.matmul(a.astype(dot_dtype), b.astype(dot_dtype), precision=preset)

mm_update = partial(mm, preset='F32_F32_F32', dot_dtype=jnp.float32)
mm_loss = partial(mm, preset='TF32_TF32_F32', dot_dtype=jnp.float32)

class IBProbit(eqx.Module):
    """
    Bayesian final layer for a multinomial probit model with a recursive
    (streaming) CAVI posterior and a learnable predictive temperature.

    Generative model (per observation i, class k):
        z_ik ~ N(w_k^T f_i, sigma_z^2),   sign-pattern (pseudo-)likelihood
        w_k  ~ carried Gaussian posterior, inflated by process noise q I each step.

    The posterior ``q(W) = N(mu, Sigma)`` is built by ``update`` as a recursive
    Bayesian filter: each minibatch's posterior is the next minibatch's prior,
    inflated by process noise ``q I`` (the PREDICT step). There is no persistent
    fixed prior -- ``mu_0`` and the seed ``Sigma = I`` are just the initial state
    and wash out with data.

    The predictive (used identically for the gradient step AND at eval) is the
    moderated probit
        logit_ik = mu_k^T f / sqrt( sigma_z^2 + f^T (q I + Sigma) f ).
    The two scalars ``q`` (process noise / covariance floor) and ``sigma_z^2``
    (latent noise) are the gradient-trained knobs; the process-noise floor
    ``q ||f||^2`` keeps the predictive variance from collapsing as ``Sigma`` does,
    so moderation survives abundant data.
    """
    # --- recursive posterior state (set by `update`; not gradient-trained) ---
    mu: Array              # (d, K) posterior mean
    Sigma: Array           # (d, d) posterior covariance
    step: Array            # int32 scalar; number of update() steps (drives BMR warmup)
    eps_alpha: Array       # scalar; Beta posterior count a (contamination mass)
    eps_beta: Array        # scalar; Beta posterior count b (model mass); eps = a/(a+b)
    # --- gradient-trained scalars ---
    log_q: Array           # scalar; q = exp(log_q)         (process noise / cov floor)
    log_sigma_z2: Array    # scalar; sigma_z^2 = exp(...)   (latent noise variance)
    beta: Array            # scalar temperature (calibrate only)
    # --- static-ish (non-array leaves) ---
    use_bias: bool
    cdf: Callable
    logcdf: Callable
    posterior_ready: bool
    key: PRNGKey
    # --- in-update q-learning config (static) ---
    learn_q: bool = eqx.field(static=True)
    hyper_lr: float = eqx.field(static=True)
    anchor_weight: float = eqx.field(static=True)
    anchor_mode: float = eqx.field(static=True)
    loss_type: int = eqx.field(static=True)
    px: bool = eqx.field(static=True)
    # --- per-data-point BMR gate config (static) ---
    bmr_gate: bool = eqx.field(static=True)
    bmr_warmup: int = eqx.field(static=True)
    # --- epsilon-contamination mixture config (static) ---
    learn_eps: bool = eqx.field(static=True)
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
        sigma_z2_init: float = 1.0,
        learn_q: bool = False,
        hyper_lr: float = 0.1,
        anchor_weight: float = 1e-4,
        anchor_mode: Optional[float] = None,
        loss_type: int = 2,
        px: bool = False,
        bmr_gate: bool = False,
        bmr_warmup: int = 0,
        learn_eps: bool = False,
        eps_init: float = 0.0,
        rho: float = 1.0,
    ):
        d = input_dim + int(use_bias)
        # Process noise q defaults to 1/D (D = embedding dim): the seed predictive
        # variance sigma_z^2 + q ||f||^2 then sits at the feature-RMS scale, so the
        # untrained predictive starts at the LayerNorm-equivalent normalization.
        q0 = (1.0 / input_dim) if q_init is None else float(q_init)
        self.key = key
        self.log_q = jnp.log(jnp.asarray(q0, dtype=jnp.float32))
        self.log_sigma_z2 = jnp.log(jnp.asarray(sigma_z2_init, dtype=jnp.float32))
        self.beta = jnp.ones((), dtype=jnp.float32)
        # recursive posterior seed: mu = 0, Sigma = I (so the first PREDICT step
        # gives Sigma_pred = (1 + q) I, the effective initial covariance).
        self.mu = jnp.zeros((d, num_classes), dtype=jnp.float32)
        self.Sigma = jnp.eye(d, dtype=jnp.float32)
        self.step = jnp.zeros((), dtype=jnp.int32)
        self.use_bias = use_bias
        self.cdf = approx_cdf if use_approx_cdf else norm.cdf
        self.logcdf = approx_logcdf if use_approx_cdf else norm.logcdf
        self.posterior_ready = False
        # in-update q-learning config (see update(): prequential SGD on log_q)
        self.learn_q = bool(learn_q)
        self.hyper_lr = float(hyper_lr)
        self.anchor_weight = float(anchor_weight)
        self.anchor_mode = float(q0 if anchor_mode is None else anchor_mode)
        self.loss_type = int(loss_type)
        self.px = bool(px)
        # per-data-point BMR gate: down-weight badly-predicted points in the update;
        # skipped for the first ``bmr_warmup`` steps (the mu=0 seed gate is a useless 0.5).
        self.bmr_gate = bool(bmr_gate)
        self.bmr_warmup = int(bmr_warmup)
        # epsilon-contamination mixture: p_eps = (1-eps) p_model + eps/K. The contamination
        # rate eps is estimated by CONJUGATE BETA updating inside update() (not by gradients):
        # each point's responsibility r_i is a soft Bernoulli count for a Beta-Bernoulli, so
        # eps = eps_alpha / (eps_alpha + eps_beta) is the posterior mean. The learnable prior
        # is Beta(1, 1/eps_init) (mean ~ eps_init); ``rho`` is the forgetting factor on the
        # accumulated counts (scripts set rho = 1 - batch/data_size to cap the memory at one
        # pass; rho=1 = unbounded accumulation). eps drives the floored predictive (scoring)
        # always, and gates assimilation only when bmr_gate is on. Active iff learn_eps or
        # eps_init>0 (``use_eps``); otherwise every eps code path is skipped and the model is
        # bitwise the plain IBProbit.
        self.learn_eps = bool(learn_eps)
        self.eps_init = float(eps_init)
        self.rho = float(rho)
        a0, b0 = self._eps_seed()
        self.eps_alpha = jnp.asarray(a0, dtype=jnp.float32)
        self.eps_beta = jnp.asarray(b0, dtype=jnp.float32)

    # --- gradient-trained predictive scalars ---
    @property
    def q(self) -> Array:
        return jnp.exp(self.log_q)

    @property
    def sigma_z2(self) -> Array:
        return jnp.exp(self.log_sigma_z2)

    @property
    def eps(self) -> Array:
        """Mixture contamination weight eps = eps_alpha / (eps_alpha + eps_beta) in (0, 1),
        the mean of the conjugate Beta posterior (fixed at eps_init when eps is not learned)."""
        return self.eps_alpha / (self.eps_alpha + self.eps_beta)

    @property
    def use_eps(self) -> bool:
        """Static: whether the eps-contamination mixture is active. Off => every eps code
        path is skipped and the model is bitwise the plain IBProbit."""
        return self.learn_eps or self.eps_init > 0.0

    def _eps_seed(self) -> Tuple[float, float]:
        """Seed (a0, b0) for the Beta contamination posterior. learn_eps: the prior
        Beta(1, 1/eps_init) (mean ~ eps_init; eps_init<=0 falls back to Beta(1, 100)). Fixed
        eps (eps_init>0, not learned): a unit-mass point (eps_init, 1-eps_init) so eps ==
        eps_init exactly. Mixture off: an unread (1, 1) sentinel."""
        if self.learn_eps:
            return 1.0, (1.0 / self.eps_init if self.eps_init > 0.0 else 100.0)
        if self.eps_init > 0.0:
            return self.eps_init, 1.0 - self.eps_init
        return 1.0, 1.0

    def reset_posterior(self) -> "IBProbit":
        """Reset the recursive posterior to the seed state (mu = 0, Sigma = I, Beta counts
        back to the prior), used before a fresh online pass over data."""
        d, nc = self.mu.shape
        a0, b0 = self._eps_seed()
        return eqx.tree_at(
            lambda m: (m.mu, m.Sigma, m.step, m.posterior_ready, m.eps_alpha, m.eps_beta),
            self,
            (jnp.zeros((d, nc), dtype=jnp.float32),
             jnp.eye(d, dtype=jnp.float32),
             jnp.zeros((), dtype=jnp.int32),
             False,
             jnp.asarray(a0, dtype=jnp.float32),
             jnp.asarray(b0, dtype=jnp.float32))
        )

    def reset(self, key: PRNGKey) -> "IBProbit":
        """Re-initialize to defaults (mu = 0, Sigma = I seed, q = 1/D, sigma_z^2 = 1, Beta
        counts to prior) and clear the posterior. Signature kept for `bllarse.utils` / tests;
        ``key`` unused."""
        d, num_classes = self.mu.shape
        input_dim = d - int(self.use_bias)
        q0 = 1.0 / input_dim
        a0, b0 = self._eps_seed()
        return eqx.tree_at(
            lambda m: (m.log_q, m.log_sigma_z2, m.eps_alpha, m.eps_beta, m.beta, m.mu, m.Sigma, m.step, m.posterior_ready),
            self,
            (jnp.log(jnp.asarray(q0, dtype=jnp.float32)),
             jnp.zeros((), dtype=jnp.float32),                  # sigma_z^2 = 1
             jnp.asarray(a0, dtype=jnp.float32),
             jnp.asarray(b0, dtype=jnp.float32),
             jnp.ones((), dtype=jnp.float32),
             jnp.zeros((d, num_classes), dtype=jnp.float32),
             jnp.eye(d, dtype=jnp.float32),
             jnp.zeros((), dtype=jnp.int32), False),
        )

    def _cavi_scan(self, fts: Array, x: Array, y_one_hot: Array, eta_init: Array, eta_anchor: Array, num_iters: int, sigma_z: Array, g: Optional[Array] = None) -> Array:
        """CAVI fixed-point scan for the posterior mean. Returns the natural
        parameter eta = eta_anchor + (1/sigma_z^2) X^T (g ⊙ E_q[z]), where E_q[z] is the
        sign-truncated latent mean E[z] = m + sigma_z * Mills(m / sigma_z),
        m = mu^T f. ``x = fts @ Lambda_post^{-1}`` so ``x @ eta`` recovers m. ``g`` is an
        optional per-point (n,1) BMR weight (None = unweighted)."""
        inv_sz2 = 1.0 / (sigma_z * sigma_z)

        def step_fn(eta_carry, *args):
            pred = mm_update(x, eta_carry)           # mean scores m = fts @ mu_carry
            alpha = pred / sigma_z                    # standardized by the latent SD

            # E_q[z_ik] = m + sigma_z * Mills(m / sigma_z) via log-space Mills ratios
            log_pdf = norm.logpdf(alpha)
            logcdf1 = norm.logcdf(alpha)
            logcdf2 = norm.logcdf(-alpha)
            mills_pos = jnp.exp(log_pdf - logcdf1)
            mills_neg = jnp.exp(log_pdf - logcdf2)
            correction = sigma_z * jnp.where(y_one_hot, mills_pos, -mills_neg)
            E_q_z = pred + correction
            contrib = E_q_z if g is None else g * E_q_z

            return eta_anchor + inv_sz2 * mm_update(fts.T, contrib), None

        eta_new, _ = lax.scan(step_fn, eta_init, jnp.arange(num_iters))
        return eta_new

    def _cavi_scan_px(self, fts: Array, x: Array, y_one_hot: Array, eta_pred: Array,
                      L_post, Lambda_pred: Array, num_iters: int, sigma_z: Array,
                      g: Optional[Array] = None) -> Array:
        """Parameter-expanded (PX) CAVI scan -- accelerates the slow scale mode.

        We expand the prior+latent by a single working scale ``a`` (latent variance
        and prior covariance by a^2, prior mean by a), which is a flat redundancy of
        the expanded free energy. Optimizing ``a`` in closed form and reducing back to
        the canonical gauge (a=1) every iteration turns the eta-space update into

            eta_new = eta_pred + b / a,     b = (1/sigma_z^2) X^T E[z],

        i.e. PX just rescales the *data increment* by 1/a. ``a`` solves the scalar
        quadratic  nK a^2 + B a - C = 0  with  eta_hat = eta_pred + b  (the plain
        M-step natural parameter, a=1) and m_hat = X Lambda_post^{-1} eta_hat = x eta_hat:
            B = sum(p * eta_hat),                         p   = Lambda_post^{-1} eta_pred,
            C = S_z/sigma_z^2 + sum(eta_hat * M_c eta_hat), M_c = Lambda_post^{-1} Lambda_pred Lambda_post^{-1},
            S_z = sum_ik (E[z^2] - 2 E[z] m_hat + m_hat^2).
        Both B and C are written in terms of eta_hat (= eta_pred + b) and the two
        scan-constant matrices p, M_c, so the per-iteration mu_hat / cho_solve are not
        needed -- the loop is pure matmuls. At the plain-CAVI fixed point nK = C - B,
        so a=1 and the reduction is the identity -> PX shares the plain fixed point,
        only converging faster."""
        inv_sz2 = 1.0 / (sigma_z * sigma_z)
        sz2 = sigma_z * sigma_z
        n, K = fts.shape[0], y_one_hot.shape[-1]
        # latent-normalizer count: n*K ungated; the effective sample size K*sum(g) gated.
        nK = float(n * K) if g is None else K * jnp.sum(g)
        # scan-constant matrices: p = Lambda_post^{-1} eta_pred,
        #   M_c = Lambda_post^{-1} Lambda_pred Lambda_post^{-1}.
        p = cho_solve(L_post, eta_pred)                          # (d,K)
        M_c = cho_solve(L_post, cho_solve(L_post, Lambda_pred).T)  # (d,d)

        def step_fn(eta_carry, *args):
            m = mm_update(x, eta_carry)              # canonical mean scores  (n,K)
            alpha = m / sigma_z

            log_pdf = norm.logpdf(alpha)
            mills_pos = jnp.exp(log_pdf - norm.logcdf(alpha))
            mills_neg = jnp.exp(log_pdf - norm.logcdf(-alpha))
            E_q_z = m + sigma_z * jnp.where(y_one_hot, mills_pos, -mills_neg)
            contrib = E_q_z if g is None else g * E_q_z

            b = inv_sz2 * mm_update(fts.T, contrib)  # data increment  (d,K)
            eta_hat = eta_pred + b                   # plain M-step nat-param (a=1)
            m_hat = mm_update(x, eta_hat)            # = fts @ Lambda_post^{-1} eta_hat  (n,K)

            E_q_z2 = m * E_q_z + sz2                  # truncated 2nd moment
            resid = E_q_z2 - 2.0 * E_q_z * m_hat + m_hat * m_hat
            S_z = jnp.sum(resid if g is None else g * resid)   # per-point g-weight (m_hat NOT re-gated)
            B = jnp.sum(p * eta_hat)
            Cq = S_z * inv_sz2 + jnp.sum(eta_hat * mm_update(M_c, eta_hat))
            a = (-B + jnp.sqrt(B * B + 4.0 * nK * Cq)) / (2.0 * nK)

            return eta_pred + b / a, None

        eta_new, _ = lax.scan(step_fn, eta_pred, jnp.arange(num_iters))
        return eta_new

    def calibrate(self, features: Array, y: Array, *, num_iters: int = 8, loss_type: int = 2) -> "IBProbit":
        """Temperature scaling via Newton steps on beta, using held-out (val) data.
        """
        def nll(b):
            lf = eqx.tree_at(lambda m: m.beta, self, b)
            return lf(features, y, loss_type=loss_type).mean()

        def newton_step(b, _):
            g = grad(nll)(b)
            h = grad(grad(nll))(b)
            return jnp.clip(b - 0.25 * g / (h + 1e-8), 0.01, 100.0), None

        beta_new, _ = lax.scan(newton_step, self.beta, jnp.arange(num_iters))
        return eqx.tree_at(lambda m: m.beta, self, beta_new)

    def update(self, features: Array, y: Array, *, num_iters: int = 32, px: Optional[bool] = None, bmr_gate: Optional[bool] = None) -> "IBProbit":
        """One recursive-Bayesian / CAVI minibatch update.

        PREDICT: the previous posterior, inflated by process noise, is this
        batch's prior -- ``Sigma_pred = Sigma + q I`` (no persistent anchor).
        UPDATE: assimilate the batch under latent noise ``sigma_z^2``; the
        posterior precision is closed-form, the mean comes from the CAVI scan:
            Lambda_post = inv(Sigma_pred) + (1/sigma_z^2) X^T X
            mu_post     = Lambda_post^{-1} ( inv(Sigma_pred) mu + (1/sigma_z^2) X^T E[z] )
        Sets (mu, Sigma) and posterior_ready=True.

        ``px=True`` runs the parameter-expanded CAVI scan (``_cavi_scan_px``), which
        collapses the slow scale mode and converges in far fewer iterations to the
        SAME posterior as the plain scan.

        If ``self.learn_q`` is set, a single prequential SGD step is taken on
        ``log_q`` BEFORE the PREDICT/CAVI assimilation: the gradient of the
        pre-update predictive NLL (``self.loss_type``) plus an optional Inverse-Gamma
        anchor (``anchor_weight * (log_q + anchor_mode * exp(-log_q))``, mode at
        ``anchor_mode``) is descended with step size ``hyper_lr``. The stepped ``q``
        then drives the PREDICT step and is persisted. This folds the external
        ``ood_evaluation`` hyper-optimizer into the recursion so the training loop
        needs no separate optimizer.

        ``px`` selects the parameter-expanded scan; if left ``None`` it falls back to
        the instance field ``self.px`` (so the training loop enables PX via
        construction without passing a kwarg).

        If ``self.bmr_gate`` (or the ``bmr_gate`` kwarg) is set, each observation i is
        softly down-weighted in BOTH the precision and the CAVI mean, giving a data-driven
        effective sample size. With the eps-mixture active the weight is the per-point
        responsibility ``r_i`` (probability the point came from the model, not the
        contamination); otherwise it is the committed fixed-1/2 BMR gate
        ``g_i = Phi(mu~_i) / (Phi(mu~_i) + 1/2)``. It is skipped for the first
        ``bmr_warmup`` update steps -- on the mu=0 seed the gate is uninformative.

        If ``self.learn_eps``, the contamination rate ``eps`` is updated by a single
        CONJUGATE BETA step per call (no gradient): the responsibilities ``r_i`` are soft
        Bernoulli counts, ``eps_alpha += sum_i (1 - r_i)`` (contamination mass) and
        ``eps_beta += sum_i r_i`` (model mass), so ``eps = eps_alpha/(eps_alpha+eps_beta)``
        is the mean of the Beta(1, 1/eps_init) posterior. ``r_i`` is formed whenever the
        mixture is active, independent of ``bmr_gate`` (which only decides whether it also
        gates assimilation). The forgetting factor ``self.rho`` decays the accumulated excess
        over the prior each step (``rho = 1 - B/N`` gives a one-pass memory; ``rho = 1``
        accumulates over the whole recursive pass); counts are reset by ``reset_posterior``.
        """
        use_px = self.px if px is None else px
        use_bmr = self.bmr_gate if bmr_gate is None else bmr_gate
        use_eps = self.use_eps                                    # static: mixture active?

        # Pre-step eps drives the responsibility (the gate / Beta counts read pre-step q/eps,
        # like the committed BMR gate; assimilation below uses the stepped q).
        eps_pre = self.eps if use_eps else None

        # Optional prequential SGD step on log_q, taken on the PRE-update predictive; the
        # per-point predictive loss is reused (has_aux) for the responsibility so the step,
        # the gate and the eps update all share one predictive pass.
        raw = None
        loss_vec = None
        if self.learn_q:
            def preq_obj(log_q):
                model = eqx.tree_at(lambda m: m.log_q, self, log_q)
                raw_l = model._predictive_logits(features)
                y_oh = nn.one_hot(y, self.mu.shape[-1])
                loss_l = model._loss_from_logits(raw_l, y_oh)[0]   # per-point plain predictive NLL
                return self._preq_objective(loss_l, log_q), (raw_l, loss_l)
            g_grad, (raw, loss_vec) = grad(preq_obj, has_aux=True)(self.log_q)
            self = eqx.tree_at(lambda m: m.log_q, self, self.log_q - self.hyper_lr * g_grad)
        elif use_eps or use_bmr:
            raw = self._predictive_logits(features)                # one pre-update pass
            if use_eps:
                loss_vec = self._loss_from_logits(raw, nn.one_hot(y, self.mu.shape[-1]))[0]

        # Per-point responsibility r_i = P(point i from the model | label); formed whenever
        # the mixture is active (drives both the Beta eps update and, if on, the gate).
        r = self._eps_responsibility(loss_vec, eps_pre) if use_eps else None   # (n,1) in (0,1)

        # Conjugate Beta update of the contamination rate (learn_eps only): the r_i are soft
        # Bernoulli counts -- eps_alpha += sum(1-r_i) (contamination mass), eps_beta += sum(r_i)
        # (model mass) -- so eps = eps_alpha/(eps_alpha+eps_beta) is the Beta posterior mean.
        # The prior Beta(1, 1/eps_init) is the seed. The forgetting factor rho decays the
        # accumulated EXCESS over the prior before adding the batch counts: with rho = 1 - B/N
        # (batch B, data size N) the steady-state concentration is ~N (a one-pass memory), so
        # eps tracks recent data; rho=1 recovers unbounded accumulation. Replaces the former
        # in-update SGD on eps.
        eps_alpha_new, eps_beta_new = self.eps_alpha, self.eps_beta
        if self.learn_eps:
            r_flat = r[:, 0]
            a0, b0 = self._eps_seed()
            eps_alpha_new = a0 + self.rho * (self.eps_alpha - a0) + jnp.sum(1.0 - r_flat)
            eps_beta_new = b0 + self.rho * (self.eps_beta - b0) + jnp.sum(r_flat)

        # Per-point gate (pre-step); active only when bmr_gate is on. With the mixture active
        # the gate is the responsibility r_i, else the committed fixed-1/2 BMR gate. Held at
        # g == 1 (un-gated) for the first bmr_warmup steps (the mu=0 seed gate is useless).
        g = None
        if use_bmr:
            g_real = r if use_eps else self._gate_from_logits(raw, y)   # (n,1)
            g = jnp.where(self.step >= self.bmr_warmup, g_real, jnp.float32(1.0))

        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        d = fts.shape[-1]

        sigma_z = jnp.sqrt(self.sigma_z2)
        inv_sz2 = 1.0 / self.sigma_z2

        # PREDICT: previous posterior + process noise is this batch's prior.

        indices = jnp.diag_indices(self.Sigma.shape[0])
        Sigma_pred = self.Sigma.at[indices].add(self.q)
        Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T)   # symmetrize for stability
        L = cho_factor(Sigma_pred)
        tmp = cho_solve(L, jnp.concatenate([self.mu, jnp.eye(d)], axis=-1))
        eta_pred = tmp[:, :-d]
        Lambda_pred = tmp[:, -d:]

        # UPDATE: posterior precision is closed-form; mean via the CAVI scan.
        # The BMR gate g (if any) weights point i in both moments: (g ⊙ X)^T X and g ⊙ E[z].
        gated_fts = fts if g is None else g * fts
        xxT = mm_update(gated_fts.T, fts)
        Lambda_new = Lambda_pred + inv_sz2 * xxT
        Lambda_new = 0.5 * (Lambda_new + Lambda_new.T)   # symmetrize for stability
        y_one_hot = nn.one_hot(y, self.mu.shape[-1])

        L = cho_factor(Lambda_new)
        x = cho_solve(L, fts.T).T
        if use_px:
            eta_post = self._cavi_scan_px(fts, x, y_one_hot, eta_pred, L,
                                          Lambda_pred, num_iters, sigma_z, g)
        else:
            eta_post = self._cavi_scan(fts, x, y_one_hot, eta_pred, eta_pred, num_iters, sigma_z, g)

        tmp = cho_solve(L, jnp.concatenate([eta_post, jnp.eye(d)], axis=-1))
        mu_new = tmp[:, :-d]
        Sigma_new = tmp[:, -d:]

        return eqx.tree_at(
            lambda m: (m.mu, m.Sigma, m.step, m.posterior_ready, m.eps_alpha, m.eps_beta),
            self,
            (mu_new, Sigma_new, self.step + 1, True, eps_alpha_new, eps_beta_new),
        )

    @property
    def params(self) -> Tuple[Array, Array]:
        params = self.mu
        return (params[:-1], params[-1]) if self.use_bias else (params, None)

    def predictive_variance(self, features: Array) -> Array:
        """Per-input predictive variance of the latent score w^T f:
            v(f) = sigma_z^2 + f^T (q I + Sigma) f  =  sigma_z^2 + q ||f||^2 + f^T Sigma f.
        This is the (class-shared) denominator squared of the moderated-probit predictive
        (Sigma is class-independent, so v is one scalar per input, not per class). It sums
        epistemic (f^T Sigma f), process-noise-floor (q ||f||^2) and aleatoric (sigma_z^2)
        terms; large for inputs in feature directions the posterior does not constrain, so
        it doubles as an epistemic OOD score (higher => more OOD). (n,)."""
        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        return self.sigma_z2 + jnp.sum((self.q * fts + (fts @ self.Sigma)) * fts, axis=-1)

    def _predictive_logits(self, features: Array) -> Array:
        """Raw moderated-probit predictive logit (BEFORE the loss_type transform):
            logit = mu^T f / sqrt( sigma_z^2 + f^T (q I + Sigma) f ).
        The process-noise floor q ||f||^2 keeps the denominator from collapsing as the
        posterior covariance Sigma does. Shared by ``__call__``, the ``learn_q`` gradient
        step, and the BMR gate so they all consume the SAME predictive. (n, K)."""
        fts = features.astype(jnp.float32)
        fts = jnp.pad(fts, [(0, 0), (0, int(self.use_bias))], constant_values=1.0)
        scores = mm_loss(fts, self.mu)
        var = jnp.sum((self.q * fts + (fts @ self.Sigma)) * fts, axis=-1)
        return scores / jnp.sqrt(self.sigma_z2 + var)[:, None]

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

    def _gate_from_logits(self, raw: Array, y: Array) -> Array:
        """Per-data-point BMR gate from the raw moderated predictive, evaluated at the
        TRUE label: g_i = Phi(mu~_i) / (Phi(mu~_i) + 1/2),  mu~_i = raw[i, y_i].
        In (0, 2/3): well-predicted -> 2/3, badly-predicted -> 0. Returns (n, 1)."""
        s = jnp.take_along_axis(raw, y[:, None], axis=1)   # (n,1) true-class score
        p1 = self.cdf(s)
        return p1 / (p1 + 0.5)

    def _eps_responsibility(self, loss_vec: Array, eps: Array) -> Array:
        """Per-point responsibility r_i = P(point from model | label) under the
        eps-contamination mixture: (1-eps) p / ((1-eps) p + eps/K), p = exp(-loss_i)
        the model's prob at the true label. In (0,1): well-predicted -> ~1, badly-predicted
        -> 0; uniform (p=1/K) -> 1-eps. log-space for stability. Returns (n,1)."""
        log_K = jnp.log(self.mu.shape[-1])
        log_num = jnp.log1p(-eps) - loss_vec                          # log((1-eps) p_model(y))
        log_den = jnp.logaddexp(log_num, jnp.log(eps) - log_K)
        return jnp.exp(log_num - log_den)[:, None]

    def _preq_objective(self, loss_vec: Array, log_q: Array) -> Array:
        """Prequential objective for the in-update log_q SGD step: the mixture predictive NLL
        at the current (fixed) eps when the mixture is active, else the plain predictive NLL,
        plus the Inverse-Gamma anchor on q. ``loss_vec`` is the per-point plain predictive
        NLL. eps is no longer gradient-trained -- it is set by the conjugate Beta update."""
        if self.use_eps:
            eps = self.eps
            log_K = jnp.log(self.mu.shape[-1])
            obj = -jnp.logaddexp(jnp.log1p(-eps) - loss_vec, jnp.log(eps) - log_K).mean()
        else:
            obj = loss_vec.mean()
        if self.anchor_weight > 0.0:
            obj = obj + self.anchor_weight * (log_q + self.anchor_mode * jnp.exp(-log_q))
        return obj

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
        if self.use_eps:
            # Score the floored mixture: return log p_eps as the "logits" (softmax is then a
            # no-op), so loss = -log p_eps(y) and any downstream NLL/Brier/ECE see the mixture.
            log_p_eps = self._mixture_logp(logits)
            loss = -jnp.take_along_axis(log_p_eps, y[:, None], axis=1)[:, 0]
            return (loss, log_p_eps) if with_logits else loss
        y_one_hot = nn.one_hot(y, self.mu.shape[-1])
        loss, logits_t = self._loss_from_logits(logits, y_one_hot)
        return (loss, logits_t) if with_logits else loss

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
