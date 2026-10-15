import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import nn
import equinox as eqx
from bllarse.losses import IBProbit, IBPolyaGamma, MultinomialPolyaGamma


def _pad(x):
    return jnp.pad(x, [(0, 0), (0, 1)], constant_values=1.0)


def test_independent_binary_probit_loss():
    key = jr.PRNGKey(0)
    D, C, N = 10, 5, 1_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))

    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)

    key, model_key = jr.split(key)
    bmp = IBProbit(input_dim=D, num_classes=C, key=model_key, use_bias=True, q_init=1.0 / D)
    bmp = bmp.update(x, y, num_iters=64)

    W_recovered, b_recovered = bmp.params

    assert jnp.allclose(W, W_recovered, atol=10.0)  # try to improve atol
    assert jnp.allclose(b, b_recovered, atol=10.0)  # try to imporve atol

    for loss_type in range(4):
        loss, logits = bmp(x, y, with_logits=True, loss_type=loss_type)
        print(loss_type, loss.mean())

    acc = jnp.mean(logits.argmax(-1) == y)
    print('acc = ', acc)


def test_ibprobit_reset_preserves_parameter_shapes():
    key = jr.PRNGKey(0)
    model = IBProbit(input_dim=10, num_classes=3, key=key, use_bias=True, q_init=0.1)
    key, reset_key = jr.split(key)
    reset_model = model.reset(reset_key)

    assert reset_model.mu.shape == model.mu.shape
    assert reset_model.Sigma.shape == model.Sigma.shape
    assert jnp.isfinite(reset_model.mu).all()
    assert jnp.isfinite(reset_model.Sigma).all()
    assert reset_model.posterior_ready is False


def test_ibprobit_posterior_precision_uses_process_noise():
    """Fresh posterior (Sigma=I seed) -> Lambda_pred = inv(I + Qhat) = 1/(1+q0) I at the
    isotropic seed, and the M-step accumulates the raw data precision (sigma_z^2 = 1):
        Lambda_post = 1/(1+q0) I + X^T X.
    This pins the process-noise seed and the gauge-fixed latent scale in CAVI."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 200
    d = D + 1
    m = IBProbit(D, C, key=key, q_init=0.5)
    x = jr.normal(jr.PRNGKey(1), (N, D))
    y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    fts = _pad(x)

    updated = m.update(x, y, num_iters=8)

    expected = jnp.eye(d) / 1.5 + fts.T @ fts   # Lambda_post = inv(Sigma_post)
    # Lambda is not stored; recover the precision as inv(Sigma). The double
    # inversion adds float32 error, so use a slightly looser absolute slack.
    assert jnp.allclose(jnp.linalg.inv(updated.Sigma), expected, rtol=2e-2, atol=1e-1)


def test_ibprobit_recursion_uses_posterior_as_prior():
    """Iterative Bayesian update: the batch-1 posterior (plus the process noise Qhat) is
    the batch-2 prior, so Lambda_post(2) = inv(Sigma_1 + Qhat) + X2^T X2. warmup=2 keeps
    both steps un-gated, so the second precision is the plain accumulation."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key, q_init=1.0 / D, warmup=2)
    x1 = jr.normal(jr.PRNGKey(1), (N, D)); y1 = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    x2 = jr.normal(jr.PRNGKey(3), (N, D)); y2 = jr.randint(jr.PRNGKey(4), (N,), 0, C)
    fts2 = _pad(x2)

    m1 = m.update(x1, y1, num_iters=8)
    m2 = m1.update(x2, y2, num_iters=8)

    Lambda_pred2 = jnp.linalg.inv(m1.Sigma + m1.Qhat)
    expected = Lambda_pred2 + fts2.T @ fts2       # Lambda_post(2)
    # Lambda is not stored; recover the precision as inv(Sigma).
    assert jnp.allclose(jnp.linalg.inv(m2.Sigma), expected, rtol=2e-2, atol=1e-1)


def _near_separable(D, C, N):
    """Near-separable labels (low truncation -> feedback factor D->1, spectral radius
    near 1), which is the slow-scale regime where the Picard scan crawls -- mirrors the
    confident-point behaviour on real DINOv3 features. Uniform-random labels instead
    damp the scale mode and converge in a handful of iters, hiding the PX speed-up."""
    x = jr.normal(jr.PRNGKey(1), (N, D))
    W_true = jr.normal(jr.PRNGKey(7), (D, C))
    y = jr.categorical(jr.PRNGKey(2), 3.0 * (x @ W_true))
    return x, y


def test_ibprobit_px_matches_plain_fixed_point():
    """Parameter-expanded CAVI (px=True) must converge to the SAME posterior as the
    plain Picard scan run to convergence. PX rescales the data increment by 1/a each
    iteration with a solved from a scalar quadratic; at the plain fixed point a=1
    exactly (nK = C - B), so the reduction is the identity there -> identical mu/Sigma."""
    m = IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6)
    x, y = _near_separable(6, 3, 200)

    plain = m.update(x, y, num_iters=6000)           # plain Picard, run to convergence
    px = m.update(x, y, num_iters=300, px=True)       # PX, far fewer iters

    assert jnp.allclose(px.mu, plain.mu, rtol=3e-2, atol=2e-2)
    assert jnp.allclose(px.Sigma, plain.Sigma, rtol=3e-2, atol=2e-2)


def test_ibprobit_px_converges_faster():
    """PX collapses the slow scale mode, so after few iters it is strictly closer to
    the converged solution than the plain Picard scan with the same iter budget."""
    m = IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6)
    x, y = _near_separable(6, 3, 200)

    target = m.update(x, y, num_iters=6000).mu        # converged reference
    K = 30
    err_plain = jnp.linalg.norm(m.update(x, y, num_iters=K).mu - target)
    err_px = jnp.linalg.norm(m.update(x, y, num_iters=K, px=True).mu - target)

    assert float(err_px) < float(err_plain)


def test_ibprobit_px_instance_field_drives_scan():
    """px can be set at construction so the training loop (which calls update without
    a px kwarg) gets the PX scan; an explicit px kwarg still overrides the field."""
    x, y = _near_separable(6, 3, 200)
    base = IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6)   # px field default False
    field_px = IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6, px=True)

    # field px=True with no kwarg == explicit px=True on the default model
    a = field_px.update(x, y, num_iters=300)
    b = base.update(x, y, num_iters=300, px=True)
    assert jnp.allclose(a.mu, b.mu) and jnp.allclose(a.Sigma, b.Sigma)

    # an explicit px kwarg overrides the field
    c = field_px.update(x, y, num_iters=300, px=False)
    d = base.update(x, y, num_iters=300)                      # plain scan
    assert jnp.allclose(c.mu, d.mu) and jnp.allclose(c.Sigma, d.Sigma)


def test_ibprobit_predictive_denominator():
    """Predictive score (train grad AND eval) is mu^T f / sqrt(1 + f^T (Qhat+Sigma) f),
    the raw moderated logit _predictive_logits computes and every internal consumer
    (the responsibility, the gate, __call__) shares. __call__ always scores the
    floored contamination mixture on top of this, so the bare formula is checked
    directly against _predictive_logits rather than through __call__."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    # loss_type is irrelevant here -- _predictive_logits (unlike __call__) does not
    # consume it at all -- so any supported value does (loss_type=0 is not
    # constructible, see test_loss_type_zero_and_four_are_rejected).
    m = IBProbit(D, C, key=key, q_init=1.0 / D, loss_type=1).update(
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (32, D))
    fts = _pad(f)

    scores = fts @ m.mu
    var = jnp.sum((fts @ (m.Qhat + m.Sigma)) * fts, -1)
    expected = scores / jnp.sqrt(1.0 + var)[:, None]

    logits = m._predictive_logits(f)
    # mm_loss uses the TF32 dot preset, so tolerate ~1e-3 vs a full-precision matmul.
    assert jnp.allclose(logits, expected, rtol=1e-2, atol=1e-2)


def test_ibprobit_process_noise_prevents_collapse():
    """With N >> d and repeated passes, the bare posterior variance f^T Sigma f
    collapses, but the process-noise floor f^T Qhat f keeps the predictive variance
    f^T (Qhat+Sigma) f bounded away from zero -> moderation survives."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 4000
    m = IBProbit(D, C, key=key, q_init=1.0 / D)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    fts = _pad(x)

    for _ in range(8):  # repeated passes -> steady state, not collapse
        m = m.update(x, y, num_iters=8)

    floor = jnp.sum((fts @ m.Qhat) * fts, -1)   # f^T Qhat f
    bare = jnp.sum((fts @ m.Sigma) * fts, -1)   # f^T Sigma f  (collapses with data)
    epi = floor + bare                          # f^T (Qhat+Sigma) f

    assert (bare >= -1e-4).all()                            # Sigma PSD
    assert float(bare.mean()) < float(floor.mean())         # posterior var collapsed below the floor
    assert float(epi.mean()) >= float(floor.mean()) - 1e-5  # floor preserved
    assert float(epi.mean()) > 0.0


def test_ibprobit_reset_restores_seed():
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 64
    d = D + 1
    m = IBProbit(D, C, key=key, q_init=1.0 / D)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    updated = m.update(x, y, num_iters=8)
    assert updated.posterior_ready is True

    cleared = updated.reset(jr.PRNGKey(3))
    assert cleared.posterior_ready is False
    assert jnp.allclose(cleared.mu, jnp.zeros((d, C)))     # seed mu = 0
    assert jnp.allclose(cleared.Sigma, jnp.eye(d))         # seed Sigma = I


def test_ibprobit_reset_clears_every_recursive_field():
    """There is a single `reset` because two near-identical resets would let a state
    field be added to one and forgotten in the other, and a head carrying
    stale state into a fresh pass still trains and still scores -- it just silently
    starts from the previous pass's posterior. Enumerate the recursive state so a new
    field has to be handled here rather than discovered in a run."""
    D, C, N = 8, 4, 64
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.02)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)

    updated = m.update(x, y, num_iters=8)
    updated = updated.calibrate(x, y, num_iters=4)     # moves beta off its seed
    fresh = updated.reset(jr.PRNGKey(3))
    seed = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.02)

    for field in ("mu", "Sigma", "step", "eps_alpha", "eps_beta",
                  "Psi", "nu", "p_a", "p_b", "beta"):
        assert jnp.allclose(getattr(fresh, field), getattr(seed, field)), (
            f"{field} survived reset(): it is recursive state the reset does not clear"
        )
    assert fresh.posterior_ready is False


def test_ibprobit_loss_type_is_instance_field_not_call_arg():
    """loss_type is fixed at construction; a per-call loss_type is accepted but
    ignored (enforces a single convergence target). Instance field (1, CBM) and
    passed-but-ignored arg (3, mixed) are kept deliberately different supported
    values -- loss_type=0 does not construct, see
    test_loss_type_zero_and_four_are_rejected."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key, q_init=1.0 / D, loss_type=1).update(
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (16, D)); y = jr.randint(jr.PRNGKey(6), (16,), 0, C)

    _, logits_field = m(f, y, with_logits=True)               # uses instance loss_type=1
    _, logits_ignored = m(f, y, with_logits=True, loss_type=3)  # passed 3 is ignored
    assert jnp.allclose(logits_field, logits_ignored)


@pytest.mark.parametrize("bad_loss_type", [0, 4])
def test_loss_type_zero_and_four_are_rejected(bad_loss_type):
    """loss_type must be 1 (CBM), 2 (CBC) or 3 (mixed) -- constructor-time, not just at
    call time. _eps_responsibility reads exp(-loss) as a normalised p_model(y), which
    holds for the softmax losses (1/2/3) but not loss_type=0 (a per-class BCE sum,
    not a normalised distribution over classes): the gate collapses and the head stops
    assimilating. loss_type=4 is simply out of range and would otherwise fall through
    _loss_from_logits' if/elif chain with no else, raising UnboundLocalError instead of
    a clear error."""
    with pytest.raises(ValueError, match="loss_type must be 1"):
        IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6, loss_type=bad_loss_type)


def test_ibprobit_call_with_logits_unchanged_after_refactor():
    """Two separate properties, deliberately not collapsed into one assertion:

    1. the loss_type=2 transform is logcdf(x) - logcdf(-x), reconstructed BY HAND here
       so a wrong sign, swapped operands or a dropped beta would be caught. Deriving
       `expected` from _loss_from_logits (or from _mixture_logp, which calls it) would
       make this f(x) == f(x);
    2. __call__ wires _predictive_logits into _mixture_logp in that order.
    """
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key, q_init=1.0 / D, loss_type=2).update(   # pin loss_type=2
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (16, D)); y = jr.randint(jr.PRNGKey(6), (16,), 0, C)

    # beta defaults to 1.0, which would make the `m.beta *` below a no-op and leave a
    # dropped-beta regression in _loss_from_logits undetectable. Set it off 1 so the
    # scaling is load-bearing in the comparison.
    m = eqx.tree_at(lambda t: t.beta, m, jnp.float32(1.7))
    raw = m._predictive_logits(f)

    # (1) the transform, hand-derived -- independent of _loss_from_logits
    hand = m.beta * (m.logcdf(raw) - m.logcdf(-raw))
    assert jnp.allclose(m._loss_from_logits(raw, jnp.zeros_like(raw))[1], hand, atol=1e-5)

    # (2) the wiring
    loss, logits = m(f, y, with_logits=True)
    assert jnp.allclose(logits, m._mixture_logp(raw), atol=1e-6)
    assert jnp.allclose(loss, -jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0], atol=1e-6)


# --- per-data-point assimilation gate ----------------------------------------

def test_ibprobit_gate_warmup_skips_early_steps():
    """With warmup=2 the first two updates are un-gated (step 0,1 < 2) and equal a
    reference that never leaves warmup; the step counter advances; the third update
    (step 2 >= 2) gates and diverges from the reference."""
    D, C, N = 6, 4, 200
    gated = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, warmup=2)
    ref = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, warmup=1000)   # never gates
    xs = [(_near_separable(D, C, N)) for _ in range(3)]

    for step, (x, y) in enumerate(xs[:2]):
        gated = gated.update(x, y, num_iters=64)
        ref = ref.update(x, y, num_iters=64)
        assert int(gated.step) == step + 1
        assert jnp.allclose(gated.mu, ref.mu, rtol=1e-4, atol=1e-4)   # warmup -> un-gated

    x, y = xs[2]
    gated3 = gated.update(x, y, num_iters=64)                       # step 2 >= warmup -> gate on
    ref3 = ref.update(x, y, num_iters=64)
    assert int(gated3.step) == 3
    assert not jnp.allclose(gated3.mu, ref3.mu, atol=1e-4)


def test_ibprobit_gate_runs_under_scan_jit():
    """The gate carried through lax.scan under filter_jit: the traced step counter, the
    K*sum(g) PX count and the responsibility must all trace. Qhat/step stay finite, and
    the gate really fires after warmup -- a run that never leaves warmup ends elsewhere."""
    key = jr.PRNGKey(0)
    D, C, N, B = 16, 4, 64, 6
    xs = jr.normal(jr.PRNGKey(1), (B, N, D))
    ys = jr.randint(jr.PRNGKey(2), (B, N), 0, C)

    def run_scan(m):
        params, static = eqx.partition(m, eqx.is_array)

        @eqx.filter_jit
        def run(params, xs, ys):
            def body(p, batch):
                x, yb = batch
                loss = eqx.combine(p, static).update(x, yb, num_iters=8)
                return eqx.filter(loss, eqx.is_array), (loss.Qhat, loss.step)
            return jax.lax.scan(body, params, (xs, ys))

        return run(params, xs, ys)

    gated, (qs, steps) = run_scan(
        IBProbit(D, C, key=key, q_init=1e-2, warmup=1, px=True))
    assert jnp.all(jnp.isfinite(qs))
    assert int(steps[-1]) == B            # step advanced once per batch

    never, _ = run_scan(
        IBProbit(D, C, key=key, q_init=1e-2, warmup=B + 1, px=True))   # gate never fires
    assert jnp.isfinite(gated.mu).all()
    assert not jnp.allclose(gated.mu, never.mu, atol=1e-4)


# --- epsilon-contamination mixture (robust likelihood / responsibility gate) ----------


def test_ibprobit_eps_responsibility_shape():
    """r_i = sigmoid(E[log(1-eps)] - E[log eps] - loss_i + log K), eps ~ Beta(a,b), the
    CAVI stationary point (it uses the Beta's expected log, NOT its mean). In (0,1),
    decreasing in the per-point loss; at the uniform point (loss = log K) it reduces to
    sigmoid(E[log(1-eps)] - E[log eps])."""
    from jax.scipy.special import digamma
    D, C = 6, 5
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.1)
    loss_vec = jnp.array([0.0, jnp.log(C), 20.0])            # p = 1, 1/K, ~0
    r = m._eps_responsibility(loss_vec)[:, 0]
    assert jnp.all((r > 0.0) & (r < 1.0))
    assert r[0] > r[1] > r[2]                                # decreasing in loss
    dig_ab = digamma(m.eps_alpha + m.eps_beta)
    E_log_eps = digamma(m.eps_alpha) - dig_ab
    E_log_1meps = digamma(m.eps_beta) - dig_ab
    assert jnp.allclose(r[1], nn.sigmoid(E_log_1meps - E_log_eps), atol=1e-6)  # uniform point


def test_ibprobit_eps_mixture_normalized_and_uniform_seed():
    """The mixture predictive log p_eps sums to 1 over classes; at the mu=0 seed it is the
    uniform 1/K for any eps."""
    D, C, N = 6, 5, 40
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.15)
    log_p = m._mixture_logp(m._predictive_logits(jr.normal(jr.PRNGKey(1), (N, D))))
    assert jnp.allclose(jnp.exp(log_p).sum(-1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.exp(log_p), 1.0 / C, atol=1e-5)


def test_ibprobit_eps_tail_bound():
    """The mixture floors p_eps(y) >= eps/K, so -log p_eps(y) <= log(K/eps) even for a
    confidently-wrong prediction -- the unbounded-NLL tail the floor exists to fix. The
    bound is read off the model's own eps because the conjugate Beta step moves it
    during update(). The bare (un-floored) probit loss -- computed directly via
    _loss_from_logits on the SAME predictive, since __call__ always scores the
    mixture -- exceeds that same bound somewhere on the same
    wrong labels, showing the floor is load-bearing rather than vacuous."""
    D, C, N = 4, 5, 200
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D,
                 eps_init=0.1).update(x, y, num_iters=200)
    y_wrong = (y + 1) % C
    loss, _ = m(x, y_wrong, with_logits=True)
    bound = float(jnp.log(C / m.eps))
    assert jnp.all(loss <= bound + 1e-4)

    raw = m._predictive_logits(x)
    bare_loss = m._loss_from_logits(raw, nn.one_hot(y_wrong, C))[0]
    assert float(bare_loss.max()) > bound


def test_ibprobit_eps_beta_prior_seed():
    """eps_init seeds the conjugate Beta(1, 1/eps_init) prior: the counts are the prior and
    eps starts at its mean 1/(1 + 1/eps_init) before any update."""
    m = IBProbit(6, 4, key=jr.PRNGKey(0), q_init=1e-2, eps_init=0.01)
    assert float(m.eps_alpha) == 1.0 and float(m.eps_beta) == 100.0
    assert jnp.allclose(m.eps, 1.0 / 101.0, atol=1e-6)


def test_ibprobit_eps_beta_conjugate_update():
    """update() moves eps by ONE Beta-conjugate step: the responsibilities r_i (from the
    pre-step eps) are soft counts -- eps_alpha += sum(1-r_i) (contamination mass),
    eps_beta += sum(r_i) (model mass) -- and eps is the posterior mean a/(a+b)."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    m0 = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.01)
    m1 = m0.update(x, y, num_iters=8)                  # nonzero mu -> non-uniform responsibility
    m2 = m1.update(x, y, num_iters=8)

    raw = m1._predictive_logits(x)
    loss_vec = m1._loss_from_logits(raw, nn.one_hot(y, C))[0]
    r = m1._eps_responsibility(loss_vec)[:, 0]
    a_exp = m1.eps_alpha + jnp.sum(1.0 - r)
    b_exp = m1.eps_beta + jnp.sum(r)
    assert jnp.allclose(m2.eps_alpha, a_exp, atol=1e-4)
    assert jnp.allclose(m2.eps_beta, b_exp, atol=1e-4)
    assert jnp.allclose(m2.eps, a_exp / (a_exp + b_exp), atol=1e-6)
    # the two soft counts split the batch: (alpha + beta) grows by exactly N each step
    assert jnp.allclose((m2.eps_alpha + m2.eps_beta) - (m1.eps_alpha + m1.eps_beta),
                        float(N), atol=1e-3)


def test_ibprobit_eps_forgetting_factor_decays_excess():
    """The forgetting factor rho decays the accumulated excess over the prior before adding
    the batch's soft counts: eps_alpha = a0 + rho*(eps_alpha - a0) + sum(1-r_i) (beta
    likewise). Scripts pass rho = 1 - B/N; rho=1 (default) recovers pure accumulation."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    rho = 0.5
    a0, b0 = 1.0, 1.0 / 0.01                           # prior Beta(1, 1/eps_init), eps_init=0.01
    m0 = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.01, rho=rho)
    m1 = m0.update(x, y, num_iters=8)                 # nonzero mu -> non-uniform responsibility
    m2 = m1.update(x, y, num_iters=8)

    raw = m1._predictive_logits(x)
    loss_vec = m1._loss_from_logits(raw, nn.one_hot(y, C))[0]
    r = m1._eps_responsibility(loss_vec)[:, 0]
    a_exp = a0 + rho * (m1.eps_alpha - a0) + jnp.sum(1.0 - r)
    b_exp = b0 + rho * (m1.eps_beta - b0) + jnp.sum(r)
    assert jnp.allclose(m2.eps_alpha, a_exp, atol=1e-4)
    assert jnp.allclose(m2.eps_beta, b_exp, atol=1e-4)


def test_ibprobit_eps_forgetting_bounds_effective_sample_size():
    """rho<1 caps the Beta concentration near the steady state a0+b0+B/(1-rho) instead of
    growing without bound, so eps keeps tracking recent data; rho=1 (default) blows past it."""
    D, C, N = 6, 3, 150
    x, y = _near_separable(D, C, N)
    rho = 0.8
    a0, b0 = 1.0, 1.0 / 0.02                           # eps_init=0.02 -> prior Beta(1, 50)
    cap = a0 + b0 + N / (1.0 - rho)                    # steady-state concentration (batch B = N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.02, rho=rho)
    for _ in range(50):
        m = m.update(x, y, num_iters=8)
    assert float(m.eps_alpha + m.eps_beta) < cap + 1.0        # bounded near steady state
    m_acc = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.02)  # rho=1 default
    for _ in range(50):
        m_acc = m_acc.update(x, y, num_iters=8)
    assert float(m_acc.eps_alpha + m_acc.eps_beta) > cap      # pure accumulation unbounded


def test_ibprobit_eps_reset_restores_prior():
    """reset restores the Beta counts to the prior seed (the contamination estimate
    restarts alongside the wiped posterior)."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.01)
    m1 = m.update(x, y, num_iters=8)
    assert not jnp.allclose(m1.eps_beta, m.eps_beta)          # counts moved
    m2 = m1.reset(jr.PRNGKey(9))
    assert float(m2.eps_alpha) == 1.0 and float(m2.eps_beta) == 100.0
    assert jnp.allclose(m2.mu, 0.0)                           # posterior wiped too


def test_ibprobit_eps_decreases_on_wellfit_data():
    """On near-separable data the head fits well (r_i -> 1), so the accumulated Beta mean
    falls below the prior mean over successive updates."""
    D, C, N = 6, 3, 300
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1.0 / D, eps_init=0.05)
    prior_mean = float(m.eps)
    for _ in range(5):
        m = m.update(x, y, num_iters=16)
    assert float(m.eps) < prior_mean


def test_ibprobit_eps_gated_px_matches_plain():
    """The responsibility-gated update shares the gated-PX fixed point: px (few iters) and
    plain (many iters) give the same (mu, Sigma) with the r_i gate active. Both run from
    m1 (step 1 >= the default warmup 1), so the gate is live in both."""
    m0 = IBProbit(6, 3, key=jr.PRNGKey(0), q_init=1.0 / 6, eps_init=0.1)
    x, y = _near_separable(6, 3, 200)
    m1 = m0.update(x, y, num_iters=8)                  # nonzero mu -> non-uniform responsibility
    plain = m1.update(x, y, num_iters=6000)
    px = m1.update(x, y, num_iters=300, px=True)
    assert jnp.allclose(px.mu, plain.mu, rtol=3e-2, atol=2e-2)
    assert jnp.allclose(px.Sigma, plain.Sigma, rtol=3e-2, atol=2e-2)


def test_ibprobit_eps_runs_under_scan_jit():
    """Qhat + the Beta-conjugate eps carried through lax.scan under filter_jit: both trace,
    eps evolves, mixture predictive / responsibility stay finite, eps in (0,1)."""
    D, C, N, B = 16, 4, 64, 6
    m = IBProbit(D, C, key=jr.PRNGKey(0), q_init=1e-2, eps_init=0.01, px=True)
    xs = jr.normal(jr.PRNGKey(1), (B, N, D)); ys = jr.randint(jr.PRNGKey(2), (B, N), 0, C)
    params, static = eqx.partition(m, eqx.is_array)

    @eqx.filter_jit
    def run(params, xs, ys):
        def body(p, batch):
            x, yb = batch
            loss = eqx.combine(p, static).update(x, yb, num_iters=8)
            return eqx.filter(loss, eqx.is_array), (loss.Qhat, loss.eps)
        return jax.lax.scan(body, params, (xs, ys))

    _, (qs, es) = run(params, xs, ys)
    assert jnp.all(jnp.isfinite(qs)) and jnp.all(jnp.isfinite(es))
    assert jnp.all((es > 0.0) & (es < 1.0))
    assert not jnp.allclose(es[-1], m.eps)


def test_independent_binary_polyagamma_loss():
    key = jr.PRNGKey(0)
    D, C, N = 10, 5, 1_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))

    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)

    key, model_key = jr.split(key)
    ibpg = IBPolyaGamma(input_dim=D, num_classes=C, key=model_key, use_bias=True)
    ibpg = ibpg.update(x, y, num_iters=64)

    W_recovered, b_recovered = ibpg.params

    assert jnp.allclose(W, W_recovered, atol=10.0)  # try to improve atol
    assert jnp.allclose(b, b_recovered, atol=10.0)  # try to imporve atol

    for loss_type in range(4):
        loss, logits = ibpg(x, y, with_logits=True, loss_type=loss_type)
        print(loss_type, loss.mean())

    acc = jnp.mean(logits.argmax(-1) == y)
    print('acc = ', acc)


def test_independent_multinomial_polya_gamma_loss():
    key = jr.PRNGKey(0)
    D, C, N = 10, 5, 1_000

    key, W_key, b_key, x_key, noise_key = jr.split(key, 5)
    W = jr.normal(W_key, (D, C))
    b = jr.normal(b_key, (C,))
    x = jr.normal(x_key, (N, D))

    logits = x @ W + b
    probs = nn.softmax(logits)
    y = jr.categorical(noise_key, logits)

    key, model_key = jr.split(key)
    mnpg = MultinomialPolyaGamma(input_dim=D, num_classes=C, use_bias=True, key=model_key)

    mnpg = mnpg.update(x, y, num_iters=64)

    for loss_type in range(2):
        loss, logits = mnpg(x, y, with_logits=True, loss_type=loss_type)
        print(loss.mean())

        print(jnp.mean(logits.argmax(-1) == y))


def test_ibprobit_predictive_variance_matches_formula_and_is_positive():
    """predictive_variance(f) = 1 + f^T (Qhat + Sigma) f: the shared (across classes)
    moderated-probit denominator, strictly positive, and monotonically increasing under
    feature scaling (its natural use as an epistemic OOD score)."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 200
    key, x_key, y_key, m_key = jr.split(key, 4)
    x = jr.normal(x_key, (N, D))
    y = jr.randint(y_key, (N,), 0, C)

    model = IBProbit(input_dim=D, num_classes=C, key=m_key, use_bias=True, q_init=1.0 / D)
    model = model.update(x, y, num_iters=32)

    var = model.predictive_variance(x)
    assert var.shape == (N,)

    # matches the explicit moderated-probit denominator (with the bias pad). The
    # reference asks for full f32 because predictive_variance pins F32_F32_F32 while a
    # bare @ would fall back to the GPU's TF32 default, a ~1e-4 relative gap.
    fp = _pad(x)
    ref = 1.0 + jnp.sum(jnp.matmul(fp, model.Qhat + model.Sigma, precision="highest") * fp, axis=-1)
    assert jnp.allclose(var, ref, rtol=1e-5, atol=1e-5)

    # strictly positive (the gauge-fixed sigma_z^2 = 1, Qhat PD, Sigma PSD).
    assert jnp.all(var > 0)

    # monotone under scaling: with no bias, var(c f) = 1 + c^2 f^T (Qhat + Sigma) f.
    nb = IBProbit(input_dim=D, num_classes=C, key=m_key, use_bias=False, q_init=1.0 / D)
    nb = nb.update(x, y, num_iters=32)
    assert jnp.all(nb.predictive_variance(2.0 * x) > nb.predictive_variance(x))
