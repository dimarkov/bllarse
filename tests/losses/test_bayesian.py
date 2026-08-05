import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
import equinox as eqx
import optax
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
    bmp = IBProbit(input_dim=D, num_classes=C, key=model_key, use_bias=True)
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
    model = IBProbit(input_dim=10, num_classes=3, key=key, use_bias=True)
    key, reset_key = jr.split(key)
    reset_model = model.reset(reset_key)

    assert reset_model.mu.shape == model.mu.shape
    assert reset_model.Sigma.shape == model.Sigma.shape
    assert jnp.isfinite(reset_model.mu).all()
    assert jnp.isfinite(reset_model.Sigma).all()
    assert reset_model.posterior_ready is False


def test_ibprobit_posterior_precision_uses_process_noise_and_sigmaz():
    """Fresh posterior (Sigma=I seed) -> Lambda_pred = inv((1+q)I) = 1/(1+q) I, and
    the M-step accumulates the data precision scaled by the latent noise variance:
        Lambda_post = 1/(1+q) I + (1/sigma_z^2) X^T X.
    This pins both the process-noise seed and the sigma_z^2 scaling in CAVI."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 200
    d = D + 1
    m = IBProbit(D, C, key=key)
    m = eqx.tree_at(lambda t: (t.log_q, t.log_sigma_z2), m,
                    (jnp.log(jnp.array(0.5)), jnp.log(jnp.array(2.0))))
    x = jr.normal(jr.PRNGKey(1), (N, D))
    y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    fts = _pad(x)

    updated = m.update(x, y, num_iters=8)

    expected = jnp.eye(d) / (1.0 + m.q) + (fts.T @ fts) / m.sigma_z2  # Lambda_post = inv(Sigma_post)
    # Lambda is no longer stored; recover the precision as inv(Sigma). The double
    # inversion adds float32 error, so use a slightly looser absolute slack.
    assert jnp.allclose(jnp.linalg.inv(updated.Sigma), expected, rtol=2e-2, atol=1e-1)


def test_ibprobit_recursion_uses_posterior_as_prior():
    """Iterative Bayesian update: the batch-1 posterior (plus process noise qI) is
    the batch-2 prior, so Lambda_post(2) = inv(Sigma_1 + qI) + (1/sigma_z^2) X2^T X2."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    d = D + 1
    m = IBProbit(D, C, key=key)
    x1 = jr.normal(jr.PRNGKey(1), (N, D)); y1 = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    x2 = jr.normal(jr.PRNGKey(3), (N, D)); y2 = jr.randint(jr.PRNGKey(4), (N,), 0, C)
    fts2 = _pad(x2)

    m1 = m.update(x1, y1, num_iters=8)
    m2 = m1.update(x2, y2, num_iters=8)

    Lambda_pred2 = jnp.linalg.inv(m1.Sigma + m1.q * jnp.eye(d))
    expected = Lambda_pred2 + (fts2.T @ fts2) / m1.sigma_z2       # Lambda_post(2)
    # Lambda is no longer stored; recover the precision as inv(Sigma).
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
    m = IBProbit(6, 3, key=jr.PRNGKey(0))
    x, y = _near_separable(6, 3, 200)

    plain = m.update(x, y, num_iters=6000)           # plain Picard, run to convergence
    px = m.update(x, y, num_iters=300, px=True)       # PX, far fewer iters

    assert jnp.allclose(px.mu, plain.mu, rtol=3e-2, atol=2e-2)
    assert jnp.allclose(px.Sigma, plain.Sigma, rtol=3e-2, atol=2e-2)


def test_ibprobit_px_converges_faster():
    """PX collapses the slow scale mode, so after few iters it is strictly closer to
    the converged solution than the plain Picard scan with the same iter budget."""
    m = IBProbit(6, 3, key=jr.PRNGKey(0))
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
    base = IBProbit(6, 3, key=jr.PRNGKey(0))                  # px field default False
    field_px = IBProbit(6, 3, key=jr.PRNGKey(0), px=True)

    # field px=True with no kwarg == explicit px=True on the default model
    a = field_px.update(x, y, num_iters=300)
    b = base.update(x, y, num_iters=300, px=True)
    assert jnp.allclose(a.mu, b.mu) and jnp.allclose(a.Sigma, b.Sigma)

    # an explicit px kwarg overrides the field
    c = field_px.update(x, y, num_iters=300, px=False)
    d = base.update(x, y, num_iters=300)                      # plain scan
    assert jnp.allclose(c.mu, d.mu) and jnp.allclose(c.Sigma, d.Sigma)


def test_ibprobit_predictive_denominator():
    """Predictive (train grad AND eval) is mu^T f / sqrt(sigma_z^2 + f^T (qI+Sigma) f)."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key, loss_type=0).update(   # loss_type 0 -> raw moderated score (instance field)
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (32, D)); y = jr.randint(jr.PRNGKey(6), (32,), 0, C)
    fts = _pad(f)

    scores = fts @ m.mu
    var = m.q * jnp.sum(fts * fts, -1) + jnp.sum((fts @ m.Sigma) * fts, -1)
    expected = scores / jnp.sqrt(m.sigma_z2 + var)[:, None]

    _, logits = m(f, y, with_logits=True)  # beta=1, instance loss_type=0 -> raw moderated score
    # mm_loss uses the TF32 dot preset, so tolerate ~1e-3 vs a full-precision matmul.
    assert jnp.allclose(logits, expected, rtol=1e-2, atol=1e-2)


def test_ibprobit_process_noise_prevents_collapse():
    """With N >> d and repeated passes, the bare posterior variance f^T Sigma f
    collapses, but the process-noise floor q||f||^2 keeps the predictive variance
    f^T (qI+Sigma) f bounded away from zero -> moderation survives."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 4000
    m = IBProbit(D, C, key=key)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    fts = _pad(x)

    for _ in range(8):  # repeated passes -> steady state, not collapse
        m = m.update(x, y, num_iters=8)

    floor = m.q * jnp.sum(fts * fts, -1)        # q ||f||^2
    bare = jnp.sum((fts @ m.Sigma) * fts, -1)   # f^T Sigma f  (collapses with data)
    epi = floor + bare                          # f^T (qI+Sigma) f

    assert (bare >= -1e-4).all()                            # Sigma PSD
    assert float(bare.mean()) < float(floor.mean())         # posterior var collapsed below the floor
    assert float(epi.mean()) >= float(floor.mean()) - 1e-5  # floor preserved
    assert float(epi.mean()) > 0.0


def test_ibprobit_trainable_params_are_q_and_sigmaz():
    """Only (log_q, log_sigma_z2) are gradient-trained; the CAVI posterior state
    (mu, Lambda, Sigma) and beta stay frozen under an optimizer step."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 128
    m = IBProbit(D, C, key=key)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    m = m.update(x, y, num_iters=8)  # populate posterior so the loss depends on q, sigma_z2

    filt = jax.tree_util.tree_map(lambda _: False, m)
    filt = eqx.tree_at(lambda t: (t.log_q, t.log_sigma_z2), filt, replace=(True, True))
    trainable, static = eqx.partition(m, filt)
    assert trainable.mu is None and trainable.Sigma is None and trainable.beta is None

    opt = optax.adamw(1e-1, weight_decay=0.0)
    opt_state = opt.init(eqx.filter(m, filt))

    @eqx.filter_value_and_grad
    def lf(tr):
        return eqx.combine(tr, static)(x, y, loss_type=3).mean()

    _, grads = lf(trainable)
    updates, opt_state = opt.update(grads, opt_state, trainable)
    trainable = eqx.apply_updates(trainable, updates)
    stepped = eqx.combine(trainable, static)

    moved = not (jnp.allclose(stepped.log_q, m.log_q)
                 and jnp.allclose(stepped.log_sigma_z2, m.log_sigma_z2))
    assert moved
    assert jnp.allclose(stepped.mu, m.mu)        # posterior frozen
    assert jnp.allclose(stepped.Sigma, m.Sigma)
    assert jnp.allclose(stepped.beta, m.beta)


def test_ibprobit_reset_posterior_restores_seed():
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 64
    d = D + 1
    m = IBProbit(D, C, key=key)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    updated = m.update(x, y, num_iters=8)
    assert updated.posterior_ready is True

    cleared = updated.reset_posterior()
    assert cleared.posterior_ready is False
    assert jnp.allclose(cleared.mu, jnp.zeros((d, C)))     # seed mu = 0
    assert jnp.allclose(cleared.Sigma, jnp.eye(d))         # seed Sigma = I


def test_ibprobit_learn_q_optout_leaves_log_q_fixed():
    """learn_q=False (default): update() must not touch log_q -> opt-out is bitwise
    identical for every existing (non-opted-in) call site."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 128
    m = IBProbit(D, C, key=key)  # learn_q defaults False
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    upd = m.update(x, y, num_iters=8)
    assert upd.log_q == m.log_q  # exactly unchanged


def test_ibprobit_learn_q_first_batch_is_noop():
    """learn_q=True, anchor off: on the mu=0 seed batch the predictive logits are 0,
    so the prequential NLL gradient is exactly 0 -> log_q unchanged."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 128
    m = IBProbit(D, C, key=key, learn_q=True, anchor_weight=0.0)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    upd = m.update(x, y, num_iters=8)
    assert upd.log_q == m.log_q  # mu=0 seed -> grad 0 -> bitwise no-op


def test_ibprobit_learn_q_matches_manual_prequential_sgd():
    """The in-update log_q step equals an independent manual prequential SGD step
    (old_log_q - hyper_lr * grad of the pre-update predictive NLL), and the returned
    posterior equals a learn_q=False update at that stepped q -- pins the fold-in to
    the ood_evaluation external reference."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 256
    lr = 0.1
    m = IBProbit(D, C, key=key, learn_q=True, hyper_lr=lr, anchor_weight=0.0)
    x1 = jr.normal(jr.PRNGKey(1), (N, D)); y1 = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    x2 = jr.normal(jr.PRNGKey(3), (N, D)); y2 = jr.randint(jr.PRNGKey(4), (N,), 0, C)

    m1 = m.update(x1, y1, num_iters=8)   # batch1: mu was 0 -> log_q unchanged; mu now nonzero
    assert m1.log_q == m.log_q

    def preq_nll(lq):
        model = eqx.tree_at(lambda t: t.log_q, m1, lq)
        return model(x2, y2).mean()
    g = jax.grad(preq_nll)(m1.log_q)
    expected_log_q = m1.log_q - lr * g

    m2 = m1.update(x2, y2, num_iters=8)
    assert jnp.allclose(m2.log_q, expected_log_q, rtol=1e-5, atol=1e-6)

    # posterior must equal a learn_q=False update at the stepped q
    ref = IBProbit(D, C, key=key, learn_q=False)
    ref = eqx.tree_at(
        lambda t: (t.mu, t.Sigma, t.log_q, t.log_sigma_z2, t.posterior_ready),
        ref, (m1.mu, m1.Sigma, expected_log_q, m1.log_sigma_z2, m1.posterior_ready))
    ref = ref.update(x2, y2, num_iters=8)
    assert jnp.allclose(m2.mu, ref.mu, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(m2.Sigma, ref.Sigma, rtol=1e-4, atol=1e-4)


def test_ibprobit_learn_q_anchor_pulls_toward_mode():
    """With mu=0 (NLL grad 0) the anchor alone acts: q>m -> log_q decreases toward m,
    by exactly hyper_lr * lambda * (1 - m/q)."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 128
    lr, lam, m_mode = 0.1, 1.0, 0.01
    m = IBProbit(D, C, key=key, q_init=0.1, learn_q=True, hyper_lr=lr,
                 anchor_weight=lam, anchor_mode=m_mode)   # q=0.1 > mode=0.01
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    upd = m.update(x, y, num_iters=8)   # mu=0 seed -> only the anchor moves log_q

    q0 = float(m.q)
    expected = float(m.log_q) - lr * lam * (1.0 - m_mode / q0)
    assert float(upd.log_q) < float(m.log_q)             # pulled down toward the mode
    assert jnp.allclose(upd.log_q, expected, rtol=1e-5, atol=1e-6)


def test_ibprobit_loss_type_is_instance_field_not_call_arg():
    """loss_type is fixed at construction; a per-call loss_type is accepted but
    ignored (enforces a single convergence target)."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key, loss_type=0).update(
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (16, D)); y = jr.randint(jr.PRNGKey(6), (16,), 0, C)

    _, logits_field = m(f, y, with_logits=True)               # uses instance loss_type=0
    _, logits_ignored = m(f, y, with_logits=True, loss_type=3)  # passed 3 is ignored
    assert jnp.allclose(logits_field, logits_ignored)


def test_ibprobit_learn_q_runs_under_scan_jit():
    """The in-update log_q step must work inside the finetuning path: loss params
    carried through lax.scan under filter_jit (mirrors utils.batch_update). q should
    evolve across batches and stay finite -- guards grad-inside-scan-inside-jit."""
    key = jr.PRNGKey(0)
    D, C, N, B = 16, 4, 64, 6
    m = IBProbit(D, C, key=key, learn_q=True, hyper_lr=0.1, anchor_weight=1e-3)
    xs = jr.normal(jr.PRNGKey(1), (B, N, D))
    ys = jr.randint(jr.PRNGKey(2), (B, N), 0, C)

    params, static = eqx.partition(m, eqx.is_array)

    @eqx.filter_jit
    def run(params, xs, ys):
        def body(p, batch):
            x, yb = batch
            loss = eqx.combine(p, static).update(x, yb, num_iters=8)
            return eqx.filter(loss, eqx.is_array), loss.q
        return jax.lax.scan(body, params, (xs, ys))

    _, qs = run(params, xs, ys)
    assert jnp.all(jnp.isfinite(qs))
    assert not jnp.allclose(qs[-1], m.q)   # q evolved across batches (mu nonzero after batch 0)


# --- BMR per-data-point gate ------------------------------------------------

def test_ibprobit_bmr_gate_off_is_identity():
    """bmr_gate=False (default) is bitwise-identical to the un-gated update, for both
    the plain and PX scans -> opt-out leaves every existing call site untouched."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 200
    m = IBProbit(D, C, key=key)
    x1 = jr.normal(jr.PRNGKey(1), (N, D)); y1 = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    m1 = m.update(x1, y1, num_iters=8)   # mu now nonzero so a leak would show
    x2 = jr.normal(jr.PRNGKey(3), (N, D)); y2 = jr.randint(jr.PRNGKey(4), (N,), 0, C)

    a = m1.update(x2, y2, num_iters=16)
    b = m1.update(x2, y2, num_iters=16, bmr_gate=False)
    assert jnp.array_equal(a.mu, b.mu) and jnp.array_equal(a.Sigma, b.Sigma)

    a_px = m1.update(x2, y2, num_iters=64, px=True)
    b_px = m1.update(x2, y2, num_iters=64, px=True, bmr_gate=False)
    assert jnp.array_equal(a_px.mu, b_px.mu) and jnp.array_equal(a_px.Sigma, b_px.Sigma)


def test_ibprobit_call_with_logits_unchanged_after_refactor():
    """Factoring __call__ into _predictive_logits + _loss_from_logits must leave the
    public with_logits output identical: for loss_type=2 it is logcdf(raw)-logcdf(-raw),
    raw being the moderated score from _predictive_logits."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 128
    m = IBProbit(D, C, key=key).update(   # default loss_type=2
        jr.normal(jr.PRNGKey(1), (N, D)), jr.randint(jr.PRNGKey(2), (N,), 0, C), num_iters=8)
    f = jr.normal(jr.PRNGKey(5), (16, D)); y = jr.randint(jr.PRNGKey(6), (16,), 0, C)

    _, logits_t = m(f, y, with_logits=True)
    raw = m._predictive_logits(f)
    expected = m.logcdf(raw) - m.logcdf(-raw)   # loss_type 2 transform, beta = 1
    assert jnp.allclose(logits_t, expected, rtol=1e-5, atol=1e-5)


def test_ibprobit_bmr_gate_range_and_discriminates():
    """g_i = Phi(mu~_i)/(Phi(mu~_i)+1/2) in (0, 2/3); a wrong (flipped) label scores the
    true class low -> lower gate than the correct label."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 5, 256
    m = IBProbit(D, C, key=key)
    x, y = _near_separable(D, C, N)
    m1 = m.update(x, y, num_iters=200)         # fit so mu != 0

    raw = m1._predictive_logits(x)
    g_true = m1._gate_from_logits(raw, y)
    g_wrong = m1._gate_from_logits(raw, (y + 1) % C)

    assert (g_true > 0.0).all() and (g_true < 2.0 / 3.0 + 1e-6).all()
    assert float(g_wrong.mean()) < float(g_true.mean())


def test_ibprobit_bmr_gate_effective_sample_size():
    """Effective N = sum(g) is strictly positive and strictly below N (each g < 2/3)."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 4, 200
    m = IBProbit(D, C, key=key)
    x, y = _near_separable(D, C, N)
    m1 = m.update(x, y, num_iters=100)
    g = m1._gate_from_logits(m1._predictive_logits(x), y)
    assert 0.0 < float(g.sum()) < float(N)


def test_ibprobit_bmr_seed_gate_is_half():
    """On the mu=0 seed the moderated score is 0 -> p1=Phi(0)=1/2 -> g = 0.5 everywhere."""
    key = jr.PRNGKey(0)
    D, C, N = 6, 3, 64
    m = IBProbit(D, C, key=key)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    g = m._gate_from_logits(m._predictive_logits(x), y)
    assert jnp.allclose(g, 0.5, atol=1e-5)


def test_ibprobit_bmr_gated_px_matches_gated_plain():
    """Gated PX must converge to the SAME posterior as the gated plain scan (the gate
    g-weights both moments and the PX scale-quadratic carries the same weights, so a=1
    at the gated fixed point) -- the load-bearing PX-gating correctness check."""
    m0 = IBProbit(6, 3, key=jr.PRNGKey(0), bmr_gate=True, bmr_warmup=0)
    x, y = _near_separable(6, 3, 200)
    m1 = m0.update(x, y, num_iters=8)            # step=1, mu != 0 -> gate is non-uniform

    plain = m1.update(x, y, num_iters=6000)               # gated plain, to convergence
    px = m1.update(x, y, num_iters=300, px=True)           # gated PX, few iters

    assert jnp.allclose(px.mu, plain.mu, rtol=3e-2, atol=2e-2)
    assert jnp.allclose(px.Sigma, plain.Sigma, rtol=3e-2, atol=2e-2)


def test_ibprobit_bmr_gate_instance_field_drives_update():
    """bmr_gate set at construction drives update() with no kwarg; an explicit kwarg
    overrides the field (matches the px field/kwarg convention)."""
    x, y = _near_separable(6, 3, 200)
    base = IBProbit(6, 3, key=jr.PRNGKey(0))                       # bmr_gate False
    field_on = IBProbit(6, 3, key=jr.PRNGKey(0), bmr_gate=True)

    a = field_on.update(x, y, num_iters=64)                        # field True, no kwarg
    b = base.update(x, y, num_iters=64, bmr_gate=True)             # explicit True
    assert jnp.allclose(a.mu, b.mu) and jnp.allclose(a.Sigma, b.Sigma)

    c = field_on.update(x, y, num_iters=64, bmr_gate=False)        # kwarg overrides field
    d = base.update(x, y, num_iters=64)                           # un-gated
    assert jnp.array_equal(c.mu, d.mu) and jnp.array_equal(c.Sigma, d.Sigma)


def test_ibprobit_bmr_warmup_skips_early_steps():
    """With bmr_warmup=2 the first two updates are un-gated (step 0,1 < 2) and equal the
    un-gated reference; the step counter advances; the third update (step 2 >= 2) gates
    and diverges from the reference."""
    D, C, N = 6, 4, 200
    gated = IBProbit(D, C, key=jr.PRNGKey(0), bmr_gate=True, bmr_warmup=2)
    ref = IBProbit(D, C, key=jr.PRNGKey(0))                        # un-gated
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


def test_ibprobit_bmr_runs_under_scan_jit():
    """bmr_gate (+ learn_q) carried through lax.scan under filter_jit: the traced step
    counter, the K*sum(g) PX count, the true-class gather and the has_aux logit reuse
    must all trace; q/step evolve and stay finite, gate activates after warmup."""
    key = jr.PRNGKey(0)
    D, C, N, B = 16, 4, 64, 6
    m = IBProbit(D, C, key=key, bmr_gate=True, bmr_warmup=1,
                 learn_q=True, hyper_lr=0.1, anchor_weight=1e-3, px=True)
    xs = jr.normal(jr.PRNGKey(1), (B, N, D))
    ys = jr.randint(jr.PRNGKey(2), (B, N), 0, C)
    params, static = eqx.partition(m, eqx.is_array)

    @eqx.filter_jit
    def run(params, xs, ys):
        def body(p, batch):
            x, yb = batch
            loss = eqx.combine(p, static).update(x, yb, num_iters=8)
            return eqx.filter(loss, eqx.is_array), (loss.q, loss.step)
        return jax.lax.scan(body, params, (xs, ys))

    _, (qs, steps) = run(params, xs, ys)
    assert jnp.all(jnp.isfinite(qs))
    assert int(steps[-1]) == B            # step advanced once per batch
    assert not jnp.allclose(qs[-1], m.q)  # q still evolves with the gate on


# --- epsilon-contamination mixture (robust likelihood / learned BMR gate) -------------

def test_ibprobit_eps_off_call_matches_plain():
    """eps off (default) -> __call__(with_logits) returns the plain loss_type-2 CBC logits
    (bitwise opt-out of the mixture predictive); use_eps is False."""
    D, C, N = 6, 4, 64
    m = IBProbit(D, C, key=jr.PRNGKey(0)).update(*_near_separable(D, C, N), num_iters=16)
    x, y = _near_separable(D, C, N)
    _, logits = m(x, y, with_logits=True)
    raw = m._predictive_logits(x)
    expected = m.beta * (m.logcdf(raw) - m.logcdf(-raw))
    assert jnp.array_equal(logits, expected)
    assert m.use_eps is False


def test_ibprobit_eps_responsibility_shape():
    """r_i = (1-eps) p / ((1-eps) p + eps/K) in (0,1), decreasing in the per-point loss,
    and equal to 1-eps at the uniform point p = 1/K."""
    D, C = 6, 5
    m = IBProbit(D, C, key=jr.PRNGKey(0), eps_init=0.1)
    eps = float(m.eps)
    loss_vec = jnp.array([0.0, jnp.log(C), 20.0])            # p = 1, 1/K, ~0
    r = m._eps_responsibility(loss_vec, m.eps)[:, 0]
    assert jnp.all((r > 0.0) & (r < 1.0))
    assert r[0] > r[1] > r[2]                                # decreasing in loss
    assert jnp.allclose(r[1], 1.0 - eps, atol=1e-5)          # uniform -> 1-eps


def test_ibprobit_eps_seed_gate_is_one_minus_eps():
    """On the mu=0 seed every predictive is uniform -> responsibility = 1-eps everywhere."""
    D, C, N = 6, 4, 50
    eps0 = 0.2
    m = IBProbit(D, C, key=jr.PRNGKey(0), eps_init=eps0)
    x = jr.normal(jr.PRNGKey(1), (N, D)); y = jr.randint(jr.PRNGKey(2), (N,), 0, C)
    loss_vec = m._loss_from_logits(m._predictive_logits(x), nn.one_hot(y, C))[0]
    r = m._eps_responsibility(loss_vec, m.eps)
    assert jnp.allclose(r, 1.0 - eps0, atol=1e-5)


def test_ibprobit_eps_mixture_normalized_and_uniform_seed():
    """The mixture predictive log p_eps sums to 1 over classes; at the mu=0 seed it is the
    uniform 1/K for any eps."""
    D, C, N = 6, 5, 40
    m = IBProbit(D, C, key=jr.PRNGKey(0), eps_init=0.15)
    log_p = m._mixture_logp(m._predictive_logits(jr.normal(jr.PRNGKey(1), (N, D))))
    assert jnp.allclose(jnp.exp(log_p).sum(-1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.exp(log_p), 1.0 / C, atol=1e-5)


def test_ibprobit_eps_tail_bound():
    """The mixture floors p_eps(y) >= eps/K, so -log p_eps(y) <= log(K/eps) even for a
    confidently-wrong prediction -- the unbounded-NLL tail the floor exists to fix. The
    un-floored probit loss exceeds the bound somewhere on the same wrong labels."""
    D, C, N = 4, 5, 200
    eps0 = 0.1
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), eps_init=eps0).update(x, y, num_iters=200)
    y_wrong = (y + 1) % C
    loss, _ = m(x, y_wrong, with_logits=True)
    bound = float(jnp.log(C / eps0))
    assert jnp.all(loss <= bound + 1e-4)
    m_plain = IBProbit(D, C, key=jr.PRNGKey(0)).update(x, y, num_iters=200)
    assert float(m_plain(x, y_wrong).max()) > bound


def test_ibprobit_eps_beta_prior_seed():
    """learn_eps seeds the conjugate Beta(1, 1/eps_init) prior: the counts are the prior and
    eps starts at its mean 1/(1 + 1/eps_init) before any update."""
    m = IBProbit(6, 4, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.01)
    assert float(m.eps_alpha) == 1.0 and float(m.eps_beta) == 100.0
    assert jnp.allclose(m.eps, 1.0 / 101.0, atol=1e-6)
    assert m.use_eps is True


def test_ibprobit_eps_beta_conjugate_update():
    """learn_eps updates eps by ONE Beta-conjugate step per update(): the responsibilities
    r_i (from the pre-step eps) are soft counts -- eps_alpha += sum(1-r_i) (contamination
    mass), eps_beta += sum(r_i) (model mass) -- and eps is the posterior mean a/(a+b)."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    m0 = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.01)
    m1 = m0.update(x, y, num_iters=8)                  # nonzero mu -> non-uniform responsibility
    m2 = m1.update(x, y, num_iters=8)

    raw = m1._predictive_logits(x)
    loss_vec = m1._loss_from_logits(raw, nn.one_hot(y, C))[0]
    r = m1._eps_responsibility(loss_vec, m1.eps)[:, 0]
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
    m0 = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.01, rho=rho)
    m1 = m0.update(x, y, num_iters=8)                 # nonzero mu -> non-uniform responsibility
    m2 = m1.update(x, y, num_iters=8)

    raw = m1._predictive_logits(x)
    loss_vec = m1._loss_from_logits(raw, nn.one_hot(y, C))[0]
    r = m1._eps_responsibility(loss_vec, m1.eps)[:, 0]
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
    m = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.02, rho=rho)
    for _ in range(50):
        m = m.update(x, y, num_iters=8)
    assert float(m.eps_alpha + m.eps_beta) < cap + 1.0        # bounded near steady state
    m_acc = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.02)  # rho=1 default
    for _ in range(50):
        m_acc = m_acc.update(x, y, num_iters=8)
    assert float(m_acc.eps_alpha + m_acc.eps_beta) > cap      # pure accumulation unbounded


def test_ibprobit_eps_learn_without_bmr_gate_leaves_assimilation_ungated():
    """eps is always estimated when learn_eps, but r_i gates the CAVI moments ONLY when
    bmr_gate is on: with bmr_gate off the posterior equals the plain ungated update while
    the Beta counts still move."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.01)
    m1 = m.update(x, y, num_iters=8)
    plain = IBProbit(D, C, key=jr.PRNGKey(0)).update(x, y, num_iters=8)
    assert jnp.allclose(m1.mu, plain.mu, atol=1e-6)           # assimilation ungated
    assert jnp.allclose(m1.Sigma, plain.Sigma, atol=1e-6)
    assert not jnp.allclose(m1.eps_beta, m.eps_beta)          # but eps was estimated


def test_ibprobit_eps_bmr_gate_toggles_responsibility_gate():
    """With the mixture active, the bmr_gate override gates the CAVI moments by r_i, so the
    gated posterior differs from the ungated one taken from the same state."""
    D, C, N = 6, 3, 200
    x, y = _near_separable(D, C, N)
    seed = IBProbit(D, C, key=jr.PRNGKey(0), eps_init=0.2).update(x, y, num_iters=8)  # nonzero mu
    gated = seed.update(x, y, num_iters=8, bmr_gate=True)
    ungated = seed.update(x, y, num_iters=8, bmr_gate=False)
    assert not jnp.allclose(gated.mu, ungated.mu, atol=1e-4)


def test_ibprobit_eps_reset_restores_prior():
    """reset_posterior restores the Beta counts to the prior seed (the contamination estimate
    restarts alongside the wiped posterior)."""
    D, C, N = 6, 4, 200
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.01)
    m1 = m.update(x, y, num_iters=8)
    assert not jnp.allclose(m1.eps_beta, m.eps_beta)          # counts moved
    m2 = m1.reset_posterior()
    assert float(m2.eps_alpha) == 1.0 and float(m2.eps_beta) == 100.0
    assert jnp.allclose(m2.mu, 0.0)                           # posterior wiped too


def test_ibprobit_eps_decreases_on_wellfit_data():
    """On near-separable data the head fits well (r_i -> 1), so the accumulated Beta mean
    falls below the prior mean over successive updates."""
    D, C, N = 6, 3, 300
    x, y = _near_separable(D, C, N)
    m = IBProbit(D, C, key=jr.PRNGKey(0), learn_eps=True, eps_init=0.05)
    prior_mean = float(m.eps)
    for _ in range(5):
        m = m.update(x, y, num_iters=16)
    assert float(m.eps) < prior_mean


def test_ibprobit_eps_gated_px_matches_plain():
    """The responsibility-gated update (bmr_gate + mixture) shares the gated-PX fixed point:
    px (few iters) and plain (many iters) give the same (mu, Sigma) with the r_i gate active."""
    m0 = IBProbit(6, 3, key=jr.PRNGKey(0), eps_init=0.1, bmr_gate=True)
    x, y = _near_separable(6, 3, 200)
    m1 = m0.update(x, y, num_iters=8)                  # nonzero mu -> non-uniform responsibility
    plain = m1.update(x, y, num_iters=6000)
    px = m1.update(x, y, num_iters=300, px=True)
    assert jnp.allclose(px.mu, plain.mu, rtol=3e-2, atol=2e-2)
    assert jnp.allclose(px.Sigma, plain.Sigma, rtol=3e-2, atol=2e-2)


def test_ibprobit_eps_runs_under_scan_jit():
    """learn_q + learn_eps carried through lax.scan under filter_jit: q and the Beta-conjugate
    eps both trace and evolve, mixture predictive / responsibility stay finite, eps in (0,1)."""
    D, C, N, B = 16, 4, 64, 6
    m = IBProbit(D, C, key=jr.PRNGKey(0), learn_q=True, learn_eps=True,
                 eps_init=0.01, hyper_lr=0.1, anchor_weight=1e-3, px=True)
    xs = jr.normal(jr.PRNGKey(1), (B, N, D)); ys = jr.randint(jr.PRNGKey(2), (B, N), 0, C)
    params, static = eqx.partition(m, eqx.is_array)

    @eqx.filter_jit
    def run(params, xs, ys):
        def body(p, batch):
            x, yb = batch
            loss = eqx.combine(p, static).update(x, yb, num_iters=8)
            return eqx.filter(loss, eqx.is_array), (loss.q, loss.eps)
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
    """predictive_variance(f) = sigma_z^2 + f^T (q I + Sigma) f: the shared (across
    classes) moderated-probit denominator, strictly positive, and monotonically
    increasing under feature scaling (its natural use as an epistemic OOD score)."""
    key = jr.PRNGKey(0)
    D, C, N = 8, 4, 200
    key, x_key, y_key, m_key = jr.split(key, 4)
    x = jr.normal(x_key, (N, D))
    y = jr.randint(y_key, (N,), 0, C)

    model = IBProbit(input_dim=D, num_classes=C, key=m_key, use_bias=True)
    model = model.update(x, y, num_iters=32)

    var = model.predictive_variance(x)
    assert var.shape == (N,)

    # matches the explicit moderated-probit denominator (with the bias pad).
    fp = _pad(x)
    ref = model.sigma_z2 + jnp.sum((model.q * fp + fp @ model.Sigma) * fp, axis=-1)
    assert jnp.allclose(var, ref, rtol=1e-5, atol=1e-5)

    # strictly positive (sigma_z^2 > 0, q > 0, Sigma PSD).
    assert jnp.all(var > 0)

    # monotone under scaling: with no bias, var(c f) = sigma_z^2 + c^2 (q||f||^2 + f^T Sigma f).
    nb = IBProbit(input_dim=D, num_classes=C, key=m_key, use_bias=False)
    nb = nb.update(x, y, num_iters=32)
    assert jnp.all(nb.predictive_variance(2.0 * x) > nb.predictive_variance(x))
