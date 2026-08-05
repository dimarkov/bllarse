# IBProbit Merge and Wishart-q Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the process-noise `IBProbit` onto `infra/results-protocol` (unbreaking three scripts that already call its API), then replace the scalar SGD-trained `q` with a conjugate Inverse-Wishart posterior on the full process-noise matrix, as specified in `docs/superpowers/specs/2026-08-05-ibprobit-merge-and-wishart-q-design.md`.

**Architecture:** `src/bllarse/losses/bayesian.py` holds a recursive-Bayesian last layer. Its posterior over head weights `q(W) = N(mu, Sigma)` is carried across minibatches; each batch runs PREDICT (`Sigma_pred = Sigma + Qhat`) then a CAVI UPDATE. Tasks 1-3 port that machinery from the `ibprobit-process-noise` branch. Tasks 4-9 replace the gradient-trained scalar `q` with a conjugate Inverse-Wishart posterior on `Q` whose forgetting rate is inferred from a closed-form Bayes Factor Surprise, deleting every gradient step from the head.

**Tech Stack:** JAX, Equinox (`eqx.Module` with `eqx.field(static=True)` for non-array config), `jax.lax.scan` for CAVI inner loops, pytest.

## Global Constraints

- **GPU required.** Tests use `DotAlgorithmPreset('TF32_TF32_F32')`, unsupported on CPU. Every pytest invocation in this plan must be prefixed with `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false`. GPUs 0 and 1 belong to other jobs; never use them.
- **Baseline to preserve:** 283 passed, 2 skipped on the full suite at branch point.
- Run everything from the worktree root `/home/dima/projects/bllarse/.claude/worktrees/ibprobit-merge`. Do not `cd` to the main checkout.
- Python is driven through `uv run`. There is no `pytest-timeout` plugin — do not pass `--timeout`.
- `IBProbit` state arrays are `float32` throughout. Symmetrize every `d x d` matrix that mathematics says is symmetric (`0.5 * (M + M.T)`) before a Cholesky.
- Reuse the module-level `mm_update = partial(mm, preset='F32_F32_F32', dot_dtype=jnp.float32)` for all posterior algebra and `mm_loss` (TF32) only for predictive logits. Do not call `jnp.matmul` directly in new code.
- Everything must survive `jax.jit` and `lax.scan` over batches: no Python branching on traced values, use `jnp.where`. Python branching on *static* config (`self.learn_q`, `self.px`) is fine and intended.
- Notation in comments and docstrings: `'` transpose, `*` matmul, `.*` elementwise. Keep the existing house style of long explanatory docstrings on the mathematical methods.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/bllarse/losses/bayesian.py` | `IBProbit` (the only class that changes; `MultinomialPolyaGamma` and `IBPolyaGamma` are byte-identical across branches and must not be touched) | 1-8 |
| `tests/losses/test_bayesian.py` | Ported process-noise suite, then the Wishart-`q` suite | 1, 4-8 |
| `tests/losses/test_cavi_dual.py` | Dual-vs-primal CAVI equivalence and dispatch | 2 |
| `tests/losses/test_wishart_q.py` | New: the `q` recursion in isolation (`S_t`, surprise, IW update) | 5-7 |
| `scripts/alternatives/ood_evaluation.py` | CLI + construction | 3, 9 |
| `scripts/alternatives/full_finetune_comparison.py` | CLI + construction | 3, 9 |
| `scripts/alternatives/vit_classification.py` | CLI + construction | 3, 9 |

`IBProbit` reaches roughly 700 lines after Task 8. That is large but it is one coherent object — the recursion, its conjugate hyper-updates, and the predictive that feeds them all read the same state. Splitting it across files would put `_q_recursion` out of sight of the `update()` that sequences it. Leave it whole.

---

## Commit 1 — port the process-noise head

### Task 1: Port `IBProbit` and its test suite

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` (whole-file replacement)
- Modify: `tests/losses/test_bayesian.py` (whole-file replacement)

**Interfaces:**
- Consumes: nothing.
- Produces: `IBProbit(input_dim, num_classes, *, key, use_bias=True, use_approx_cdf=True, q_init=None, sigma_z2_init=1.0, learn_q=False, hyper_lr=0.1, anchor_weight=1e-4, anchor_mode=None, loss_type=2, px=False, bmr_gate=False, bmr_warmup=0, learn_eps=False, eps_init=0.0, rho=1.0)`; state fields `mu (d,K)`, `Sigma (d,d)`, `step ()`, `eps_alpha ()`, `eps_beta ()`, `log_q ()`, `log_sigma_z2 ()`, `beta ()`; methods `update(features, y, *, num_iters=32, px=None, bmr_gate=None)`, `reset_posterior()`, `reset(key)`, `calibrate(features, y, *, num_iters=8, loss_type=2)`, `predictive_variance(features)`, `params` property, `__call__(features, y, *, with_logits=False, loss_type=None)`.

The two branches' `MultinomialPolyaGamma` and `IBPolyaGamma` were verified byte-identical (`diff` of `bayesian.py:211-417` on this branch against `579-785` on the other returns nothing), so a whole-file copy is safe and is the smallest correct change. The process-noise version already has no `eqx.nn.LayerNorm` (removed in `e79fc20`) and no `alpha` kwarg on `update()` (replaced by process noise), so §3.2 and §3.3 of the spec need no separate edit. Nothing in the repo passes `alpha=` to `update()` — verified by grep — so dropping it breaks no caller.

- [ ] **Step 1: Replace the module and the test file from the other branch**

```bash
git show ibprobit-process-noise:src/bllarse/losses/bayesian.py > src/bllarse/losses/bayesian.py
git show ibprobit-process-noise:tests/losses/test_bayesian.py > tests/losses/test_bayesian.py
```

- [ ] **Step 2: Run the ported suite**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_bayesian.py`
Expected: PASS, 44 tests. If any fail, they are genuine port breakage — fix `bayesian.py`, not the test.

- [ ] **Step 3: Confirm the dual-CAVI suite is now the only breakage**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/`
Expected: `test_bayesian.py` passes; `test_cavi_dual.py` fails — it references `IBProbit._prefers_dual` and `_expected_z`, which the ported file does not have. This is expected and is Task 2's job. Do not fix it here.

- [ ] **Step 4: Confirm the rest of the suite is unaffected**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q --ignore=tests/losses/test_cavi_dual.py`
Expected: PASS. Compare the count against the 283-passed baseline; the delta should be exactly the 40 new `test_bayesian.py` tests minus the 6 ignored dual tests.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/test_bayesian.py
git commit -m "feat(ibprobit): port the process-noise head onto the protocol branch

Brings Sigma-carrying recursive state, process noise q I, the PX-CAVI scan,
the epsilon-contamination mixture with conjugate Beta updating, and the
per-point BMR gate across from ibprobit-process-noise. The three scripts
consolidated by 726bc78 already call this API and were dead without it.

The LayerNorm and alpha-forgetting do not come across: e79fc20 replaced the
former with the probit marginalization, and process noise subsumes the latter.

tests/losses/test_cavi_dual.py is knowingly left failing; the next commit
restores the dual form it covers."
```

---

### Task 2: Restore the dual CAVI form

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — add `_prefers_dual` and `_expected_z` static methods, rewrite `_cavi_scan`
- Modify: `tests/losses/test_cavi_dual.py:137-149` — recompute the end-to-end pin

**Interfaces:**
- Consumes: `IBProbit` from Task 1.
- Produces: `IBProbit._prefers_dual(N, D, K, num_iters) -> bool` (staticmethod), `IBProbit._expected_z(pred, y_one_hot) -> Array` (staticmethod), and `_cavi_scan(fts, x, y_one_hot, eta_init, eta_anchor, num_iters, sigma_z, g=None)` with an added dual branch. Signature is unchanged, so `update()` needs no edit.

`3f5e5c2` added a second algebraically identical CAVI form that carries `pred (N,K)` against a precomputed hat matrix instead of `eta (d,K)`; it wins on narrow batches with wide features. The port dropped it. The derivation survives the process-noise anchor unchanged: the carry is `eta_anchor + sigma_z^-2 * fts'(g .* E[z])`, so `x*eta = c + sigma_z^-2 * S * (g .* E[z])` with `c = x*eta_anchor` and `S = x*fts'`. Both `x` and `S` are built from *ungated* `fts`, matching the primal — the gate enters only through the contribution, and `Lambda_post` (which defines `x`) already uses gated `fts` on one side only.

The dual does not compose with PX, which needs `eta_hat` in d-space for `B = sum(p .* eta_hat)`. So `_prefers_dual` is consulted only inside `_cavi_scan` and `_cavi_scan_px` is left alone.

- [ ] **Step 1: Write the failing test**

Append to `tests/losses/test_cavi_dual.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_cavi_dual.py::test_dual_matches_primal_under_process_noise_and_gate`
Expected: FAIL with `AttributeError: type object 'IBProbit' has no attribute '_prefers_dual'`.

- [ ] **Step 3: Add the two static methods**

Insert into `IBProbit`, immediately above `_cavi_scan`:

```python
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
    def _expected_z(alpha: Array, y_one_hot: Array) -> Array:
        """Standardized truncated-latent mean: E_q[z]/sigma_z given alpha = m/sigma_z,
        via log-space Mills ratios. Caller rescales by sigma_z."""
        log_pdf = norm.logpdf(alpha)
        mills_pos = jnp.exp(log_pdf - norm.logcdf(alpha))
        mills_neg = jnp.exp(log_pdf - norm.logcdf(-alpha))
        return alpha + jnp.where(y_one_hot, mills_pos, -mills_neg)
```

- [ ] **Step 4: Rewrite `_cavi_scan` with the dual branch**

Replace the whole `_cavi_scan` body (keep the signature and extend the docstring):

```python
    def _cavi_scan(self, fts: Array, x: Array, y_one_hot: Array, eta_init: Array, eta_anchor: Array, num_iters: int, sigma_z: Array, g: Optional[Array] = None) -> Array:
        """CAVI fixed-point scan for the posterior mean. Returns the natural
        parameter eta = eta_anchor + (1/sigma_z^2) fts' (g .* E_q[z]), where E_q[z] is
        the sign-truncated latent mean E[z] = m + sigma_z * Mills(m / sigma_z),
        m = mu' f. ``x = fts * Lambda_post^-1`` so ``x * eta`` recovers m. ``g`` is an
        optional per-point (n,1) weight (None = unweighted).

        Two algebraically identical forms, picked by whichever is cheaper at these
        shapes. The primal carries eta (d,K) at 4NdK per iteration. Since the carry is
        always eta_anchor + (1/sigma_z^2) fts'(g .* E[z]), the prediction it feeds back
        satisfies

            x * eta = x * eta_anchor + (1/sigma_z^2)(x * fts')(g .* E[z]) = c + S(g .* E[z])/sigma_z^2,

        so the dual carries pred (N,K) against the precomputed hat matrix S = x * fts'
        at 2N^2K per iteration. Both x and S are built from UNGATED fts, matching the
        primal -- the gate enters only through the contribution. The dual wins on wide
        features and modest batches (roughly N < 2D) and loses on large batches, where
        the N x N matrix is both slower and far bigger than eta. PX (``_cavi_scan_px``)
        has no dual form: it needs eta_hat in d-space for its scalar quadratic.
        """
        inv_sz2 = 1.0 / (sigma_z * sigma_z)

        if self._prefers_dual(x.shape[0], x.shape[1], y_one_hot.shape[-1], num_iters):
            S = mm_update(x, fts.T)
            c = mm_update(x, eta_anchor)

            def step_fn(pred, *args):
                E_q_z = sigma_z * self._expected_z(pred / sigma_z, y_one_hot)
                contrib = E_q_z if g is None else g * E_q_z
                return c + inv_sz2 * mm_update(S, contrib), None

            # one prediction is consumed by the final projection below, so the
            # scan is one iteration shorter than the primal loop
            pred, _ = lax.scan(step_fn, mm_update(x, eta_init), jnp.arange(num_iters - 1))
            E_q_z = sigma_z * self._expected_z(pred / sigma_z, y_one_hot)
            contrib = E_q_z if g is None else g * E_q_z
            return eta_anchor + inv_sz2 * mm_update(fts.T, contrib)

        def step_fn(eta_carry, *args):
            pred = mm_update(x, eta_carry)
            E_q_z = sigma_z * self._expected_z(pred / sigma_z, y_one_hot)
            contrib = E_q_z if g is None else g * E_q_z
            return eta_anchor + inv_sz2 * mm_update(fts.T, contrib), None

        eta_new, _ = lax.scan(step_fn, eta_init, jnp.arange(num_iters))
        return eta_new
```

- [ ] **Step 5: Run the new test**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_cavi_dual.py::test_dual_matches_primal_under_process_noise_and_gate`
Expected: PASS.

- [ ] **Step 6: Repair the pre-existing dual tests**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_cavi_dual.py`

Five of the six existing tests exercise `_prefers_dual` and `_cavi_scan` directly and should now pass; they call `_cavi_scan` with the old 6-argument signature and need `sigma_z=jnp.float32(1.0)` appended. `test_update_is_unchanged_end_to_end` pins a numeric value produced by the pre-process-noise `update()` and is no longer meaningful: replace its assertion with an invariance check, which is what it was actually protecting.

```python
def test_update_is_unchanged_end_to_end():
    """update() must give the same posterior whichever CAVI form the dispatch picks."""
    import jax.numpy as jnp
    import jax.random as jr
    from bllarse.losses.bayesian import IBProbit

    D, K, N = 40, 3, 12                      # narrow batch -> dual
    keys = jr.split(jr.key(7), 3)
    lf = IBProbit(D, K, key=keys[0], use_bias=True, q_init=1e-2)
    fts = jr.normal(keys[1], (N, D))
    y = jr.randint(keys[2], (N,), 0, K)

    assert IBProbit._prefers_dual(N, D + 1, K, 8)
    dual = lf.update(fts, y, num_iters=8)

    # force the primal by asking for so few iterations that the hat matrix cannot amortise
    assert not IBProbit._prefers_dual(N, D + 1, K, 1)
    ref = lf.update(fts, y, num_iters=1)
    one_step_dual = lf.update(fts, y, num_iters=1)
    assert jnp.allclose(ref.mu, one_step_dual.mu)

    assert jnp.isfinite(dual.mu).all() and jnp.isfinite(dual.Sigma).all()
```

- [ ] **Step 7: Run the full suite**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q`
Expected: PASS, all tests.

- [ ] **Step 8: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/test_cavi_dual.py
git commit -m "perf(ibprobit): restore the dual CAVI form under process noise

The 3f5e5c2 dual re-association survives the eta_pred anchor and the BMR gate
unchanged: x and S are built from ungated fts, so the gate enters only through
the contribution term. PX keeps no dual -- its scalar quadratic needs eta_hat
in d-space."
```

---

### Task 3: Smoke-test the three consuming scripts

**Files:**
- Create: `tests/test_ibprobit_script_construction.py`

**Interfaces:**
- Consumes: `IBProbit` from Tasks 1-2, plus `build_ibprobit` from `scripts/alternatives/full_finetune_comparison.py:117`.
- Produces: nothing consumed downstream; this is a regression guard so Task 9's CLI surgery cannot silently re-break construction.

The whole reason for Commit 1 is that these three scripts were dead. Nothing in the suite would have caught that, and nothing will catch it regressing. This test constructs a head through the same kwargs each script passes.

- [ ] **Step 1: Write the failing test**

```python
"""Guards the API contract between IBProbit and its three consuming scripts.

726bc78 landed ood_evaluation.py, full_finetune_comparison.py and
vit_classification.py on this branch without the IBProbit they call, and the
suite did not notice. These assertions are cheap and would have.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

from bllarse.losses.bayesian import IBProbit


def _kwargs():
    """Exactly the kwarg set the three scripts construct with."""
    return dict(q_init=1e-3, sigma_z2_init=1.0, learn_q=True, hyper_lr=0.1,
                anchor_weight=1e-3, px=True, bmr_gate=True, bmr_warmup=1,
                learn_eps=True, eps_init=0.01, rho=0.9)


def test_scripts_kwargs_construct_a_head():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs())
    assert lf.mu.shape == (17, 4)
    assert lf.Sigma.shape == (17, 17)


def test_scripts_kwargs_survive_a_streaming_pass():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs()).reset_posterior()
    keys = jr.split(jr.key(1), 2)
    fts = jr.normal(keys[0], (24, 16))
    y = jr.randint(keys[1], (24,), 0, 4)
    for _ in range(3):
        lf = lf.update(fts, y, num_iters=4)
    assert jnp.isfinite(lf.mu).all()
    assert jnp.isfinite(lf.Sigma).all()
    assert lf.step == 3
    loss = lf(fts, y)
    assert loss.shape == (24,) and jnp.isfinite(loss).all()


def test_predictive_variance_is_positive_and_per_input():
    lf = IBProbit(16, 4, key=jr.key(0), **_kwargs())
    v = lf.predictive_variance(jr.normal(jr.key(2), (7, 16)))
    assert v.shape == (7,) and (v > 0).all()
```

- [ ] **Step 2: Run it**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/test_ibprobit_script_construction.py`
Expected: PASS. Tasks 1-2 already provide everything; this test exists to fail loudly in Task 9 and in the future, not now. If it fails here, the port is incomplete — fix `bayesian.py`.

- [ ] **Step 3: Verify the scripts import and parse their arguments**

```bash
for s in ood_evaluation full_finetune_comparison vit_classification; do
  echo "=== $s"; CUDA_VISIBLE_DEVICES=2 uv run python scripts/alternatives/$s.py --help > /dev/null && echo OK
done
```
Expected: three `OK` lines. These scripts import `IBProbit` at module scope, so `--help` succeeding proves the import contract holds.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ibprobit_script_construction.py
git commit -m "test(ibprobit): guard the script-facing constructor contract

726bc78 landed three scripts calling an IBProbit API this branch did not have,
and the suite did not notice. It would now."
```

---

## Commit 2 — the Wishart `q` recursion

From here the target is `docs/superpowers/specs/2026-08-05-ibprobit-merge-and-wishart-q-design.md` §4, which in turn implements `~/projects/papers/bllarse/q-update-spec.md` §5.5 generalized from diagonal to full `Q`.

### Task 4: Inverse-Wishart state, gauge-fixed `sigma_z`, and the `learn_q` switch

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — `IBProbit` fields, `__init__`, `reset`, `reset_posterior`, `q`/`sigma_z2` properties, `_cavi_scan`, `_cavi_scan_px`, `predictive_variance`, `_predictive_logits`, `update`
- Modify: `tests/losses/test_bayesian.py` — delete the tests for deleted features
- Modify: `tests/test_ibprobit_script_construction.py` — `_kwargs()` follows the new signature

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: state fields `Psi (d,d)`, `nu ()`, `p_a ()`, `p_b ()` replacing `log_q` and `log_sigma_z2`; property `Qhat -> (d,d)`; static config `learn_q: bool`, `q0: float`, `kappa: float`, `warmup: int`, `alpha: float`; constructor `IBProbit(input_dim, num_classes, *, key, use_bias=True, use_approx_cdf=True, q_init=None, alpha=1e-3, batch_size=None, kappa=None, learn_q=True, loss_type=2, px=False, warmup=1, eps_init=0.01, rho=1.0)`; helpers `_q_seed() -> (Psi_0, nu_0)` and `_pc_seed() -> (p_a0, p_b0)`.

`sigma_z^2 === 1` (q-update-spec §5.7) removes the `sigma_z` argument from both scans; `_expected_z(pred, y_one_hot)` then takes the raw prediction, exactly as it did on this branch before the port. `learn_q=False` needs no branch in `Qhat`: since `Psi_0/(nu_0 - d - 1) = q0 * I` exactly, never updating `(Psi, nu)` holds `Qhat` at `q0 * I` by construction.

- [ ] **Step 1: Write the failing tests**

Create `tests/losses/test_wishart_q.py`:

```python
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

    L = jnp.linalg.cholesky(V)
    w = lf.mu[None, :, :] + jnp.einsum("ij,sjk->sik", L, jr.normal(keys[3], (200_000, d, K)))
    mc = jnorm.cdf(jnp.einsum("nd,sdk->snk", f, w)).mean(axis=0)

    assert jnp.abs(closed - mc).max() < 3e-3, jnp.abs(closed - mc).max()
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'alpha'`.

- [ ] **Step 3: Replace the field block and `__init__`**

In `IBProbit`, replace the `log_q` / `log_sigma_z2` fields and the q-learning config block with:

```python
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
    cdf: Callable
    logcdf: Callable
    posterior_ready: bool
    key: PRNGKey
    # --- static config ---
    learn_q: bool = eqx.field(static=True)
    q0: float = eqx.field(static=True)       # prior E[Q] scale: alpha^2 / B, or q_init
    kappa: float = eqx.field(static=True)    # Inverse-Wishart prior pseudo-count
    warmup: int = eqx.field(static=True)
    loss_type: int = eqx.field(static=True)
    px: bool = eqx.field(static=True)
    eps_init: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)
```

and the constructor:

```python
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        key: PRNGKey,
        use_bias: bool = True,
        use_approx_cdf: bool = True,
        q_init: Optional[float] = None,
        alpha: float = 1e-3,
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
        if q_init is not None:
            q0 = float(q_init)
        elif batch_size is not None:
            q0 = float(alpha) ** 2 / float(batch_size)
        else:
            raise ValueError("pass q_init or batch_size (q0 = alpha^2 / batch_size)")

        self.key = key
        self.q0 = q0
        self.kappa = float(num_classes if kappa is None else kappa)
        self.learn_q = bool(learn_q)
        self.warmup = int(warmup)
        self.loss_type = int(loss_type)
        self.px = bool(px)
        self.eps_init = float(eps_init)
        self.rho = float(rho)

        self.mu = jnp.zeros((d, num_classes), dtype=jnp.float32)
        self.Sigma = jnp.eye(d, dtype=jnp.float32)
        self.step = jnp.zeros((), dtype=jnp.int32)
        self.beta = jnp.ones((), dtype=jnp.float32)
        self.use_bias = use_bias
        self.cdf = approx_cdf if use_approx_cdf else norm.cdf
        self.logcdf = approx_logcdf if use_approx_cdf else norm.logcdf
        self.posterior_ready = False

        Psi0, nu0 = self._q_seed(d)
        self.Psi, self.nu = Psi0, nu0
        self.p_a, self.p_b = self._pc_seed()
        a0, b0 = self._eps_seed()
        self.eps_alpha = jnp.asarray(a0, dtype=jnp.float32)
        self.eps_beta = jnp.asarray(b0, dtype=jnp.float32)
```

- [ ] **Step 4: Add the seed helpers and the `Qhat` property**

```python
    def _q_seed(self, d: Optional[int] = None) -> Tuple[Array, Array]:
        """Isotropic Inverse-Wishart prior on Q with E_0[Q] = q0 * I.

        With nu_0 = d + 1 + kappa the mean is Psi_0 / (nu_0 - d - 1) = Psi_0 / kappa,
        so Psi_0 = q0 * kappa * I. kappa is the prior pseudo-count, defaulting to K so
        the prior carries the weight of one batch's worth of transition evidence."""
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
```

Delete the `q` and `sigma_z2` properties. Simplify `_eps_seed` to always return the learning prior, since the mixture is now structural:

```python
    def _eps_seed(self) -> Tuple[float, float]:
        """Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init)."""
        return 1.0, (1.0 / self.eps_init if self.eps_init > 0.0 else 100.0)
```

- [ ] **Step 5: Drop `sigma_z` from both scans and from the predictive**

In `_cavi_scan` and `_cavi_scan_px`, remove the `sigma_z` parameter and every `inv_sz2` / `sz2` / `sigma_z *` factor (they are all 1). `_cavi_scan`'s steps become `E_q_z = self._expected_z(mm_update(x, eta_carry), y_one_hot)`; `_cavi_scan_px`'s becomes `E_q_z = self._expected_z(m, y_one_hot)` with `E_q_z2 = m * E_q_z + 1.0` and `Cq = S_z + jnp.sum(eta_hat * mm_update(M_c, eta_hat))`. Change `_expected_z`'s parameter name back from `alpha` to `pred` — `alpha` now means the prior-scale constructor argument and the collision is a trap.

In `predictive_variance` and `_predictive_logits`, replace `self.q * fts` with `fts @ self.Qhat` and `self.sigma_z2` with `1.0`:

```python
        var = jnp.sum((mm_update(fts, self.Qhat + self.Sigma)) * fts, axis=-1)
        return 1.0 + var                      # predictive_variance
        ...
        return scores / jnp.sqrt(1.0 + var)[:, None]   # _predictive_logits
```

In `update`, replace the PREDICT line `Sigma_pred = self.Sigma.at[indices].add(self.q)` with `Sigma_pred = self.Sigma + self.Qhat`, drop `sigma_z`/`inv_sz2`, and drop the `learn_q` SGD block and the `bmr_gate`/`learn_eps` branching for now — the gate becomes unconditional `g = jnp.where(self.step >= self.warmup, r, 1.0)`. Update `reset` and `reset_posterior` to restore `(Psi, nu)` from `_q_seed()` and `(p_a, p_b)` from `_pc_seed()` instead of `log_q`/`log_sigma_z2`.

- [ ] **Step 6: Delete the tests for deleted features**

From `tests/losses/test_bayesian.py`, delete these 15 tests, whose subjects no longer exist:

```
test_ibprobit_trainable_params_are_q_and_sigmaz
test_ibprobit_learn_q_optout_leaves_log_q_fixed
test_ibprobit_learn_q_first_batch_is_noop
test_ibprobit_learn_q_matches_manual_prequential_sgd
test_ibprobit_learn_q_anchor_pulls_toward_mode
test_ibprobit_learn_q_runs_under_scan_jit
test_ibprobit_bmr_gate_off_is_identity
test_ibprobit_bmr_gate_range_and_discriminates
test_ibprobit_bmr_seed_gate_is_half
test_ibprobit_bmr_gate_instance_field_drives_update
test_ibprobit_bmr_gated_px_matches_gated_plain
test_ibprobit_eps_off_call_matches_plain
test_ibprobit_eps_seed_gate_is_one_minus_eps
test_ibprobit_eps_learn_without_bmr_gate_leaves_assimilation_ungated
test_ibprobit_eps_bmr_gate_toggles_responsibility_gate
```

Keep `test_ibprobit_bmr_effective_sample_size`, `test_ibprobit_bmr_warmup_skips_early_steps`, `test_ibprobit_bmr_runs_under_scan_jit` and `test_ibprobit_bmr_gated_px_matches_plain`, renaming `bmr` to `gate` in their names — the behaviour survives, only the switch is gone. Update every remaining construction in the file to the new signature: drop `sigma_z2_init`, `learn_q=` as an SGD flag, `hyper_lr`, `anchor_weight`, `bmr_gate`, `learn_eps`, and add `q_init=` where a specific `q` is wanted.

Task 2's `test_dual_matches_primal_under_process_noise_and_gate` also needs updating: it reads `lf.q` and `lf.sigma_z2` and calls `_cavi_scan` with a `sigma_z` argument, all three of which are gone. Rewrite its middle section as:

```python
    Sigma_pred = lf.Sigma + lf.Qhat
    tmp = cho_solve(cho_factor(0.5 * (Sigma_pred + Sigma_pred.T)),
                    jnp.concatenate([lf.mu, jnp.eye(d)], axis=-1))
    eta_pred, Lambda_pred = tmp[:, :-d], tmp[:, -d:]

    Lambda_new = Lambda_pred + mm_update((g * fts).T, fts)
    L = cho_factor(0.5 * (Lambda_new + Lambda_new.T))
    x = cho_solve(L, fts.T).T

    assert IBProbit._prefers_dual(N, d, K, iters), "shapes must select the dual path"
    dual = lf._cavi_scan(fts, x, y_one_hot, eta_pred, eta_pred, iters, g)

    eta = eta_pred
    for _ in range(iters):
        E_z = IBProbit._expected_z(mm_update(x, eta), y_one_hot)
        eta = eta_pred + mm_update(fts.T, g * E_z)

    assert jnp.allclose(dual, eta, atol=1e-4), jnp.abs(dual - eta).max()
```

- [ ] **Step 7: Update the script-contract test**

In `tests/test_ibprobit_script_construction.py`, `_kwargs()` becomes:

```python
def _kwargs():
    """Exactly the kwarg set the three scripts construct with."""
    return dict(alpha=1e-3, batch_size=32, learn_q=True, px=True,
                warmup=1, eps_init=0.01, rho=0.9)
```

- [ ] **Step 8: Run the tests**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/ tests/test_ibprobit_script_construction.py`
Expected: PASS. The three scripts are knowingly broken until Task 9; `--help` will fail and that is fine.

- [ ] **Step 9: Commit**

```bash
git add -A src/bllarse/losses/bayesian.py tests/
git commit -m "feat(ibprobit): Inverse-Wishart state on Q, gauge-fixed sigma_z

Replaces the gradient-trained scalar log_q with (Psi, nu), a conjugate
Inverse-Wishart posterior on the full process-noise matrix, seeded
isotropically at E_0[Q] = q0 I with q0 = alpha^2/B (q-update-spec 5.5e).
sigma_z^2 is gauge-fixed to 1 and log_sigma_z2 deleted (5.7), so both CAVI
scans lose their sigma_z factor.

learn_q now switches the recursion rather than an SGD step; off holds Qhat at
q0 I by construction, which is the ablation REWRITE-PLAN Risk 3 asks for.
The recursion itself lands in the next three commits."
```

---

### Task 5: The expected squared increment `S_t`

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — add `_increment_statistic`
- Modify: `tests/losses/test_wishart_q.py`

**Interfaces:**
- Consumes: Task 4.
- Produces: `_increment_statistic(Qhat, Lambda_pred, Sigma_new, Delta) -> Array (d,d)`.

This is the substance of q-update-spec §5.5(a), kept in matrix form. The naive mean-field statistic returns `2K*Sigma_jj > 0` even at `Q = 0` — a model that cannot represent "the head is not moving" always reports that it is. Conditioning on `W_t` fixes it exactly and for free, because the emission touches only `W_t`, so the increment's posterior follows from PREDICT alone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/losses/test_wishart_q.py`:

```python
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

    G = Qhat @ Lambda_pred
    expected = (jnp.sum((G @ Delta) ** 2, axis=-1)
                + K * (jnp.diagonal(G @ Sigma_prev) + jnp.diagonal(G @ Sigma_new @ G.T)))
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k increment`
Expected: FAIL with `AttributeError: 'IBProbit' object has no attribute '_increment_statistic'`.

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k increment`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/test_wishart_q.py
git commit -m "feat(ibprobit): expected squared increment S_t in matrix form

q-update-spec 5.5(a) generalized from its diagonal to the full matrix. The
G-form is what lets the estimator represent a stationary head: at Qhat = 0 it
returns exactly zero, where the naive mean-field statistic returns 2K Sigma_jj."
```

---

### Task 6: The Bayes Factor Surprise and the adaptive rate

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — add `_logdet_pd`, `_log_bayes_factor`, `_surprise_rate`
- Modify: `tests/losses/test_wishart_q.py`

**Interfaces:**
- Consumes: Tasks 4-5.
- Produces: module-level `_logdet_pd(M) -> Array`; methods `_log_bayes_factor(S, Psi, nu, K) -> Array` (scalar) and `_surprise_rate(S, Psi, nu, p_a, p_b, K) -> Array` (scalar gamma in (0,1)).

Where the diagonal spec compounds a Gamma belief with a scaled-chi-squared observation, the matrix version compounds an Inverse-Wishart belief with `S_t | Q ~ Wishart_d(Q, K)`. The compound marginal is closed form and the `(Psi, nu)`-free part cancels in the ratio, as does the `d(d-1)/4 * log(pi)` constant inside the multivariate gamma.

- [ ] **Step 1: Write the failing tests**

Append to `tests/losses/test_wishart_q.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k "bayes_factor or surprise"`
Expected: FAIL with `AttributeError: 'IBProbit' object has no attribute '_log_bayes_factor'`.

- [ ] **Step 3: Add the import and the log-determinant helper**

Extend the existing import at the top of `bayesian.py`:

```python
from jax.scipy.special import gammaln, digamma
```

and add beside `mm`:

```python
def _logdet_pd(M):
    """log|M| for symmetric positive-definite M, via Cholesky. Cheaper and better
    conditioned than slogdet, and every matrix it is called on here is PD by
    construction (see IBProbit._increment_statistic)."""
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(jnp.linalg.cholesky(M))))
```

- [ ] **Step 4: Implement the two methods**

```python
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
```

- [ ] **Step 5: Run to verify they pass**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k "bayes_factor or surprise"`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/test_wishart_q.py
git commit -m "feat(ibprobit): Wishart Bayes Factor Surprise and adaptive rate

q-update-spec 5.5(b,c) with the Gamma-Gamma compound marginal replaced by the
Wishart / Inverse-Wishart one. The surprise is normalized by d*K rather than
the spec's K: one matrix node absorbs d*K scalar observations, and dividing by
K alone saturates gamma at 1 for any realistic d."
```

---

### Task 7: The Inverse-Wishart update, the `p_c` node, and wiring

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — add `_q_recursion`, rewrite `update`
- Modify: `tests/losses/test_wishart_q.py`

**Interfaces:**
- Consumes: Tasks 4-6.
- Produces: `_q_recursion(Qhat, Lambda_pred, Sigma_new, Delta, K) -> (Psi_new, nu_new, p_a_new, p_b_new, gamma)`; `update(features, y, *, num_iters=32, px=None)` sequenced per q-update-spec §8.

- [ ] **Step 1: Write the failing tests**

Append to `tests/losses/test_wishart_q.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k "recursion or isotropic or contracts or warmup or pc_posterior or jit_and_scan"`
Expected: FAIL with `AttributeError: 'IBProbit' object has no attribute '_q_recursion'`.

- [ ] **Step 3: Implement `_q_recursion`**

```python
    def _q_recursion(self, Qhat: Array, Lambda_pred: Array, Sigma_new: Array,
                     Delta: Array, K: int) -> Tuple[Array, Array, Array, Array, Array]:
        """Steps 5-8 of q-update-spec sec 8: the conjugate hyper-updates.

        S_t | Q ~ Wishart_d(Q, K), whose conjugate is the Inverse-Wishart, so absent
        forgetting the update is nu += K, Psi += S_t. VarSMiLe replaces the fixed EMA
        discount with the inferred rate gamma:

            nu_t  = (1 - gamma) nu_{t-1}  + gamma nu_0  + K
            Psi_t = (1 - gamma) Psi_{t-1} + gamma Psi_0 + S_t

        At large gamma this lands on (Psi_0 + S_t, nu_0 + K) -- the prior plus this
        batch -- so a detected change discards the stale drift estimate. gamma is the
        variational responsibility q(c_t = 1), so it doubles as the soft count for the
        Beta on the change rate p_c.

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
```

- [ ] **Step 4: Rewrite `update` to the §8 order**

```python
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

        # 1-2. q(s) then q(eps). The two-argument _eps_responsibility is the ported
        # Beta-mean form; Task 8 replaces it with the psi-form and drops the argument.
        r = self._eps_responsibility(loss_vec, self.eps)             # (n,1)
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
```

- [ ] **Step 5: Run the new tests**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py`
Expected: PASS, all tests.

- [ ] **Step 6: Run the whole losses suite**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/`
Expected: PASS. `_eps_responsibility` still takes its old `(loss_vec, eps)` signature at this point — if a test fails on it, that is Task 8's job; note it and move on only if the failure is exactly that signature.

- [ ] **Step 7: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/test_wishart_q.py
git commit -m "feat(ibprobit): conjugate Inverse-Wishart update on Q, sequenced per spec 8

Completes the q recursion: S_t -> surprise -> q(Q) -> q(p_c), deferred through
warmup because the mu=0 seed makes the first increment a full-size jump.
update() now follows q-update-spec 8 exactly and contains no gradient step.

The Beta on p_c takes one soft count per step where the diagonal formulation
took d, so p_c identifies ~d times slower; documented at the call site."
```

---

### Task 8: psi-form responsibilities and the last deletions

**Files:**
- Modify: `src/bllarse/losses/bayesian.py` — rewrite `_eps_responsibility`, delete `_gate_from_logits`, `_preq_objective`
- Modify: `tests/losses/test_wishart_q.py`

**Interfaces:**
- Consumes: Tasks 4-7.
- Produces: `_eps_responsibility(loss_vec) -> Array (n,1)` — one argument, reads `(eps_alpha, eps_beta)` from state.

q-update-spec §5.2: the code uses the Beta mean where CAVI wants `exp E[log eps]`. Fixing it makes the update an exact stationary point of the free energy. `_gate_from_logits` and `_preq_objective` are now unreachable and come out.

- [ ] **Step 1: Write the failing tests**

Append to `tests/losses/test_wishart_q.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py -k "psi_form or orders_by_fit or dead_helpers"`
Expected: FAIL — `_eps_responsibility` still takes two arguments, and the two helpers still exist.

- [ ] **Step 3: Rewrite `_eps_responsibility` and delete the dead helpers**

```python
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
```

Change the call site in `update()` from `self._eps_responsibility(loss_vec, self.eps)` to `self._eps_responsibility(loss_vec)`.

Delete `_gate_from_logits` and `_preq_objective` entirely. Delete the `use_eps` property and every `if self.use_eps` branch — the mixture is structural now, so `__call__` always scores the floored mixture through `_mixture_logp`, and `_mixture_logp` loses its guard.

- [ ] **Step 4: Run to verify they pass**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_wishart_q.py`
Expected: PASS, all tests.

- [ ] **Step 5: Repair the remaining ported eps tests**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q tests/losses/test_bayesian.py`

`test_ibprobit_eps_responsibility_shape`, `test_ibprobit_eps_mixture_normalized_and_uniform_seed`, `test_ibprobit_eps_tail_bound`, `test_ibprobit_eps_beta_prior_seed`, `test_ibprobit_eps_beta_conjugate_update`, `test_ibprobit_eps_forgetting_factor_decays_excess`, `test_ibprobit_eps_forgetting_bounds_effective_sample_size`, `test_ibprobit_eps_reset_restores_prior`, `test_ibprobit_eps_decreases_on_wellfit_data`, `test_ibprobit_eps_gated_px_matches_plain` and `test_ibprobit_eps_runs_under_scan_jit` all survive in substance. Drop the `learn_eps=True` kwarg from each construction and update the two that call `_eps_responsibility(loss, eps)` to the one-argument form. Any test asserting a specific numeric responsibility must have its expected value recomputed from the psi-form — the Beta mean and `exp E[log]` differ, and that difference is the point of this task.

- [ ] **Step 6: Run the full suite**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q --ignore=tests/test_probe_smoke.py`
Expected: PASS except for failures traceable to the three scripts' CLIs, which Task 9 fixes.

- [ ] **Step 7: Commit**

```bash
git add src/bllarse/losses/bayesian.py tests/losses/
git commit -m "feat(ibprobit): psi-form responsibilities; drop the ad-hoc gate

q-update-spec 5.2. The Beta mean is replaced by exp E[log eps] via digamma,
making the eps update an exact stationary point of the free energy. The
committed fixed-1/2 BMR gate has no generative meaning and is deleted -- under
the contamination model the gate IS r_i, always -- and the mixture becomes
structural rather than opt-in. _preq_objective goes with the SGD it served."
```

---

### Task 9: Update the three scripts' CLIs

**Files:**
- Modify: `scripts/alternatives/ood_evaluation.py:321-326, 367-373, 451-472`
- Modify: `scripts/alternatives/full_finetune_comparison.py:117-129, 598, 679-684, 1099-1118`
- Modify: `scripts/alternatives/vit_classification.py:620-627, 644-651, 806-825`

**Interfaces:**
- Consumes: Tasks 4-8.
- Produces: no new API; the three scripts construct `IBProbit` with the Task 4 signature and log the new hyperparameters.

The three flag blocks are near-identical. Delete `--fix-q`, `--hyper-lr`, `--anchor-weight`, `--sigma-z2-init`, `--bmr-gate`, `--learn-eps`; rename `--bmr-warmup` to `--warmup`; add `--alpha` and `--no-learn-q`. Old sweep configs passing a removed flag now fail loudly, which is the intent — a flag that silently does nothing is worse.

- [ ] **Step 1: Replace the flag block in all three scripts**

The block is `ood_evaluation.py:451-472`, `full_finetune_comparison.py:1099-1118`, `vit_classification.py:806-825`. In each, replace the flags between `--num-update-iters` and the next section comment with (using `p` or `parser` to match the surrounding file):

```python
    p.add_argument("--px", action="store_true",
                   help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    p.add_argument("--no-learn-q", action="store_true",
                   help="hold Q at its isotropic prior q0*I (the ablation arm); default infers it")
    p.add_argument("--alpha", type=float, default=1e-3,
                   help="prior scale on the process noise: q0 = alpha^2 / batch_size (q-update-spec 5.5e)")
    p.add_argument("--q-init", type=float, default=None,
                   help="override q0 directly; pass 1/D to reproduce the pre-Wishart regime")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart prior pseudo-count (default: num_classes, i.e. one batch's weight)")
    p.add_argument("--warmup", type=int, default=1,
                   help="defer the gate and the q recursion for the first N steps (the mu=0 seed is uninformative)")
    p.add_argument("--eps-init", type=float, default=0.01,
                   help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). Forgetting rho = 1 - B/N set automatically.")
```

- [ ] **Step 2: Update the construction sites**

`ood_evaluation.py:321-326` becomes:

```python
        base = IBProbit(embed_dim, num_classes, key=jr.split(key)[1],
                        q_init=args.q_init, alpha=args.alpha, batch_size=args.batch_size,
                        kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
                        warmup=args.warmup, eps_init=args.eps_init, rho=eps_rho)
```

`full_finetune_comparison.py:117-129` becomes:

```python
def build_ibprobit(args, embed_dim, num_classes, key, rho=1.0, batch_size=None):
    return IBProbit(
        embed_dim, num_classes, key=key,
        q_init=args.q_init, alpha=args.alpha, batch_size=batch_size,
        kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
        warmup=args.warmup, eps_init=args.eps_init, rho=rho,
    )
```

and its call at line 598 gains `batch_size=effective_batch` (the same value line 684 already uses for `eps_rho`). `vit_classification.py:644-651` follows the `ood_evaluation` pattern with `batch_size=batch_size`, the local already in scope at line 627.

- [ ] **Step 3: Update the three params-logging dicts**

`ood_evaluation.py:367-373`, `full_finetune_comparison.py:679-684` and `vit_classification.py:620-627` log the old knobs to MLflow. Replace the removed keys with the new ones in each:

```python
                          num_update_iters=args.num_update_iters, px=args.px,
                          learn_q=not args.no_learn_q, alpha=args.alpha,
                          q_init=args.q_init, kappa=args.kappa, warmup=args.warmup,
                          eps_init=args.eps_init,
                          eps_rho=max(0.0, 1.0 - args.batch_size / len(train_f)),
```

adjusting the batch/dataset expression to whichever the surrounding file already uses.

- [ ] **Step 4: Update the script-contract test to the shipped kwargs**

In `tests/test_ibprobit_script_construction.py`, `_kwargs()` becomes exactly what the scripts now pass:

```python
def _kwargs():
    """Exactly the kwarg set the three scripts construct with."""
    return dict(q_init=None, alpha=1e-3, batch_size=32, kappa=None,
                learn_q=True, px=True, warmup=1, eps_init=0.01, rho=0.9)
```

- [ ] **Step 5: Verify the scripts parse**

```bash
for s in ood_evaluation full_finetune_comparison vit_classification; do
  echo "=== $s"; CUDA_VISIBLE_DEVICES=2 uv run python scripts/alternatives/$s.py --help > /dev/null && echo OK
done
grep -rn "fix_q\|hyper_lr\|anchor_weight\|sigma_z2_init\|bmr_gate\|learn_eps\|bmr_warmup" \
  --include=*.py scripts/ src/ tests/ bllarse_sweeps/
```
Expected: three `OK` lines, and the grep returns nothing.

- [ ] **Step 6: Run the full suite**

Run: `CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q`
Expected: PASS. Compare against the 283-passed baseline: the delta is the ported and new IBProbit tests minus the 15 deleted ones.

- [ ] **Step 7: Commit**

```bash
git add scripts/alternatives/ tests/test_ibprobit_script_construction.py
git commit -m "refactor(scripts): retire the SGD-q and BMR-gate flags

Deletes --fix-q, --hyper-lr, --anchor-weight, --sigma-z2-init, --bmr-gate and
--learn-eps, whose fields no longer exist; renames --bmr-warmup to --warmup and
adds --alpha, --kappa and --no-learn-q. Sweep configs passing a removed flag now
fail loudly rather than silently doing nothing."
```

---

## What this plan does not do

- **No experimental runs.** q-update-spec §12.5 makes the old-`q` versus new-`q` comparison the measurement the paper rewrite is blocked on: `q0` moves from `1/D ~ 7.8e-4` to `alpha^2/B ~ 3.9e-9`, five orders of magnitude, and §5.5(f) predicts in-distribution ECE will move because the `q||f||^2` floor stops contributing. Commit 1 exists so that comparison can be run in-branch. Running it is a separate task.
- **No ImageNet conditioning check.** q-update-spec §7.1 and §12.3 flag that the precision bound scales as `1/Qhat`, reaching ~2.6e8 at the new prior, at the edge of float32. Unchanged by the Wishart generalization and still needed before the paper claims alpha-forgetting can go.
- **No paper edits.** Three things in `~/projects/papers/bllarse` now disagree with the code and need updating by whoever writes §3.7: the complexity claim (this is ~3x the per-batch dense linear algebra, not "a handful of lgamma calls"), the "ARD-like, axis-aligned" description of the moderation denominator (it is now a full learned Mahalanobis form), and the `p_c` identification argument in §5.5(d) (one soft count per step, not `d`).
