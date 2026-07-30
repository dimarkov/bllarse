# In-`update()` prequential `log_q` gradient (self-tuning process noise)

**Date:** 2026-06-24
**Branch:** `ibprobit-process-noise`
**File:** `src/bllarse/losses/bayesian.py` (`IBProbit`), wired from `scripts/finetuning.py`

## Goal

Let `IBProbit.update()` self-tune the process noise `q` via a single
**prequential SGD step on `log_q`**, so the Deep MLP full-network finetuning
loop (`bllarse.utils.batch_update` -> `loss_fn.update(...)`) learns `q` with
**zero changes to the training loop** and no second optimizer.

This is motivated by running the full-network finetuning experiment on the
Deep MLP large model (`B=12, Wi=1024`, mlpox `B_12-Wi_1024_res_64_in21k`):
there the outer AdamW optimizer trains the **backbone**, and the IBProbit
head's `q`/`sigma_z2` are never touched. Folding the q-step inside `update()`
keeps `finetuning.py` / `run_training` / `batch_update` untouched (they already
just call `loss_fn.update(...)`).

The mechanism is exactly `ood_evaluation.py:stream_train_ibprobit`'s external
prequential SGD step (`optax.sgd(hyper_lr)` on `log_q` against the pre-update
predictive `model(f, y, loss_type=2).mean()`), folded into `update()` and
specialized to `q` only.

This is **stage 1** of a two-stage plan. **Stage 2 (deferred, separate design):**
reuse the same pre-update predictive logits to compute a per-point BMR gate
`g_i = Phi(mu_tilde_i,true) / (Phi(mu_tilde_i,true) + 1/2)` and down-weight
badly-predicted observations in the CAVI assimilation. Stage 1 is architected so
the pre-update predictive pass is the single, reusable hook stage 2 extends.

## Non-goals (stage 1)

- `log_sigma_z2` is **not** learned (user scope: `q` only).
- No change to `_cavi_scan` / `_cavi_scan_px` (those are stage 2's surface).
- No file edits to `utils.py`, `run_training`, `batch_update`, or any test that
  doesn't opt in. (`ood_evaluation.py` is untouched but inherits the `loss_type`
  default shift -- see the loss_type scope below.)
- The px-scan default in `finetuning` is orthogonal and out of scope here.

## Design

### New `IBProbit` fields (all `eqx.field(static=True)`)

| field | type | default | meaning |
|-------|------|---------|---------|
| `learn_q` | bool | `False` | enable the in-update `log_q` gradient step |
| `hyper_lr` | float | `0.1` | SGD step size on `log_q` |
| `anchor_weight` | float | `1e-3` | anchor strength `lambda` (`0` = off); concept `lambda = (alpha+1)/N` |
| `anchor_mode` | float | init `q` (`= 1/D`) | prior mode `m` the anchor pulls `q` toward |
| `loss_type` | int | `2` | fixed loss branch in `__call__`; enforced per instance |

### `loss_type` as an instance field

`loss_type` is promoted from a per-call `__call__` kwarg to a construction-time
`IBProbit` field, because varying it across forward passes changes the
gradient/convergence target (e.g. `ood_evaluation.py` uses `2` for the
prequential grad and `3` for eval on one instance). `__call__` **keeps** the
`loss_type` kwarg in its signature but **ignores** it (accept-and-ignore, exactly
like the classical `MSE`/`CrossEntropy` losses already do) and always branches on
`self.loss_type`. So no call site breaks syntactically; the instance value
governs.

Default `2`. Consequences: the prequential q-grad objective uses
`self.loss_type` (`= 2` -> the predictive NLL, the proper-scoring objective, and
it matches the external reference), and IBProbit's network gradient in
`batch_update` collapses from the passed `3` to the instance `2` -- the
intended consistency. `calibrate`'s own `loss_type` param likewise becomes inert
(it routes through `__call__`).

**Scope:** this pass changes `IBProbit` only. The other Bayesian loss classes in
`bayesian.py` and the per-script `loss_type=...` call sites
(`full_finetune_comparison`, `sample_efficiency`, `ood_evaluation`,
`segmentation`, `language_modeling`, `experiment_ibprobit`) are left as-is and
cleaned up later (per user: "we can clean the code later"). Those that relied on
passing a non-default `loss_type` to an IBProbit instance now inherit the field
default `2`; this is an accepted, deferred follow-up, not part of stage 1.

### Simplified IG anchor

The original `neg_log_inv_gamma(q, shape, scale) = (shape+1) log q + scale/q`
is reparameterized to the clean two-parameter form (mode `m`, weight `lambda`),
dropping `shape`/`scale`:

```
penalty(log_q) = lambda * ( log_q + m * exp(-log_q) )           # convex in log_q, min at q = m
```

This is `-log p(q)` for the IG prior (your `lambda*(-log_q - m*exp(-log_q))` is
the log-prior; the minimization objective subtracts it). Its gradient
`lambda*(1 - m*exp(-log_q)) = lambda*(1 - m/q)` is `0` at `q=m`, `>0` for `q>m`
(pushes `log_q` down toward `m`) and `<0` for `q<m` -- an **attractive** anchor.

`anchor_mode` is set in `__init__` to `exp(log_q)` at construction (i.e. `q0`,
either `q_init` or `1/D`). It is **static** and never moves, so the mode stays
pinned at the nominal init `q` even as `log_q` is stepped. Because `log_q` also
*starts* at `log(q0) = log(anchor_mode)`, the anchor gradient is `0` at the seed
-- so with the default `mu=0` seed the first-batch step is a **no-op even with
the anchor on** (both the NLL gradient and the anchor gradient vanish).

`lambda` defaults to `1e-3` (anchor on, gentle). Conceptually it is the
per-sample prior weight `(alpha+1)/N`; as a field it is just a scalar, `0` = off.

Static fields are not pytree leaves, so they do not interfere with any existing
`eqx.filter(loss_fn, eqx.is_array)` (utils `batch_update`) or
`eqx.partition(loss_fn, filt)` / `tree_map` (ood_evaluation) machinery.

`__init__` gains keyword-only params `learn_q=False, hyper_lr=0.1,
anchor_weight=1e-3, anchor_mode=None, loss_type=2`; `anchor_mode=None` resolves
to the init `q` (`q0`). `reset` / `reset_posterior` rebuild via `eqx.tree_at` on
named leaves only, so they carry the new static fields through unchanged (no
edits needed there).

### The gradient step (start of `update()`, before PREDICT/CAVI)

Inserted at the very top of `update()`, **before** the existing PREDICT and
CAVI blocks (per the prequential ordering: score the batch under the pre-update
predictive, step `log_q`, then assimilate):

```python
if self.learn_q:
    def preq_obj(log_q):
        model = eqx.tree_at(lambda m: m.log_q, self, log_q)
        obj = model(features, y).mean()                       # pre-update predictive NLL, self.loss_type (=2)
        if self.anchor_weight > 0.0:
            pen = log_q + self.anchor_mode * jnp.exp(-log_q)   # -log p(q) up to const
            obj = obj + self.anchor_weight * pen
        return obj
    g = jax.grad(preq_obj)(self.log_q)
    self = eqx.tree_at(lambda m: m.log_q, self, self.log_q - self.hyper_lr * g)
# ...existing PREDICT (Sigma_pred = Sigma + self.q * I, stepped q) + CAVI unchanged...
```

Notes:
- Objective = pre-update predictive NLL (`loss_type=2`, softmax CE on the
  moderated-probit predictive) + the simplified IG anchor on `q` (q-only;
  the external trainer's `sigma_z2` penalty term is dropped).
- Plain SGD has no momentum -> **no optimizer state to carry**; manual
  `log_q -= hyper_lr * g` is bitwise-equal to `optax.sgd(hyper_lr)`.
- Differentiates **only** `log_q` (scalar closure via `eqx.tree_at`);
  `sigma_z2`, `mu`, `Sigma`, `beta` are untouched.
- Runs fine traced inside `lax.scan` / `eqx.filter_jit` (it is the finetuning
  path).
- The stepped `self` (new `log_q`) flows into the existing PREDICT and into the
  final `eqx.tree_at(lambda m: (m.mu, m.Sigma, m.posterior_ready), self, ...)`
  return, so the new `log_q` is persisted automatically.

### Correctness invariants (-> tests)

1. **Regression / opt-out:** `learn_q=False` -> `update()` is identical to
   current `(mu, Sigma, log_q)` on a synthetic batch. Protects all non-opted-in
   call sites.
2. **First-batch no-op (exact):** `learn_q=True, anchor_weight=0` with the
   `mu=0` seed -> scores=0 -> logits=0 -> softmax-CE constant -> `grad=0` ->
   `log_q` **bitwise unchanged**. With the default anchor on
   (`anchor_weight=1e-3`, `anchor_mode=`init `q`) the seed anchor gradient
   `lambda*(1 - m/q)` is `0` up to float32 rounding (`q=m` at the seed), so the
   step is negligible (~1e-11), not a bug.
3. **Exact equivalence to the external trainer (one batch):** with
   `learn_q=True, anchor_weight=0`, after one prior `update()` (nonzero `mu`),
   `m.update(f, y).log_q == old_log_q - hyper_lr * jax.grad(preq_nll)(old_log_q)`
   computed independently (`preq_nll = model(f, y).mean()`, `loss_type=2`),
   **and** the returned `(mu, Sigma)` equals `update()` run at that stepped `q`.
   Pins the fold-in against the `ood_evaluation.py` reference.
4. **Anchor pull:** with `learn_q=True, anchor_weight>0`, `mu=0` (isolate the
   anchor), and `log_q` displaced from `log(anchor_mode)`, the gradient
   `lambda*(1 - m/q)` pushes `q` back toward `anchor_mode` (sign check: `q>m` ->
   `log_q` decreases); `anchor_weight=0` recovers the exact no-op.

## Experiment wiring (`scripts/finetuning.py` only)

- Construct (line ~201):
  `IBProbit(embed_dim, num_classes, key=_key, learn_q=args.learn_q,
  hyper_lr=args.hyper_lr, anchor_weight=args.anchor_weight)`
  (`anchor_mode` defaults to init `q`; `loss_type` defaults to `2`).
- New CLI flags: `--learn-q` (store_true), `--hyper-lr` (default `0.1`),
  `--anchor-weight` (default `1e-3`).
- Log `loss_fn.q` per epoch (console / MLflow params already plumbed).
- `utils.py` / `run_training` / `batch_update` **unchanged**.

Run target: Deep MLP large, full-network finetuning, IBProbit head with
`--learn-q` (anchor on by default at `--anchor-weight 1e-3`), `B=12, Wi=1024`,
`--pretrained in21k` + fresh head (no `_cifar` checkpoint exists for the large
model).

## Deferred / tunable

- `anchor_weight` value for the experiment (default `1e-3`; retune by watching
  the `q` trajectory -- raise it if `q` wanders, set `0` to disable the anchor).
- `log_sigma_z2` learning (same fold-in, trivial extension).
- `loss_type`-as-field cleanup across the other Bayesian loss classes and the
  per-script call sites (deferred; see loss_type scope).
- px-scan default in `finetuning` (orthogonal).
- **Stage 2:** BMR per-point gate reusing the pre-update predictive logits.
