# In-`update()` self-tuning ε-contamination mixture (robust likelihood / learned BMR gate)

**Date:** 2026-06-30
**Branch:** `ibprobit-process-noise`
**File:** `src/bllarse/losses/bayesian.py` (`IBProbit`), wired from `scripts/finetuning.py` and reported by `scripts/alternatives/ood_evaluation.py`

> **Superseded in part (2026-07-03):** `eps` is no longer learned by an in-`update()` SGD
> step on `logit_eps` with a Beta *anchor penalty*. It is now estimated by **conjugate Beta
> updating**: the responsibilities `r_i` are soft Bernoulli counts that update a carried
> `Beta(1, 1/eps_init)` posterior (default `Beta(1, 100)`) with a forgetting factor `rho`
> (scripts set `rho = 1 - B/N` for a one-pass memory), and `eps` is its mean
> `eps_alpha/(eps_alpha+eps_beta)`. Gating of assimilation by `r_i` now happens **only** when
> `bmr_gate` is on (`r_i` is always computed when `learn_eps`). The **motivation, mixture
> likelihood, responsibility, tail-bound, and eval-scoring** sections below are unchanged and
> still current; the **"New fields", "In-`update()` joint prequential step", and "Beta anchor
> on `eps`"** sections are replaced by the Revision at the end of this doc.

## Goal

Give `IBProbit` an **ε-contamination observation model**

```
p_eps_i(k) = (1 - eps) * p_model_i(k)  +  eps * (1 / K)
```

whose contamination weight `eps in [0, 1)` is **self-tuned inside `update()`** by the
same in-update prequential SGD step that already learns `log_q` (see
`2026-06-24-in-update-q-gradient-design.md`). One extra scalar, one extra term in the
existing gradient closure, no new optimizer and no training-loop change.

The single `eps` plays two consistent roles:

1. **Assimilation (training):** each point is down-weighted in the CAVI moments by its
   posterior **responsibility** `r_i` (the probability it came from the model, not the
   noise). This is the existing per-point BMR gate, with the hard-coded mixing constant
   promoted to a learned parameter.
2. **Scoring (eval):** the reported predictive is the floored mixture `p_eps`, which
   bounds the per-example log-loss and fixes the OOD-NLL tail (below).

## Motivation

The OOD calibration run (`ood_evaluation.py`, now logging `acc/ece/nll/brier` at `T=1`
and an ID-val-fit temperature `T*`) shows, for IBProbit vs a CrossEntropy head:

- **ACC, ECE, Brier** all favor IBProbit, ID and OOD.
- **NLL** is worse for IBProbit, and **stays worse after each model gets its own
  ID-val-fit temperature** (`nll_ts`).

Two proper scores disagreeing (Brier favors IB, NLL favors CE) localizes the gap to the
**tail**: a few points where IBProbit assigns catastrophically low probability to the
true class. Brier caps each point's contribution at <= 2; NLL integrates the full
`-log p(true)`, which is unbounded. Temperature scaling (each model at its NLL-optimal
global `T`) does **not** close it, which rules out a global scale / `beta` problem.

The cause is structural to the likelihood family. IBProbit's eval logits are the CBC
transform `s = logPhi(x) - logPhi(-x)` of the moderated score `x`. For a confidently
wrong point (true-class score `x << 0`):

```
logPhi(x) ~ -x^2 / 2     (Gaussian log-tail)   =>  -log p(true) grows like x^2 / 2
```

versus softmax's exponential tail (`-log p(true)` grows ~ linearly in the logit gap). So
on confident mistakes IBProbit pays a **quadratic** log-loss where CE pays **linear**. A
scalar temperature only rescales the logit; it cannot turn a Gaussian tail into an
exponential one. A mixture floor can: `p_eps(true) >= eps/K`, so

```
-log p_eps(true)  <=  log(K / eps)        # bounded tail
```

i.e. mixing `eps` mass onto the `mu=0` uniform predictive thickens the probit tail
toward softmax's. This is the degree of freedom temperature lacks.

## Key idea: the BMR gate is already this mixture, with `eps` pinned

The committed per-point BMR gate is

```
g_i = p1_i / (p1_i + 1/2),     p1_i = Phi(x_i,true)        # binary probit prob at true class
```

This is the posterior responsibility of a per-point binary mixture "true-probit vs
fair-coin(1/2)" **with the mixing weight fixed at `eps = 1/2`**: substituting
`(1-eps) p1 / [(1-eps) p1 + eps*(1/2)]` and solving `g_i = p1/(p1+1/2)` gives `eps = 1/2`.

So:

- The **null** component (`1/K`, or coin-`1/2` per class) is *literally* the IBProbit
  head at `mu = 0` (the seed state: `x_k = 0 => softmax = 1/K`, `Phi(0) = 1/2`).
- The committed gate is the **`eps = 1/2`** special case of an ε-contamination model.
- Learning `eps` un-pins the constant: the data choose the contamination rate. A learned
  `eps` (typically << 1/2) makes the gate *less* aggressive than the fixed-1/2 gate on
  well-predicted points (seed gate `= 1 - eps`, near 1 for small `eps`).

This spec adopts the **label-level (K-way) mixture** as the principled core (one `eps`,
identical for assimilation and scoring), and recovers the committed gate as its
binary-probit approximation.

## Non-goals

- `log_sigma_z2` is still **not** learned (`sigma_z^2` fixed at its init, the moderation
  floor; see the OOD-eval port).
- No change to the `_cavi_scan` / `_cavi_scan_px` *signatures* -- they already accept a
  per-point gate `g`; only the value fed in changes (responsibility instead of the
  fixed-1/2 gate).
- The committed `bmr_gate` flag (fixed-1/2 gate, no predictive floor) stays as-is for
  back-compat; it is **not** combined with the ε-path (they parameterize the same gate).
  Subsuming `bmr_gate` into `eps=0.5` is a deferred cleanup.
- `eps = 0` + `learn_eps = False` MUST be **bitwise-identical** to the current plain
  `IBProbit` (`update` and `__call__`). Protects every existing call site.

## Design

### Mixture likelihood and responsibility (from the per-example predictive loss)

The pre-update predictive already yields, per point, the loss-type-2 softmax-CE at the
true label inside `_loss_from_logits`:

```
loss_i = -log p_model_i(y_i)      =>    p_model_i(y_i) = exp(-loss_i)
```

The mixture quantities are cheap functions of `loss_i` (no extra forward pass):

```
mixture NLL   :  nll_eps_i = -log( (1 - eps) * exp(-loss_i) + eps / K )
responsibility:  r_i       = (1 - eps) * exp(-loss_i) / ( (1 - eps) * exp(-loss_i) + eps / K )   in (0, 1)
```

`r_i` is the per-point gate fed to the (already gated) CAVI moments:

```
Lambda_post = Lambda_pred + (1/sigma_z^2) (r .* X)^T X
eta        += (1/sigma_z^2) X^T (r .* E[z])
# PX scan: nK -> K * sum_i r_i ,  S_z weighted by r_i   (exactly the committed gated path)
```

Properties (-> tests): well-predicted `loss_i -> 0` gives `r_i -> ~1` (for small `eps`);
badly predicted `loss_i -> inf` gives `r_i -> 0`; the `mu=0` seed (`p_model = 1/K`) gives
`r_i = 1 - eps` uniformly. The committed binary gate is the `Phi`-based, `eps=1/2`
approximation of this `r_i` (same `1 - eps` seed value).

### New `IBProbit` fields

State leaf (array, mirrors `log_q`):

| leaf | init | meaning |
|------|------|---------|
| `logit_eps` | `logit(eps_init)` | `eps = sigmoid(logit_eps) in (0,1)`; stepped in-update and persisted |

Static config (`eqx.field(static=True)`, mirrors the q-anchor pattern):

| field | default | meaning |
|-------|---------|---------|
| `learn_eps` | `False` | enable the in-update prequential gradient on `logit_eps` |
| `eps_init` | `0.0` | initial / fixed contamination weight |
| `eps_anchor_weight` | `1e-3` | Beta-anchor strength `lambda_eps` (`0` = off) |
| `eps_anchor_mode` | `0.02` | Beta prior mode `m_eps` that `eps` is pulled toward |

`eps` property `= jax.nn.sigmoid(self.logit_eps)`. `eps_init = 0` => `logit_eps = -inf`;
represent "off" as a guarded literal (when `eps_init == 0 and not learn_eps`, skip all ε
code so the leaf can stay at a sentinel and the path is bitwise the current model). When
`learn_eps` or `eps_init > 0`, seed `logit_eps = logit(max(eps_init, tiny))`.

`reset` / `reset_posterior` reset `logit_eps -> logit(eps_init)` (they wipe `mu -> 0`, so
the contamination estimate restarts with the posterior), alongside the existing
`mu/Sigma/step` resets.

### In-`update()` joint prequential step (`log_q`, `logit_eps`)

Extend the existing `learn_q` closure to differentiate the **mixture NLL** w.r.t. both
scalars (q only changes the moderation denominator inside `loss_i`; `eps` enters the
floor):

```python
if self.learn_q or self.learn_eps:
    def preq_obj(log_q, logit_eps):
        model = eqx.tree_at(lambda m: m.log_q, self, log_q)
        raw   = model._predictive_logits(features)
        loss  = model._loss_per_example(raw, y_oh)          # -log p_model(y) per point (loss_type)
        eps   = jax.nn.sigmoid(logit_eps)
        nll   = -jnp.log((1.0 - eps) * jnp.exp(-loss) + eps / K)   # mixture NLL per point
        obj   = nll.mean()
        if self.anchor_weight > 0.0:                         # IG anchor on q (unchanged)
            obj = obj + self.anchor_weight * (log_q + self.anchor_mode * jnp.exp(-log_q))
        if self.eps_anchor_weight > 0.0:                     # Beta anchor on eps (below)
            obj = obj + self.eps_anchor_weight * beta_pen(eps, self.eps_anchor_mode)
        return obj, (raw, loss)
    (g_q, g_e), (raw, loss) = jax.grad(preq_obj, argnums=(0, 1), has_aux=True)(self.log_q, self.logit_eps)
    if self.learn_q:   self = eqx.tree_at(lambda m: m.log_q,     self, self.log_q     - self.hyper_lr * g_q)
    if self.learn_eps: self = eqx.tree_at(lambda m: m.logit_eps, self, self.logit_eps - self.hyper_lr * g_e)
elif use_gate:
    raw  = self._predictive_logits(features)
    loss = self._loss_per_example(raw, y_oh)
```

The gate then uses the **pre-step** `eps`/`loss` (the aux from the grad point), exactly as
the committed code uses pre-step `q` for the gate; assimilation uses the **stepped** `q`.
During the first `bmr_warmup` steps the gate is held at `g = 1` (the `mu=0` seed gives
the uninformative `r = 1 - eps`), reusing the committed warmup mechanism.

Notes:
- Plain SGD, no momentum -> no optimizer state. Single `jax.grad` over `(log_q,
  logit_eps)`; `argnums=(0,1)`. Runs traced under `lax.scan` / `eqx.filter_jit`.
- The objective is the **proper scoring rule for the mixture** -- the same loss eval
  reports -- so `eps` is fit to the held-out predictive, not a heuristic.
- `_loss_per_example` is `_loss_from_logits` returning the **un-reduced** loss vector
  (today it `.mean()`s); a one-line refactor exposes the per-point loss the responsibility
  and mixture NLL both need.

### Beta anchor on `eps`

To stop `eps` collapsing to 0 (no robustness) or drifting up to absorb what `q` should do,
anchor it with a Beta prior, mode `m_eps`, analogous to the IG anchor on `q`:

```
beta_pen(eps, m_eps) = -(a - 1) log eps - (b - 1) log(1 - eps)      # -log Beta(eps; a, b), up to const
```

with `(a, b)` chosen from `(mode = m_eps, gentle concentration)` -- e.g. fix `a = 1 + c`,
`b = 1 + c*(1 - m_eps)/m_eps` for a small `c` so the mode sits at `m_eps` and the barrier
is weak. Default `m_eps = 0.02`, `lambda_eps = 1e-3` (anchor on, gentle). `lambda_eps = 0`
disables it. Identifiability note: `eps` and `q` both encode "trust in the data"; the
anchor is what keeps the joint step well-posed.

### Predictive consistency (eval must score the mixture)

When `eps > 0`, the reported predictive is `p_eps`, not the bare probit. Return the **log
mixture probabilities** as the "logits", so the downstream softmax is a no-op
(`softmax(log p) = p` for a normalized `p`) and `nll/ece/brier` are all computed on the
mixture:

```
log_p_model = log_softmax(beta * s)                                   # s = CBC transform (loss_type 2)
log_p_eps   = jnp.logaddexp( jnp.log1p(-eps) + log_p_model,           # (n, K)
                             jnp.log(eps) - jnp.log(K) )
return (loss_eps, log_p_eps)        # loss_eps = -log_p_eps[y]
```

Gate behind `eps > 0`: when `eps == 0` (and `learn_eps` off) the code path is unchanged
(bitwise). This touches `_loss_from_logits` / `__call__` only in the `eps > 0` branch.
Temperature scaling composes on top (`log_p_eps / T` then re-softmax) and stays bounded;
with a learned `eps` it should be largely redundant.

### Tail bound (the fix, restated)

`p_eps(y) >= eps/K` => `-log p_eps(y) <= log(K/eps)`. The Gaussian `-x^2/2` tail is capped
at `log(K/eps)`; this is the property a global temperature cannot provide and the reason
the mixture is the right knob for the OOD-NLL gap.

## Correctness invariants (-> tests)

1. **Regression / opt-out (bitwise):** `eps_init = 0, learn_eps = False` -> `update()` and
   `__call__(with_logits=True)` are identical to the current model on a synthetic batch.
2. **Responsibility shape:** `r_i` is monotone decreasing in `loss_i`; `loss_i -> 0` =>
   `r_i -> (1-eps)K/((1-eps)K + eps)`; `loss_i -> inf` => `r_i -> 0`; `mu=0` seed =>
   `r_i = 1 - eps` for all `i`.
3. **Committed-gate correspondence:** the binary-probit responsibility at `eps = 1/2`
   equals the committed gate `p1/(p1 + 1/2)` (and seed `= 1 - eps = 1/2`).
4. **Tail bound:** for any logits, `-log p_eps(y) <= log(K/eps) + 1e-5`.
5. **Mixture predictive normalization:** `exp(log_p_eps).sum(-1) == 1`; at `mu=0`,
   `p_eps = 1/K` (uniform) for any `eps`.
6. **Exact joint step (one batch):** with `learn_q = learn_eps = True`, anchors off, after
   one prior `update()` (nonzero `mu`), the returned `(log_q, logit_eps)` equal an
   independently computed joint SGD step on the mixture-NLL objective, **and** the returned
   `(mu, Sigma)` equal the responsibility-gated `update` evaluated at the stepped
   `(q, eps)`.
7. **Anchor pull:** with `eps_anchor_weight > 0`, `mu=0`, `logit_eps` displaced from
   `logit(m_eps)`, the gradient pushes `eps` back toward `m_eps`; `eps_anchor_weight = 0`
   recovers the no-op.
8. **PX == plain under the ε-gate:** `px=True` (few iters) vs plain (many iters) give the
   same `(mu, Sigma)` with `r_i` as the gate (the committed gated-PX `a = 1` proof carries
   over; `r_i` is just a different `g`).
9. **Runs under `filter_jit` + `lax.scan`:** `learn_q = learn_eps = True`, `bmr_warmup = 1`
   carried through `lax.scan`; finite, `step`/`logit_eps` evolve, gate activates after
   warmup.

## Experiment wiring

`scripts/finetuning.py`:
- Construct: add `learn_eps=args.learn_eps, eps_init=args.eps_init,
  eps_anchor_weight=args.eps_anchor_weight, eps_anchor_mode=args.eps_anchor_mode`.
- CLI: `--learn-eps` (store_true), `--eps-init` (default `0.0`), `--eps-anchor-weight`
  (default `1e-3`), `--eps-anchor-mode` (default `0.02`).
- Log `loss_fn.eps` per epoch alongside `loss_fn.q`.
- `utils.py` / `run_training` / `batch_update` unchanged (field-driven).

`scripts/alternatives/ood_evaluation.py`:
- Construct the IBProbit head with the ε flags; pass through `--learn-eps` etc.
- The eval already softmaxes `with_logits` output, so reported `nll/brier/ece` (and the
  `T*` variants) automatically score the mixture once `__call__` returns `log_p_eps`. Log
  the learned `eps` as `mlflow.log_metric("eps", float(loss_fn.eps))`.
- This is the direct test of the diagnosis: a learned `eps` should close the IBProbit-vs-CE
  `nll` / `nll_ts` gap (tail floored) while `acc/ece/brier` stay better.

## Deferred / tunable

- `eps_anchor_mode` / `eps_anchor_weight` values (watch the `eps` trajectory; raise the
  weight if `eps` wanders, set `0` to free it).
- Subsume the committed `bmr_gate` flag into `eps = 0.5, learn_eps = False` (back-compat
  cleanup; keep both paths for now).
- Binary-probit responsibility (`Phi`-based, reuses `raw` directly) as a cheaper gate
  variant vs the label-level `r_i` specced here.
- A **separate eval-time `eps`** fit on ID-val (like the temperature `T*`) to floor the
  residual OOD tail on top of the in-loop (ID-rate) `eps`: in-loop `eps` robustifies
  *assimilation* (the posterior over `mu`), an eval `eps` robustifies *scoring*; they
  compose.
- `log_sigma_z2` learning (still out of scope; `sigma_z^2` fixed).

## Revision (2026-07-03): conjugate Beta updating replaces the in-`update()` SGD

`eps` is a Bernoulli mixing weight and `r_i` is exactly the posterior probability that point
`i` came from the model. That is a **Beta-Bernoulli conjugate pair**, so `eps` has a
closed-form recursive update — no gradient, no learning rate, no anchor-penalty tuning.

### State (replaces "New `IBProbit` fields")

Two array leaves join the recursive posterior state (mirroring `mu`/`Sigma`, not `log_q`):

| leaf | init | meaning |
|------|------|---------|
| `eps_alpha` | `1.0` | Beta count `a` — accumulated contamination mass |
| `eps_beta`  | `1/eps_init` (default `100`) | Beta count `b` — accumulated model mass |

`eps` property `= eps_alpha / (eps_alpha + eps_beta)` (the Beta posterior **mean**). Static
config: `learn_eps`, `eps_init` (now doubles as the prior knob) and `rho` (the forgetting
factor, replacing `eps_anchor_weight`/`eps_anchor_mode`). `_eps_seed()` gives the prior
`Beta(1, 1/eps_init)` (mean ≈ `eps_init`; `eps_init<=0` falls back to `Beta(1, 100)`) when
`learn_eps`, a unit-mass point `(eps_init, 1-eps_init)` for a fixed `eps` (so `eps ==
eps_init` exactly — preserves the tail bound), and a `(1,1)` sentinel when the mixture is
off. `reset` / `reset_posterior` restore the counts to that seed, so the contamination
estimate restarts with the wiped posterior.

### In-`update()` conjugate step with forgetting (replaces the joint prequential step / anchor)

Per `update()`, with the pre-step `eps` used to form `r_i` (`_eps_responsibility`, unchanged),
and prior seed `(a0, b0)`:

```
eps_alpha = a0 + rho*(eps_alpha - a0) + sum_i (1 - r_i)   # decay excess, add contaminated count
eps_beta  = b0 + rho*(eps_beta  - b0) + sum_i r_i         # decay excess, add "from model" count
eps       = eps_alpha / (eps_alpha + eps_beta)
```

The **forgetting factor** `rho` decays the accumulated *excess over the prior* each step; the
driver scripts pass `rho = 1 - B/N` (batch `B`, train size `N`), which caps the steady-state
concentration at `a0 + b0 + B/(1-rho) = a0 + b0 + N` — a **one-pass memory**, so `eps` keeps
tracking a model that improves over epochs instead of freezing. `rho = 1` recovers unbounded
accumulation across the whole recursive pass; `rho = 0` is `prior + current batch` only.
`log_q` is still stepped by its prequential SGD (`_preq_objective`, now single-argument,
mixture NLL at the current fixed `eps`); the eps gradient / `logit_eps` leaf / `_eps_anchor_pen`
are gone.

### Assimilation gating is decoupled from the eps estimate

`r_i` gates the CAVI moments **iff `bmr_gate` is on** (previously any active mixture gated).
`r_i` is always computed when `learn_eps` (for the Beta counts), so a `learn_eps` run *without*
`bmr_gate` estimates `eps` and floors the predictive (scoring) while leaving assimilation
ungated — the posterior over `mu` equals the plain update. This makes the robust *scoring*
and robust *assimilation* roles independently switchable.

### Tunable (supersedes the `eps_anchor_*` tuning note)

- **`eps_init`** sets the prior `Beta(1, 1/eps_init)` (default `0.01` → `Beta(1, 100)`, prior
  mean ≈ 0.0099). Larger `eps_init` = a looser "more contamination expected" prior that lets
  data move `eps` up sooner; smaller = a tighter low-contamination prior.
- **`rho` = 1 − B/N** is wired automatically in every driver script (`finetuning.py`,
  `vit_classification.py`, `full_finetune_comparison.py`, `ood_evaluation.py`) from the run's
  batch size and train-set size, giving the one-pass memory. To freeze the estimate after the
  first pass instead, pass `rho = 1` (unbounded accumulation); the tests pin both regimes.
