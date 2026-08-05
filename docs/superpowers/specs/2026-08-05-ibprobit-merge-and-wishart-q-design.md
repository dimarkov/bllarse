# IBProbit: merge the process-noise head, then align it to the paper

Two commits on `ibprobit/merge-process-noise`, branched from `infra/results-protocol`.
Commit 1 ports the process-noise implementation onto this branch's base. Commit 2
rewrites the `q` machinery to the paper's target design, with an Inverse-Wishart
posterior on the full process-noise matrix in place of the spec's per-component
Gammas.

Source of truth for the target algorithm: `~/projects/papers/bllarse/q-update-spec.md`
(v3) and `REWRITE-PLAN.md`. This document records where we follow the spec, where
we generalize it, and why.

---

## 1. The problem

`infra/results-protocol` carries the process-noise *callers* but not the
*implementation*. Commit `726bc78` ("consolidate alternatives and vbll_pytorch onto
the protocol branch") brought `ood_evaluation.py`, `full_finetune_comparison.py` and
`vit_classification.py` across from `ibprobit-process-noise`; all three construct
`IBProbit` with `q_init=`, `learn_q=`, `learn_eps=`, `px=`, `bmr_gate=` and friends.
`src/bllarse/losses/bayesian.py` stayed at the pre-process-noise version, whose
constructor takes only `(input_dim, num_classes, key, use_bias, use_approx_cdf)`.

All three scripts are therefore dead on this branch:

```
IBProbit.__init__() got an unexpected keyword argument 'q_init'
```

A git merge of the two branches is not the fix. They are 5978 files and ~112k
insertions apart — `infra/results-protocol` owns the entire tuning, slurm, analysis
and results-protocol stack that `ibprobit-process-noise` never saw. The work is a
targeted port of one module and its tests.

Baseline before any change: 283 passed, 2 skipped (full suite, GPU 2).

## 2. Scope

In scope: `src/bllarse/losses/bayesian.py` (class `IBProbit` only —
`MultinomialPolyaGamma` and `IBPolyaGamma` are byte-identical on both branches),
`tests/losses/test_bayesian.py`, `tests/losses/test_cavi_dual.py`, and the argparse
surface of the three consuming scripts.

Out of scope: the VBLL comparison, the ImageNet float32 conditioning check
(q-update-spec §12.3), and any experimental run. This branch lands the
implementation; measurements are a separate task.

## 3. Commit 1 — port the process-noise head

Take `IBProbit` from `ibprobit-process-noise` wholesale: `Sigma` state, process
noise `q I`, the PX-CAVI scan, the epsilon-contamination mixture with conjugate
Beta updating, the per-point BMR gate, `reset_posterior`, `predictive_variance`,
`_predictive_logits`, `_mixture_logp`. Three reconciliations against this branch.

### 3.1 The dual CAVI form survives

`3f5e5c2` added `_prefers_dual` and a second algebraically identical CAVI form that
carries `pred (N,K)` against a precomputed hat matrix instead of `eta (d,K)`. It
wins on narrow batches with wide features. It must not be lost to the port.

The derivation carries over to the process-noise anchor unchanged. The carry is
still `eta_anchor + sigma_z^-2 * X'(g .* E[z])`, so

```
x*eta = c + sigma_z^-2 * S * (g .* E[z]),    c = x*eta_pred,  S = x*fts'
```

with `x = fts * Lambda_post^-1` exactly as now. The BMR gate enters only through the
contribution term; `x` and `S` are built from ungated `fts`, matching the primal,
where `Lambda_post` uses gated `fts` on one side only.

The dual does **not** compose with PX: PX needs `eta_hat` in d-space for
`B = sum(p .* eta_hat)`. So `_prefers_dual` is consulted only on the plain path, and
PX remains the separate scan it already is.

### 3.2 The LayerNorm goes

`e79fc20` ("IBProbit learnable prior + marginalization normalization") deliberately
removed `eqx.nn.LayerNorm` from the head and replaced it with the probit
marginalization `mu'f / sqrt(1 + f'Sf)`. q-update-spec §5.8 assumes that. So
`self.norm` is deleted and `update` / `__call__` stop calling `vmap(self.norm)`.
Per-coordinate feature scale is supplied by the backbone's own final LayerNorm,
which is what q-update-spec §5.5(e) relies on when it approximates `lambda_j ~ B`.

### 3.3 Alpha-forgetting goes

Process noise replaces it (q-update-spec §7.1: `Sigma_pred >= Q` bounds the
precision, which is what alpha was introduced to do). `update()` loses the `alpha=`
kwarg and the carried state changes from `Lambda` to `Sigma`.

### 3.4 Verification

The ~40 IBProbit tests from `ibprobit-process-noise/tests/losses/test_bayesian.py`
are ported and must pass. `test_cavi_dual.py::test_update_is_unchanged_end_to_end`
pins a numeric value against the old update and needs recomputing; the other five
dual tests are agnostic to the anchor and should pass unmodified.

After this commit the three scripts run again, under the *old* algorithm. That is
deliberate: it gives an in-branch baseline for the five-orders-of-magnitude `q`
change in commit 2, which q-update-spec §12.5 identifies as the measurement the
paper rewrite is blocked on.

## 4. Commit 2 — align to the paper

### 4.1 Deletions (q-update-spec §9)

| removed | reason |
|---|---|
| `log_sigma_z2`, `sigma_z2_init` | sigma_z^2 === 1, gauge-fixed (§5.7) |
| `log_q`, `learn_q`-as-SGD, `hyper_lr`, `anchor_weight`, `anchor_mode`, `_preq_objective` | replaced by the conjugate recursion below |
| `_gate_from_logits` | the fixed-1/2 gate has no generative meaning; the gate *is* `r_i`, always (§5.2) |
| `bmr_gate` switch | folded into "mixture on" |
| `learn_eps` switch | epsilon is now always inferred conjugately; `eps_init` survives as its prior mean |

Matching argparse flags come out of `ood_evaluation.py`,
`full_finetune_comparison.py` and `vit_classification.py`. Old sweep configs passing
them will fail loudly rather than silently do nothing.

`bmr_warmup` survives, renamed `warmup`: BMR is gone from the vocabulary and it now
defers the `q` recursion (§4.5 below) rather than the gate.

`learn_q` survives as a name but changes meaning — see §4.6.

### 4.2 Why Inverse-Wishart rather than per-component Gamma

q-update-spec §5.5 puts an independent Gamma on each `tau_j = 1/q_j`, making `Q`
diagonal. We generalize to a full `Q` with a conjugate Inverse-Wishart posterior.

This does not break the structural property the spec is careful about (§3, point 1):
`Q` still carries no class index, so `Sigma` stays shared across `k`, and the one
Cholesky per batch and K-parallel inner loop are untouched.

It strengthens the OOD argument. The spec sells per-component `q` as an "ARD-like,
axis-aligned uncertainty inflation". A full `Qhat` makes the moderation denominator
a learned Mahalanobis form, `s(f) = sqrt(1 + f'(Qhat + Sigma)f)`, which is strictly
more expressive along the same argument. It also collapses `d` independent nodes
into one, which is the honest model: feature directions under a drifting backbone do
not drift independently.

### 4.3 State and prior

Replaces the spec's `(a_q, b_q)` length-d vectors:

```
Psi          (d,d)   Inverse-Wishart scale on Q, symmetric PD
nu           scalar  Inverse-Wishart degrees of freedom
p_a, p_b     scalar  Beta on the change rate p_c   (form unchanged from spec §5.5d)
```

carried alongside the unchanged `mu`, `Sigma`, `eps_a`, `eps_b`, `step`.

The working point estimate consumed by PREDICT and by the predictive is the
Inverse-Wishart mean, which is the correct object: `Sigma_pred` is
`Cov(W_t | data_{1..t-1}) = Sigma_{t-1} + E_q(Q)[Q]` by the law of total covariance,
since `q(Q)` is independent of `q(W_{t-1})` in the variational family.

```
Qhat = Psi / (nu - d - 1)          requires nu > d + 1
```

Seeded isotropically at the spec's derived scale. With `nu_0 = d + 1 + kappa`:

```
Psi_0 = q_0 * kappa * I            so  E_0[Q] = q_0 * I
q_0   = alpha^2 / B                (q-update-spec §5.5e), alpha default 1e-3
kappa = K                          prior carries the weight of one batch
```

`q_init` overrides `q_0` so the old `1/D` regime stays reachable for the §12.5 A/B.
`nu_t` only ever grows from `nu_0` by convex combination plus `K`, so `nu > d + 1`
holds for all `t` by construction and the mean never fails to exist.

Constructor changes, relative to the ported signature:

```
added:    alpha: float = 1e-3          prior scale, q_0 = alpha^2 / batch_size
          batch_size: int              B; required when q_init is not given
          kappa: float = None          prior pseudo-count, defaults to num_classes
removed:  sigma_z2_init, hyper_lr, anchor_weight, anchor_mode, bmr_gate, learn_eps
renamed:  bmr_warmup -> warmup         default 1, so the mu=0 seed step is skipped
kept:     q_init (overrides q_0), eps_init, rho, px, loss_type, learn_q (new meaning)
```

`rho_p`, the forgetting factor on the `p_c` node, reuses `rho` rather than adding a
knob. q-update-spec §5.5(d) only requires it be fixed, and `rho = 1 - B/N` already
encodes the one-pass memory convention the `p_c` prior mean is seeded from.

### 4.4 Update equations

Written in ASCII; `'` is transpose, `*` is matmul, `.*` is elementwise.

**Sufficient statistic** (q-update-spec §5.5a, kept in matrix form). With
`Delta = mu_t - mu_{t-1}` and `Lambda_pred = (Sigma_{t-1} + Qhat)^-1`:

```
G  = Qhat * Lambda_pred                       (d,d), not symmetric
A  = Qhat - Qhat * Lambda_pred * Qhat         (= G * Sigma_{t-1}, symmetric by construction)
Cv = G * Sigma_t * G'
S_t = (G*Delta) * (G*Delta)' + K * (A + Cv)   (d,d) symmetric PSD
```

`diag(S_t)` is exactly the spec's `S_{t,j}`. Compute `A` as `Qhat - Qhat*Lambda_pred*Qhat`
rather than `G*Sigma_{t-1}`: it is symmetric by inspection and avoids differencing
near-equal matrices.

Both anchor properties survive the generalization. At `Qhat = 0`, `G = 0` so
`S_t = 0` — the estimator can represent a stationary head, which is the entire
reason for the `G`-form over the naive mean-field statistic. And `S_t >= 0`: the
first term is a rank-<=K outer product, and
`A = Qhat^(1/2) (I - Qhat^(1/2) Lambda_pred Qhat^(1/2)) Qhat^(1/2)` is PSD because
`Lambda_pred <= Qhat^-1`. So `Psi` stays PD with no floor or jitter.

**Conjugate update.** `S_t | Q ~ Wishart_d(Q, K)`, whose conjugate is the
Inverse-Wishart, so absent forgetting the update is `nu += K`, `Psi += S_t`.

**Bayes Factor Surprise** (q-update-spec §5.5b, matrix form). The
Wishart / Inverse-Wishart compound marginal is closed form:

```
log p(S | Psi, nu) = [ (K-d-1)/2 * logdet(S) - logGamma_d(K/2) ]   <- free of (Psi,nu)
                   + block(Psi, nu)

block(Psi, nu) = (nu/2) * logdet(Psi)
               + sum_{j=1..d} [ lgamma((nu+K)/2 + (1-j)/2) - lgamma(nu/2 + (1-j)/2) ]
               - ((nu+K)/2) * logdet(Psi + S)
```

The `(Psi,nu)`-free part cancels in the ratio, as does the `d(d-1)/4 * log(pi)`
constant inside the multivariate gamma. Evaluated at the **pre-update** `(Psi, nu)`:

```
log S_BF = block(Psi_0, nu_0) - block(Psi_{t-1}, nu_{t-1})
log Sbar = log S_BF / (d * K)
```

**Normalization is our call, not the spec's.** The spec divides by `K` because each
scalar node received `K` observations. Here one node absorbs `d*K` scalar
observations, and `log S_BF` grows with `d` through both logdets and the `d` lgamma
pairs. Dividing by `K` alone would saturate `gamma` at 1 for any realistic `d` —
the same trap §5.5(b) warns about one level down. `1/(d*K)` keeps `gamma` free of
both `K` and `d` and numerically comparable to the diagonal version.

**Adaptive rate and the Q update** (q-update-spec §5.5c):

```
m       = p_a / p_b
gamma_t = m * Sbar / (1 + m * Sbar)           scalar

nu_t  = (1 - gamma_t) * nu_{t-1}  + gamma_t * nu_0  + K
Psi_t = (1 - gamma_t) * Psi_{t-1} + gamma_t * Psi_0 + S_t
Qhat_t = Psi_t / (nu_t - d - 1)
```

Anchors hold unchanged: at `t = 0` the two blocks are identical, so `log S_BF = 0`,
`Sbar = 1`, `gamma = E[p_c]`. Large `gamma` sends `(Psi, nu) -> (Psi_0 + S_t, nu_0 + K)`
— prior plus this batch — so a detected change discards the stale drift estimate.

**Change rate** (q-update-spec §5.5d), seeded at `E_0[p_c] = B/N = 1 - rho`:

```
p_a = p_a0 + rho_p*(p_a - p_a0) + gamma_t
p_b = p_b0 + rho_p*(p_b - p_b0) + (1 - gamma_t)
```

*Known consequence.* The diagonal version fed `d` soft counts per step, which is why
q-update-spec §5.5(d) could say `p_c` is very well determined. One node feeds one
count per step, so `p_c` identifies roughly `d` times slower and the prior mean
`B/N` does more of the work for longer. `rho_p` bounds the memory either way, so
this is a rate-of-identification change, not a stability one. Worth logging `p_c`'s
trajectory as a diagnostic.

**Contamination responsibilities** switch from the Beta mean to the psi-form
(q-update-spec §5.2), making the update an exact stationary point of the free
energy:

```
r_i = sigmoid( E[log(1-eps)] - E[log eps] + log phat(y_i | f_i) + log K )
E[log eps] = digamma(eps_a) - digamma(eps_a + eps_b)
```

`phat` stays the pre-update, marginalised, one-step-ahead predictive — deliberately
prequential, so a point cannot certify its own inlier status.

### 4.5 Per-batch order

Follows q-update-spec §8 exactly:

```
0. PREDICTIVE  ungated, pre-update: logit = mu_{t-1}'f / sqrt(1 + f'(Qhat + Sigma_{t-1})f)
1. q(s)        r_i from the psi-form and that predictive
2. q(eps)      Beta discount-update with counts (1-r_i), r_i
3. PREDICT     Sigma_pred = Sigma_{t-1} + Qhat  (full matrix add); Lambda_pred, eta_pred
4. UPDATE      Lambda_t, eta_t with r_i gating both moments; PX or plain CAVI scan
5. S_t         G, A, Cv as above
6. SURPRISE    log S_BF from the two blocks; log Sbar = log S_BF / (d*K); gamma_t
7. q(Q)        Inverse-Wishart update at rate gamma_t -> Qhat_t
8. q(p_c)      Beta discount-update with counts gamma_t, 1 - gamma_t
```

Step 0 must be ungated — the gate is derived from it. Steps 5-8 follow step 4
because `S_t` needs `mu_t, Sigma_t`; `Qhat_t` is consumed at `t+1`. Step 6 reads the
pre-update `(Psi, nu)`, otherwise the surprise is measured against a belief already
updated to explain it.

Steps 5-8 are deferred for the first `warmup` steps (q-update-spec §12.4): at
`t = 0`, `mu = 0`, the predictive is uniform and `Delta = mu_1` is a full-size jump
off the seed, so the first batch reports a huge `S` and drives `gamma -> 1`.

### 4.6 The learn_q switch

```
learn_q = True    the recursion above
learn_q = False   Qhat === q_0 * I for all t; steps 5-8 skipped entirely,
                  Psi / nu / p_a / p_b never touched
```

Off is exactly the fixed-`q` algorithm with `q_0` in place of `1/D`. This is the
clean ablation REWRITE-PLAN §7 (Risk 3) asks for — isolate one component, cheapest
being with/without learned `q` — and the cheap arm of the §12.5 A/B.

### 4.7 Cost

Two extra `d x d` Choleskys per batch, for `logdet(Psi_0 + S)` and
`logdet(Psi_{t-1} + S)`, plus three or four `d x d` matmuls for `S_t`.
`logdet(Psi_0)` is a constant and `logdet(Psi_{t-1})` is carried.

This is same-order-as, not negligible-against, the Cholesky already being done —
roughly 3x the per-batch dense linear algebra. At `d = 1281` that is ~1.5 GFLOP and
irrelevant on GPU, but it is no longer the "handful of lgamma calls" the diagonal
spec advertised. **The paper's complexity table must say so.**

## 5. Testing

Commit 1 is verified by the ported suite passing, minus the one recomputed pin.

Commit 2 deletes the tests for deleted features (5 `learn_q`-SGD tests, 9
`bmr_gate` tests) and adds tests written from the spec's own anchors, which are
unusually testable:

1. `S_t == 0` when `Qhat == 0` — the property the naive mean-field statistic fails,
   and the reason for the `G`-form.
2. `S_t` symmetric and PSD for random states.
3. At `t == 0`, `log S_BF == 0`, hence `Sbar == 1` and `gamma == E[p_c]`.
4. Isotropic reduction: with `Psi` and `Sigma` isotropic, the matrix `S_t` diagonal
   and the resulting `Qhat` agree with the scalar recursion of q-update-spec §5.5
   computed directly. This pins the generalization against the spec.
5. Prior seeding: `E_0[Q] == q_0 * I` for the chosen `(kappa, nu_0, Psi_0)`.
6. `Psi` stays PD and `nu > d + 1` over a multi-step run.
7. Marginalization is exact (q-update-spec §5.8): Monte-Carlo
   `E_{w ~ N(mu, Sigma + Qhat)}[Phi(w'f)]` matches
   `Phi(mu'f / sqrt(1 + f'(Sigma + Qhat)f))` to sampling error. This is the paper's
   central claim and the one worth pinning directly. (Note the gauge-invariance
   property of §5.7 is *not* testable after commit 2 — deleting `sigma_z2` is what
   fixes the gauge, so there is no longer a free scale to vary. It is an argument
   about the model, discharged by the deletion itself.)
8. Contraction (q-update-spec §6): under a stationary head `Qhat` decreases. The
   contraction is cubic in the Kalman gain, so it is invisible at realistic drift
   rates — the test must construct a large-gain state deliberately, and should say
   so, rather than running a long realistic loop and hoping.
9. `learn_q=False` holds `Qhat` at `q_0 * I` and leaves `Psi`, `nu`, `p_a`, `p_b`
   untouched across steps.
10. Beta updates for `eps` and `p_c` match their closed forms; psi-form `r_i`
    matches a direct digamma computation.

Retained from the port: PX shares the plain-CAVI fixed point; dual matches primal;
probit-Gaussian marginalization is exact; the whole update runs under `scan` + `jit`.

## 6. Risks

**The five-orders-of-magnitude `q` change.** q-update-spec §5.5(f) predicts
moderation gets much weaker at `q_0 = alpha^2/B` (the `q||f||^2` floor drops from
~1 to ~5e-6) and warns that any in-distribution calibration currently coming from
that floor is temperature scaling in disguise. ECE will move. Commit 1 exists
precisely so the comparison can be made in-branch. This is a measurement task, not
a bug.

**`nu > d + 1` and the Inverse-Wishart mean.** Guaranteed by construction here, but
it is a real constraint that the diagonal formulation did not have. If a future
change lets `nu` be reset below `d + 1` the mean silently blows up; the test at §5.6
guards it.

**Conditioning.** q-update-spec §7.1 flags that the precision bound scales as
`1/Qhat`, reaching ~2.6e8 at the new prior, at the edge of float32. Unchanged by the
Wishart generalization, still needs the ImageNet check before the paper claims
alpha-forgetting can go. Out of scope here.
