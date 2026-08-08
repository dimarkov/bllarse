# ImageNet-1k full-network fine-tuning: tuning campaign

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure for, and launch, an Optuna tuning campaign
over four fine-tuning arms on ImageNet-1k — under a fixed 2M-sample budget with
evaluation spaced every 16,384 samples.

**Architecture:** Amends `docs/superpowers/specs/2026-08-06-full-network-finetuning-design.md`
§8. The probing campaign's sample-spaced evaluation protocol (commit `dfeccfe`)
is ported to fine-tuning; the log-spaced `dump_steps` grid is retained only for
the final Fig 5/6/7 campaign, which is out of scope here.

**Tech Stack:** JAX/equinox (three arms), PyTorch/HF-transformers (VBLL arm),
Optuna TPE on `JournalStorage`, SLURM on Raven.

---

## Amendments to the 2026-08-06 spec

Recorded because each contradicts something the approved spec says.

**A1. Tuning budget is 2,000,000 samples, not `--epoch-fraction 0.5`.**
2M / 1,281,167 = **1.561 epochs** (the spec assumed 0.5 = 640,583). Steps are
still derived from the tuned effective batch, so D4 (fixed samples-seen, not
fixed steps) is unchanged — only the constant moves.

**A2. Evaluation is spaced every 16,384 samples, replacing the log-spaced
`dump_steps` grid.** 2,000,000 / 16,384 = **122 evaluations per trial**,
identical for every rung of the batch ladder. This is the same defect and the
same fix as probing commit `dfeccfe`: on a geometric grid one "count" means
~150 steps early and ~7,000 late, so it is unusable as a pruner key, a patience
clock, or an argmin search. Every ladder rung (256…4096) divides 16,384, so the
grid is realisable as a whole number of steps everywhere.

**A2b. The geometric grid is removed from the codebase entirely — fine-tuning
AND linear probing.** No surviving caller, no fallback, no "only for logging".

`evaldump.dump_schedule` (`np.geomspace`) is **deleted**, along with
`protocol.dump_steps`, `protocol.ood_dump_steps`, and the `report_grid` in
`scripts/dinov3/probe.py:357` and `scripts/vbll_pytorch/probing_vbll.py:218`.
Everything that needed a schedule now takes a **stride of the sample-spaced
evaluation index**, which is uniform in samples by construction:

| was | becomes |
| --- | --- |
| `dump_steps` (geomspace over steps) | `--dump-every-k-evals` |
| `ood_dump_steps` (stride of the above) | `--ood-every-k-dumps`, now a stride of a uniform grid |
| `report_grid` (LP MLflow/stdout throttle) | `eval_index % report_every == 0` |

The final Fig 5/6 campaign therefore gets a *uniform* trajectory rather than a
log-spaced one. That is the correct axis anyway: results-plan puts samples-seen
on Fig 5's x-axis, and a geometric grid puts most of its resolution where the
x-axis has least.

**Deployment note.** The LP change touches `probe.py` while the Stage 4 campaign
is running on Raven. It is safe scientifically — `report_grid` throttles MLflow
and stdout only, and never touched evaluation, dumps, scoring or early stopping
— but it must NOT be deployed to the Raven checkout mid-campaign, or runs
before and after the pull would log at different step sets. Commit it; pull it
after Stage 4 drains.

**A3. Tuning evaluates val ONLY; the 40k test partition is never forwarded.**
`val_fraction` rises from 0.2 to **0.4**, giving 20k val / 30k test, and tuning
forwards only the 20k. This is what makes the eval overhead ~41% instead of
~100%: at 122 evals a 50k eval set would cost 6.1M forwards against 2M training
samples.

**A4. The fine-tuning objective scores the minimum val_nll observed in the run's
own evaluation stream, NOT `early_stop(run_curve(...))`.**

`objective.py`'s docstring makes a point of the score BEING
`analysis.thresholds.early_stop` so the two cannot drift. That property is
preserved, not abandoned: `early_stop(curve, "val_nll")` is defined as the
argmin of val_nll over finite rows of the curve, and the eval stream is that
curve. The equivalence is exact, minus one fp16 dump round-trip.

The change is forced by A3. `curves.run_curve` reads `test_record.meta` (see
`analysis/curves.py:52`), so it raises on a run that has no test dump — and
producing one would mean forwarding 30k images at a step we cannot identify
until the run is over.

**A5. `wishart_scale` is pinned at 0.13 on the VBLL arm; every other head knob
is searched in a probing-narrowed box.** See "Head parameter transfer" below.

**A5b. `patience = 20` evaluations, overriding spec §8's "patience off".**

§8 turned patience off on the reasoning that "a 0.5-epoch trial is truncated by
construction, so patience fires on noise or never." A2 removes the premise: 122
uniformly sample-spaced evaluations over 1.56 epochs is a real signal, not the
bounded geometric grid that made the probing campaign's `--patience 20`
structurally unable to fire (it could only fire if the argmin landed at dump
<= 19; verified on 133/133 trials).

20 evaluations = **327,680 samples** (0.256 epochs) with no 0.1% relative
improvement in val_nll. With `warmup_evals = 10` the earliest possible stop is
eval 30 = 491,520 samples, so a doomed config costs ~1.5 h instead of 6.3 h.
The configs this is aimed at are trunk learning rates high enough to cause
catastrophic forgetting, which plateau or diverge well inside the first epoch.

`patience` is NOT the pruner and the two are not redundant: `MedianPruner`
kills trials that are LOSING against their cohort, patience stops trials that
have FINISHED CONVERGING — whose score is a perfectly good result. That is why
`EarlyConvergence` is deliberately not `optuna.TrialPruned`.

Read `patience_stopped_at_eval` and `truncated` off the journal after the first
worker batch. `patience_stopped_at_eval == {'200': 30}` on many trials means 20
is too tight; `truncated` true nearly everywhere means the 2M budget is the
binding constraint instead, and the two call for opposite corrections.

**A6. A fourth arm, `ibprobit_noeps`, joins the three the spec names.** Free —
it is `--arm ibprobit --eps-init 1e-12`, exactly as on the probing side
(`space.IBPROBIT_NOEPS_INIT`).

**A8. The micro-batch is measured once and PINNED per arm; the per-trial
auto-probe is removed.**

The ladder is (256, 512, 1024, 2048, 4096), so `gcd = 256`. For `num_accum` to
be an exact integer at *every* rung the micro-batch must be a power of two
`<= 256`. The rule is therefore:

```
MICRO_BATCH[arm] = min(2 ** floor(log2(largest_that_fits)), 256)
```

— the largest measured-fitting micro-batch, floored to a power of two, capped
at the ladder's gcd. If the largest that fits exceeds 256 we take 256 anyway
(the accumulation is then 1 micro-step at the bottom rung), because a
micro-batch that does not divide the smallest macro batch is unusable there.

Three things this buys:

1. **`effective == target`, always.** `accum_plan` currently computes
   `num_accum = max(1, round(target / micro))`, so a non-dividing micro-batch
   silently yields an effective batch that is not the tuned one — and its own
   docstring warns callers to log the returned value for exactly that reason.
   With a pinned divisor the rounding branch is dead and becomes an assertion.
2. **Trials stop being nondeterministic in cost.** The auto-probe doubles until
   it OOMs, once per run, on a GPU shared with nothing but luck. 140 trials pay
   that; a constant pays it once.
3. **It closes §11's last IBProbit item.** `--micro-batch-size` is required for
   `--arm ibprobit` only because that arm has no auto-probe. With the value
   pinned, no arm needs one.

Per-arm rather than global because the three arms have different activation
profiles and two different frameworks. IBProbit's *head* state is not the
constraint — `Sigma` is `(d, d)` shared across classes, so at D=1281, K=1000 the
whole head is ~20 MB — but its micro forward retains features for the CAVI
update, which CE's does not.

Measured on Raven's `--gres=gpu:a100:1`, not locally: the pinned constant is
only valid for the hardware it was measured on, and a different GPU would make
every trial OOM or leave memory unused.

**A7. §11's two IBProbit blockers are closed** and the spec section is stale.
`finetune_jax.main` now builds `IBProbit` with the full knob set
(`q_init`, `alpha`, `kappa`, `learn_q`, `px`, `warmup`, `eps_init`, `rho`), and
dumps `logit_kind="softmax_logits"`, which `run_curve` accepts. What remains
open from §11 is only the missing ibprobit micro-batch auto-probe
(`--micro-batch-size` stays required for that arm).

## Head parameter transfer (answers the launch question)

Only `wishart_scale` transfers as a value. Spread across the five probing
datasets is the evidence — a knob that is a property of the *head* agrees
across datasets; one that is a property of the *dataset* does not.

| knob | probing optima | spread | decision |
| --- | --- | ---: | --- |
| `wishart_scale` | 0.076 – 0.204 | 2.7x | **pin 0.13** |
| `eps_init` | 1.1e-3 – 3.3e-2 | 30x | box `[1e-6, 1e-1]` (floor dropped 3 decades) |
| `kl_ratio` | 2.4e-7 – 7.1e-5 | 300x | box `[1e-7, 1e-4]` (narrowed from 1e-2) |
| `q_init` | 2.4e-5 – 4.2e-3 | 175x | box `[1e-5, 1e-2]` (narrowed from 1e-6/1e-1) |
| `prior_scale` | 0.14 – 352 | 2500x | box `[1e-1, 1e3]` unchanged |
| head `learning_rate` | 1.5e-3 – 2.8e-2 | 19x | not transferable; reparameterized |

Two reasons not to push past `wishart_scale`:

1. **The spreads are wider than the narrowed boxes.** Pinning any of them
   transplants one probing dataset's answer under the name of a head property.
2. **The frozen-feature geometry does not survive fine-tuning.** Probing's head
   is conjugate against a *fixed* feature second moment. Here the trunk
   co-adapts, so `prior_scale`, `q_init` and the head learning rate are all
   defined against a distribution that stops existing after step 1.

**The `eps_init` floor drops from 1e-3 to 1e-6**, three decades below anything
probing selected. This buys a diagnostic the `ibprobit_noeps` ablation cannot
produce on its own: if the tuned `eps_init` lands on or near the 1e-6 floor,
the *searched* arm has answered the ablation's question by itself — the head
chose to switch contamination off when it was free to keep it. The pinned arm
stays at `IBPROBIT_NOEPS_INIT = 1e-12`, still six decades below the new floor,
so "searched but tiny" and "numerically absent" remain distinguishable.

`num_update_iters` pinned to its ceiling (192) in **4 of 5** IBProbit studies.
That is a signal that the ladder is too low, not a value to adopt: the ladder
moves up to **(128, 192, 256)** and stays searched. 64 and 32 are dropped —
the PX operating point is ~128 iterations (it matches ~1000 plain iterations
there), so rungs below it are known-underconverged and would spend trials
measuring an unconverged head rather than the method. (On `ibprobit_noeps` only
2 of 5 studies hit the ceiling, which is why it stays an axis rather than being
pinned for symmetry.)

Head learning rate keeps the spec's reparameterization — `trunk_lr` x
`head_lr_ratio`, not two independent rates — because AdamW moves weights by
~lr/step and a fresh head needs ~100x the trunk rate, so independent axes spend
trials sliding along that diagonal. Probing's absolute head rates imply ratios
of 15–280 at a trunk_lr of 1e-4, comfortably interior to `[1, 1e3]`.
**IBProbit has no head learning rate at all** — the head is assimilated in
closed form — which is a Fig 7 result, not a gap.

## Global Constraints

- `IMAGENET_N_TRAIN = 1_281_167`; tuning budget `MAX_SAMPLES = 2_000_000`.
- `EVAL_QUANTUM = 16_384`. Every batch on the ladder MUST divide it exactly;
  a non-integral ratio raises rather than rounding.
- Batch ladder: `(256, 512, 1024, 2048, 4096)`, effective (accumulated).
- `MICRO_BATCH[arm]` is a pinned power-of-two `<= 256`, measured once on Raven
  (A8). `accum_plan` requires exact divisibility and raises otherwise.
- NO geometric grid anywhere (A2b). `evaldump.dump_schedule` is deleted.
- `img_size = 224`. 512 is refused unless `--allow-non-224`.
- `val_fraction = 0.4` -> 20,000 val / 30,000 test.
- Head weight decay is 0 on every arm (spec D6).
- One tuning seed per trial (spec §8), seed 200.
- `patience = 20` evaluations, `warmup_evals = 10`,
  `min_relative_improvement = 1e-3`.
  `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`.
- OOD disabled during tuning (`--ood-every-k-dumps 0`).
- Tuning writes exactly ONE dump per run: the buffered argmin-val_nll val dump.
- Campaign namespace: `imagenet_ft`. `open_study` loads by name, so an
  un-namespaced study would resume a differently-shaped search space.

## Cost

Anchored on the spec §9 figure of 11.4 h per 4-epoch dinov3_huge run
(5.125M samples) = 2.224 h/Msample training; eval forward-only is ~1/3 that.

| | per trial |
| --- | ---: |
| training, 2.0M samples | 4.45 h |
| eval, 122 x 20,000 = 2.44M forwards | 1.81 h |
| **total** | **~6.3 h** |

140 trials (CE 30, VBLL 50, IBProbit 30, IBProbit-noeps 30) = **~880 A100-h**
as a CEILING — that is the cost with every trial running its full 2M samples.
`patience = 20` (A5b) and `MedianPruner` both cut into it, by an amount that
cannot be predicted before the first trials report; a patience stop at the
earliest possible eval costs ~1.5 h instead of 6.3 h.

At Raven's `MaxJobs=8` the ceiling is ~4.6 days wall-clock if perfectly packed.
Size worker tasks against the ceiling, not the hoped-for average: a 24 h SLURM
job fits **3 full-length trials**, so tasks run `--n-trials 3` and each study
needs 10–17 of them, drip-fed by the existing `stage4_submit.sh` pattern. A
task that finishes its three trials early simply exits.

Dump volume: one 20,000 x 1000 fp16 val dump = 40 MB/run, 5.6 GB total. Without
the argmin buffer it would be 122 x 40 MB = **4.9 GB per run**.

## Out of scope

The final Fig 5/6/7 campaign protocol — its epoch cap, eval quantum, dump
stride, OOD stride and seed count. Two notes for that decision: evaluating
every 16,384 samples over a 4-epoch run would be 313 evals x 50k images =
**~100% overhead**, so the final runs need a larger quantum than tuning does;
and per A2b that grid will be uniform in samples, with `--dump-every-k-evals`
selecting the Fig 5 trajectory points.

---

## File Structure

| file | responsibility |
| --- | --- |
| `scripts/finetune/probe_micro_batch.py` | measure the per-arm micro-batch ceiling (new, A8) |
| `src/bllarse/evaldump.py` | **&minus;** `dump_schedule` (A2b) |
| `scripts/dinov3/probe.py`, `scripts/vbll_pytorch/probing_vbll.py` | **&minus;** `report_grid`; throttle on eval index (A2b) |
| `src/bllarse/finetune/protocol.py` | + `steps_for_samples`, `eval_every_steps`, `EVAL_QUANTUM`, `MAX_SAMPLES`; **&minus;** `dump_steps`, `ood_dump_steps`; `accum_plan` exact |
| `scripts/finetune/finetune_jax.py` | + `--max-samples`, sample-spaced eval, buffered argmin dump, 4-arg `on_dump`, `--eval-splits` |
| `src/bllarse/tuning/finetune_space.py` | ladder, boxes, `suggest`, `ARMS` |
| `src/bllarse/tuning/objective.py` | + `seeds=`, `scripts=`, `score=` kwargs (probing behaviour bit-identical) |
| `src/bllarse/tuning/finetune_worker.py` | Optuna worker for one (arm) study |
| `src/slurm/jobs/slurm_finetune_tune.sh` | SLURM entry point |
| `scripts/finetune/finetune_vbll.py` | PyTorch arm (new) |
| `scripts/finetune/check_backbone_parity.py` | HF-vs-JAX trunk parity (new) |
| `src/slurm/images/bllarse-torch.def` | + `transformers` |

---

### Task 1: `protocol.py` — sample budget and sample-spaced eval grid

**Files:**
- Modify: `src/bllarse/finetune/protocol.py`
- Test: `tests/finetune/test_protocol.py`

**Interfaces produced:**
- `MAX_SAMPLES: int = 2_000_000`
- `EVAL_QUANTUM: int = 16_384`
- `steps_for_samples(max_samples: int, effective_batch: int) -> int`
- `eval_every_steps(effective_batch: int, quantum: int = EVAL_QUANTUM) -> int`
- `accum_plan(target, micro)` — now raises unless `target % micro == 0`

**Removed:** `dump_steps`, `ood_dump_steps` (A2b).

- [ ] **Step 1: failing tests** — `steps_for_samples(2_000_000, 1024) == 1953`;
  monotone decreasing in batch; raises at 0 steps.
  `eval_every_steps(1024) == 16`; `eval_every_steps(4096) == 4`;
  `eval_every_steps(768)` raises (768 does not divide 16,384);
  every rung of the ladder yields `2_000_000 // 16_384 == 122` evaluations.
  `accum_plan(1024, 256) == (256, 4, 1024)`; `accum_plan(1024, 768)` raises.
  Delete the `dump_steps`/`ood_dump_steps` tests
  (`tests/finetune/test_protocol.py:82-104`).
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** Both new functions raise rather than rounding.
  `eval_every_steps`: a rounded grid gives per-trial `samples_seen` values,
  which is exactly the defect the quantum exists to remove but now invisible.
  `accum_plan`: rounding silently yields an effective batch that is not the
  tuned one, which then mislabels every point of Fig 5's samples-seen axis.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit**

---

### Task 1b: delete the geometric grid (fine-tuning AND linear probing)

**Files:**
- Modify: `src/bllarse/evaldump.py` (delete `dump_schedule`)
- Modify: `scripts/dinov3/probe.py:357,495`, `scripts/vbll_pytorch/probing_vbll.py:218,332`
- Test: `tests/test_evaldump.py` (delete its four `dump_schedule` tests)

**DO NOT deploy to the Raven checkout until Stage 4 drains** — see A2b.

- [ ] **Step 1: failing test** — `grep -rn "geomspace\|dump_schedule"` over
  `src/` and `scripts/` returns nothing. Add it as a real test so the grid
  cannot come back by accident.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** In both LP scripts, replace
  `report_grid = set(evaldump.dump_schedule(...))` / `if step in report_grid`
  with a stride on the evaluation index: `if eval_index % args.report_every == 0`.
  `--n-dumps` becomes `--report-every` (default 1; these are MLflow/stdout
  throttles only, and have not driven evaluation, dumps, scoring or early
  stopping since commit `dfeccfe`).
- [ ] **Step 4: run the LP test subset, verify pass**
- [ ] **Step 5: commit**

---

### Task 2: `finetune_jax.py` — sample budget, sample-spaced eval, buffered dump

**Files:**
- Modify: `scripts/finetune/finetune_jax.py`
- Test: `tests/finetune/test_finetune_jax_plan.py`, `test_finetune_jax_contracts.py`

**Interfaces produced:**
- `--max-samples INT` (overrides `--epochs` when set)
- `--eval-quantum INT` (default `protocol.EVAL_QUANTUM`)
- `--eval-splits {val,val+test}` (default `val+test`; tuning passes `val`)
- `--dump-every-k-evals INT` (default 0 = buffered argmin only). Replaces
  `--n-dumps`; `--ood-every-k-dumps` now strides a uniform grid.
- `on_dump(eval_index, step, samples_seen, scalars)` — 4-arg, matching probing
- `main` returns `None`; the argmin val_nll is delivered through `on_dump`

**Removed:** `probe_micro_batch_size_ce`, `check_micro_batch_required`, and the
auto-probe branch in `main`. `--micro-batch-size` defaults to
`finetune_space.MICRO_BATCH[arm]` (A8) and both arms treat it identically, so
§11's "required for `--arm ibprobit`" caveat is gone.

- [ ] **Step 1: failing tests.** `resolve_plan` with `--max-samples 2000000`
  `--batch-size 1024` gives `steps == 1953`, `eval_every == 16`, `n_evals == 122`.
  `--eval-splits val` produces no test dump. A run whose val_nll is non-finite
  at the first eval does not latch (regression guard for the nan-latch bug,
  commit `3ddbc85`). `_write_best` is reached when `on_dump` raises.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** Mirror `scripts/dinov3/probe.py` exactly:
  - `eval_every = protocol.eval_every_steps(plan["effective_batch"], quantum)`
  - inside the loop, `if step % eval_every: continue`
  - `eval_index = samples_seen // quantum`
  - `if math.isfinite(val_nll) and (best is None or val_nll < best["val_nll"])`
    — the `isfinite` guard is load-bearing, not defensive: without it a
    non-finite FIRST value becomes `best` and every later `v < nan` is False.
  - `try: ... finally: _write_best(args, best, eval_index)`. This is the
    ordinary path, not an edge case: with `patience = 20` (A5b) an
    `EarlyConvergence` raised out of `on_dump` escapes `main` on every
    converged trial, and a `_write_best` outside the `finally` would silently
    produce no dump for exactly those runs.
  - `at_last_eval=best["eval_index"] == last_eval_index` recorded at write time.
  - `dump_set` / `ood_step_set` are built from eval-index strides, not from
    `protocol.dump_steps` (deleted in Task 1).
  Keep the existing non-finite-val_nll `RuntimeError` divergence guard, but move
  it after the `best` update so a single non-finite point does not discard a
  good buffered dump. Delete the `dump_steps`/`ood_steps` plan tests in
  `tests/finetune/test_finetune_jax_plan.py:80-89`,
  `test_finetune_jax_contracts.py:205`, `test_finetune_jax_ood.py:54,372-383`
  and rewrite them against the uniform grid.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit**

---

### Task 2b: `probe_micro_batch.py` — measure and pin the micro-batch

**Files:**
- Create: `scripts/finetune/probe_micro_batch.py`
- Test: `tests/finetune/test_probe_micro_batch.py`

**Interfaces produced:**
- `floor_to_ladder_divisor(probed: int, gcd: int = 256) -> int`
  = `min(2 ** floor(log2(probed)), gcd)`

**Blocks Task 3** — its output is `finetune_space.MICRO_BATCH`.

- [ ] **Step 1: failing tests** (pure function; no GPU needed for these):
  `floor_to_ladder_divisor(300) == 256`; `(200) == 128`; `(64) == 64`;
  `(100) == 64`; `(1000) == 256`. Result always divides every rung of the
  ladder exactly.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** Reuse the existing doubling/OOM-substring/
  `jax.clear_caches()+gc.collect()` structure of
  `finetune_jax.probe_micro_batch_size_ce`, but run it for the ibprobit arm too
  (the constructor knobs it needs have landed) and report
  `jax.local_devices()[0].memory_stats()` alongside, so the pinned constant
  carries the device it was measured on.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: run it on Raven** (`--gres=gpu:a100:1`, dinov3_huge, 224, bf16)
  for `ce` and `ibprobit`; record both numbers here before Task 3 hardcodes
  them. The PyTorch number is measured in Task 7 against the same rule.

---

### Task 3: `finetune_space.py` — the search space

**Files:**
- Create: `src/bllarse/tuning/finetune_space.py`
- Test: `tests/tuning/test_finetune_space.py`

**Interfaces produced:**
- `ARMS = ("ce", "vbll", "ibprobit", "ibprobit_noeps")`
- `BATCH_SIZES = (256, 512, 1024, 2048, 4096)`
- `TUNING_MODEL = "dinov3_huge"`, `TUNING_SEED = 200`
- `WISHART_SCALE = 0.13`, `IBPROBIT_NOEPS_INIT = 1e-12`
- `IBPROBIT_UPDATE_ITERS = (128, 192, 256)`
- `EPS_INIT_BOX = (1e-6, 1e-1)`
- `MICRO_BATCH: dict[str, int]` — from Task 2b's Raven run
- `suggest(trial, arm) -> dict[str, Any]`

Axes per arm: CE 4, VBLL 6, IBProbit 6, IBProbit-noeps 5.

- [ ] **Step 1: failing tests.** Every rung of `BATCH_SIZES` divides
  `EVAL_QUANTUM`, and every `MICRO_BATCH` value divides every rung.
  `suggest` returns no `head_lr_ratio` for either ibprobit arm
  (no head optimizer exists — accepting one would present a tuned axis that
  silently does nothing). `eps_init` is absent from `ibprobit_noeps`'s
  suggestions, not sampled-and-overridden — a sampled axis that never reaches
  the script makes TPE model a dimension the runs do not have. `ARMS` is pinned
  against `finetune_worker.SCRIPTS`.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement**, with boxes from the transfer table above.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit**

---

### Task 4: `objective.py` — `seeds=`, `scripts=`, `score=` kwargs

**Files:**
- Modify: `src/bllarse/tuning/objective.py`
- Test: `tests/tuning/test_objective.py`

**Interfaces produced:**
- `evaluate(..., seeds=TUNING_SEEDS, scripts=SCRIPTS, score=score_dumps)`

All three default so probing behaviour is bit-identical.

- [ ] **Step 1: failing tests.** Probing call path unchanged (existing tests
  must pass untouched). `seeds=(200,)` runs exactly one seed. A custom `score`
  callable receives the run uids and its return value is the objective.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement**
- [ ] **Step 4: run full `tests/tuning/`, verify pass**
- [ ] **Step 5: commit**

---

### Task 5: `finetune_worker.py` — the Optuna worker

**Files:**
- Create: `src/bllarse/tuning/finetune_worker.py`
- Test: `tests/tuning/test_finetune_worker.py`

**Interfaces consumed:** Task 3's `suggest`, Task 4's `evaluate` kwargs.

- [ ] **Step 1: failing tests.** `trial_config` sets `max_samples=2_000_000`,
  `eval_quantum=16_384`, `eval_splits="val"`, `ood_every_k_dumps=0`,
  `val_fraction=0.4`, `img_size=224`, `model="dinov3_huge"`.
  `arm="ibprobit_noeps"` maps to script flag `--arm ibprobit` **and**
  `eps_init=1e-12` (the script's argparser accepts only `{ce, ibprobit}`, so
  passing the study name through would abort every trial at argparse).
  `arm="ce"` carries `head_lr = trunk_lr * head_lr_ratio`; both ibprobit arms
  carry none. `study_name` includes the campaign suffix.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** `PRUNER_STARTUP_TRIALS = 5` (spec §8: 20 was a
  fifth of ~80 probing trials; at 30 it would barely engage, and pruning is
  where the saving is at 6 h/trial). `patience=20` (A5b), `warmup_evals=10`.
  Objective is `min(val_nll)` over the eval stream (amendment A4) — a
  patience-stopped trial still scores its argmin, which is why
  `EarlyConvergence` must not be caught as a failure. Record `argmin_eval`,
  `total_evals`, `truncated`, `patience_stopped`, `patience_stopped_at_eval`
  user attrs; those last two and `truncated` are what say whether patience 20
  and the 2M budget are set sensibly, and they call for opposite corrections.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit**

---

### Task 6: SLURM entry point and launch

**Files:**
- Create: `src/slurm/jobs/slurm_finetune_tune.sh`
- Test: `tests/tuning/test_finetune_tune_job_script.py`

- [ ] **Step 1: failing test** — script exists, is executable, passes
  `--campaign imagenet_ft`, `--n-trials 3`, and `--patience 20`.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement**, modelled on `slurm_tune.sh` (env-var parameterized,
  SIF via `BLLARSE_SIF`). `--time=24:00:00`, `--n-trials 3`,
  `--patience "${BLLARSE_FT_PATIENCE:-20}"`.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit, then launch the three JAX arms** (CE, IBProbit,
  IBProbit-noeps = 90 trials) through the capacity-aware drip-feed submitter.

---

### Task 7: `finetune_vbll.py` — the PyTorch arm

**Files:**
- Create: `scripts/finetune/finetune_vbll.py`
- Modify: `src/slurm/images/bllarse-torch.def` (+ `transformers`, `datasets`)
- Test: `tests/finetune/test_finetune_vbll.py`

Trunk is HF `transformers` DINOv3 (`facebook/dinov3-*-pretrain-lvd1689m`, gated
— needs the existing `HF_TOKEN`). Same released weights as the JAX arm,
different implementation, so parity is verified in Task 8 rather than assumed.

- [ ] **Step 1: failing tests.**
  - **H1 (spec §6):** one accumulated step equals one full-batch step to
    tolerance. `vbll`'s loss is `NLL + regularization_weight * KL` computed
    per micro-batch; averaging over micro-batches leaves the KL correctly
    weighted, summing multiplies it by `num_accum` — up to 16x — silently
    retuning `kl_ratio` past the whole searched range.
  - **H2:** the head param group has `weight_decay == 0` while the trunk
    carries `--weight-decay`, and the two groups carry different `lr`.
  - Batch order matches the JAX arm: `protocol.shuffled_indices(n, seed, epoch)`
    is the shared numpy permutation, so both frameworks consume an identical
    stream.
  - Same `--max-samples` / `--eval-quantum` / argmin-buffered dump contract as
    Task 2, verified by asserting the same `(steps, eval_every, n_evals)`.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** `(loss / num_accum).backward()` in the micro loop,
  one `optimizer.step()` per macro batch, `torch.autocast` bfloat16 to match
  the JAX arm's activation precision.
- [ ] **Step 4: run, verify pass**
- [ ] **Step 5: commit**

---

### Task 8: `check_backbone_parity.py` and launch VBLL

**Files:**
- Create: `scripts/finetune/check_backbone_parity.py`
- Test: `tests/finetune/test_check_backbone_parity.py`

- [ ] **Step 1: failing test** — verdict fails on injected divergence, passes
  on identical features.
- [ ] **Step 2: run, verify fail**
- [ ] **Step 3: implement.** Forward the pinned eval set through the PyTorch
  trunk and compare against the JAX `.npz` cached features: per-sample cosine
  similarity plus delta linear-probe accuracy. Pooling and final-LayerNorm
  differences between implementations are the likely failure and are cheap to
  find this way rather than after a 6 h trial.
- [ ] **Step 4: run, verify pass; then run the real parity check on Raven**
- [ ] **Step 5: commit, then launch the VBLL arm** (50 trials).
