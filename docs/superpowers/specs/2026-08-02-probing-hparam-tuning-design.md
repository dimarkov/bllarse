# Hyperparameter tuning for the probing arms

**Date:** 2026-08-02
**Status:** design, approved
**Scope:** VBLL and AdamW+CE linear probing (Fig 1 and Fig 2). Full-network
finetuning is explicitly out of scope and will be retuned separately.

## Problem

Neither probing arm's hyperparameters were ever chosen. `probe_vbll.py` and
`probe_adamw_ce.py` restate the argparse defaults, which are themselves
`build_vbll_head`'s defaults. Three pieces of evidence say that is not tenable.

**1. The AdamW learning rate is at the bottom of the range the project already
searched.** `scripts/alternatives/run_vit_classifciation_ce_sweep.sh` sweeps
`CE_LRS=(1e-2 5e-3 1e-3)`. Both probing sweeps hardcode `1e-3` — the smallest of
the three.

**2. VBLL's defaults collapse outside the regime they were picked in.** Measured
on Raven, `dinov3_huge`, 40 epochs:

| dataset | classes | D | test_acc | test_nll | test_ece |
|---|---:|---:|---:|---:|---:|
| food101 (AdamW+CE) | 101 | 1280 | 0.9541 | 0.1879 | 0.0340 |
| food101 (VBLL) | 101 | 1280 | **0.7801** | **2.8051** | **0.6863** |
| cifar10 (VBLL, smoke) | 10 | 384 | 0.9420 | 0.2225 | 0.0722 |

This is not a bug. 860 historical VBLL runs on the tracking server reach
`test_accuracy=0.9423`, `ECE=0.0320` on cifar10 with the *same*
`prior_scale=1.0, wishart_scale=0.1`. Every one of those runs was cifar10 or
cifar100 on a 512-dim `scaling_mlps` backbone with 10 or 100 classes. The
defaults were "good enough" in that regime and are carried unchanged into one
with 25x more posterior entries (`1280x101` vs `512x10`), where the prior is far
too diffuse — the temperature fit of 0.17 says the head is badly underconfident.

**3. `prior_scale` and `wishart_scale` have never been varied.** Across all 860
historical runs both are constant (`1.0`, `0.1`). Only `parameterization` and
`cov_rank` ever moved, and rank-52 lowrank gave *better* calibration than
diagonal at matched accuracy (ECE 0.018–0.027 vs 0.031–0.034) — i.e. the one
axis anybody varied showed the richer posterior winning on exactly the quantity
Fig 1 reports.

## Non-goals

- Tuning full-network finetuning. Separate campaign, separate spec.
- imagenet1k. Dropped from linear probing entirely; it is a full-network
  finetuning dataset only.
- Changing the reported metrics, the dump protocol, or the early-stopping rule.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Comparison basis | Each arm tuned per dataset, best-vs-best | Neither arm handicapped; matches the existing IBProbit/CE precedent |
| Backbone | `dinov3_huge` only, applied to the whole ladder | Arguments are made around `huge`. Stated as an explicit limitation, not hidden |
| Search | Optuna TPE, one study per (arm, dataset) | `batch_size` interacts with `lr`, and VBLL adds two continuous axes — where a grid wastes most of its budget |
| Objective | `min` over dump grid of `val_nll`, mean of 2 seeds | Reuses `analysis.thresholds.early_stop`; no test or OOD leakage |
| Workers | 4 per study | TPE degrades under parallelism — workers sample a posterior not yet updated by in-flight trials |
| Truncated trials | Raise epoch budget until rare, then tune | Comparing a truncated slow config against a converged fast one biases toward high LR |
| `cov_rank` | **Fixed `diagonal` in the figures; rank swept as an appendix Pareto front** | See below — rank is a cost/quality dial the other arms do not have |

### Why `cov_rank` is not a tuned axis

`cov_rank` buys calibration with memory. Neither AdamW+CE nor IBProbit has an
equivalent knob, so letting TPE spend it freely would let VBLL win Fig 1 partly
by being larger. Head parameters at `D=1280`:

| dataset | C | CE linear | VBLL diag | VBLL r=2 | VBLL r=4 | VBLL r=64 | IBProbit |
|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 10 | 12.8K | 25.6K | 51.2K | 76.8K | 845K | 819K |
| dtd | 47 | 60.2K | 120K | 241K | 361K | 4.0M | 819K |
| food101 | 101 | 129K | 259K | 517K | 776K | 8.5M | 819K |
| stanford_cars | 196 | 251K | 502K | 1.00M | 1.51M | 16.6M | 819K |

IBProbit costs `D(D-1)/2`, **independent of C**. VBLL costs `D*C*(2+r)`, linear
in C. So the rank matching IBProbit's budget is ~64 at cifar10 but ~1 at
stanford_cars: there is no single fair rank across one figure's panels. At
stanford_cars, rank-64 is 16.6M parameters against CE's 251K — a 66x gap no
calibration gain argues its way out of.

The figures therefore fix `parameterization="diagonal"`, the cheapest VBLL
variant, so any win is cost-free to defend. The rank/calibration trade-off is
reported separately as a Pareto front (stage 5), which is where it belongs.

### Backbone limitation

The four backbones emit **different** feature dimensions — 384 / 768 / 1024 /
1280 — so a config tuned on `huge` is not guaranteed optimal on `small`. This is
accepted deliberately: the paper's arguments centre on `huge`, and the
alternative (24 studies per arm) does not earn its cost. The limitation is to be
stated in the paper, not buried.

## Search space

Suggested per trial. Shared axes apply to both arms because VBLL's head is also
optimised by AdamW.

| Param | Arms | Distribution | Range |
|---|---|---|---|
| `learning_rate` | both | log-uniform | `1e-4` … `3e-2` |
| `weight_decay` | both | log-uniform | `1e-5` … `1e-1` |
| `batch_size` | both | categorical, powers of 2 | 64 … dataset cap |
| `prior_scale` | VBLL | log-uniform | `1e-2` … `1e2` |
| `wishart_scale` | VBLL | log-uniform | `1e-3` … `1e1` |

`parameterization` is **fixed to `diagonal`** and `cov_rank` is not searched —
see "Why `cov_rank` is not a tuned axis". VBLL therefore has 5 free axes, not 7.

### batch_size caps

`probing_vbll.py` raises when `len(x_train) // batch_size == 0` (deliberately,
commit `1650e54`), and a batch near `n_train` makes epochs degenerate under
drop-remainder. Cap at `n_train // 2` rounded down to a power of two, so every
config gets at least 2 steps per epoch.

| dataset | figure | res | n_train | batch cap | steps/epoch at cap |
|---|---|---:|---:|---:|---:|
| oxford_pets | 1 | 512 | 3,680 | 1024 | 3 |
| dtd | 1 | 512 | 3,760 | 1024 | 3 |
| flowers102 | 1 | 512 | 7,169 | 2048 | 3 |
| stanford_cars | 1 | 512 | 8,144 | 2048 | 3 |
| food101 | 1 | 512 | 75,750 | 32768 | 2 |
| cifar10 | 2 | 224 | 50,000 | 16384 | 3 |
| cifar100 | 2 | 224 | 50,000 | 16384 | 3 |

The floor is 2 rather than 4 steps per epoch so that `1024` is reachable on
`dtd` and `oxford_pets`, the two smallest training sets. IBProbit is expected to
favour larger batches, and the batch axis has to span the same ladder for every
arm or a cross-arm comparison is confounded by the search space rather than by
the methods. A floor of 4 would have excluded 1024 on exactly the datasets where
IBProbit's preference is most likely to bite.

Two steps per epoch is still a meaningful epoch: the permutation is redrawn each
epoch, so consecutive epochs see different partitions of the training set.

The rule is applied uniformly, which also doubles the caps on the larger
datasets. That is deliberate — special-casing the small datasets would make the
`batch_size` axis a per-dataset judgement call rather than a mechanical
consequence of `n_train`. TPE will not spend many trials at the top of the
ladder if large batches do not pay.

The `batch_size` axis therefore differs per dataset. Studies remain comparable
on every other axis but not on this one; that is inherent to the constraint.

CIFAR sits at res224 by design — its native resolution is 32x32 and upscaling to
512 would be meaningless. Fig 1 datasets are at res512. Different feature files,
so potentially different optima; this is a further reason to tune per dataset.

## Objective

```
TUNING_SEEDS = (200, 201)          # disjoint from Fig 1/2's range(137, 147)

run probe(config, seed=200) -> step-indexed dumps
    report intermediate val_nll to the pruner at each dump
    if pruned: raise TrialPruned, seed 201 never runs
run probe(config, seed=201) -> step-indexed dumps

objective = mean over both seeds of
            early_stop(run_curve(dumps), "val_nll").val_nll     # minimise
```

Intermediates are reported from the first seed only; pruning a hopeless config
before its second seed is where most of the saving comes from.

**Tuning seeds are disjoint from the reported ones.** Fig 1 and Fig 2 use
`range(137, 147)`. Selecting hyperparameters on seeds that later produce the
reported error bars would tune against the same random draws being reported.
The earlier IBProbit/CE sweeps used 143 and 145, which are *inside* that range;
this spec deliberately does not repeat that.

The objective *is* `early_stop`, not a reimplementation of it, so the tuning
criterion and the reported stopping rule cannot drift apart. Selection is on
val, reporting on test, and `bllarse.splits` makes them disjoint by
construction. The OOD split never enters tuning — for Fig 2, studies are keyed
on the ID dataset only, since `probe.py` loads the OOD features after the
training loop purely for the final dump. cifar10's two pairs therefore share one
tuned config.

## Stages

### Stage 0 — plumbing

- Add `cov_rank` to `build_vbll_head` and a `--cov-rank` flag to
  `probing_vbll.py`. **Lowrank is currently unreachable from the probing path** —
  `build_vbll_head` does not accept the argument and the script has no flag — so
  stage 5's Pareto front cannot be produced without this.
- Log `head_params` from both arms, so cost is a recorded quantity rather than
  something a plotting script recomputes.
- Add `optuna` to both `.def` images; rebuild.
- Remove `imagenet1k` from `probe_vbll.py` and `probe_adamw_ce.py`.

### Stage 1 — pilot: find an epoch budget

Both arms were still improving at the final dump at 40 epochs (CE `val_nll`
0.1895 → 0.1806, VBLL 2.8315 → 2.8264 between steps 4707 and 5880), so
`at_last_step` is true and 40 epochs truncates.

Run the LR extremes (`1e-4`, `3e-2`) x 7 datasets x 2 arms at a generous epoch
cap. Raise the per-dataset budget until `at_last_step` is false for >=90% of
pilot configs, subject to a ceiling so a pathological LR cannot drive the budget
without bound. Budgets are per dataset — food101 and dtd differ 21x in
steps/epoch, so one shared budget would badly over-train the small ones.

`n_dumps` stays at 40 everywhere. Only the spacing changes, so every Fig 1 panel
keeps the same curve resolution.

### Stage 2 — tuning

| Arm | Free params | Trials/study | Studies | Trials |
|---|---:|---:|---:|---:|
| AdamW+CE | 3 (lr, wd, bs) | 64 | 7 | 448 |
| VBLL | 5 (lr, wd, bs, prior, wishart) | 96 | 7 | 672 |

Storage is Optuna `JournalStorage` on `JournalFileBackend`, one journal per
study under `/ptmp/cheins/optuna/<arm>_<dataset>.log`. **Not SQLite** — SQLite
over GPFS with concurrent writers is unreliable; the journal backend is built
for network filesystems. One file per study means the 14 studies never contend.

Pruning is `MedianPruner`, `n_startup_trials=20`, warmup set so it cannot fire
before ~25% of the dump grid.

This does **not** go through `run_sweep`. Its contract is
`create_configs() -> list[dict]`, enumerated upfront; TPE generates
configurations adaptively. A separate ask-and-tell worker path is required.

### Stage 3 — collect

Emit `bllarse_sweeps/tuned_hparams.json`: `(arm, dataset) -> best config`,
committed to the repo. Tuned values are **not** hand-copied into any sweep's
`BASE`; the sweeps read the lookup. This keeps submission reproducible without a
live MLflow dependency and makes the provenance of every number auditable.

### Stage 4 — figures

`probe_vbll.py`, `probe_adamw_ce.py`, `probe_ood_vbll.py` and
`probe_ood_adamw_ce.py` resolve their per-dataset hyperparameters from
`tuned_hparams.json`. Which datasets constitute Fig 1's final grid is a later
decision; tuning covers all seven so that choice stays open.

### Stage 5 — appendix: rank/cost Pareto front

Takes each dataset's tuned diagonal config and re-runs it with
`parameterization="lowrank"`, `cov_rank` in {2, 4, 8, 16, 32, 64}, everything
else held at the tuned values. Reports test ECE and NLL against **head
parameter count**, with CE's and IBProbit's counts marked on the same axis, so
the cost of any calibration gain is visible rather than asserted.

Datasets: cifar10 (C=10), food101 (C=101), stanford_cars (C=196). These span the
regime where the cost story inverts — at C=10 rank-64 is IBProbit-sized, at
C=196 it is 20x IBProbit.

Seeds: 3 from the reported range, so the appendix carries error bars comparable
to the figures.

Two honest caveats to state in the appendix text:

- `prior_scale` and `wishart_scale` are **not** re-tuned per rank, though
  changing rank changes posterior capacity and so plausibly moves their optimum.
  The front is therefore a lower bound on what each rank could achieve, not a
  tuned comparison. Re-tuning per rank would multiply the cost by the number of
  ranks and is not justified for an appendix.
- Head parameter count must be logged (`head_params`) by both arms so the front
  is derivable from runs rather than recomputed by hand in a plotting script.

Cost: 3 datasets x 6 ranks x 3 seeds = 54 runs, VBLL only. Negligible next to
stage 2.

## Cost

Measured on Raven, `dinov3_huge`, food101, 40 epochs (5,880 steps):

| Arm | Wall per run |
|---|---:|
| AdamW+CE | **101.9 s** |
| VBLL | **101.4 s** |

The two arms are at parity. They were not: AdamW+CE originally took **460 s**,
4.5x VBLL, because `probe.py`'s optimiser step ran in JAX's op-by-op eager mode
— it was the only JAX training script in the repo with no `jit`. At these shapes
the un-jitted step cost 58.5 ms against 2.2 ms jitted (26x), i.e. 344 s of the
460 s. Hoisting the step into an `eqx.filter_jit`ed `_ce_step` brought the run
to 101.9 s with **every reported metric bit-identical** (`test_acc=0.9541`,
`test_ece=0.0340`, `test_nll=0.1879`, `temperature=0.7668`). Fixed before this
campaign, so no tuning result depends on the slow path.

With all 14 studies submitted concurrently (4 workers each = 56 GPU tasks), the
critical path is now food101 at roughly **1 h** per arm before pruning savings,
down from ~4 h. Raising the epoch budget in stage 1 scales this proportionally;
a 2–3x increase puts the campaign in the 2–3 h range. These are projections from
one measured datapoint per arm, not measurements of the campaign itself.

`_logits_for` remains un-jitted. It is the same root cause but a separate,
smaller cost — 40 eval passes against 5,880 training steps — and was left alone
rather than bundled into the fix. Worth measuring before changing.

## Risks

| Risk | Mitigation |
|---|---|
| Tuned-on-`huge` config transfers poorly to `small` (D 1280 → 384) | Accepted and stated. Cheap to check afterwards by rerunning the best config on `small` for one dataset |
| TPE quality degrades with parallel workers | Capped at 4 |
| Pilot budget ceiling still leaves some configs truncated | Truncation flagged per trial; if the winning config is truncated, the budget was wrong and the study is not trustworthy |
| `imagenet1k` features already staged (17.7 GB) go unused | No action. Full-network finetuning needs raw images, not these features, so they likely will not serve that either |

## Open questions

- Is `_logits_for` worth jitting too? Same root cause, unmeasured share of the
  remaining ~100 s. Measure before changing.
- Should `prior_scale`/`wishart_scale` be parameterised relative to `D*C` rather
  than absolutely? The evidence says the good value scales with problem size, so
  an absolute range must be wide enough to cover all seven datasets. Tuning per
  dataset sidesteps this, but a scale-aware parameterisation would transfer
  better across the backbone ladder.
