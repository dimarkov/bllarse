#!/usr/bin/env python3
"""Linear probing on cached DINOv3 features, with logit dumps.

One head, one dataset, one seed. Loads features that precompute_features.py has
already extracted, trains a head, evaluates val_nll every `eval_quantum`
SAMPLES, and writes one dump: the evaluation that minimised it. All reported
metrics are computed later from that dump -- nothing here decides what the
figures show.

Evaluation is spaced in samples rather than steps so that index i is the same
samples_seen for every trial in a study regardless of batch size, and the
spacing is sub-epoch by construction (0.22-0.33 epochs across these datasets),
which is the point of a step-based protocol. See tuning.space.eval_quantum.

Arms: adamw_ce and ibprobit. They share everything outside the training step --
the same cached features, the same pinned val/test split, the same sample-spaced
evaluation grid -- so the only thing that differs between their dumps is how the
head was fitted.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr, vmap

from bllarse import evaldump, metrics, splits
from bllarse.features import eval_split_name, load_cached_features, train_split_name
from bllarse.losses import CrossEntropy, IBProbit
from bllarse.mlflow_utils import child_run
from bllarse.reporting import report_step
from bllarse.tuning import space


def steps_per_epoch(n_train: int, batch_size: int) -> int:
    """Optimiser steps in one pass. Drop-remainder, so every step sees exactly
    `batch_size` samples -- which is what makes samples_seen exact."""
    n_batches = n_train // batch_size
    if n_batches == 0:
        # Silently clamping to one batch would dump a plausible-looking but
        # untrained head -- its accuracy/NLL/Brier would be indistinguishable
        # from a real result after the fact. Fail loudly instead.
        raise ValueError(
            f"batch_size={batch_size} exceeds n_train={n_train}; this would "
            f"train zero gradient steps per epoch. Lower --batch-size or "
            f"provide more training data."
        )
    return n_batches


def head_param_count(head) -> int:
    """Total parameter count of the head (memory cost, trainable or frozen).

    Counts every stored parameter the head holds, including any frozen parameters,
    because the Pareto front plots calibration against *memory cost*, not just
    trainable degrees of freedom. (For the adamw_ce arm, all parameters are
    trainable, so this is the number that appears in the figures. Both arms must
    report this identically or the Pareto front's cost axis is not a single axis.)

    Logged per run because Fig-appendix's rank/cost Pareto front plots calibration
    against head size: VBLL costs D*C*(2+r) and IBProbit costs D*(D-1)/2,
    independent of C, so the ranks that match IBProbit's budget differ by two
    orders of magnitude across one figure's panels. Reading the count off the run
    keeps that axis a recorded quantity.
    """
    return int(
        sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(head, eqx.is_array)))
    )


@eqx.filter_jit
def _ce_step(head, opt_state, optimizer, loss_fn, fb, yb):
    """One AdamW step. Hoisted out of the epoch loop so it can be compiled ONCE.

    Inlined in the loop this ran in JAX's op-by-op eager mode: every primitive
    dispatched separately from Python with no fusion. Measured on an A100 at the
    Fig-1 working shapes (D=1280, C=101, batch=512), that cost 58.5 ms/step
    against 2.2 ms jitted -- 26x, or 343 s of the 460 s a 5,880-step food101 run
    took. probe.py was the only JAX training script in the repo without a jit;
    vit_classification, sample_efficiency and full_finetune_comparison all use
    eqx.filter_jit.

    `optimizer` and `loss_fn` are passed rather than closed over: filter_jit
    traces their array leaves (CrossEntropy.beta) and holds the rest static, so
    the signature stays explicit and the function stays importable for testing.
    Shapes are constant across steps because steps_per_epoch drops the
    remainder, so this compiles once per run, not once per step.
    """
    def loss(h):
        return loss_fn(vmap(h)(fb), yb).mean()

    grads = eqx.filter_grad(loss)(head)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(head, eqx.is_array)
    )
    return eqx.apply_updates(head, updates), opt_state


def _ce_epoch(head, opt_state, optimizer, loss_fn, feats, labels, batch_size, key):
    """One AdamW pass, yielding (head, opt_state) after every optimiser step.

    A generator rather than a whole-epoch call because the dump schedule is now
    indexed on steps: an epoch of ImageNet is 2,502 steps, and dumping only at
    epoch boundaries would put the first measurement 1.28M samples in -- past
    the entire early regime the sample-efficiency claim lives in.
    """
    perm = jr.permutation(key, len(feats))
    for i in range(steps_per_epoch(len(feats), batch_size)):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        fb, yb = jnp.asarray(feats[idx]), jnp.asarray(labels[idx])
        head, opt_state = _ce_step(head, opt_state, optimizer, loss_fn, fb, yb)
        yield head, opt_state


@eqx.filter_jit
def _ibprobit_step(loss_fn, fb, yb, num_iters):
    """One recursive-Bayesian CAVI update. Jitted for the same reason `_ce_step`
    is: eager op-by-op dispatch measured 26x slower there, and this step runs
    just as many times.

    `num_iters` arrives as a Python int, which filter_jit holds static -- it has
    to, since `_cavi_scan` builds a `lax.scan` over `jnp.arange(num_iters)`.
    `px` is not passed: it is a static constructor field, so `update` reads it
    off `self` and a per-call override could only make the two disagree.
    Shapes are constant across steps (steps_per_epoch drops the remainder), so
    this compiles once per run.
    """
    return loss_fn.update(fb, yb, num_iters=num_iters)


def _ibprobit_epoch(loss_fn, feats, labels, batch_size, num_iters, key):
    """One streaming pass, yielding the head after every CAVI update.

    Structurally parallel to `_ce_epoch` -- same drop-remainder step count, same
    permutation source, same yield per optimiser-equivalent step -- so a dump
    scheduled at step N lands on the same `samples_seen` for both arms. Nothing
    else would make the two trajectories comparable point-for-point.

    There is no optimiser state to thread through: IBProbit's `update` IS the
    training step, and it returns the whole head.
    """
    perm = jr.permutation(key, len(feats))
    for i in range(steps_per_epoch(len(feats), batch_size)):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        fb, yb = jnp.asarray(feats[idx]), jnp.asarray(labels[idx])
        loss_fn = _ibprobit_step(loss_fn, fb, yb, num_iters)
        yield loss_fn


def _logits_for(head, loss_fn, feats, labels, chunk=4096):
    """Dumped logits for either arm.

    `head` is None on the ibprobit arm: IBProbit consumes features directly --
    its posterior mean IS the linear map -- so there is no separate module to
    push them through first.
    """
    out = []
    for start in range(0, len(feats), chunk):
        fb = jnp.asarray(feats[start:start + chunk])
        yb = jnp.asarray(labels[start:start + chunk])
        inputs = fb if head is None else vmap(head)(fb)
        _, logits = loss_fn(inputs, yb, with_logits=True)
        out.append(np.asarray(logits))
    return np.concatenate(out, axis=0)


def _scores_for(loss_fn, feats, chunk=4096):
    """IBProbit's native epistemic score, 1 + f'(Qhat + Sigma)f.

    The direct analogue of the VBLL arm's dumped score (that one sums the logit
    predictive's covariance diagonal), so `curves.run_native_detection` compares
    like with like across the two Bayesian arms. AdamW+CE has no such quantity
    and dumps none.
    """
    out = []
    for start in range(0, len(feats), chunk):
        fb = jnp.asarray(feats[start:start + chunk])
        out.append(np.asarray(loss_fn.predictive_variance(fb)))
    return np.concatenate(out, axis=0)


def _derive_quantum(n_train: int, batch_size: int) -> int:
    """The eval quantum when --eval-quantum is not given.

    Derived from the training set actually loaded rather than from
    `space.eval_quantum(dataset)`, so this works for any dataset -- the search
    space's N_TRAIN table only knows the campaign's seven, and probe.py is also
    run standalone and under test on synthetic ones.

    Falls back to `batch_size` (evaluate every step, the finest grid there is)
    whenever the ladder cap is not a whole number of steps -- an off-ladder
    batch_size, or a training set too small to have a ladder at all. The
    campaign never takes that path: worker.trial_config passes eval_quantum
    explicitly.
    """
    try:
        cap = space.batch_cap(n_train)
    except ValueError:
        return batch_size
    return cap if cap >= batch_size and cap % batch_size == 0 else batch_size


def _write_best(args, best, last_eval_index, n_evals):
    """Write the buffered argmin-val_nll evaluation. The run's ONLY disk write.

    Called from a `finally`, so it runs whether the loop finished its budget,
    was stopped by an EarlyConvergence raised out of `on_dump`, or DIED. That
    last case is why `n_evals` (the scheduled count) is written alongside
    `last_eval_index` (the count that actually ran): a crashed run still writes
    a dump here, and without those two numbers it is indistinguishable from a
    completed one. See evaldump.write_dump.

    `best` is None for TWO reasons that need telling apart, because they call
    for opposite responses. Either no evaluation ran at all (a budget shorter
    than one eval interval -- a misconfiguration), or evaluations ran and every
    single one was non-finite (the run diverged -- a result). The message used
    to name only the first, and duly sent a reader hunting for a budget bug on
    a run that had completed 30,000 evaluations of pure nan.
    """
    if best is None:
        if not last_eval_index:
            raise ValueError(
                f"{args.uid}: no evaluation ran, so there is no dump to write. "
                f"The step budget is shorter than one eval interval."
            )
        raise ValueError(
            f"{args.uid}: {last_eval_index} evaluations ran and EVERY ONE had "
            f"a non-finite val_nll, so there is no dump to write. This is a "
            f"diverged run, not a misconfigured budget -- the argmin buffer "
            f"only ever accepts finite values."
        )
    for split_name, logits, labels, score in best["splits"]:
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, best["step"],
            logits, labels,
            arm=args.arm, split=split_name, logit_kind="softmax_logits",
            seed=args.seed, samples_seen=best["samples_seen"],
            score=score,
            score_kind=None if score is None else "predictive_variance",
            eps_hat=best["eps_hat"],
            # Recorded here because only this loop knows how many evaluations
            # there were. A run writes one dump, so nothing downstream can tell
            # a converged optimum from a budget that ran out -- which is the
            # signal tuning.budgets uses to decide the budget is too small.
            at_last_eval=best["eval_index"] == last_eval_index,
            eval_index=best["eval_index"],
            evals_run=last_eval_index,
            evals_planned=n_evals,
        )
    truncated = best["eval_index"] == last_eval_index
    died = last_eval_index < n_evals
    print(
        f"[probe] {args.uid}: best val_nll={best['val_nll']:.4f} at step "
        f"{best['step']} (eval {best['eval_index']} of {last_eval_index} run, "
        f"{n_evals} scheduled, epoch {best['epoch']}, "
        f"{best['samples_seen']} samples)"
        f"{' AT LAST EVAL -- truncated, not converged' if truncated else ''}"
        f"{' INCOMPLETE -- the run did not reach the end of its budget' if died else ''}"
        f"; wrote {len(best['splits'])} dumps",
        flush=True,
    )


def main(args, on_dump=None):
    """Train one head. `on_dump(eval_index, step, samples_seen, scalars)` fires
    after every scheduled evaluation, if given.

    `eval_index` counts evaluations rather than optimiser steps because it is
    Optuna's pruning key: MedianPruner compares trials at the SAME key, and
    trials vary batch_size. Evaluations are spaced `eval_quantum` SAMPLES apart
    (see tuning.space.eval_quantum), so index i is the same samples_seen for
    every trial in a study -- which the old dump index was not: it was a
    geometric grid over STEPS, so index i meant 256x more data for a batch-16384
    trial than a batch-64 one, and the pruner compared them there anyway.
    """
    key = jr.PRNGKey(args.seed)
    hf_repo = None if args.no_hf_cache else args.hf_repo
    img_size = args.img_size

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size,
        train_split_name(args.dataset), hf_repo=hf_repo,
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size,
        eval_split_name(args.dataset), hf_repo=hf_repo,
    )

    # Pinned split: val is for post-hoc temperature scaling, test for reporting.
    # Seed-independent by construction (bllarse.splits) -- the key below names
    # the cache file only, it does not enter the partition, so passing the eval
    # split's real name rather than the literal "test" renames food101's cache
    # and leaves its indices untouched. flowers102 needs it: its eval pool is
    # the 7,169-image "train" file, which under the old key would collide with
    # a cached 1,020-entry split and raise on size drift.
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, eval_split_name(args.dataset),
        len(eval_f), args.val_fraction,
    )

    embed_dim = train_f.shape[-1]
    key, head_key = jr.split(key)

    if args.arm == "ibprobit":
        # No head module and no optimizer. IBProbit's posterior mean is the
        # linear map, and `update` assimilates each batch in closed form -- so
        # --learning-rate and --weight-decay are inert on this arm and the
        # search space does not sample them.
        #
        # rho is the eps forgetting factor, 1 - B/N: one pass' worth of memory,
        # the convention the class documents and the other three IBProbit
        # scripts pass. batch_size is handed over as well so that omitting
        # --q-init falls back to the derived q0 = alpha^2 / B rather than
        # raising.
        head, optimizer, opt_state = None, None, None
        loss_fn = IBProbit(
            embed_dim, args.num_classes, key=head_key,
            q_init=args.q_init, alpha=args.alpha, batch_size=args.batch_size,
            kappa=args.kappa, learn_q=not args.no_learn_q, px=args.px,
            warmup=args.warmup, eps_init=args.eps_init,
            rho=max(0.0, 1.0 - args.batch_size / len(train_f)),
        )
    else:
        head = eqx.nn.Linear(embed_dim, args.num_classes, key=head_key)
        loss_fn = CrossEntropy(args.label_smooth, args.num_classes)
        optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
        opt_state = optimizer.init(eqx.filter(head, eqx.is_array))

    # Loaded once, before the loop. NOT because the OOD set is dumped at every
    # eval step -- it is not, and this comment used to say so. The OOD forward
    # pass runs only when an evaluation IMPROVES on the running val_nll argmin,
    # and only the last such payload is ever written (see `_write_best`). It is
    # hoisted because "on every improvement" is still many times per run, and
    # re-reading the feature file each time would cost more than the forward
    # pass it feeds.
    ood_f = ood_l = None
    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, img_size,
            eval_split_name(args.ood_dataset), hf_repo=hf_repo,
        )
        print(f"[probe] OOD set {args.ood_dataset}: {len(ood_f)} samples")

    per_epoch = steps_per_epoch(len(train_f), args.batch_size)
    total_steps = args.epochs * per_epoch

    # Evaluation is scheduled on SAMPLES, not steps. `quantum` samples pass
    # between consecutive evaluations for every trial in a study, so the trial
    # at the ladder's top evaluates every step and one at batch = cap/k
    # evaluates every k steps -- and both land on identical samples_seen. A
    # step grid does not: at a fixed step budget a step is `batch_size`
    # samples, so "step 5,000" was 256x more data for one trial than another
    # within a single study, and Optuna's pruner was comparing them there.
    quantum = args.eval_quantum or _derive_quantum(len(train_f), args.batch_size)
    if quantum % args.batch_size:
        raise ValueError(
            f"--eval-quantum {quantum} is not a whole number of "
            f"batch_size={args.batch_size} steps. Evaluation happens between "
            f"optimiser steps, so a fractional interval is unschedulable. The "
            f"ladder is powers of two (see tuning.space.batch_sizes), so this "
            f"means batch_size is off-ladder for {args.dataset}."
        )
    eval_every = quantum // args.batch_size

    n_evals = total_steps // eval_every
    print(
        f"[probe] {args.uid}: {args.epochs} epochs x {per_epoch} steps "
        f"= {total_steps} steps ({total_steps * args.batch_size} samples), "
        f"eval every {eval_every} steps ({quantum} samples, "
        f"{quantum / len(train_f):.2f} epochs) = {n_evals} evals, "
        f"every {args.report_every} reported"
    )

    with child_run(vars(args), run_name=args.uid, tags={
        "arm": args.arm, "group_id": args.group_id, "uid": args.uid,
        "dump_dir": str(Path(args.dump_root) / args.group_id / args.uid),
    }) as run:
        run.log_params({
            "steps_per_epoch": per_epoch,
            "total_steps": total_steps,
            # On the ibprobit arm the head IS the loss_fn, so that is what the
            # memory-cost axis has to count: mu (d,K) plus the two (d,d) matrices
            # Sigma and Psi. Counting `head` (None) would report 0.
            "head_param_count": head_param_count(loss_fn if head is None else head),
        })

        step = 0
        # The single best-val_nll evaluation, held in memory until the run ends.
        # Nothing is written during training: the old loop wrote 2-3 arrays to
        # GPFS at every one of its 40 points, and this grid has up to 30,000.
        # Only the argmin is ever read downstream -- `analysis.thresholds.
        # early_stop` selects it and `tuning.objective.score_dumps` scores it --
        # so the rest was written and then deleted unread.
        best: dict[str, Any] | None = None
        eval_index = 0
        # The val_nll curve, scalars only. `best` keeps ONE evaluation because
        # logits are large; this keeps every evaluation because floats are not,
        # so questions about the stopping rule stay answerable without a re-run.
        # See evaldump.write_trajectory.
        curve: dict[str, list] = {"eval_index": [], "step": [],
                                  "samples_seen": [], "val_nll": []}

        # Hoisted: fancy indexing copies, and at up to 30,000 evaluations a
        # per-eval copy of the val block would cost more than the forward pass
        # it feeds.
        val_f, val_l = eval_f[val_idx], eval_l[val_idx]
        test_f, test_l = eval_f[test_idx], eval_l[test_idx]

        # try/finally, not a trailing call: `on_dump` raises EarlyConvergence to
        # stop a converged run (see tuning.objective), and with the dump now
        # buffered rather than streamed, an escaping exception would leave the
        # run with NOTHING on disk -- scoring would then fail on a trial that
        # had in fact finished successfully.
        try:
            for epoch in range(1, args.epochs + 1):
                key, epoch_key = jr.split(key)
                if args.arm == "ibprobit":
                    stream = _ibprobit_epoch(
                        loss_fn, train_f, train_l, args.batch_size,
                        args.num_update_iters, epoch_key,
                    )
                else:
                    stream = _ce_epoch(
                        head, opt_state, optimizer, loss_fn,
                        train_f, train_l, args.batch_size, epoch_key,
                    )
                for produced in stream:
                    if args.arm == "ibprobit":
                        loss_fn = produced
                    else:
                        head, opt_state = produced
                    step += 1
                    if step % eval_every:
                        continue

                    # Exact: drop-remainder means every step is exactly batch_size.
                    samples_seen = step * args.batch_size
                    # Defined as the position on the SAMPLE grid, not a running
                    # counter: index i means i * quantum samples for every trial
                    # in the study, which is what makes it a sound pruning key.
                    eval_index = samples_seen // quantum

                    # Val only, and only the logits. This runs up to 30,000
                    # times, so everything the selection rule does not need
                    # stays out of it -- `_scores_for` alone is O(N d^2) and
                    # would dominate the run at this cadence.
                    val_logits = _logits_for(head, loss_fn, val_f, val_l)
                    val_nll = metrics.nll(val_logits, val_l)

                    # `isfinite` first, and NOT folded into the comparison:
                    # `best is None or val_nll < best[...]` latches on a
                    # non-finite FIRST evaluation, because `best is None` is
                    # True so nan becomes the best, and every later
                    # `v < nan` is False so nothing can replace it. The run
                    # then stalls by construction, patience fires exactly
                    # `1 + patience` evaluations in, and the trial writes a
                    # nan dump that scores nan. Two ibprobit oxford_pets
                    # trials died this way at num_update_iters=192, both with
                    # seed 201 stopping at eval 41 against a patience of 40.
                    # A non-finite evaluation is not a candidate optimum; it
                    # is simply no information.
                    if math.isfinite(val_nll) and (
                        best is None or val_nll < best["val_nll"]
                    ):
                        # A new argmin: build the full payload NOW, because it
                        # is a function of head state the next step overwrites.
                        # Logits rather than the head itself -- for IBProbit the
                        # head carries two (d,d) matrices, 134 MB at d=4096, an
                        # order of magnitude more than the arrays they produce.
                        eps_hat = float(loss_fn.eps) if args.arm == "ibprobit" else None
                        test_logits = _logits_for(head, loss_fn, test_f, test_l)
                        val_scores = test_scores = ood_scores = None
                        if args.arm == "ibprobit":
                            # IBProbit's contamination estimate and its native
                            # epistemic score ride along with the dump. Both are
                            # absent on adamw_ce, which has neither quantity.
                            val_scores = _scores_for(loss_fn, val_f)
                            test_scores = _scores_for(loss_fn, test_f)
                        best = {
                            "val_nll": float(val_nll),
                            "step": step,
                            "samples_seen": samples_seen,
                            "epoch": epoch,
                            "eval_index": eval_index,
                            "eps_hat": eps_hat,
                            "splits": [
                                ("val", np.asarray(val_logits), np.asarray(val_l), val_scores),
                                ("test", np.asarray(test_logits), np.asarray(test_l), test_scores),
                            ],
                        }
                        # OOD at the val-optimal point rather than at the last
                        # step. It used to be dumped at total_steps on the
                        # grounds that "Fig 2 is a converged-endpoint figure" --
                        # but AdamW+CE on cifar100 peaks at step 1,786 of 30,000
                        # and then nearly triples its NLL, so the endpoint
                        # measured that overfitting rather than the method.
                        if ood_f is not None:
                            if args.arm == "ibprobit":
                                ood_scores = _scores_for(loss_fn, ood_f)
                            best["splits"].append((
                                "ood",
                                np.asarray(_logits_for(head, loss_fn, ood_f, ood_l)),
                                np.asarray(ood_l),
                                ood_scores,
                            ))

                    curve["eval_index"].append(eval_index)
                    curve["step"].append(step)
                    curve["samples_seen"].append(samples_seen)
                    curve["val_nll"].append(float(val_nll))

                    scalars = {"val_nll": float(val_nll)}
                    if eval_index % args.report_every == 0:
                        # A stride of the EVALUATION grid, so reported points
                        # are uniform in samples like everything else. The
                        # throttle exists because the full metric set needs
                        # test logits and a temperature fit, and one MLflow
                        # point per evaluation would be its own outage.
                        scalars = report_step(
                            run, step, samples_seen, epoch,
                            _logits_for(head, loss_fn, test_f, test_l), test_l,
                            val_logits, val_l,
                        )
                        print(
                            f"[probe] step {step} (epoch {epoch}, "
                            f"{samples_seen} samples): "
                            + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()),
                            flush=True,
                        )
                    if on_dump is not None:
                        on_dump(eval_index, step, samples_seen, scalars)
        finally:
            # Trajectory FIRST: _write_best raises when no evaluation ran, and
            # the curve is exactly the evidence needed to see why.
            evaldump.write_trajectory(
                args.dump_root, args.group_id, args.uid,
                eval_index=curve["eval_index"], step=curve["step"],
                samples_seen=curve["samples_seen"], val_nll=curve["val_nll"],
                arm=args.arm, seed=args.seed, evals_planned=n_evals,
            )
            _write_best(args, best, eval_index, n_evals)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="adamw_ce", choices=["adamw_ce", "ibprobit"])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ood-dataset", default=None)
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--hf-repo", default="dimarkov/bllarse-features")
    p.add_argument("--no-hf-cache", action="store_true")
    p.add_argument("--group-id", default="probe")
    p.add_argument("--uid", default="probe_0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-quantum", type=int, default=0,
                   help="Samples between evaluations. 0 derives it from the "
                        "dataset's batch ladder (tuning.space.eval_quantum), "
                        "which is the only value that is a whole number of "
                        "steps at every rung. Set explicitly only for a "
                        "batch_size that is off-ladder.")
    p.add_argument("--report-every", type=int, default=10,
                   help="Report to MLflow and stdout on every k-th EVALUATION. "
                        "A stride of the evaluation grid, so reported points "
                        "are uniform in samples like everything else -- this "
                        "replaced a geometric grid, which put most of its "
                        "resolution where the samples-seen axis has least. "
                        "Throttling only: it has never driven evaluation, "
                        "dumps, scoring or early stopping.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # --- IBProbit head knobs (arm=ibprobit only) ---
    # Flag names, types and defaults copied VERBATIM from
    # scripts/dinov3/ood_evaluation.py and scripts/finetune/finetune_jax.py
    # so all the IBProbit entry points share one operating point. Do not invent
    # defaults here.
    p.add_argument("--num-update-iters", type=int, default=128,
                   help="CAVI iterations per batch (ibprobit arm). Tuned over "
                        "{32, 64, 128, 192}; see bllarse.tuning.space.")
    p.add_argument("--px", action="store_true",
                   help="parameter-expanded CAVI scan (collapses the slow scale mode; far fewer iters)")
    p.add_argument("--no-learn-q", action="store_true",
                   help="pin Qhat at its q0 seed (skip the Inverse-Wishart recursion on Q)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="process-noise scale: q0 = alpha^2 / batch_size. Used only when "
                        "--q-init is omitted; at 1e-3 the recursion is measurably inert")
    p.add_argument("--q-init", type=float, default=None,
                   help="override q0 directly (bypasses alpha^2 / B; 1/D is the pre-Wishart "
                        "value). This, not --alpha, is the tuned axis: q0 is what the "
                        "campaign compares across datasets, and alpha alone does not pin it "
                        "once batch size is tuned too.")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart prior pseudo-count (default: 0.5, i.e. half a "
                        "pseudo-observation against the K the likelihood adds per step). "
                        "Fixed by decision, not tuned.")
    p.add_argument("--warmup", type=int, default=1,
                   help="hold the contamination gate at 1 for the first N update steps")
    p.add_argument("--eps-init", type=float, default=0.01,
                   help="Beta(1, 1/eps_init) prior on the contamination rate (mean ~ eps_init). "
                        "Forgetting rho = 1 - B/N set automatically.")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
