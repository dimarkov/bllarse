#!/usr/bin/env python3
"""Linear probing on cached DINOv3 features, with logit dumps.

One head, one dataset, one seed. Loads features that precompute_features.py has
already extracted, trains a head, and writes a dump at each scheduled step. All
reported metrics are computed later from those dumps -- nothing here decides
what the figures show.

Arms: adamw_ce and ibprobit. They share everything outside the training step --
the same cached features, the same pinned val/test split, the same log-spaced
dump schedule -- so the only thing that differs between their dumps is how the
head was fitted.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr, vmap

from bllarse import evaldump, splits
from bllarse.features import eval_split_name, load_cached_features, train_split_name
from bllarse.losses import CrossEntropy, IBProbit
from bllarse.mlflow_utils import child_run
from bllarse.reporting import report_step


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


def main(args, on_dump=None):
    """Train one head. `on_dump(dump_index, step, scalars)` fires after every
    scheduled dump, if given.

    The index counts dumps rather than optimiser steps because it is Optuna's
    pruning key: MedianPruner compares trials at the SAME key, and trials vary
    batch_size, so two configs would share almost no raw step values. n_dumps is
    40 for every run, so the dump index is a common grid by construction.
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

    # Loaded once, before the loop: the OOD set is dumped at EVERY eval step
    # (see the write below), and re-reading the feature file per dump would cost
    # more than the forward pass it feeds.
    ood_f = ood_l = None
    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, img_size,
            eval_split_name(args.ood_dataset), hf_repo=hf_repo,
        )
        print(f"[probe] OOD set {args.ood_dataset}: {len(ood_f)} samples")

    per_epoch = steps_per_epoch(len(train_f), args.batch_size)
    total_steps = args.epochs * per_epoch
    schedule = set(evaldump.dump_schedule(total_steps, n_dumps=args.n_dumps).tolist())
    print(
        f"[probe] {args.uid}: {args.epochs} epochs x {per_epoch} steps "
        f"= {total_steps} steps ({total_steps * args.batch_size} samples), "
        f"{len(schedule)} dumps"
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
        dump_index = 0
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
                if step not in schedule:
                    continue

                # Exact: drop-remainder means every step is exactly batch_size.
                samples_seen = step * args.batch_size
                logits = _logits_for(head, loss_fn, eval_f, eval_l)
                # IBProbit's contamination estimate and its native epistemic
                # score ride along with the dump. Both are absent on adamw_ce,
                # which has neither quantity.
                eps_hat = float(loss_fn.eps) if args.arm == "ibprobit" else None
                scores = _scores_for(loss_fn, eval_f) if args.arm == "ibprobit" else None
                for split_name, idx in (("val", val_idx), ("test", test_idx)):
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        logits[idx], eval_l[idx],
                        arm=args.arm, split=split_name, logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=samples_seen,
                        score=None if scores is None else scores[idx],
                        score_kind=None if scores is None else "predictive_variance",
                        eps_hat=eps_hat,
                    )
                # OOD at every eval step, not only the last. This used to be a
                # single dump at total_steps, on the grounds that "Fig 2 is a
                # converged-endpoint figure" -- but these runs carry no early
                # stopping, and the endpoint is not where the configuration was
                # tuned. AdamW+CE on cifar100 peaks at step 1,786 of 30,000 and
                # then nearly triples its NLL, so scoring detection at the last
                # step measured that overfitting rather than the method. With a
                # dump per step, detection can be read at the same step as any
                # other metric, including the val-optimal one.
                if ood_f is not None:
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        _logits_for(head, loss_fn, ood_f, ood_l), ood_l,
                        arm=args.arm, split="ood", logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=samples_seen,
                        score=None if scores is None else _scores_for(loss_fn, ood_f),
                        score_kind=None if scores is None else "predictive_variance",
                        eps_hat=eps_hat,
                    )
                scalars = report_step(
                    run, step, samples_seen, epoch,
                    logits[test_idx], eval_l[test_idx],
                    logits[val_idx], eval_l[val_idx],
                )
                print(
                    f"[probe] step {step} (epoch {epoch}, {samples_seen} samples): "
                    + " ".join(f"{k}={v:.4f}" for k, v in scalars.items())
                )
                if on_dump is not None:
                    on_dump(dump_index, step, scalars)
                dump_index += 1

        if ood_f is not None:
            print(f"[probe] wrote {len(schedule)} OOD dumps for {args.ood_dataset}")


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
    p.add_argument("--n-dumps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # --- IBProbit head knobs (arm=ibprobit only) ---
    # Flag names, types and defaults copied VERBATIM from
    # scripts/alternatives/ood_evaluation.py and scripts/finetune/finetune_jax.py
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
