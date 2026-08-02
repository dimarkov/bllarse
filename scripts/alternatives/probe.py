#!/usr/bin/env python3
"""Linear probing on cached DINOv3 features, with logit dumps.

One head, one dataset, one seed. Loads features that precompute_features.py has
already extracted, trains a head, and writes a dump at each scheduled step. All
reported metrics are computed later from those dumps -- nothing here decides
what the figures show.

Arms: adamw_ce now; ibprobit is wired the same way and enters in Phase 3.
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
from bllarse.features import eval_split_name, load_cached_features
from bllarse.losses import CrossEntropy
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


def _logits_for(head, loss_fn, feats, labels, chunk=4096):
    out = []
    for start in range(0, len(feats), chunk):
        fb = jnp.asarray(feats[start:start + chunk])
        yb = jnp.asarray(labels[start:start + chunk])
        _, logits = loss_fn(vmap(head)(fb), yb, with_logits=True)
        out.append(np.asarray(logits))
    return np.concatenate(out, axis=0)


def main(args):
    key = jr.PRNGKey(args.seed)
    hf_repo = None if args.no_hf_cache else args.hf_repo
    img_size = args.img_size

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size, "train", hf_repo=hf_repo
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, img_size,
        eval_split_name(args.dataset), hf_repo=hf_repo,
    )

    # Pinned split: val is for post-hoc temperature scaling, test for reporting.
    # Seed-independent by construction (bllarse.splits).
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, "test", len(eval_f), args.val_fraction
    )

    embed_dim = train_f.shape[-1]
    key, head_key = jr.split(key)

    if args.arm != "adamw_ce":
        raise NotImplementedError(
            f"arm '{args.arm}' is not wired yet; adamw_ce only in Phase 1"
        )

    head = eqx.nn.Linear(embed_dim, args.num_classes, key=head_key)
    loss_fn = CrossEntropy(args.label_smooth, args.num_classes)
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(eqx.filter(head, eqx.is_array))

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
            "head_param_count": head_param_count(head),
        })

        step = 0
        for epoch in range(1, args.epochs + 1):
            key, epoch_key = jr.split(key)
            for head, opt_state in _ce_epoch(
                head, opt_state, optimizer, loss_fn,
                train_f, train_l, args.batch_size, epoch_key,
            ):
                step += 1
                if step not in schedule:
                    continue

                # Exact: drop-remainder means every step is exactly batch_size.
                samples_seen = step * args.batch_size
                logits = _logits_for(head, loss_fn, eval_f, eval_l)
                for split_name, idx in (("val", val_idx), ("test", test_idx)):
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        logits[idx], eval_l[idx],
                        arm=args.arm, split=split_name, logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=samples_seen,
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

        # OOD dump: final point only. Fig 2 is a converged-endpoint figure.
        if args.ood_dataset:
            ood_f, ood_l = load_cached_features(
                args.cache_dir, args.model, args.ood_dataset, img_size,
                eval_split_name(args.ood_dataset), hf_repo=hf_repo,
            )
            ood_logits = _logits_for(head, loss_fn, ood_f, ood_l)
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, total_steps,
                ood_logits, ood_l,
                arm=args.arm, split="ood", logit_kind="softmax_logits",
                seed=args.seed, samples_seen=total_steps * args.batch_size,
            )
            print(f"[probe] wrote OOD dump for {args.ood_dataset} ({len(ood_f)} samples)")


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
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
