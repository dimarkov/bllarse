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
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr, vmap

from bllarse import evaldump, metrics, splits
from bllarse.features import eval_split_name, load_cached_features
from bllarse.losses import CrossEntropy


def _ce_epoch(head, opt_state, optimizer, loss_fn, feats, labels, batch_size, key):
    """One AdamW pass over the features."""
    perm = jr.permutation(key, len(feats))
    n_batches = len(feats) // batch_size
    if n_batches == 0:
        # Silently clamping to one batch would dump a plausible-looking but
        # untrained head -- its accuracy/NLL/Brier would be indistinguishable
        # from a real result after the fact. Fail loudly instead.
        raise ValueError(
            f"batch_size={batch_size} exceeds len(feats)={len(feats)}; this "
            f"would train zero gradient steps per epoch. Lower --batch-size "
            f"or provide more training data."
        )
    for i in range(n_batches):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        fb, yb = jnp.asarray(feats[idx]), jnp.asarray(labels[idx])

        def loss(h):
            return loss_fn(vmap(h)(fb), yb).mean()

        grads = eqx.filter_grad(loss)(head)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(head, eqx.is_array)
        )
        head = eqx.apply_updates(head, updates)
    return head, opt_state


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

    schedule = set(evaldump.dump_schedule(args.epochs, n_dumps=args.n_dumps).tolist())
    print(f"[probe] {args.uid}: {args.epochs} epochs, dumping at {sorted(schedule)}")

    for epoch in range(1, args.epochs + 1):
        key, epoch_key = jr.split(key)
        head, opt_state = _ce_epoch(
            head, opt_state, optimizer, loss_fn,
            train_f, train_l, args.batch_size, epoch_key,
        )
        if epoch not in schedule:
            continue

        logits = _logits_for(head, loss_fn, eval_f, eval_l)
        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, epoch,
                logits[idx], eval_l[idx],
                arm=args.arm, split=split_name, logit_kind="softmax_logits",
                seed=args.seed, samples_seen=epoch * len(train_f),
            )
        scalars = metrics.classification_metrics(logits[test_idx], eval_l[test_idx])
        print(f"[probe] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()))

    # OOD dump: final point only. Fig 2 is a converged-endpoint figure.
    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, img_size,
            eval_split_name(args.ood_dataset), hf_repo=hf_repo,
        )
        ood_logits = _logits_for(head, loss_fn, ood_f, ood_l)
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, args.epochs,
            ood_logits, ood_l,
            arm=args.arm, split="ood", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=args.epochs * len(train_f),
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
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
