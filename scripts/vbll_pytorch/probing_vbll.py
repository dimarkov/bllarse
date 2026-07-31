#!/usr/bin/env python3
"""VBLL linear probing on cached DINOv3 features.

Reads the same .npz feature files the JAX arm reads, so the backbone is
byte-identical across arms and Figs 1-4 carry no cross-framework parity risk.
Writes dumps under the shared contract (bllarse.evaldump).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bllarse import evaldump, splits
from bllarse.features import load_cached_features
from vbll_experiments.models import build_vbll_head


def _forward(head, feats, device, chunk=4096):
    """Return (logits, epistemic_score) as numpy arrays.

    `out.predictive` (a `torch.distributions.Categorical` built from VBLL's
    Monte-Carlo-integrated softmax) exposes `.logits` as documented, but it has
    no `.covariance` -- that attribute lives on the pre-softmax logit
    distribution instead, which the public `forward()` does not return.
    `head.logit_predictive(x)` gives that distribution directly; its
    `covariance_diagonal`, summed over classes, is the analytic epistemic
    variance from the last-layer weight posterior -- VBLL's actual native
    uncertainty quantity. (Categorical's own `.variance` is degenerate --
    it evaluates to NaN, since "variance of a category index" is not a
    meaningful quantity for a nominal outcome.)
    """
    head.eval()
    logits_out, score_out = [], []
    with torch.no_grad():
        for start in range(0, len(feats), chunk):
            x = torch.from_numpy(feats[start:start + chunk]).float().to(device)
            out = head(x)
            logits_out.append(out.predictive.logits.cpu().numpy())
            score_out.append(
                head.logit_predictive(x).covariance_diagonal.sum(dim=-1).cpu().numpy()
            )
    return np.concatenate(logits_out), np.concatenate(score_out)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hf_repo = None if args.no_hf_cache else args.hf_repo

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "train", hf_repo=hf_repo
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "test", hf_repo=hf_repo
    )
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, "test", len(eval_f), args.val_fraction
    )

    head = build_vbll_head(
        train_f.shape[-1], args.num_classes,
        regularization_weight=1.0 / len(train_f),
        prior_scale=args.prior_scale,
        wishart_scale=args.wishart_scale,
        parameterization=args.parameterization,
        vbll_type=args.vbll_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    x_train = torch.from_numpy(train_f).float()
    y_train = torch.from_numpy(train_l).long()
    schedule = set(evaldump.dump_schedule(args.epochs, n_dumps=args.n_dumps).tolist())
    print(f"[vbll] {args.uid}: {args.epochs} epochs, dumping at {sorted(schedule)}")

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(x_train))
        n_batches = len(x_train) // args.batch_size
        if n_batches == 0:
            # Silently clamping to one batch would dump a plausible-looking
            # but untrained head -- its accuracy/NLL/Brier would be
            # indistinguishable from a real result after the fact. Fail
            # loudly instead.
            raise ValueError(
                f"--batch-size {args.batch_size} exceeds len(x_train)="
                f"{len(x_train)}; this would train zero gradient steps per "
                f"epoch. Lower --batch-size or provide more training data."
            )
        for i in range(n_batches):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            head(xb).train_loss_fn(yb).backward()
            optimizer.step()

        if epoch not in schedule:
            continue

        logits, score = _forward(head, eval_f, device)
        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, epoch,
                logits[idx], eval_l[idx],
                arm="vbll", split=split_name, logit_kind="softmax_logits",
                seed=args.seed, samples_seen=epoch * len(train_f),
                score=score[idx], score_kind="predictive_variance",
            )
        print(f"[vbll] epoch {epoch}: dumped {len(val_idx)} val / {len(test_idx)} test")

    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, args.img_size, "test", hf_repo=hf_repo
        )
        ood_logits, ood_score = _forward(head, ood_f, device)
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, args.epochs,
            ood_logits, ood_l,
            arm="vbll", split="ood", logit_kind="softmax_logits",
            seed=args.seed, samples_seen=args.epochs * len(train_f),
            score=ood_score, score_kind="predictive_variance",
        )
        print(f"[vbll] wrote OOD dump for {args.ood_dataset} ({len(ood_f)} samples)")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--group-id", default="probe_vbll")
    p.add_argument("--uid", default="vbll_0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-dumps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--wishart-scale", type=float, default=0.1)
    p.add_argument("--parameterization", default="diagonal")
    p.add_argument(
        "--vbll-type", default="discriminative", choices=["discriminative"],
        help=(
            "Only 'discriminative' is supported: GenClassification.logit_predictive "
            "returns a plain tensor (not a distribution with .covariance_diagonal) "
            "in the installed vbll version, so _forward's score computation would "
            "raise AttributeError at score time rather than failing loudly here."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=137)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
