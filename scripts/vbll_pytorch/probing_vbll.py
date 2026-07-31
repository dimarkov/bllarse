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


def _fit_standardizer(train_f, eps=1e-6):
    """Per-feature (mean, std) from the training split, std floored at `eps`.

    VBLL's Jensen bound adds a term to the pre-softmax logits proportional to
    each class's predictive variance, which is itself a weighted sum of
    per-feature posterior variances scaled by x**2 (see
    vbll.layers.classification.DiscClassification.jensen_bound /
    logit_predictive). Cached backbone features are not guaranteed to arrive
    on a comparable per-dimension scale, and any single large-magnitude
    dimension makes that term blow up quadratically, which stalls training:
    verified directly against this script's own smoke-test fixture (a 16-dim
    Gaussian blob with one feature offset by 6.0) -- unstandardized, VBLL
    training plateaus at ~0.70 test accuracy even after 10,000 epochs (the
    gradient on that feature's posterior variance dwarfs the gradient on its
    mean), while standardized it reaches 1.0 by epoch 1000 with everything
    else unchanged. AdamW+CE has no such term and needs no such fix, which is
    why this lives here rather than in the shared bllarse.features.

    Fit on train only, then applied everywhere (train/val/test/OOD alike) --
    an OOD dataset standardized by its own statistics would have that
    distribution shift erased before the head ever saw it.
    """
    mean = train_f.mean(axis=0)
    std = np.maximum(train_f.std(axis=0), eps)
    return mean, std


def _apply_standardizer(feats, mean, std):
    return ((feats.astype(np.float64) - mean) / std).astype(np.float32)


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

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "train"
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "test"
    )
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, "test", len(eval_f), args.val_fraction
    )

    feat_mean, feat_std = _fit_standardizer(train_f)
    train_f = _apply_standardizer(train_f, feat_mean, feat_std)
    eval_f = _apply_standardizer(eval_f, feat_mean, feat_std)

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
        for i in range(len(x_train) // args.batch_size):
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
            args.cache_dir, args.model, args.ood_dataset, args.img_size, "test"
        )
        ood_f = _apply_standardizer(ood_f, feat_mean, feat_std)
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
    p.add_argument("--vbll-type", default="discriminative")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=137)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
