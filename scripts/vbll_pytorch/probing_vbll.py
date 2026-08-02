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
from bllarse.features import eval_split_name, load_cached_features
from bllarse.mlflow_utils import child_run
from bllarse.reporting import report_step
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


def head_param_count(head) -> int:
    """Total parameter count of the head (memory cost, trainable or frozen).

    Counts every stored parameter, including vbll's frozen noise_mean (C elements),
    because the Pareto front plots calibration against *memory cost*, not just
    trainable degrees of freedom. This is why the count differs from a trainable-only
    tally: a frozen parameter still occupies memory on the device.

    Both arms must report this identically or the Pareto front's cost axis is not a
    single axis. See probe.head_param_count for the parallel implementation and
    motivation.
    """
    return int(sum(p.numel() for p in head.parameters()))


def main(args, on_dump=None):
    """Train one head. `on_dump(dump_index, step, scalars)` fires after every
    scheduled dump, if given.

    The index counts dumps rather than optimiser steps because it is Optuna's
    pruning key: MedianPruner compares trials at the SAME key, and trials vary
    batch_size, so two configs would share almost no raw step values. n_dumps is
    40 for every run, so the dump index is a common grid by construction.
    """
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hf_repo = None if args.no_hf_cache else args.hf_repo

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size, "train", hf_repo=hf_repo
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size,
        eval_split_name(args.dataset), hf_repo=hf_repo,
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
        cov_rank=args.cov_rank,
        vbll_type=args.vbll_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    x_train = torch.from_numpy(train_f).float()
    y_train = torch.from_numpy(train_l).long()

    n_batches = len(x_train) // args.batch_size
    if n_batches == 0:
        # Silently clamping to one batch would dump a plausible-looking but
        # untrained head -- its accuracy/NLL/Brier would be indistinguishable
        # from a real result after the fact. Fail loudly instead.
        raise ValueError(
            f"--batch-size {args.batch_size} exceeds len(x_train)="
            f"{len(x_train)}; this would train zero gradient steps per epoch. "
            f"Lower --batch-size or provide more training data."
        )
    total_steps = args.epochs * n_batches
    schedule = set(evaldump.dump_schedule(total_steps, n_dumps=args.n_dumps).tolist())
    print(
        f"[vbll] {args.uid}: {args.epochs} epochs x {n_batches} steps "
        f"= {total_steps} steps ({total_steps * args.batch_size} samples), "
        f"{len(schedule)} dumps"
    )

    with child_run(vars(args), run_name=args.uid, tags={
        "arm": "vbll", "group_id": args.group_id, "uid": args.uid,
        "dump_dir": str(Path(args.dump_root) / args.group_id / args.uid),
    }) as run:
        run.log_params({
            "steps_per_epoch": n_batches,
            "total_steps": total_steps,
            "head_param_count": head_param_count(head),
        })

        step = 0
        dump_index = 0
        for epoch in range(1, args.epochs + 1):
            head.train()
            perm = torch.randperm(len(x_train))
            for i in range(n_batches):
                idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
                xb, yb = x_train[idx].to(device), y_train[idx].to(device)
                optimizer.zero_grad()
                head(xb).train_loss_fn(yb).backward()
                optimizer.step()

                step += 1
                if step not in schedule:
                    continue

                # Exact: drop-remainder means every step is exactly batch_size.
                samples_seen = step * args.batch_size
                logits, score = _forward(head, eval_f, device)
                # _forward leaves the head in eval mode. Evaluating mid-epoch
                # rather than at the epoch boundary means the remaining steps
                # of this epoch would otherwise train in eval mode.
                head.train()
                for split_name, sp_idx in (("val", val_idx), ("test", test_idx)):
                    evaldump.write_dump(
                        args.dump_root, args.group_id, args.uid, step,
                        logits[sp_idx], eval_l[sp_idx],
                        arm="vbll", split=split_name, logit_kind="softmax_logits",
                        seed=args.seed, samples_seen=samples_seen,
                        score=score[sp_idx], score_kind="predictive_variance",
                    )
                scalars = report_step(
                    run, step, samples_seen, epoch,
                    logits[test_idx], eval_l[test_idx],
                    logits[val_idx], eval_l[val_idx],
                )
                print(
                    f"[vbll] step {step} (epoch {epoch}, {samples_seen} samples): "
                    + " ".join(f"{k}={v:.4f}" for k, v in scalars.items())
                )
                if on_dump is not None:
                    on_dump(dump_index, step, scalars)
                dump_index += 1

        if args.ood_dataset:
            ood_f, ood_l = load_cached_features(
                args.cache_dir, args.model, args.ood_dataset, args.img_size,
                eval_split_name(args.ood_dataset), hf_repo=hf_repo,
            )
            ood_logits, ood_score = _forward(head, ood_f, device)
            evaldump.write_dump(
                args.dump_root, args.group_id, args.uid, total_steps,
                ood_logits, ood_l,
                arm="vbll", split="ood", logit_kind="softmax_logits",
                seed=args.seed, samples_seen=total_steps * args.batch_size,
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
        "--cov-rank", type=int, default=2,
        help="Rank of the lowrank posterior. Inert unless "
             "--parameterization lowrank. Head parameters scale as "
             "D*C*(2+cov_rank), so this is a memory dial, not a free knob -- "
             "which is why the figures fix diagonal and only the appendix "
             "Pareto front varies it.",
    )
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
    p.add_argument("--enable-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
