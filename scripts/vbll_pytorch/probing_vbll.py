#!/usr/bin/env python3
"""VBLL linear probing on cached DINOv3 features.

Reads the same .npz feature files the JAX arm reads, so the backbone is
byte-identical across arms and Figs 1-4 carry no cross-framework parity risk.
Writes dumps under the shared contract (bllarse.evaldump).
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bllarse import evaldump, metrics, splits
from bllarse.features import eval_split_name, load_cached_features, train_split_name
from bllarse.mlflow_utils import child_run
from bllarse.reporting import report_step
from bllarse.tuning import space
from vbll_experiments.models import build_vbll_head


def _forward(head, feats, device, n_samples):
    """Return (logits, epistemic_score) as numpy arrays.

    Calls `head.predictive(x, n_samples=...)` rather than reading
    `head(x).predictive`: vbll's `forward` builds that Categorical from
    `self.predictive(x)` at its default `n_samples = 20` with no pass-through
    (classification.py:147), and 20 samples is a badly biased estimate of
    `E_w[softmax(w.x)]` whenever the posterior is wide relative to the decision
    margin. Measured on stanford_cars, argmax through that noise cost 8.9 points
    of accuracy at the old campaign's operating point and 15.2 at prior_scale
    1e3, against a 2,000-sample reference. The bias is one-directional -- it
    penalises wide posteriors -- so leaving it in place would bias the tuner's
    own objective against posterior width, which is the quantity the paper
    measures.

    The score comes from `head.logit_predictive(x)` instead, because
    `out.predictive` (a `torch.distributions.Categorical`) has no `.covariance`
    -- that attribute lives on the pre-softmax logit distribution, which the
    public `forward()` does not return. Its `covariance_diagonal`, summed over
    classes, is the analytic epistemic variance from the last-layer weight
    posterior -- VBLL's actual native uncertainty quantity. (Categorical's own
    `.variance` is degenerate -- it evaluates to NaN, since "variance of a
    category index" is not meaningful for a nominal outcome.)
    """
    head.eval()
    # rsample materialises [n_samples, chunk, C], so the chunk has to fall as
    # n_samples rises or the eval OOMs where training did not: a fixed chunk of
    # 4096 allocated 16 GiB once n_samples passed a few thousand.
    n_classes = head.W_mean.shape[0]
    chunk = max(64, int(2e8 / (n_samples * n_classes)))
    logits_out, score_out = [], []
    with torch.no_grad():
        for start in range(0, len(feats), chunk):
            x = torch.from_numpy(feats[start:start + chunk]).float().to(device)
            probs = head.predictive(x, n_samples=n_samples)
            # log, so the returned array stays log-probabilities as before.
            # clamp_min guards the log: predictive() clips to [0, 1] and a
            # class no sample lands on comes back exactly 0.
            logits_out.append(torch.log(probs.clamp_min(1e-12)).cpu().numpy())
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


def _derive_quantum(n_train: int, batch_size: int) -> int:
    """The eval quantum when --eval-quantum is not given. The vbll twin of the
    helper in scripts/dinov3/probe.py; see the note there."""
    try:
        cap = space.batch_cap(n_train)
    except ValueError:
        return batch_size
    return cap if cap >= batch_size and cap % batch_size == 0 else batch_size


def _write_best(args, best, last_eval_index):
    """Write the buffered argmin-val_nll evaluation. The run's ONLY disk write.

    The vbll twin of `_write_best` in scripts/dinov3/probe.py; see the note
    there. Called from a `finally` so an EarlyConvergence raised out of
    `on_dump` still leaves the run scoreable.
    """
    if best is None:
        raise ValueError(
            f"{args.uid}: no evaluation ran, so there is no dump to write. "
            f"The step budget is shorter than one eval interval."
        )
    for split_name, logits, labels, score in best["splits"]:
        evaldump.write_dump(
            args.dump_root, args.group_id, args.uid, best["step"],
            logits, labels,
            arm="vbll", split=split_name, logit_kind="softmax_logits",
            seed=args.seed, samples_seen=best["samples_seen"],
            score=score, score_kind="predictive_variance",
            # See the matching note in scripts/dinov3/probe.py.
            at_last_eval=best["eval_index"] == last_eval_index,
        )
    print(
        f"[vbll] {args.uid}: best val_nll={best['val_nll']:.4f} at step "
        f"{best['step']} (eval {best['eval_index']} of {last_eval_index}, "
        f"epoch {best['epoch']}, {best['samples_seen']} samples)"
        f"{' AT LAST EVAL -- truncated, not converged' if best['eval_index'] == last_eval_index else ''}"
        f"; wrote {len(best['splits'])} dumps",
        flush=True,
    )


def main(args, on_dump=None):
    """Train one head. `on_dump(eval_index, step, samples_seen, scalars)` fires
    after every scheduled evaluation, if given.

    `eval_index` counts evaluations rather than optimiser steps because it is
    Optuna's pruning key: MedianPruner compares trials at the SAME key, and
    trials vary batch_size. Evaluations are `eval_quantum` SAMPLES apart (see
    tuning.space.eval_quantum), so index i is the same samples_seen for every
    trial in a study.
    """
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hf_repo = None if args.no_hf_cache else args.hf_repo

    train_f, train_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size,
        train_split_name(args.dataset), hf_repo=hf_repo,
    )
    eval_f, eval_l = load_cached_features(
        args.cache_dir, args.model, args.dataset, args.img_size,
        eval_split_name(args.dataset), hf_repo=hf_repo,
    )
    # The cache key is the eval split's real name, not the literal "test". The
    # indices themselves do not depend on it -- splits.make_pinned_split seeds
    # from the dataset name alone -- so this only renames food101's cache file
    # and leaves its partition byte-identical. It matters for flowers102, whose
    # eval pool is now the 7,169-image "train" file: under the old hardcoded key
    # this would collide with a cached 1,020-entry split and raise on size drift.
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.dataset, eval_split_name(args.dataset),
        len(eval_f), args.val_fraction,
    )

    head = build_vbll_head(
        train_f.shape[-1], args.num_classes,
        # eta / N, not 1 / N. eta = kl_ratio * prior_scale is the KL multiplier;
        # the ELBO's principled eta = 1 is recovered at kl_ratio = 1 /
        # prior_scale. Shrinkage of the posterior mean goes as eta / prior_scale
        # = kl_ratio, so searching the ratio directly makes the two axes
        # near-orthogonal: kl_ratio controls fit, prior_scale controls width.
        regularization_weight=args.kl_ratio * args.prior_scale / len(train_f),
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
    # Loaded once, before the loop: the OOD set is dumped at EVERY eval step, so
    # re-reading the feature file per dump would cost more than the forward pass
    # it feeds.
    ood_f = ood_l = None
    if args.ood_dataset:
        ood_f, ood_l = load_cached_features(
            args.cache_dir, args.model, args.ood_dataset, args.img_size,
            eval_split_name(args.ood_dataset), hf_repo=hf_repo,
        )
        print(f"[vbll] OOD set {args.ood_dataset}: {len(ood_f)} samples")

    total_steps = args.epochs * n_batches

    # Evaluation is scheduled on SAMPLES, matching scripts/dinov3/probe.py
    # exactly -- see the long note there. The two arms must share the grid or
    # Fig 1 compares them at different points of their own training.
    quantum = args.eval_quantum or _derive_quantum(len(train_f), args.batch_size)
    if quantum % args.batch_size:
        raise ValueError(
            f"--eval-quantum {quantum} is not a whole number of "
            f"batch_size={args.batch_size} steps. Evaluation happens between "
            f"optimiser steps, so a fractional interval is unschedulable."
        )
    eval_every = quantum // args.batch_size
    report_grid = set(evaldump.dump_schedule(total_steps, n_dumps=args.n_dumps).tolist())
    print(
        f"[vbll] {args.uid}: {args.epochs} epochs x {n_batches} steps "
        f"= {total_steps} steps ({total_steps * args.batch_size} samples), "
        f"eval every {eval_every} steps ({quantum} samples, "
        f"{quantum / len(train_f):.2f} epochs) = {total_steps // eval_every} evals, "
        f"{len(report_grid)} reported"
    )

    with child_run(vars(args), run_name=args.uid, tags={
        "arm": "vbll", "group_id": args.group_id, "uid": args.uid,
        "dump_dir": str(Path(args.dump_root) / args.group_id / args.uid),
    }) as run:
        run.log_params({
            "steps_per_epoch": n_batches,
            "total_steps": total_steps,
            "head_param_count": head_param_count(head),
            # Derived, not searched: eta is the departure from the ELBO, so it
            # must be visible per trial rather than reconstructed from two axes.
            "eta": args.kl_ratio * args.prior_scale,
            "regularization_weight": args.kl_ratio * args.prior_scale / len(train_f),
            # Which pool was which, since flowers102's two are swapped.
            "train_split": train_split_name(args.dataset),
            "eval_split": eval_split_name(args.dataset),
            "n_train": len(train_f),
            "n_eval": len(eval_f),
        })

        step = 0
        # See scripts/dinov3/probe.py: the argmin evaluation is buffered and
        # written once, from a `finally` so an EarlyConvergence raised out of
        # `on_dump` cannot leave the run with nothing on disk.
        best = None
        eval_index = 0
        val_f, val_l = eval_f[val_idx], eval_l[val_idx]
        test_f, test_l = eval_f[test_idx], eval_l[test_idx]

        try:
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
                    if step % eval_every:
                        continue

                    # Exact: drop-remainder means every step is exactly batch_size.
                    samples_seen = step * args.batch_size
                    # Position on the SAMPLE grid -- see probe.py.
                    eval_index = samples_seen // quantum

                    # Val only. `_forward` draws `predictive_samples` MC samples
                    # per call, so at this cadence the test and OOD blocks stay
                    # out of it until an evaluation actually improves.
                    val_logits, val_score = _forward(
                        head, val_f, device, args.predictive_samples
                    )
                    # _forward leaves the head in eval mode. Evaluating mid-epoch
                    # rather than at the epoch boundary means the remaining steps
                    # of this epoch would otherwise train in eval mode.
                    head.train()
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
                        test_logits, test_score = _forward(
                            head, test_f, device, args.predictive_samples
                        )
                        head.train()
                        best = {
                            "val_nll": float(val_nll),
                            "step": step,
                            "samples_seen": samples_seen,
                            "epoch": epoch,
                            "eval_index": eval_index,
                            "splits": [
                                ("val", val_logits, val_l, val_score),
                                ("test", test_logits, test_l, test_score),
                            ],
                        }
                        # OOD at the val-optimal point, not the last step -- see
                        # the matching note in scripts/dinov3/probe.py. A single
                        # endpoint dump forced Fig 2's detection metrics to be
                        # read at a step no arm was tuned at, and the arms drift
                        # from that step by very different amounts.
                        if ood_f is not None:
                            ood_logits, ood_score = _forward(
                                head, ood_f, device, args.predictive_samples
                            )
                            head.train()
                            best["splits"].append(
                                ("ood", ood_logits, ood_l, ood_score)
                            )

                    scalars = {"val_nll": float(val_nll)}
                    if step in report_grid:
                        test_logits, _ = _forward(
                            head, test_f, device, args.predictive_samples
                        )
                        head.train()
                        scalars = report_step(
                            run, step, samples_seen, epoch,
                            test_logits, test_l, val_logits, val_l,
                        )
                        print(
                            f"[vbll] step {step} (epoch {epoch}, "
                            f"{samples_seen} samples): "
                            + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()),
                            flush=True,
                        )
                    if on_dump is not None:
                        on_dump(eval_index, step, samples_seen, scalars)
        finally:
            _write_best(args, best, eval_index)


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
    p.add_argument("--eval-quantum", type=int, default=0,
                   help="Samples between evaluations. 0 derives it from the "
                        "dataset's batch ladder (tuning.space.eval_quantum).")
    p.add_argument("--n-dumps", type=int, default=40,
                   help="Points on the geometric grid that throttles MLflow "
                        "and stdout. NOT the evaluation grid.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--prior-scale", type=float, default=1.0)
    p.add_argument("--wishart-scale", type=float, default=0.1)
    p.add_argument(
        "--kl-ratio", type=float, default=1.0,
        help="eta / prior_scale, where eta is the KL multiplier in "
             "regularization_weight = eta / N. The default 1.0 combined with "
             "the default --prior-scale 1.0 gives eta = 1, the textbook ELBO "
             "and this script's previous behaviour. There is no single measured "
             "optimum: across the seven tuned datasets it spans 2.3e-7 to "
             "2.5e-3, with the coarse 10/100-class sets tolerating ~1e-3 and the "
             "fine-grained ones wanting ~1e-7. The corresponding eta spans 9e-8 "
             "to 1.19, so 'vbll needs a strongly tempered posterior' holds for "
             "most datasets but not all -- cifar10's optimum sits essentially at "
             "the untempered eta = 1.",
    )
    p.add_argument(
        "--predictive-samples", type=int, default=200,
        help="Monte Carlo samples for E_w[softmax(w.x)] at eval time. vbll's "
             "own default is 20, which is visibly biased when the posterior is "
             "wide relative to the decision margin -- see _forward.",
    )
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
