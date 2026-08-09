"""learn_q vs fixed-q across alpha, at a fixed CAVI iteration budget.

Reuses the training/eval code from scripts/dinov3/vit_classification.py, so the
numbers match that script -- this only adds the grid, multi-seed repeats, and a paired
table. Features are loaded once per dataset and shared across the whole grid.

The grid is (alpha) x (learn_q on/off), everything else held fixed. For each alpha the
two arms are printed adjacently with their difference, since that difference is the
quantity of interest.

Usage (from the worktree root):

    CUDA_VISIBLE_DEVICES=2 uv run python scripts/analysis/compare_variants.py \
        --datasets dtd --alphas 0.03 0.1 0.3 --iters 128 --seeds 0
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

# Load XLA_* before anything can initialize the JAX backend. vit_classification
# does this too, but it is imported after jax below -- do not rely on that ordering.
load_dotenv()

# Resolve the repo root from cwd -- run this from the worktree you want to test.
REPO = os.getcwd()
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "dinov3"))

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from vit_classification import (  # noqa: E402
    DATASET_CONFIGS,
    EQUIMO_MODELS,
    get_img_size,
    train_epoch,
    evaluate_model,
)
from bllarse.losses import IBProbit, CrossEntropy  # noqa: E402


def load_features(model, dataset, cache_dir):
    cfg = DATASET_CONFIGS[dataset]
    img_size = get_img_size(model, dataset)
    test_split = cfg["test_split"]
    tr = os.path.join(cache_dir, f"{model}_{dataset}_res{img_size}_train.npz")
    te = os.path.join(cache_dir, f"{model}_{dataset}_res{img_size}_{test_split}.npz")
    for p in (tr, te):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"no cached features at {p}\n"
                f"  precompute them with scripts/dinov3/vit_classification.py "
                f"--model {model} --dataset {dataset}"
            )
    a, b = np.load(tr), np.load(te)
    return a["features"], a["labels"], b["features"], b["labels"], cfg["num_classes"]


def run_one(xtr, ytr, xte, yte, num_classes, batch_size, epochs, seed,
            *, ce=False, **ib_kwargs):
    """Train one configuration for `epochs`; return the best-test-NLL metrics."""
    embed_dim = xtr.shape[-1]
    n_train = len(xtr)
    key = jr.PRNGKey(seed)
    key, loss_key = jr.split(key)

    head = opt_state = optimizer = None
    if ce:
        loss_fn = CrossEntropy(0.0, num_classes)
        key, head_key = jr.split(key)
        head = eqx.nn.Linear(embed_dim, num_classes, key=head_key)
        optimizer = optax.adamw(1e-3, weight_decay=1e-4)
        opt_state = optimizer.init(eqx.filter(head, eqx.is_array))
        iters = 0
    else:
        iters = ib_kwargs.pop("iters")
        loss_fn = IBProbit(
            embed_dim, num_classes, key=loss_key,
            batch_size=batch_size,
            # eps forgetting factor: one-pass Beta contamination memory (as in the script)
            rho=max(0.0, 1.0 - batch_size / n_train),
            **ib_kwargs,
        )

    best = (float("inf"), None)
    t0 = time.time()
    for _ in range(epochs):
        key, train_key = jr.split(key)
        loss_fn, head, opt_state, _ = train_epoch(
            loss_fn, xtr, ytr, batch_size, iters, loss_type=3, key=train_key,
            head=head, opt_state=opt_state, optimizer=optimizer,
        )
        acc, ece, nll = evaluate_model(loss_fn, xte, yte, batch_size, loss_type=3, head=head)
        if float(nll) < best[0]:
            best = (float(nll), (float(acc), float(ece), float(nll)))

    acc, ece, nll = best[1]
    out = dict(acc=acc, ece=ece, nll=nll, secs=time.time() - t0)
    if not ce:
        out["q"] = float(jnp.trace(loss_fn.Qhat) / loss_fn.Qhat.shape[0])
        out["eps"] = float(loss_fn.eps)
    return out


def agg(runs, k):
    v = np.array([r[k] for r in runs])
    return v.mean(), v.std()


def cell(runs, k, nseeds):
    m, s = agg(runs, k)
    return f"{m:.4f}" if nseeds == 1 else f"{m:.4f}±{s:.4f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov3_small", choices=list(EQUIMO_MODELS.keys()))
    p.add_argument("--datasets", nargs="+", default=["dtd"],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.1, 0.3],
                   help="process-noise scales; q0 = alpha^2 / batch_size")
    p.add_argument("--iters", type=int, default=128, help="CAVI iterations per batch")
    p.add_argument("--no-px", action="store_true", help="plain CAVI instead of PX-VB")
    p.add_argument("--kappa", type=float, default=None,
                   help="Inverse-Wishart pseudo-count (default: the library's 0.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--with-ce", action="store_true", help="add a CrossEntropy+AdamW row")
    p.add_argument("--cache-dir", default="/home/dima/projects/bllarse/.cache/features")
    args = p.parse_args()

    px = not args.no_px
    kappa = args.kappa
    nseeds = len(args.seeds)
    w = 16 if nseeds > 1 else 8

    for dataset in args.datasets:
        xtr, ytr, xte, yte, K = load_features(args.model, dataset, args.cache_dir)
        common = dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, num_classes=K,
                      batch_size=args.batch_size, epochs=args.epochs)

        print(f"\n{'='*(46 + 3*w)}")
        print(f"{args.model} / {dataset}   train={xtr.shape} test={xte.shape} K={K}")
        print(f"B={args.batch_size}  px={px}  iters={args.iters}  "
              f"epochs={args.epochs}  seeds={args.seeds}  kappa={kappa or K}")
        print(f"{'='*(46 + 3*w)}")
        print(f"{'alpha':>8} {'learn_q':<9} {'acc':>{w}} {'ece':>{w}} {'nll':>{w}} "
              f"{'q':>10} {'eps':>9} {'s/run':>7}")

        for alpha in args.alphas:
            arms = {}
            for learn_q in (True, False):
                runs = [run_one(**common, seed=s, px=px, learn_q=learn_q, alpha=alpha,
                                kappa=kappa, iters=args.iters) for s in args.seeds]
                arms[learn_q] = runs
                q, _ = agg(runs, "q")
                eps, _ = agg(runs, "eps")
                secs, _ = agg(runs, "secs")
                print(f"{alpha:>8g} {'yes' if learn_q else 'no':<9} "
                      f"{cell(runs, 'acc', nseeds):>{w}} {cell(runs, 'ece', nseeds):>{w}} "
                      f"{cell(runs, 'nll', nseeds):>{w}} {q:>10.3g} {eps:>9.3g} {secs:>7.1f}")

            # The paired difference is the quantity of interest: fixed-q minus learned-q.
            d = {k: agg(arms[False], k)[0] - agg(arms[True], k)[0] for k in ("acc", "ece", "nll")}
            print(f"{'':>8} {'Δ(no-yes)':<9} {d['acc']:>+{w}.4f} {d['ece']:>+{w}.4f} "
                  f"{d['nll']:>+{w}.4f}")

        if args.with_ce:
            runs = [run_one(**common, seed=s, ce=True) for s in args.seeds]
            secs, _ = agg(runs, "secs")
            print(f"{'-':>8} {'ce':<9} {cell(runs, 'acc', nseeds):>{w}} "
                  f"{cell(runs, 'ece', nseeds):>{w}} {cell(runs, 'nll', nseeds):>{w}} "
                  f"{'-':>10} {'-':>9} {secs:>7.1f}")


if __name__ == "__main__":
    main()
