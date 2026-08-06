"""Is VBLL's native epistemic score anything more than the feature norm?

The score dumped by probing_vbll.py is
`logit_predictive(x).covariance_diagonal.sum(-1)`. For a diagonal Gaussian
posterior over the head weights (w_k ~ N(m_k, S_k)) the logits are w_k . x, so

    score(x) = sum_k Var[w_k . x] = sum_k x^T S_k x = x^T diag(v) x,
    v_d = sum_k S_{k,d}

which involves the posterior COVARIANCE only -- never the mean. It is therefore
blind to where the decision boundaries are, and if v is near-isotropic it
degenerates to a constant times ||x||^2.

AUROC depends only on the RANKING of scores, so the decisive test is Spearman,
not Pearson: if score is a monotone function of ||x||^2 then substituting
||x||^2 for it reproduces the AUROC exactly. That is checked directly here
rather than inferred from a correlation, because a high correlation alone would
still leave room for the ranking to differ where it matters.
"""

import argparse
import os

import numpy as np

from bllarse import detection, evaldump
from bllarse.features import eval_split_name, load_cached_features
from bllarse import splits


def spearman(a, b):
    """Rank correlation, computed by hand to avoid a scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def final_dump(root, group_id, uid, split):
    best = None
    for record in evaldump.iter_dumps(root, group_id, uid):
        if record.meta["split"] != split:
            continue
        if best is None or record.meta["step"] > best.meta["step"]:
            best = record
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--group-id", default="probe_vbll_fig2")
    p.add_argument("--uid", default="dinov3_huge_cifar10_vs_svhn_seed137")
    p.add_argument("--cache-dir", default=".cache/features")
    p.add_argument("--split-cache-dir", default=".cache/splits")
    p.add_argument("--model", default="dinov3_huge")
    p.add_argument("--id-dataset", default="cifar10")
    p.add_argument("--ood-dataset", default="svhn")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    test_rec = final_dump(args.dump_root, args.group_id, args.uid, "test")
    ood_rec = final_dump(args.dump_root, args.group_id, args.uid, "ood")
    if test_rec is None or ood_rec is None:
        raise SystemExit(f"missing test/ood dump for {args.group_id}/{args.uid}")
    if test_rec.score is None or ood_rec.score is None:
        raise SystemExit("this run carries no native score; use a VBLL run")

    # ID side: the dump is eval_f[test_idx], so the same pinned split has to be
    # reproduced to line features up with scores.
    eval_f, _ = load_cached_features(
        args.cache_dir, args.model, args.id_dataset, args.img_size,
        eval_split_name(args.id_dataset),
    )
    val_idx, test_idx = splits.load_or_make_pinned_split(
        args.split_cache_dir, args.id_dataset, eval_split_name(args.id_dataset),
        len(eval_f), args.val_fraction,
    )
    id_feats = np.asarray(eval_f)[test_idx]

    # OOD side: the dump is the whole eval split, in order. No split to redo.
    ood_feats, _ = load_cached_features(
        args.cache_dir, args.model, args.ood_dataset, args.img_size,
        eval_split_name(args.ood_dataset),
    )
    ood_feats = np.asarray(ood_feats)

    id_score = np.asarray(test_rec.score, dtype=np.float64)
    ood_score = np.asarray(ood_rec.score, dtype=np.float64)
    if len(id_feats) != len(id_score) or len(ood_feats) != len(ood_score):
        raise SystemExit(
            f"length mismatch -- id {len(id_feats)} vs {len(id_score)}, "
            f"ood {len(ood_feats)} vs {len(ood_score)}; the split or the cached "
            f"features do not match the run that produced these dumps"
        )

    id_norm2 = np.einsum("ij,ij->i", id_feats, id_feats).astype(np.float64)
    ood_norm2 = np.einsum("ij,ij->i", ood_feats, ood_feats).astype(np.float64)

    print(f"run: {args.group_id}/{args.uid}")
    print(f"  ID  {args.id_dataset}: {len(id_score)} points")
    print(f"  OOD {args.ood_dataset}: {len(ood_score)} points\n")

    for name, score, norm2 in (("ID", id_score, id_norm2), ("OOD", ood_score, ood_norm2)):
        pear = float(np.corrcoef(score, norm2)[0, 1])
        print(f"{name:<4s} pearson(score, ||x||^2) = {pear:+.6f}   "
              f"spearman = {spearman(score, norm2):+.6f}")

    pooled_score = np.concatenate([id_score, ood_score])
    pooled_norm2 = np.concatenate([id_norm2, ood_norm2])
    print(f"\npooled spearman = {spearman(pooled_score, pooled_norm2):+.6f}   "
          f"(1.0 means identical ranking, hence identical AUROC)")

    native = detection.detection_metrics(id_score, ood_score)
    norm_only = detection.detection_metrics(id_norm2, ood_norm2)
    print("\n                      AUROC     FPR@95      AUPR")
    print(f"  native score     {native['auroc']:>8.5f} {native['fpr95']:>10.5f} {native['aupr']:>9.5f}")
    print(f"  ||x||^2 alone    {norm_only['auroc']:>8.5f} {norm_only['fpr95']:>10.5f} {norm_only['aupr']:>9.5f}")
    delta = abs(native["auroc"] - norm_only["auroc"])
    print(f"\n  |dAUROC| = {delta:.6f}")
    if delta < 1e-4:
        print("  => the native score carries NO information beyond the feature norm.")
    elif delta < 5e-3:
        print("  => the learned per-dimension weighting changes almost nothing.")
    else:
        print("  => the learned weighting contributes materially; not just norm.")


if __name__ == "__main__":
    main()
