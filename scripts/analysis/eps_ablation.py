"""Does IBProbit need its contamination parameter at all?

Pairs `ibprobit` (eps_init tuned per dataset) against `ibprobit_noeps`
(eps_init pinned at 1e-12, i.e. the mixture floor numerically absent) and
reports the PAIRED difference.

Covers BOTH campaigns: `--figure fig1` is the backbone ladder x five datasets,
`--figure fig2` the OOD campaign on dinov3_huge. Fig 2's tuned arm existed and
was read by nothing at all -- 30 runs of dumps whose only mention anywhere was
the sweep file that wrote them -- so the eps claim rested on the Fig 1 campaign
alone while the evidence for the other sat unexamined on disk.

Fig 2's 30 runs are 20 trainings: cifar10 is the ID set of two OOD pairs and the
OOD set is not an input to training, so those runs are submitted twice and
collapse to one. Everything compared here is an in-distribution metric, so the
pairing key is the ID dataset and the duplicates are reported, not pooled.

Paired, not two independent means: the two arms share the backbone ladder, the
five datasets and the ten reported seeds, so pairing on (model, dataset, seed)
removes the between-run variance that dominates an unpaired comparison. Runs
present in only one arm are dropped and counted rather than silently averaged
over a different population -- the two sweeps drain at different rates, so the
matched subset is what is actually comparable at any given moment.

Scoring is `fig1_curves.score_run`, not a reimplementation, so these numbers
are the ones the figure will show.

Usage:
    python scripts/analysis/eps_ablation.py --dump-root /raven/ptmp/cheins/eval_dumps
"""

import argparse
import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from bllarse import evaldump, metrics  # noqa: E402
from fig1_curves import UID_RE, score_run  # noqa: E402
from fig2_detection import UID_RE as UID_RE_FIG2  # noqa: E402

# 4 backbones x 5 datasets x 10 seeds. Enumerated so a run that produced NO
# dump at all is reported as such rather than quietly shrinking the matched
# set: a diverged run raises before writing, so its absence IS the result.
MODELS = ("dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge")
DATASETS = ("oxford_pets", "dtd", "flowers102", "stanford_cars", "food101")
SEEDS = tuple(range(137, 147))

# Fig 2 is dinov3_huge only -- it has no backbone axis, so the other three rungs
# were never run (see bllarse_sweeps/probe_ood_adamw_ce.py).
OOD_MODELS = ("dinov3_huge",)
OOD_PAIRS = (("cifar10", "svhn"), ("cifar10", "cifar100"), ("cifar100", "svhn"))

# The tuned arm first: deltas below are (noeps - tuned), so a POSITIVE delta on
# a lower-is-better metric means dropping eps HURT.
TUNED_GROUP = "probe_ibprobit_fig1"
NOEPS_GROUP = "probe_ibprobit_noeps_fig1"


def _fig1_expected():
    return tuple(
        f"{m}_{d}_seed{s}" for m in MODELS for d in DATASETS for s in SEEDS
    )


def _fig2_expected():
    return tuple(
        f"{m}_{i}_vs_{o}_seed{s}"
        for m in OOD_MODELS for i, o in OOD_PAIRS for s in SEEDS
    )


# What each campaign's ablation is made of. `key` is what pairing happens on,
# and its second slot is what `summarize` groups by -- the dataset for Fig 1,
# the ID/OOD pair for Fig 2, so one summariser serves both.
FIGURES = {
    "fig1": {
        "tuned": TUNED_GROUP,
        "noeps": NOEPS_GROUP,
        "uid_re": UID_RE,
        "expected": _fig1_expected,
        "key": lambda m: (m["model"], m["dataset"], int(m["seed"])),
        "label": "backbone ladder x 5 datasets",
    },
    "fig2": {
        "tuned": "probe_ibprobit_fig2",
        "noeps": "probe_ibprobit_noeps_fig2",
        "uid_re": UID_RE_FIG2,
        "expected": _fig2_expected,
        # Keyed on the ID DATASET, not on the ID/OOD pair. Every metric this
        # ablation compares (nll, acc, ece, brier, and their TS variants) is an
        # in-distribution metric, and the OOD set is not an input to training --
        # so cifar10's two pairs are one training run submitted twice, not two.
        #
        # Keying on the pair made that visible and wrong: cifar10_vs_svhn and
        # cifar10_vs_cifar100 came out byte-identical on every column, and the
        # pooled counts read "n=30, noeps better on 2" when only 20 of those
        # rows were independent. Same double-count as fig2_detection.summarize_id.
        "key": lambda m: (m["model"], m["id"], int(m["seed"])),
        "label": "2 ID datasets on dinov3_huge (3 OOD pairs, deduplicated)",
    },
}

# Lower is better for all but acc.
#
# No `ts_*`: both arms here are IBProbit, and temperature scaling is reported
# for AdamW+CE alone (see curves.TS_ARMS). An earlier version of this list
# carried ts_nll and ts_ece, which meant the ablation's headline "0 of 20 pairs
# favour eps-free after temperature scaling" was a statement about a
# temperature-scaled IBProbit -- a quantity the project does not report.
METRICS = ("val_nll", "nll", "acc", "ece", "brier")
LOWER_IS_BETTER = {"val_nll", "nll", "ece", "brier"}


def collect_group(root, group_id, uid_re=UID_RE, key_fn=None, duplicates=None):
    """{key: scored} for one group, keyed by `key_fn` over the uid match.

    `uid_re`/`key_fn` are parameters rather than module constants because the
    two campaigns name their runs differently -- `model_dataset_seedN` against
    `model_id_vs_ood_seedN` -- while everything downstream is identical.
    """
    key_fn = key_fn or FIGURES["fig1"]["key"]
    out = {}
    duplicates = [] if duplicates is None else duplicates
    directory = os.path.join(root, group_id)
    if not os.path.isdir(directory):
        return out
    for uid in sorted(os.listdir(directory)):
        match = uid_re.match(uid)
        if not match:
            continue
        scored = score_run(root, group_id, uid)
        if scored is None:
            continue
        # val_nll at the same step the test metrics are read from. This is the
        # SELECTION criterion, so it belongs beside the reported numbers -- an
        # arm can win on val and lose on test, and that gap is worth seeing.
        for record in evaldump.iter_dumps(root, group_id, uid):
            if record.meta["split"] == "val" and record.meta["step"] == scored["step"]:
                scored["val_nll"] = float(metrics.nll(record.logits, record.labels))
                break
        key = key_fn(match)
        if key in out:
            # Collapsed, not overwritten silently. Two uids mapping to one key
            # means the same training was submitted twice (see the fig2 key),
            # and a reader is entitled to know the pooled n is not the run count.
            duplicates.append(uid)
            continue
        out[key] = scored
    return out


def missing_runs(root, group_id, expected):
    """Expected uids with no dump directory at all.

    A diverged run raises inside _write_best before writing anything, so it
    leaves no dump. Counting those separately matters here: "this arm failed to
    produce a run" is an outcome of the ablation, not a gap in the data.
    """
    directory = os.path.join(root, group_id)
    present = set(os.listdir(directory)) if os.path.isdir(directory) else set()
    return sorted(uid for uid in expected if uid not in present)


def pair(tuned, noeps):
    """Matched rows, plus what each arm has that the other does not."""
    shared = sorted(set(tuned) & set(noeps))
    return (
        [(k, tuned[k], noeps[k]) for k in shared],
        sorted(set(tuned) - set(noeps)),
        sorted(set(noeps) - set(tuned)),
    )


def _stats(values):
    return {
        "mean": st.mean(values),
        "std": st.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize(pairs, by="model_dataset"):
    """Per group and overall, for each metric.

    `by="dataset"` collapses the four backbones, which is the view for deciding
    which arm a FIGURE should show: the figure has a backbone axis, so a
    per-dataset verdict is what carries across it.
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for (model, dataset, _seed), tuned, noeps in pairs:
        model = model if by == "model_dataset" else ""
        for metric in METRICS:
            if metric in tuned and metric in noeps:
                buckets[(model, dataset)][metric].append(
                    (tuned[metric], noeps[metric])
                )

    rows = {}
    for (model, dataset), by_metric in sorted(
            buckets.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        entry = {}
        for metric, values in by_metric.items():
            tuned_v = [t for t, _ in values]
            noeps_v = [n for _, n in values]
            deltas = [n - t for t, n in values]
            entry[metric] = {
                "tuned": _stats(tuned_v),
                "noeps": _stats(noeps_v),
                "delta": _stats(deltas),
                # How many of the paired seeds moved in the delta's direction.
                # A mean delta carried by one outlier reads very differently
                # from one that every seed agrees on.
                "seeds_agreeing": sum(
                    1 for d in deltas if (d > 0) == (st.mean(deltas) > 0)
                ),
            }
        rows[f"{model}|{dataset}" if model else dataset] = entry
    return rows


def overall(pairs):
    out = {}
    for metric in METRICS:
        deltas = [
            n[metric] - t[metric]
            for _k, t, n in pairs
            if metric in t and metric in n
        ]
        if deltas:
            out[metric] = {
                **_stats(deltas),
                "noeps_better": sum(
                    1 for d in deltas
                    if (d < 0) == (metric in LOWER_IS_BETTER)
                ),
            }
    return out


def format_table(rows, key_metric="nll"):
    lines = [
        f"{'model|dataset':38s} {'tuned eps':>16s} {'eps=1e-12':>16s} "
        f"{'delta':>16s} {'agree':>6s}",
        "-" * 96,
    ]
    for name, entry in rows.items():
        if key_metric not in entry:
            continue
        e = entry[key_metric]
        lines.append(
            f"{name:38s} "
            f"{e['tuned']['mean']:8.4f}+-{e['tuned']['std']:<6.4f} "
            f"{e['noeps']['mean']:8.4f}+-{e['noeps']['std']:<6.4f} "
            f"{e['delta']['mean']:+8.4f}+-{e['delta']['std']:<6.4f} "
            f"{e['seeds_agreeing']:2d}/{e['delta']['n']:<3d}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root",
                   default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--metric", default="nll", choices=METRICS)
    p.add_argument("--by", default="dataset",
                   choices=["dataset", "model_dataset"],
                   help="dataset collapses the backbone ladder -- the view for "
                        "deciding which arm a figure should show.")
    p.add_argument("--figure", default="fig1", choices=sorted(FIGURES),
                   help="Which campaign to ablate. fig2's tuned arm was "
                        "previously read by nothing at all.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    figure = FIGURES[args.figure]
    dupes = []
    tuned = collect_group(args.dump_root, figure["tuned"],
                          figure["uid_re"], figure["key"], dupes)
    noeps = collect_group(args.dump_root, figure["noeps"],
                          figure["uid_re"], figure["key"], dupes)
    pairs, only_tuned, only_noeps = pair(tuned, noeps)

    print(f"=== eps ablation, {args.figure} ({figure['label']}) ===")

    print(f"tuned-eps runs scored : {len(tuned)}")
    print(f"eps=1e-12 runs scored : {len(noeps)}")
    print(f"MATCHED pairs         : {len(pairs)}")
    if dupes:
        print(f"collapsed duplicates  : {len(dupes)} (same training submitted "
              f"under more than one uid; pooled n counts each ONCE)")
    print(f"unmatched (tuned only): {len(only_tuned)}")
    print(f"unmatched (noeps only): {len(only_noeps)}")
    expected = figure["expected"]()
    for label, group in (("tuned eps", figure["tuned"]),
                         ("eps=1e-12", figure["noeps"])):
        absent = missing_runs(args.dump_root, group, expected)
        print(f"NO DUMP AT ALL ({label:9s}): {len(absent)}"
              + (f"  {absent}" if absent else ""))
    if not pairs:
        raise SystemExit("no matched pairs; nothing to compare")

    rows = summarize(pairs, by=args.by)
    print()
    print(f"=== test {args.metric}, paired; delta = noeps - tuned "
          f"({'lower is better' if args.metric in LOWER_IS_BETTER else 'higher is better'}) ===")
    print(format_table(rows, args.metric))

    print()
    print("=== overall, pooled over every matched pair ===")
    for metric, stats in overall(pairs).items():
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        print(f"  {metric:8s} delta {stats['mean']:+.5f} +- {stats['std']:.5f} "
              f"(n={stats['n']:3d}, noeps better on {stats['noeps_better']:3d} "
              f"-- {direction} is better)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"per_cell": rows, "overall": overall(pairs),
                       "n_pairs": len(pairs)}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
