"""Does IBProbit need its contamination parameter at all?

Pairs `ibprobit` (eps_init tuned per dataset) against `ibprobit_noeps`
(eps_init pinned at 1e-12, i.e. the mixture floor numerically absent) on the
Fig 1 runs, and reports the PAIRED difference.

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

from fig1_curves import UID_RE, score_run  # noqa: E402

# The tuned arm first: deltas below are (noeps - tuned), so a POSITIVE delta on
# a lower-is-better metric means dropping eps HURT.
TUNED_GROUP = "probe_ibprobit_fig1"
NOEPS_GROUP = "probe_ibprobit_noeps_fig1"

# Lower is better for all but acc.
METRICS = ("nll", "acc", "ece", "brier", "ts_nll", "ts_ece")
LOWER_IS_BETTER = {"nll", "ece", "brier", "ts_nll", "ts_ece"}


def collect_group(root, group_id):
    """{(model, dataset, seed): scored} for one group."""
    out = {}
    directory = os.path.join(root, group_id)
    if not os.path.isdir(directory):
        return out
    for uid in sorted(os.listdir(directory)):
        match = UID_RE.match(uid)
        if not match:
            continue
        scored = score_run(root, group_id, uid)
        if scored is None:
            continue
        key = (match["model"], match["dataset"], int(match["seed"]))
        out[key] = scored
    return out


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


def summarize(pairs):
    """Per (model, dataset) and overall, for each metric."""
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for (model, dataset, _seed), tuned, noeps in pairs:
        for metric in METRICS:
            if metric in tuned and metric in noeps:
                buckets[(model, dataset)][metric].append(
                    (tuned[metric], noeps[metric])
                )

    rows = {}
    for (model, dataset), by_metric in sorted(buckets.items()):
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
        rows[f"{model}|{dataset}"] = entry
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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tuned = collect_group(args.dump_root, TUNED_GROUP)
    noeps = collect_group(args.dump_root, NOEPS_GROUP)
    pairs, only_tuned, only_noeps = pair(tuned, noeps)

    print(f"tuned-eps runs scored : {len(tuned)}")
    print(f"eps=1e-12 runs scored : {len(noeps)}")
    print(f"MATCHED pairs         : {len(pairs)}")
    print(f"unmatched (tuned only): {len(only_tuned)}")
    print(f"unmatched (noeps only): {len(only_noeps)}")
    if not pairs:
        raise SystemExit("no matched pairs; nothing to compare")

    rows = summarize(pairs)
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
