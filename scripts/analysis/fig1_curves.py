"""Fig 1: linear probing across the DINOv3 ladder, from the probe_*_fig1 dumps.

Runs where the dumps are and emits JSON; plotting is separate.

Every run is scored at its VAL-OPTIMAL step, not its last. The probe scripts
carry no early stopping and the tuning campaign selected on min-over-dumps
val_nll, so the final step is not where these configurations were chosen -- and
the arms drift from that point by very different amounts. AdamW+CE on cifar100
peaked at step 1,786 of 30,000 and then nearly tripled its NLL while VBLL moved
by 0.0003; reporting endpoints would have scored that drift as a method
difference. Selection uses the UNSCALED val_nll, matching the campaign's own
objective.

AdamW+CE+TS is not a separate arm: the temperature is fit on the val dump at the
selected step and applied to that step's test logits, so it costs no GPU time.
"""

import argparse
import collections
import json
import os
import re
import statistics as st

from bllarse import evaldump, metrics

# Four arms. `ibprobit_noeps` is the contamination ablation -- the same
# IBProbit head with eps_init pinned at 1e-12 instead of tuned -- and it is a
# separate arm here rather than a variant of `ibprobit` because the two were
# tuned independently and land on different hyperparameters.
# "ibprobit" IS THE EPS-FREE VARIANT. The label carries no qualifier because
# the method the paper presents has no contamination parameter -- so the
# figures show `probe_ibprobit_noeps_{fig}`, not `probe_ibprobit_{fig}`.
#
# Read that mapping carefully before changing it: the group_id and the arm
# label deliberately DISAGREE, which is exactly the kind of relabelling that
# looks like a typo and gets "fixed" back into a silent swap of which sweep a
# figure reports.
#
# The eps-tuned sweep is retained and reported as an ablation, not plotted
# here. Measured over 199 matched (model, dataset, seed) pairs, the two arms
# are indistinguishable on the selection criterion -- pooled val_nll delta
# +0.0017 +- 0.0047, with the eps-free arm better on 98 of 199 -- while
# accuracy and Brier marginally favour dropping eps. scripts/analysis/
# eps_ablation.py regenerates those numbers.
#
# One caveat that belongs in the caption rather than in a comment: the eps-free
# sweep has 199 of 200 runs. dinov3_huge/oxford_pets/seed144 produced non-finite
# val_nll at every evaluation and wrote no dump, so that ONE cell carries 9
# seeds instead of 10. The cause is not established -- it is not known whether
# removing the mixture floor caused it or whether the eps parameterisation
# itself is the numerically fragile part -- so it is reported, not explained.
GROUPS = {
    "adamw_ce": "probe_adamw_ce_fig1",
    "vbll": "probe_vbll_fig1",
    "ibprobit": "probe_ibprobit_noeps_fig1",
}
UID_RE = re.compile(r"^(?P<model>dinov3_[a-z0-9]+)_(?P<dataset>.+)_seed(?P<seed>\d+)$")
METRIC_KEYS = ("acc", "ece", "nll", "brier")


def score_run(root, group_id, uid):
    """Test metrics at the val-argmin step, raw and temperature-scaled."""
    indexed = collections.defaultdict(dict)
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed[record.meta["split"]][record.meta["step"]] = record
    if "test" not in indexed or "val" not in indexed:
        return None

    shared = sorted(set(indexed["test"]) & set(indexed["val"]))
    if not shared:
        return None

    best_step = min(
        shared,
        key=lambda s: metrics.nll(indexed["val"][s].logits, indexed["val"][s].labels),
    )
    test, val = indexed["test"][best_step], indexed["val"][best_step]

    out = {
        "step": best_step,
        "final_step": max(shared),
        "n_dumps": len(shared),
        **metrics.classification_metrics(test.logits, test.labels),
    }
    temperature = metrics.fit_temperature(val.logits, val.labels)
    out["temperature"] = temperature
    out["temperature_clipped"] = float(metrics.is_clipped_temperature(temperature))
    out.update({
        f"ts_{k}": v
        for k, v in metrics.classification_metrics(
            metrics.apply_temperature(test.logits, temperature), test.labels
        ).items()
    })
    # The endpoint too, so the cost of not early-stopping stays measurable per
    # arm rather than being silently corrected for by the selection above.
    last = indexed["test"][max(shared)]
    out.update({
        f"final_{k}": v
        for k, v in metrics.classification_metrics(last.logits, last.labels).items()
    })
    return out


def collect(root):
    rows, skipped = [], []
    for arm, group_id in GROUPS.items():
        directory = os.path.join(root, group_id)
        if not os.path.isdir(directory):
            continue
        for uid in sorted(os.listdir(directory)):
            m = UID_RE.match(uid)
            if not m:
                skipped.append(f"{arm}/{uid} (unparseable uid)")
                continue
            scored = score_run(root, group_id, uid)
            if scored is None:
                skipped.append(f"{arm}/{uid} (incomplete dumps)")
                continue
            rows.append({"arm": arm, "uid": uid, **m.groupdict(), **scored})
    return rows, skipped


def summarize(rows):
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for key, value in r.items():
            if isinstance(value, (int, float)) and key != "seed":
                buckets[(r["arm"], r["model"], r["dataset"])][key].append(value)

    out = {}
    for (arm, model, dataset), values in sorted(buckets.items()):
        out[f"{arm}|{model}|{dataset}"] = {
            "n_seeds": len(next(iter(values.values()))),
            **{
                k: {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0}
                for k, v in values.items()
            },
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--out", default="fig1_curves.json")
    args = p.parse_args()

    rows, skipped = collect(args.dump_root)
    if not rows:
        raise SystemExit(f"[fig1] no runs found under {args.dump_root}")
    summary = summarize(rows)

    print(f"[fig1] scored {len(rows)} runs into {len(summary)} cells")
    if skipped:
        # Never silent: a partial sweep must not read as a complete one.
        print(f"[fig1] SKIPPED {len(skipped)}:")
        for name in skipped[:10]:
            print(f"         {name}")

    ladder = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
    datasets = sorted({r["dataset"] for r in rows})
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        print(f"{'arm':<12s} {'backbone':<15s} {'n':>3s} {'acc':>8s} {'ECE':>8s} "
              f"{'NLL':>8s} {'Brier':>8s} {'T':>7s} {'dNLL_end':>9s}")
        for arm in ("adamw_ce", "vbll"):
            for model in ladder:
                cell = summary.get(f"{arm}|{model}|{dataset}")
                if not cell:
                    continue
                d_end = cell["final_nll"]["mean"] - cell["nll"]["mean"]
                print(f"{arm:<12s} {model:<15s} {cell['n_seeds']:>3d} "
                      + "".join(f"{cell[k]['mean']:>8.4f}" for k in METRIC_KEYS)
                      + f"{cell['temperature']['mean']:>7.3f}{d_end:>+9.4f}")

    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
