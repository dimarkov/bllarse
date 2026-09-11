"""Fig 2: aggregate OOD detection metrics over the probe_*_fig2 dump tree.

Runs where the dumps are (Raven, inside the container) and emits JSON, so the
logit arrays never have to move. Plotting happens separately from that JSON.

Three score families, and which apply is a property of the ARM, not a choice:

  shared   msp, entropy      -- functions of the normalized predictive, so they
                               mean the same thing in every arm. This is the
                               controlled comparison and the primary result.
  raw      energy, max_logit -- need unnormalized logits. Valid for AdamW+CE,
                               which dumps genuine pre-softmax logits. NOT valid
                               for VBLL, whose dump is log(predictive): its
                               logsumexp is identically 0, which would make
                               energy constant and max_logit a monotone
                               transform of MSP. metrics rejects those rather
                               than returning a plausible number, and this
                               script records the refusal instead of hiding it.
  native   the arm's own epistemic score -- VBLL's predictive variance. Tests
                               the section 3.7 claim that the epistemic term is
                               the OOD mechanism, which a shared score cannot.

Reporting only the shared scores would leave that claim untested; reporting only
the native one would confound model with score function. Hence all three.
"""

import argparse
import collections
import json
import os
import re
import statistics as st

from bllarse import evaldump, metrics
from bllarse.analysis import curves
from bllarse.analysis.curves import completeness

SHARED_SCORES = {"msp": metrics.msp_ood_score, "entropy": metrics.entropy_ood_score}
RAW_SCORES = {"energy": metrics.energy_ood_score, "max_logit": metrics.max_logit_ood_score}
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
# The eps-tuned sweep is retained and reported as an ablation, not plotted here.
#
# Do NOT copy Fig 1's "the two arms are indistinguishable" here: that was
# measured on the Fig 1 campaign and does not hold on THIS one. Over the 20
# matched (model, ID dataset, seed) pairs of the OOD campaign, dropping eps
# costs test NLL by +0.00142 +- 0.00118 -- tiny, but the eps-free arm is better
# on only 1 of 20 pairs. Meanwhile it is better on val_nll, the criterion it was
# SELECTED on, at 17 of 20. Accuracy, ECE and Brier sit at coin-flip agreement.
# (No temperature-scaled figure here: both arms are IBProbit, which is reported
# raw -- see curves.TS_ARMS.)
#
# So the ablation's verdict is arm-consistent in magnitude and NOT in sign: the
# eps-free head is chosen because it is the method the paper proposes and the
# gap is ~0.001 nats, not because the two are indistinguishable here.
# `scripts/analysis/eps_ablation.py --figure fig2` regenerates those numbers;
# n is 20 rather than 30 because cifar10's two OOD pairs are one training.
#
# One caveat that belongs in the caption rather than in a comment: the eps-free
# sweep has 199 of 200 runs. dinov3_huge/oxford_pets/seed144 produced non-finite
# val_nll at every evaluation and wrote no dump, so that ONE cell carries 9
# seeds instead of 10. The cause is not established -- it is not known whether
# removing the mixture floor caused it or whether the eps parameterisation
# itself is the numerically fragile part -- so it is reported, not explained.
GROUPS = {
    "adamw_ce": "probe_adamw_ce_fig2",
    "vbll": "probe_vbll_fig2",
    "ibprobit": "probe_ibprobit_noeps_fig2",
}
# The model prefix is matched explicitly rather than with a lazy `.+?`: uids look
# like dinov3_huge_cifar10_vs_svhn_seed137, and a lazy group takes "dinov3" and
# leaves id="huge_cifar10", which mislabels every row without failing.
UID_RE = re.compile(
    r"^(?P<model>dinov3_[a-z0-9]+)_(?P<id>.+)_vs_(?P<ood>.+)_seed(?P<seed>\d+)$"
)


def uids_for(root, group_id):
    path = os.path.join(root, group_id)
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def fitted_temperature(root, group_id, uid, arm=None):
    """The temperature for `uid`'s TS variant, or None if the arm has none.

    Fitted on the run's OWN val dump -- the ID dataset's validation split
    (CIFAR-10's for the cifar10 pairs, CIFAR-100's for the cifar100 one), at
    the same step the detection metrics are read from. There is no OOD-side
    fit: SVHN has no CIFAR labels, and a deployed model carries one
    temperature anyway.
    """
    if not curves.applies_temperature_scaling(arm):
        return None
    indexed = {}
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed.setdefault(record.meta["split"], {})[record.meta["step"]] = record
    if "test" not in indexed:
        return None
    step = max(indexed["test"])
    val = indexed.get("val", {}).get(step)
    if val is None:
        return None
    return float(metrics.fit_temperature(val.logits, val.labels))


def id_metrics(root, group_id, uid, arm=None):
    """In-distribution accuracy/ECE/NLL/Brier at the SAME step the OOD dump uses.

    AUROC and its companions are the only metrics definable on the OOD side --
    an SVHN digit has no correct CIFAR-10 label, so accuracy and NLL are not
    merely unreported there but undefined. The in-distribution side has all
    four, results-plan 0.1 requires them on every figure, and without them a
    detection number is uninterpretable: a model that is bad in-distribution can
    still separate ID from OOD, and the AUROC alone cannot distinguish that from
    a good model. Taken at the final dumped step so it describes the same
    checkpoint the detection metrics do, not a better one from earlier.

    For the current campaign that step is ALSO the val-optimal one: probe.py and
    probing_vbll.py write a single dump, taken at the argmin, with the OOD
    forward pass alongside it -- so the `at_best_*` block below is a no-op. See
    the note there.
    """
    indexed = {}
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed.setdefault(record.meta["split"], {})[record.meta["step"]] = record
    if "test" not in indexed:
        return None

    step = max(indexed["test"])
    test = indexed["test"][step]
    out = dict(metrics.classification_metrics(test.logits, test.labels))

    # AdamW+CE only -- see curves.TS_ARMS. VBLL and IBProbit are reported RAW:
    # their claim is calibration by construction, and their dumped array is an
    # already-normalized log-predictive, so dividing it by T is not the same
    # transform that TS applies to CE's logits.
    val = indexed.get("val", {}).get(step)
    if val is not None and curves.applies_temperature_scaling(arm):
        temperature = metrics.fit_temperature(val.logits, val.labels)
        scaled = metrics.apply_temperature(test.logits, temperature)
        out["temperature"] = temperature
        out["temperature_clipped"] = float(metrics.is_clipped_temperature(temperature))
        out.update({
            f"ts_{k}": v
            for k, v in metrics.classification_metrics(scaled, test.labels).items()
        })

    out.update(completeness(test.meta))
    out["dumped_step"] = step
    out["n_dumps"] = len(indexed["test"])

    # The same metrics at the val-optimal step, which is the point the tuning
    # campaign actually SELECTED these hyperparameters at (its objective was
    # min-over-dumps val_nll). The probe scripts carry no patience, so on a
    # TRAJECTORY dump the final step can be far past that point, and an arm that
    # overfits is then scored somewhere its own tuning never endorsed. The gap
    # between these two blocks is how much that costs, per arm.
    #
    # Guarded on there being more than one dumped step, because on this campaign
    # there is not. The OOD dump moved into the eval loop and is now written at
    # the argmin, so `best_step == step` identically -- and emitting `at_best_*`
    # anyway produced two numbers that are equal by construction. The figure
    # drew them as a "val-optimal vs final" comparison with a connector whose
    # length was necessarily zero, which reads as the measured finding "no arm
    # overfits" rather than "not measured". Verified across all 90 runs of the
    # Fig 2 sweep: val_optimal_step == final_step in every one.
    if len(indexed["test"]) > 1 and indexed.get("val"):
        best_step = min(
            indexed["val"],
            key=lambda s: metrics.nll(indexed["val"][s].logits, indexed["val"][s].labels),
        )
        best_test = indexed["test"].get(best_step)
        if best_test is not None:
            out["val_optimal_step"] = best_step
            out["final_step"] = step
            out.update({
                f"at_best_{k}": v
                for k, v in metrics.classification_metrics(
                    best_test.logits, best_test.labels
                ).items()
            })
    return out


def has_ood_dump(root, group_id, uid) -> bool:
    """A run is only scoreable once its OOD dump exists.

    The OOD dump is written once, after the final training step, so a run that
    is still training has test/val dumps and no OOD one. Without this check such
    a run is not skipped but RECORDED AS REFUSED for every score, which reads in
    the summary as "VBLL's msp was rejected" -- indistinguishable from the real
    refusals this script exists to report.
    """
    directory = os.path.join(root, group_id, uid)
    return any(f.endswith("_ood.npz") for f in os.listdir(directory))


def collect(root):
    rows, incomplete = [], []
    for arm, group_id in GROUPS.items():
        for uid in uids_for(root, group_id):
            m = UID_RE.match(uid)
            if not m:
                print(f"[fig2] SKIP unparseable uid: {uid}")
                continue
            if not has_ood_dump(root, group_id, uid):
                incomplete.append(f"{arm}/{uid}")
                continue
            row = {"arm": arm, "uid": uid, **m.groupdict()}

            scores = dict(SHARED_SCORES)
            # Offered to every arm; metrics raises on the arms where the array
            # is already normalized, and that refusal is recorded per-run rather
            # than assumed from the arm name -- if a future arm changes what it
            # dumps, this notices instead of silently mislabelling.
            scores.update(RAW_SCORES)
            for name, fn in scores.items():
                try:
                    row[name] = curves.run_detection(root, group_id, uid, fn)
                except ValueError as exc:
                    row[name] = None
                    row.setdefault("refused", {})[name] = str(exc)[:120]

            # The temperature-scaled variant of each shared score, for TS_ARMS
            # only. Fig 1 carries CE+TS as its comparator, so Fig 2 reporting
            # CE raw would compare the two figures' CE columns at different
            # operating points.
            temperature = fitted_temperature(root, group_id, uid, arm)
            if temperature is not None:
                row["temperature"] = temperature
                for name, fn in SHARED_SCORES.items():
                    try:
                        row[f"ts_{name}"] = curves.run_detection(
                            root, group_id, uid, fn, temperature=temperature
                        )
                    except ValueError as exc:
                        row[f"ts_{name}"] = None
                        row.setdefault("refused", {})[f"ts_{name}"] = str(exc)[:120]

            try:
                row["native"] = curves.run_native_detection(root, group_id, uid)
            except ValueError as exc:
                row["native"] = None
                row.setdefault("refused", {})["native"] = str(exc)[:120]

            row["id_metrics"] = id_metrics(root, group_id, uid, arm)

            rows.append(row)
            print(f"[fig2] {arm:<9s} {m['id']:>8s} vs {m['ood']:<9s} seed {m['seed']}  "
                  + "  ".join(
                      f"{k}={row[k]['auroc']:.4f}" for k in ("msp", "entropy", "energy", "native")
                      if row.get(k)))
    if incomplete:
        # Never silent: a partially-drained sweep must not be mistaken for a
        # complete one just because the table it printed looked well formed.
        print(f"\n[fig2] {len(incomplete)} run(s) SKIPPED as incomplete (no OOD dump yet):")
        for name in incomplete[:10]:
            print(f"         {name}")
        if len(incomplete) > 10:
            print(f"         ... and {len(incomplete) - 10} more")
    return rows


def summarize(rows):
    """Mean +- stdev of AUROC/FPR95/AUPR over seeds, per (arm, pair, score)."""
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        pair = f"{r['id']}_vs_{r['ood']}"
        for score in ("msp", "entropy", "ts_msp", "ts_entropy",
                      "energy", "max_logit", "native"):
            if r.get(score):
                for metric, value in r[score].items():
                    buckets[(r["arm"], pair, score)][metric].append(value)

    out = {}
    for (arm, pair, score), metric_values in sorted(buckets.items()):
        out[f"{arm}|{pair}|{score}"] = {
            "n_seeds": len(next(iter(metric_values.values()))),
            **{
                m: {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0}
                for m, v in metric_values.items()
            },
        }
    return out


def summarize_id(rows):
    """ID metrics keyed by (arm, ID dataset) -- NOT by pair.

    cifar10 is the ID set of two different pairs, but its in-distribution
    performance does not depend on which OOD set it is later scored against, so
    keying by pair would report the same model twice and invite reading a
    difference into what is one number.

    And for the same reason each (arm, ID dataset, seed) is counted ONCE. The
    OOD set is not an input to training -- same config, same seed, same
    trajectory -- so cifar10's two pairs are the same ten runs submitted twice,
    not twenty. Averaging all twenty reported `n_runs: 20` for ten independent
    seeds and put every cifar10 value into the mean twice. (Verified on the
    campaign: for ibprobit all ten cifar10 seeds are bitwise identical across
    the two pairs, and for adamw_ce seven of ten are, the rest differing in the
    sixth decimal from GPU reduction order. The duplication is real, not an
    artefact of the check.)

    Deduplication is by seed rather than by value, so two runs that genuinely
    diverged in the last decimal still collapse to one -- they are one run.
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    duplicates = 0
    for r in rows:
        if not r.get("id_metrics"):
            continue
        key = (r["arm"], r["id"], r["seed"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        for metric, value in r["id_metrics"].items():
            # id_metrics now also carries the completeness fields, which are
            # None on dumps too old to have them -- and st.mean chokes on None
            # rather than skipping it. bool passes this test, which is what
            # turns `at_last_eval` into a per-cell truncation rate.
            if isinstance(value, (int, float)):
                buckets[(r["arm"], r["id"])][metric].append(float(value))
    if duplicates:
        print(f"\n[fig2] ID control: collapsed {duplicates} duplicate "
              f"(arm, ID dataset, seed) run(s) -- an ID dataset that appears in "
              f"two pairs is trained once, not twice")

    out = {}
    for (arm, dataset), metric_values in sorted(buckets.items()):
        out[f"{arm}|{dataset}"] = {
            "n_runs": len(next(iter(metric_values.values()))),
            **{
                m: {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0}
                for m, v in metric_values.items()
            },
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--out", default="fig2_detection.json")
    p.add_argument("--publish", nargs="?", const="fig2_detection", default=None,
                   metavar="NAME",
                   help="Publish the JSON as an MLflow artifact under NAME "
                        "so plotting can fetch it with --json mlflow:NAME. "
                        "This is what lets figures be rebuilt off-cluster; "
                        "without it the JSON only exists beside the dumps.")
    args = p.parse_args()

    rows = collect(args.dump_root)
    if not rows:
        raise SystemExit(f"[fig2] no runs found under {args.dump_root}")
    summary = summarize(rows)
    id_summary = summarize_id(rows)

    print(f"\n{'arm':<10s} {'pair':<24s} {'score':<10s} {'n':>3s} "
          f"{'AUROC':>16s} {'FPR@95':>16s} {'AUPR':>16s}")
    print("-" * 100)
    for key, s in summary.items():
        arm, pair, score = key.split("|")
        print(f"{arm:<10s} {pair:<24s} {score:<10s} {s['n_seeds']:>3d} "
              + "".join(f"{s[m]['mean']:>10.4f}+-{s[m]['std']:<5.4f}"
                        for m in ("auroc", "fpr95", "aupr")))

    refusals = collections.Counter(
        (r["arm"], name) for r in rows for name in r.get("refused", {})
    )
    if refusals:
        print("\nscores refused (expected for VBLL's raw-logit scores):")
        for (arm, name), n in sorted(refusals.items()):
            print(f"  {arm:<10s} {name:<10s} {n} runs")

    print("\nIN-DISTRIBUTION control (final step -- the checkpoint the detection "
          "metrics above describe)")
    print(f"{'arm':<10s} {'dataset':<12s} {'n':>3s} {'acc':>9s} {'ECE':>9s} "
          f"{'NLL':>9s} {'Brier':>9s} {'T':>7s}")
    print("-" * 74)
    for key, s in id_summary.items():
        arm, dataset = key.split("|")
        print(f"{arm:<10s} {dataset:<12s} {s['n_runs']:>3d} "
              + "".join(f"{s[m]['mean']:>9.4f}" for m in ("acc", "ece", "nll", "brier"))
              + f"{s.get('temperature', {}).get('mean', float('nan')):>7.3f}")

    comparable = [s for s in id_summary.values() if "at_best_nll" in s]
    if comparable:
        print("\nSame runs at the VAL-OPTIMAL step -- where the tuning campaign "
              "selected these configs.")
        print("A large gap means the arm overfits without early stopping, which "
              "the probe scripts do not do.")
        print(f"{'arm':<10s} {'dataset':<12s} {'acc':>9s} {'ECE':>9s} {'NLL':>9s} "
              f"{'Brier':>9s} {'dNLL':>9s} {'step':>14s}")
        print("-" * 88)
        for key, s in id_summary.items():
            arm, dataset = key.split("|")
            if "at_best_nll" not in s:
                continue
            d_nll = s["nll"]["mean"] - s["at_best_nll"]["mean"]
            print(f"{arm:<10s} {dataset:<12s} "
                  + "".join(f"{s['at_best_' + m]['mean']:>9.4f}"
                            for m in ("acc", "ece", "nll", "brier"))
                  + f"{d_nll:>+9.4f}"
                  + f"{int(s['val_optimal_step']['mean']):>8d}/"
                    f"{int(s['final_step']['mean']):<6d}")
    else:
        # Said plainly rather than printed as an empty table under a header,
        # which is how this read while the figure beside it drew the comparison
        # as if it had happened.
        print("\nNo val-optimal-vs-final comparison: every run wrote a single "
              "dump, taken AT its val optimum, so the two steps coincide by "
              "construction. Measuring the endpoint penalty needs trajectory "
              "dumps.")

    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary, "id_summary": id_summary}, fh, indent=2)
    print(f"\nwrote {args.out} ({len(rows)} runs, {len(summary)} detection cells, "
          f"{len(id_summary)} ID cells)")

    if args.publish:
        # Raises rather than warning: a caller that asked to publish and then
        # plots from a stale local JSON, believing it came from the server, is
        # the exact failure this exists to prevent.
        from bllarse.mlflow_utils import publish_analysis
        run_id = publish_analysis(args.out, args.publish, tags={
            "n_runs": len(rows),
            "n_detection_cells": len(summary),
            "n_id_cells": len(id_summary),
        })
        print(f"published as mlflow:{args.publish} (run {run_id}) -- "
              f"plot anywhere with --json mlflow:{args.publish}")


if __name__ == "__main__":
    main()
