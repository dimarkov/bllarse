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

SHARED_SCORES = {"msp": metrics.msp_ood_score, "entropy": metrics.entropy_ood_score}
RAW_SCORES = {"energy": metrics.energy_ood_score, "max_logit": metrics.max_logit_ood_score}
GROUPS = {"adamw_ce": "probe_adamw_ce_fig2", "vbll": "probe_vbll_fig2"}
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


def id_metrics(root, group_id, uid):
    """In-distribution accuracy/ECE/NLL/Brier at the SAME step the OOD dump uses.

    AUROC and its companions are the only metrics definable on the OOD side --
    an SVHN digit has no correct CIFAR-10 label, so accuracy and NLL are not
    merely unreported there but undefined. The in-distribution side has all
    four, results-plan 0.1 requires them on every figure, and without them a
    detection number is uninterpretable: a model that is bad in-distribution can
    still separate ID from OOD, and the AUROC alone cannot distinguish that from
    a good model. Taken at the final step so it describes the same checkpoint
    the detection metrics do, not a better one from earlier in training.
    """
    indexed = {}
    for record in evaldump.iter_dumps(root, group_id, uid):
        indexed.setdefault(record.meta["split"], {})[record.meta["step"]] = record
    if "test" not in indexed:
        return None

    step = max(indexed["test"])
    test = indexed["test"][step]
    out = dict(metrics.classification_metrics(test.logits, test.labels))

    val = indexed.get("val", {}).get(step)
    if val is not None:
        temperature = metrics.fit_temperature(val.logits, val.labels)
        scaled = metrics.apply_temperature(test.logits, temperature)
        out["temperature"] = temperature
        out["temperature_clipped"] = float(metrics.is_clipped_temperature(temperature))
        out.update({
            f"ts_{k}": v
            for k, v in metrics.classification_metrics(scaled, test.labels).items()
        })

    # The same metrics at the val-optimal step, which is the point the tuning
    # campaign actually SELECTED these hyperparameters at (its objective was
    # min-over-dumps val_nll). The probe scripts carry no patience, so the final
    # step can be far past that point -- and an arm that overfits without early
    # stopping is then scored somewhere its own tuning never endorsed. The gap
    # between these two blocks is how much that costs, per arm. Detection cannot
    # be moved to the val-optimal step: the OOD dump is written once, at the end.
    if indexed.get("val"):
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

            try:
                row["native"] = curves.run_native_detection(root, group_id, uid)
            except ValueError as exc:
                row["native"] = None
                row.setdefault("refused", {})["native"] = str(exc)[:120]

            row["id_metrics"] = id_metrics(root, group_id, uid)

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
        for score in ("msp", "entropy", "energy", "max_logit", "native"):
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
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if not r.get("id_metrics"):
            continue
        for metric, value in r["id_metrics"].items():
            buckets[(r["arm"], r["id"])][metric].append(value)

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

    print("\nSame runs at the VAL-OPTIMAL step -- where the tuning campaign "
          "selected these configs.")
    print("A large gap means the arm overfits without early stopping, which the "
          "probe scripts do not do.")
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

    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary, "id_summary": id_summary}, fh, indent=2)
    print(f"\nwrote {args.out} ({len(rows)} runs, {len(summary)} detection cells, "
          f"{len(id_summary)} ID cells)")


if __name__ == "__main__":
    main()
