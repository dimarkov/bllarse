"""Fig 1: linear probing across the DINOv3 ladder, from the probe_*_fig1 dumps.

Runs where the dumps are and emits JSON; plotting is separate.

Every run is scored at its VAL-OPTIMAL step. Selection uses the UNSCALED
val_nll, matching the tuning objective.

For the Fig 1 sweeps that selection is a formality: probe.py and
probing_vbll.py buffer the argmin evaluation and write exactly ONE dump per run
(see their `_write_best`), so the argmin is the only point there is. The argmin
is still taken rather than assumed, because the same code must also work on
dumps that hold a whole trajectory of steps.

Two consequences of one-dump-per-run:

  * ENDPOINT METRICS DO NOT EXIST. `final_*` is emitted only when a run
    actually dumped more than one step. On a single-dump run the endpoint IS
    the argmin, so a `final_nll - nll` panel is identically zero -- not a
    measurement that the arms do not overfit, but no measurement at all.
  * A DUMP DOES NOT PROVE A RUN FINISHED. `_write_best` is called from a
    `finally`, so a crashed run writes its best-so-far dump too. `evals_run` /
    `evals_planned` in the dump meta are what separate the three cases; see
    `completeness` below and evaldump.write_dump.

AdamW+CE+TS is not a separate arm: the temperature is fit on the val dump at the
selected step and applied to that step's test logits, so it costs no GPU time.
"""

import argparse
import collections
import functools
import importlib
import json
import os
import re
import statistics as st
import sys

from bllarse import evaldump, metrics
from bllarse.analysis import curves
from bllarse.analysis.curves import completeness
from bllarse.tuning import space

# Three arms. "ibprobit" IS THE EPS-FREE VARIANT: the IBProbit head with its
# contamination parameter eps_init pinned at 1e-12 rather than tuned. The label
# carries no qualifier because the method the paper presents has no
# contamination parameter -- so the figures show `probe_ibprobit_noeps_{fig}`,
# not `probe_ibprobit_{fig}`.
#
# Read that mapping carefully before changing it: the group_id and the arm
# label deliberately DISAGREE, which is exactly the kind of relabelling that
# looks like a typo and gets "fixed" back into a silent swap of which sweep a
# figure reports.
#
# The eps-free head is the one plotted because, on these probing tasks, tuning
# eps buys nothing on the selection criterion: matched runs of the eps-tuned
# and eps-free heads are indistinguishable on val_nll, while accuracy and Brier
# marginally favour dropping eps. On the Fig 2 sweeps dropping eps costs a small
# but systematic amount of test NLL (~0.001 nats); see fig2_detection.py.
#
# One caveat for the caption: dinov3_huge/oxford_pets/seed144 produced a
# non-finite val_nll at every evaluation and wrote no dump, so that ONE cell
# carries 9 seeds instead of 10. The cause is not established -- it is not
# known whether removing the mixture floor caused it or whether the eps
# parameterisation itself is the numerically fragile part -- so it is
# reported, not explained.
GROUPS = {
    "adamw_ce": "probe_adamw_ce_fig1",
    "vbll": "probe_vbll_fig1",
    "ibprobit": "probe_ibprobit_noeps_fig1",
}
UID_RE = re.compile(r"^(?P<model>dinov3_[a-z0-9]+)_(?P<dataset>.+)_seed(?P<seed>\d+)$")
METRIC_KEYS = ("acc", "ece", "nll", "brier")

# Runs whose dumps exist but whose PROCESS DIED, established from their MLflow
# run status rather than from anything on disk. They are excluded from every
# reported number.
#
# They are named here, not filtered by a rule, because their dumps carry no
# evidence of the crash: they predate `evals_run`/`evals_planned`, and
# `_write_best` runs in a `finally`, so what landed on disk is a normal-looking
# argmin dump:
#
#   adamw_ce/dinov3_big_food101_seed143     FAILED after   39 s (siblings:  253 s)
#       argmin step 3232, outside the sibling range 4256-5472; its NLL 0.2643
#       is worse than all nine siblings (max 0.2633), so it biased that cell.
#   ibprobit/dinov3_big_flowers102_seed142  FAILED after  415 s (siblings: 1893 s)
#       argmin step 3759, outside the sibling range 7670-11506.
#
# (A third failure, ibprobit/dinov3_huge_oxford_pets_seed144, diverged to a
# non-finite val_nll at every evaluation and correctly wrote nothing -- it is
# the documented 9-seed cell and needs no entry here.)
#
# Delete an entry only when that run has actually been re-run to completion.
# Dumps that carry `evals_run`/`evals_planned` are caught automatically.
INCOMPLETE_RUNS = {
    ("adamw_ce", "dinov3_big_food101_seed143"): "MLflow status FAILED after 39 s",
    ("ibprobit", "dinov3_big_flowers102_seed142"): "MLflow status FAILED after 415 s",
}


# Where each arm's budget comes from. The sweep modules already own it -- an
# arm's `TUNED[dataset]` fixes batch_size and epochs, and those two ARE the
# budget -- so this reads it there rather than restating numbers that would then
# have two homes and drift apart.
#
# It exists because an older dump records only the numerator. `meta["step"]`
# and `meta["samples_seen"]` say exactly where the val optimum fell, but "step
# 3,759" is early or terminal depending on a denominator such a dump does not
# carry. That denominator needs no re-run: `space.eval_quantum` is a pure
# function of the dataset and the sweep config is in the repo, so any dump can
# be placed on its own budget after the fact.
SWEEP_MODULES = {
    "adamw_ce": "probe_adamw_ce",
    "vbll": "probe_vbll",
    "ibprobit": "probe_ibprobit_noeps",
}


@functools.lru_cache(maxsize=None)
def budget(arm, dataset):
    """The number of evaluations one sweep cell plans (`evals_planned`), or None.

    None whenever the budget cannot be established -- an arm with no sweep
    module, a dataset the sweep does not cover, an import that fails outside the
    repo. Unknown is returned rather than guessed: a wrong denominator would
    silently reclassify converged runs as truncated and vice versa.
    """
    module_name = SWEEP_MODULES.get(arm)
    if module_name is None:
        return None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "bllarse_sweeps"))
        module = importlib.import_module(module_name)
        tuned = module.TUNED[dataset]
        batch_size = tuned["batch_size"]
        per_epoch = space.steps_per_epoch(dataset, batch_size)
        eval_every = space.eval_quantum(dataset) // batch_size
        if not eval_every:
            return None
        return (tuned["epochs"] * per_epoch) // eval_every
    except (ImportError, KeyError, ValueError, ZeroDivisionError):
        return None


def score_run(root, group_id, uid, arm=None, dataset=None):
    """Test metrics at the val-argmin step; temperature-scaled for CE only.

    `arm` and `dataset` are optional and used only to place the run on its own
    budget (see `budget`); the metrics do not depend on them.
    """
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

    # Both fallbacks are reconstructions for dumps that predate the meta fields,
    # and `completeness` prefers the dump's own values wherever it has them.
    # eval_index is the argmin's position on the SAMPLE grid -- the same
    # definition the training loop uses (samples_seen // quantum), so the
    # reconstruction and the recorded value agree by construction rather than by
    # coincidence.
    evals_planned = fallback_index = None
    if dataset is not None:
        evals_planned = budget(arm, dataset)
        try:
            fallback_index = test.meta["samples_seen"] // space.eval_quantum(dataset)
        except (KeyError, ValueError, ZeroDivisionError):
            fallback_index = None

    out = {
        "step": best_step,
        "final_step": max(shared),
        "n_dumps": len(shared),
        **completeness(test.meta, eval_index=fallback_index,
                       evals_planned=evals_planned),
        **metrics.classification_metrics(test.logits, test.labels),
    }
    # AdamW+CE only -- see curves.TS_ARMS. Not computed for the Bayesian arms
    # at all: a number in the JSON gets quoted whether or not a figure plots it.
    if curves.applies_temperature_scaling(arm):
        temperature = metrics.fit_temperature(val.logits, val.labels)
        out["temperature"] = temperature
        out["temperature_clipped"] = float(
            metrics.is_clipped_temperature(temperature)
        )
        out.update({
            f"ts_{k}": v
            for k, v in metrics.classification_metrics(
                metrics.apply_temperature(test.logits, temperature), test.labels
            ).items()
        })
    # The endpoint too, so the cost of not early-stopping stays measurable per
    # arm rather than being silently corrected for by the selection above --
    # but ONLY if the run dumped a trajectory to have an endpoint in. On a
    # single-dump run the endpoint is the argmin by construction, and emitting
    # `final_*` there produces a difference that is identically zero and reads
    # downstream as "measured: no drift" instead of "not measured".
    if len(shared) > 1:
        last = indexed["test"][max(shared)]
        out.update({
            f"final_{k}": v
            for k, v in metrics.classification_metrics(last.logits, last.labels).items()
        })
    return out


def collect(root):
    """Score every run, dropping the ones that are not measurements.

    Returns (rows, skipped, dropped). `dropped` is kept separate from `skipped`
    because the two mean different things: a skipped run has no usable dump,
    while a dropped one has a perfectly readable dump that describes a process
    which died partway through its budget.
    """
    rows, skipped, dropped = [], [], []
    for arm, group_id in GROUPS.items():
        directory = os.path.join(root, group_id)
        if not os.path.isdir(directory):
            continue
        for uid in sorted(os.listdir(directory)):
            m = UID_RE.match(uid)
            if not m:
                skipped.append(f"{arm}/{uid} (unparseable uid)")
                continue
            scored = score_run(root, group_id, uid, arm, m.group("dataset"))
            if scored is None:
                skipped.append(f"{arm}/{uid} (incomplete dumps)")
                continue
            # Two independent sources for the same fact: the dump's own counts
            # (authoritative, present on anything written after they existed)
            # and the hardcoded list (the only evidence for the older runs).
            reason = INCOMPLETE_RUNS.get((arm, uid))
            if scored["incomplete"]:
                reason = (
                    f"{scored['evals_run']} of {scored['evals_planned']} "
                    f"scheduled evaluations ran"
                )
            if reason:
                dropped.append(f"{arm}/{uid} ({reason})")
                continue
            rows.append({"arm": arm, "uid": uid, **m.groupdict(), **scored})
    return rows, skipped, dropped


def summarize(rows):
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    counts = collections.Counter()
    for r in rows:
        cell = (r["arm"], r["model"], r["dataset"])
        counts[cell] += 1
        for key, value in r.items():
            # `bool` passes isinstance(_, int), which is what makes the
            # truncation flag aggregate into a per-cell RATE for free. None is
            # excluded by the same test, so a cell of dumps too old to carry the
            # flag reports no rate rather than a rate of zero.
            if isinstance(value, (int, float)) and key != "seed":
                buckets[cell][key].append(float(value))

    out = {}
    for cell, values in sorted(buckets.items()):
        arm, model, dataset = cell
        out[f"{arm}|{model}|{dataset}"] = {
            # Counted over rows, not read off whichever metric list happens to
            # come first: keys are not present on every row (`final_*` only on
            # trajectory dumps, the completeness fields only on newer ones), so
            # a per-key length is not the seed count.
            "n_seeds": counts[cell],
            **{
                k: {"mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0,
                    "n": len(v)}
                for k, v in values.items()
            },
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-root", default=os.environ.get("BLLARSE_EVAL_DUMP_ROOT", "eval_dumps"))
    p.add_argument("--out", default="fig1_curves.json")
    args = p.parse_args()

    rows, skipped, dropped = collect(args.dump_root)
    if not rows:
        raise SystemExit(f"[fig1] no runs found under {args.dump_root}")
    summary = summarize(rows)

    print(f"[fig1] scored {len(rows)} runs into {len(summary)} cells")
    if skipped:
        # Never silent: a partial sweep must not read as a complete one.
        print(f"[fig1] SKIPPED {len(skipped)}:")
        for name in skipped[:10]:
            print(f"         {name}")
    if dropped:
        print(f"[fig1] DROPPED {len(dropped)} run(s) whose process did not "
              f"finish -- their dumps exist and are readable, which is exactly "
              f"why they have to be named here:")
        for name in dropped:
            print(f"         {name}")

    ladder = ["dinov3_small", "dinov3_big", "dinov3_large", "dinov3_huge"]
    datasets = sorted({r["dataset"] for r in rows})
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        print(f"{'arm':<12s} {'backbone':<15s} {'n':>3s} {'acc':>8s} {'ECE':>8s} "
              f"{'NLL':>8s} {'Brier':>8s} {'T':>7s} {'trunc':>7s} {'dNLL_end':>9s}")
        # Derived from GROUPS, never hardcoded: a summary that silently omits
        # an arm the JSON beside it contains is worse than one that fails.
        for arm in GROUPS:
            for model in ladder:
                cell = summary.get(f"{arm}|{model}|{dataset}")
                if not cell:
                    continue
                # Both columns are blank when the underlying quantity was not
                # measured, rather than printing a 0.0000 that reads as a
                # measurement: dNLL_end needs a trajectory dump, and the
                # truncation rate needs `at_last_eval` on the dump.
                trunc = cell.get("truncated")
                d_end = cell.get("final_nll")
                print(f"{arm:<12s} {model:<15s} {cell['n_seeds']:>3d} "
                      + "".join(f"{cell[k]['mean']:>8.4f}" for k in METRIC_KEYS)
                      + (f"{cell['temperature']['mean']:>7.3f}"
                         if 'temperature' in cell else f"{'--':>7s}")
                      + (f"{trunc['mean']:>7.0%}" if trunc else f"{'--':>7s}")
                      + (f"{d_end['mean'] - cell['nll']['mean']:>+9.4f}"
                         if d_end else f"{'--':>9s}"))

    truncated = [r for r in rows if r.get("truncated")]
    if truncated:
        # A run whose val optimum is its LAST evaluation never turned the
        # corner: the reported point is where the budget stopped, not where the
        # method converges. Named per cell because a whole cell at 100% is a
        # budget that is too short for that configuration, while one seed in ten
        # is noise.
        by_cell = collections.Counter(
            (r["arm"], r["model"], r["dataset"]) for r in truncated
        )
        print(f"\n[fig1] BUDGET-TRUNCATED: {len(truncated)} of {len(rows)} runs "
              f"put their val optimum at or within the last "
              f"{1 - curves.TRUNCATION_FRACTION:.0%} of their budget. These "
              f"are still improving; the figure marks them.")
        for (arm, model, dataset), n in sorted(by_cell.items(), key=lambda kv: -kv[1]):
            cell = summary[f"{arm}|{model}|{dataset}"]
            print(f"         {arm:<10s} {dataset:<14s} {model:<14s} "
                  f"{n}/{cell['n_seeds']} seeds")

    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2)
    print(f"\nwrote {args.out}")



if __name__ == "__main__":
    main()
