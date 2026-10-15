"""Fig 2: OOD detection from frozen DINOv3-H+ features.

Reads the JSON written by scripts/analysis/fig2_detection.py (which runs where
the dumps are). Three ID/OOD pairs across the columns, the three detection
metrics down the rows, and one point per (arm, score) with a band over seeds.

The x-axis is deliberately (arm x score) rather than just arm. Which scores an
arm can offer is a property of what it dumps, not a free choice -- AdamW+CE
dumps raw logits so energy and max-logit apply to it and not to VBLL, whose dump
is already normalized; VBLL dumps a native predictive variance that CE has no
counterpart for. Collapsing that to one bar per arm would silently pick a
winner per arm and hide the comparison the figure exists to make.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style
from paper_style import DOWN, UP

# Which pair is far-OOD and which near-OOD is said in the caption.
PAIRS = [("cifar10_vs_svhn", "CIFAR-10 vs SVHN"),
         ("cifar10_vs_cifar100", "CIFAR-10 vs CIFAR-100"),
         ("cifar100_vs_svhn", "CIFAR-100 vs SVHN")]

# The two metrics the paper reports. AUPR is measured and in the JSON, and
# `--with-aupr` adds it as a third row; it ranks the methods the same way.
METRICS = [("auroc", f"AUROC {UP}"), ("fpr95", f"FPR at 95% TPR {DOWN}")]
AUPR = ("aupr", f"AUPR {UP}")

# (arm, score, label, colour).
#
# MSP and entropy only. energy/max-logit were measured and are in the JSON, but
# they apply to AdamW+CE alone -- VBLL's dump is an already-normalized
# log-predictive, on which they are degenerate -- so putting them on a
# cross-arm figure compares four CE variants against two VBLL ones and invites
# reading the extra points as a method difference. VBLL's native predictive
# variance is likewise measured and excluded here; it belongs in the text, not
# on this figure, because it answers a different question (see the module
# docstring of scripts/analysis/fig2_detection.py) and is far off this scale.
# MSP ALONE, by decision. Entropy is still computed and still in the JSON, and
# it is the stronger detector by 0.001-0.007 AUROC for every arm -- but the ARM
# ORDERING is identical under both scores in all six measured cases (3 pairs x
# {samples40, win5}), so no claim rests on it and one score per arm makes the
# x-axis just the arm.
#
# MSP rather than entropy because it is the score no arm can be said to be
# disadvantaged by: VBLL's own library ships `max_predictive` -- the max of the
# mean predictive probability, i.e. exactly MSP -- as its OOD score.
# CE is shown TEMPERATURE-SCALED and only so, matching Fig 1, where TS is CE's
# comparator because it is the cheap post-hoc fix available to a plain softmax
# head. Carrying raw CE here and TS CE there would put the two figures' CE
# columns at different operating points. VBLL and IBProbit stay raw: their
# claim is calibration by construction and their dumps are already-normalized
# log-predictives (curves.TS_ARMS).
#
# The label is the method alone: every entry is scored by MSP, which the
# caption states. Colours and markers are the ones the other figures use for
# the same methods.
SERIES = [
    ("adamw_ce", "ts_msp", "CE+TS",    paper_style.COLOURS["ce_ts"]),
    ("vbll",     "msp",    "VBLL",     paper_style.COLOURS["vbll"]),
    ("ibprobit", "msp",    "IBProbit", paper_style.COLOURS["ibprobit"]),
]
MARKERS = {"adamw_ce": "s", "vbll": "^", "ibprobit": "D"}

# ID control: the four classification metrics, keyed by ID dataset. Undefined on
# the OOD side (an SVHN digit has no correct CIFAR label), so they cannot go on
# the detection panels -- but without them a detection number is uninterpretable.
# ECE dropped here for the same reason as in Fig 1: NLL and Brier are proper
# scores and already price calibration. It remains in the JSON.
ID_METRICS = [("acc", "Accuracy ↑"), ("nll", "NLL ↓"), ("brier", "Brier ↓")]
ID_DATASETS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")]
# (arm, label, colour, metric prefix). CE carries "ts_" so the ID control
# describes the same operating point the detection panels do.
ID_ARMS = [("adamw_ce", "AdamW+CE (TS)", "#4C72B0", "ts_"),
           ("vbll", "VBLL", "#C44E52", ""),
           ("ibprobit", "IBProbit", "#8172B2", "")]


def _arm_boundaries():
    """x positions between adjacent SERIES entries belonging to different arms.

    One divider per arm change, however many arms SERIES grows to. The x-axis
    is (arm x score) and the dividers are what stop a reader attributing a
    neighbouring arm's score to the wrong method.
    """
    # None needed when SERIES holds one entry per arm: every x position IS an
    # arm then, so a divider between each adjacent pair separates nothing and
    # only adds rules to the panel. This returns to dividing as soon as any arm
    # contributes two entries again (a second score, or a second campaign).
    arms = [s[0] for s in SERIES]
    if len(set(arms)) == len(arms):
        return []
    return [
        i - 0.5
        for i in range(1, len(SERIES))
        if SERIES[i][0] != SERIES[i - 1][0]
    ]


def make_figure(summary, out_path, metrics=None):
    metrics = metrics or METRICS
    paper_style.use(font_scale=0.9)
    fig, axes = paper_style.figure(len(metrics), len(PAIRS),
                                   height=1.4 * len(metrics) + 0.2,
                                   sharex=True)

    for j, (pair_key, pair_label) in enumerate(PAIRS):
        for i, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[i, j]
            for x, (arm, score, _label, colour) in enumerate(SERIES):
                cell = summary.get(f"{arm}|{pair_key}|{score}")
                if not cell or metric_key not in cell:
                    continue
                ax.errorbar([x], [cell[metric_key]["mean"]],
                            yerr=[cell[metric_key]["std"]], color=colour,
                            marker=MARKERS.get(arm, "o"), markersize=6.5,
                            markeredgecolor="white", markeredgewidth=0.7,
                            elinewidth=1.3, capsize=3.0, capthick=1.3,
                            linestyle="none", zorder=3)

            ax.set_xlim(-0.6, len(SERIES) - 0.4)
            ax.set_xticks(range(len(SERIES)))
            ax.set_xticklabels([s[2] for s in SERIES])
            # Derived from SERIES, never hardcoded. This was a single axvline
            # at 1.5 -- correct for the two arms the figure had when it was
            # written, and silently wrong the moment IBProbit made it three:
            # the CE|VBLL divider stayed and the VBLL|IBProbit one was simply
            # absent, so four of the six points read as belonging to one arm.
            for boundary in _arm_boundaries():
                ax.axvline(boundary, color="#cccccc", linewidth=0.8,
                           linestyle="--", zorder=1)
            paper_style.few_ticks(ax, y=5)
            if i == 0:
                ax.set_title(pair_label)
            if j == 0:
                ax.set_ylabel(metric_label)

    paper_style.save(fig, out_path)


def make_id_figure(id_summary, out_path):
    """The in-distribution control for the checkpoint detection is scored on.

    One point per arm, with a band over seeds. Detection metrics say nothing
    about whether a model is any good in-distribution -- a weak model can still
    separate ID from OOD -- so results-plan 0.1 requires these four beside every
    detection panel.

    This used to draw TWO points per arm joined by a connector, hollow for the
    val-optimal step and filled for the final step, captioned "a long connector
    means that arm overfits without early stopping". It could never show one.
    Each run writes a single dump, taken at its val optimum, with the OOD
    forward pass alongside it -- so the two steps are the same step in all 90
    runs, the connector had length zero everywhere, and the hollow markers were
    hidden underneath the filled ones. A figure that cannot fail its own test
    reads as having passed it.

    `at_best_*` is emitted again the moment runs dump trajectories, and the
    two-point form should come back with it -- but as something measured.
    """
    if any("at_best_acc" in cell for cell in id_summary.values()):
        raise NotImplementedError(
            "id_summary carries at_best_*, which this single-point figure would "
            "silently ignore -- reporting the endpoint alone. Either (a) the "
            "JSON predates the fix and must be regenerated with "
            "scripts/analysis/fig2_detection.py, which no longer emits at_best_* "
            "for single-dump runs, or (b) the runs genuinely dumped "
            "trajectories, the val-optimal vs final comparison is now a real "
            "measurement, and the two-point form should be restored on purpose."
        )

    fig, axes = plt.subplots(len(ID_METRICS), len(ID_DATASETS),
                             figsize=(3.0 * len(ID_DATASETS), 2.1 * len(ID_METRICS)))

    for j, (dataset, ds_label) in enumerate(ID_DATASETS):
        for i, (metric, metric_label) in enumerate(ID_METRICS):
            ax = axes[i, j]
            for x, (arm, arm_label, colour, prefix) in enumerate(ID_ARMS):
                cell = id_summary.get(f"{arm}|{dataset}")
                key = f"{prefix}{metric}"
                if not cell or key not in cell:
                    continue
                ax.errorbar([x], [cell[key]["mean"]], yerr=[cell[key]["std"]],
                            fmt="none", ecolor="#999999", elinewidth=1,
                            capsize=3, zorder=2)
                ax.scatter([x], [cell[key]["mean"]], s=58, color=colour,
                           zorder=4, edgecolors="white", linewidths=0.8)

            ax.set_xlim(-0.6, len(ID_ARMS) - 0.4)
            ax.set_xticks(range(len(ID_ARMS)))
            ax.set_xticklabels([a[1] for a in ID_ARMS] if i == len(ID_METRICS) - 1 else [],
                               rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(ds_label, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=9)

    fig.suptitle(
        "DRAFT Fig 2 control — in-distribution performance of the same runs\n"
        "scored at each run's val-optimal step, which is where the OOD dump "
        "is taken; bands are ±1 sd over seeds",
        fontsize=9.5, y=1.055,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default="scripts/results/fig2_detection.json")
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--with-aupr", action="store_true",
                   help="Add AUPR as a third row.")
    args = p.parse_args()

    # `mlflow:<name>` fetches the published artifact; anything else is a
    # local path. This is what decouples plotting from the cluster that
    # produced the dumps.
    from bllarse.mlflow_utils import resolve_analysis_path
    json_path = resolve_analysis_path(args.json)

    with open(json_path) as fh:
        payload = json.load(fh)
    summary = payload["summary"]
    id_summary = payload.get("id_summary", {})

    metrics = METRICS + [AUPR] if args.with_aupr else METRICS
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(summary, os.path.join(args.out_dir, f"fig2_draft.{ext}"),
                    metrics=metrics)
        if id_summary:
            make_id_figure(id_summary, os.path.join(args.out_dir, f"fig2_id_control.{ext}"))

    # The headline contrast, printed so it is not read off the figure by eye.
    print("\nNative epistemic score vs MSP on the same model (VBLL, AUROC):")
    for pair_key, label in PAIRS:
        native = summary.get(f"vbll|{pair_key}|native")
        msp = summary.get(f"vbll|{pair_key}|msp")
        if native and msp:
            delta = native["auroc"]["mean"] - msp["auroc"]["mean"]
            print(f"  {pair_key:<24s} native={native['auroc']['mean']:.4f}  "
                  f"msp={msp['auroc']['mean']:.4f}  delta={delta:+.4f}")


if __name__ == "__main__":
    main()
