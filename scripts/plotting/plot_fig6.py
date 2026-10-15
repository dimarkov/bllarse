#!/usr/bin/env python3
"""Fig 6 -- the same trajectory, scored under distribution shift.

    python scripts/plotting/plot_fig6.py --scores /ptmp/$USER/large_all.json

Fig 5's axes, evaluated on the shift suite: ImageNet-R, -A, -V2 and -Sketch
(`finetune.imagenet.OOD_KEYS`). Rows are metrics, columns are shift sets. Fig 5
is in-distribution and Fig 6 is the same runs under shift; they do not merge.

ImageNet-R and -A cover only 200 of the 1000 classes and are evaluated with the
subset mask, so their accuracies are over a 200-way decision and are NOT
comparable in level to V2, Sketch or the ID panels. V2 and Sketch span all 1000
classes and are unmasked.

AdamW+CE is drawn in its temperature-scaled form, with T fit on the
IN-DISTRIBUTION val split at that step and applied unchanged to the shifted
logits. No OOD label goes into T. Refitting it on the shifted set would be
test-time adaptation, and the symmetric move would be to run IBProbit's CAVI
update or VBLL's posterior update there too -- every arm is fit
in-distribution and then evaluated under shift.

The OOD grid is a stride of the dump grid, and on the dinov3_large campaign an
irregular one -- the OOD workers were under-provisioned and the trainer skipped
scheduled checkpoints under backpressure. Points are drawn on their own x
values and marked individually, never interpolated onto the ID grid: a gap here
is a missing measurement, not a flat stretch.

ImageNet-C is deliberately absent. The available builder is severity-5 only,
which measures corruption robustness at one arbitrary severity rather than
shift. ObjectNet is absent because it is not in the registry.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paper_style  # noqa: E402
from bllarse.analysis import final_json  # noqa: E402


def make_figure(data, out_path, *, x_log=True, tuning_budget=None):
    metrics = final_json.OOD_METRICS
    sets = final_json.OOD_SETS
    # Sixteen small panels, so one step below the paper context's default size.
    paper_style.use(font_scale=0.85)
    fig, axes = paper_style.figure(len(metrics), len(sets),
                                   height=1.25 * len(metrics) + 0.45,
                                   sharex=True)
    drawn = 0

    for r, (metric, ylabel, _lb, log_y) in enumerate(metrics):
        for c, (split, title) in enumerate(sets):
            ax = axes[r][c]
            for (arm, pass_key), label, colour, marker in \
                    final_json.ood_arms(data):
                per_seed = final_json.ood_series(data, arm, split, metric,
                                                 pass_key)
                if not per_seed:
                    continue
                xs, mean, lo, hi, n = final_json.band(per_seed)
                if not xs:
                    continue
                # Markers on every point, because the grid can be irregular
                # and a bare line would imply measurements between them. Small,
                # so that a regular grid of forty points still reads as a line.
                ax.plot(xs, mean, color=colour, lw=1.3, marker=marker, ms=2.0,
                        mew=0, label=label, zorder=3)
                if max(n) > 1:
                    ax.fill_between(xs, lo, hi, color=colour, alpha=0.22, lw=0,
                                    zorder=2)
                drawn += 1
            # Where the search actually looked; everything right of it is the
            # AdamW SDE transfer extrapolating past its selection budget.
            if tuning_budget:
                ax.axvline(tuning_budget, color="0.35", ls="--", lw=0.9,
                           zorder=1)
            if x_log:
                ax.set_xscale("log")
                # A log axis spanning under two decades labels its MINOR
                # ticks too, which on a narrow panel overprints into an
                # unreadable smear.
                ax.xaxis.set_minor_formatter(NullFormatter())
            if log_y:
                ax.set_yscale("log")
                paper_style.plain_log_ticks(ax, "y")
            else:
                paper_style.few_ticks(ax, y=4)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(metrics) - 1:
                ax.set_xlabel("Samples seen")

    if not drawn:
        raise SystemExit("[fig6] no OOD points in the scored JSON")
    handles, labels = axes[0][0].get_legend_handles_labels()
    paper_style.legend_below(fig, handles, labels, markerscale=2.0)
    paper_style.save(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="fig6_shift")
    p.add_argument("--tuning-budget", type=float, default=None,
                   help="Samples the hyperparameter search scored trials at, "
                        "marked with a dashed rule. 0 to omit.")
    p.add_argument("--linear-x", action="store_true")
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig6] {final_json.provenance(data)}")
    if not args.tuning_budget:
        print("[fig6] no --tuning-budget: the selection-budget rule is OMITTED. "
              "See plot_fig5 for why there is no default.")
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    x_log=not args.linear_x,
                    tuning_budget=args.tuning_budget or None)


if __name__ == "__main__":
    main()
