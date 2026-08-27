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

from bllarse.analysis import final_json  # noqa: E402


def make_figure(data, out_path, *, x_log=True):
    metrics = final_json.METRICS
    sets = final_json.OOD_SETS
    fig, axes = plt.subplots(len(metrics), len(sets),
                             figsize=(3.5 * len(sets), 2.7 * len(metrics)),
                             sharex=True, squeeze=False)
    drawn = 0

    for r, (metric, ylabel, _lb, log_y) in enumerate(metrics):
        for c, (split, title) in enumerate(sets):
            ax = axes[r][c]
            for (arm, pass_key), label, colour, marker in final_json.ARMS:
                # Only the raw pass exists off-distribution: score_final does
                # not temperature-scale OOD dumps, since T is fit on an
                # in-distribution val split and carrying it across the shift
                # would be a different claim than the one Fig 5 makes.
                if pass_key != "id":
                    continue
                per_seed = final_json.ood_series(data, arm, split, metric)
                if not per_seed:
                    continue
                xs, mean, lo, hi, n = final_json.band(per_seed)
                # Markers on every point, because the grid is irregular and a
                # bare line would imply measurements between them.
                ax.plot(xs, mean, color=colour, lw=1.5, marker=marker, ms=3.2,
                        label=label, zorder=3)
                if max(n) > 1:
                    ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, lw=0,
                                    zorder=2)
                drawn += 1
            if x_log:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(metrics) - 1:
                ax.set_xlabel("samples seen")
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            # A log axis spanning under two decades labels its MINOR ticks too,
            # which on a narrow panel overprints into an unreadable smear.
            ax.xaxis.set_minor_formatter(NullFormatter())

    if not drawn:
        raise SystemExit("[fig6] no OOD points in the scored JSON")
    axes[0][0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Fig 6 — the same trajectory under shift "
                 "(R/A are 200-class, masked; V2/Sketch are 1000-class)\n"
                 + final_json.provenance(data), fontsize=9, y=0.997)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig6] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="fig6_shift")
    p.add_argument("--linear-x", action="store_true")
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig6] {final_json.provenance(data)}")
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    x_log=not args.linear_x)


if __name__ == "__main__":
    main()
