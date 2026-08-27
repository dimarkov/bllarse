#!/usr/bin/env python3
"""Fig 5 -- sample efficiency, in distribution.

    python scripts/plotting/plot_fig5.py --scores /ptmp/$USER/large_all.json

Four rows -- accuracy, ECE, NLL, Brier -- against samples seen, for AdamW+CE,
AdamW+CE+TS, VBLL and IBProbit, with a band over seeds. The claim is about the
CURVES, not the endpoints: IBProbit is expected to reach usable calibration far
earlier than CE, with the gap largest early.

The eval set is the 80% of ImageNet's official validation split that
`imagenet.pinned_eval_split` reserves as `test`; the other 20% is the `val`
partition, used to fit T and never reported. The split is derived from the
dataset name alone, so it is identical across arms, seeds and machines.

TS is drawn for CE ONLY, and T is refit at every step against that step's val
dump -- never carried across steps. VBLL and IBProbit are shown RAW because
their claim is calibration by construction.

Reads score_final.py's JSON, which is written after every run, so this will
happily draw a partial campaign. It says so on the figure when it does.
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
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6.4, 3.0 * len(metrics)),
                             sharex=True)
    seen_labels = []

    for ax, (metric, ylabel, _lower_better, log_y) in zip(axes, metrics):
        for (arm, pass_key), label, colour, marker in final_json.ARMS:
            per_seed = final_json.id_series(data, arm, pass_key, metric)
            if not per_seed:
                continue
            xs, mean, lo, hi, n = final_json.band(per_seed)
            ax.plot(xs, mean, color=colour, lw=1.8, label=label, zorder=3)
            # min-max over seeds, not a standard error: at five seeds the
            # spread is the object of interest, and a sigma ribbon would hide
            # which runs are the outliers.
            if max(n) > 1:
                ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, lw=0,
                                zorder=2)
            ax.plot(xs[-1], mean[-1], marker=marker, ms=5, color=colour,
                    mec="white", mew=0.9, zorder=4)
            if label not in seen_labels:
                seen_labels.append(label)
        if x_log:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        # A log axis spanning under two decades labels its MINOR ticks too,
        # which on a narrow panel overprints into an unreadable smear.
        ax.xaxis.set_minor_formatter(NullFormatter())

    axes[-1].set_xlabel("samples seen")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Fig 5 — sample efficiency (ImageNet val, held-out 80%)\n"
                 + final_json.provenance(data), fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig5] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True,
                   help="score_final.py output JSON.")
    p.add_argument("--out-dir", default="scripts/figures")
    p.add_argument("--name", default="fig5_sample_efficiency")
    p.add_argument("--linear-x", action="store_true",
                   help="Linear samples axis. The log axis is the default "
                        "because the claim is about the early trajectory.")
    args = p.parse_args()

    data = final_json.load(args.scores)
    print(f"[fig5] {final_json.provenance(data)}")
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        make_figure(data, os.path.join(args.out_dir, f"{args.name}.{ext}"),
                    x_log=not args.linear_x)


if __name__ == "__main__":
    main()
