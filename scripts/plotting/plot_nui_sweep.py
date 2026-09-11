#!/usr/bin/env python3
"""The dinov3_large num_update_iters sweep, as trajectories.

Eight IBProbit runs that differ only in `num_update_iters` -- every other
coordinate is `ft_ibprobit_noeps_large400k`'s best trial, transferred to 2
epochs by `final_config.py` -- against the two reference arms from the
`ladder2ep` campaign at the same backbone, batch and schedule.

CE is drawn TEMPERATURE-SCALED, with T refit on the val dump at every eval and
applied to that eval's test dump. That is the protocol: temperature scaling is
applied to AdamW+CE and to nothing else, because IBProbit and VBLL dump an
already-normalized log-predictive rather than a free logit scale. Raw CE is
kept as a thin dotted line so the size of that correction stays visible instead
of being folded silently into the comparison.

Usage:
    scripts/plotting/plot_nui_sweep.py --csv nui_large_trajectories.csv
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

GRID = [16, 32, 64, 128, 192, 256, 384, 512]
TUNED = 128  # what the study picked, and what the reported campaign runs

REF_STYLE = {
    "ce_ts":  dict(color="#111111", ls="--",  lw=1.8, label="AdamW+CE +TS"),
    "ce_raw": dict(color="#111111", ls=":",   lw=1.0, alpha=0.45,
                   label="AdamW+CE (no TS)"),
    "vbll":   dict(color="#c2185b", ls="-.",  lw=1.8, label="VBLL"),
}


def read(path):
    series = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            s = series[row["series"]]
            s["samples"].append(float(row["samples"]) / 1e6)
            for k in ("val_nll", "test_nll", "test_acc"):
                s[k].append(float(row[k]))
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()}
            for k, v in series.items()}


def nui_color(nui):
    """Sequential over log2(nui), so the reader sees the axis, not eight hues."""
    t = (np.log2(nui) - np.log2(16)) / (np.log2(512) - np.log2(16))
    return plt.cm.viridis(0.08 + 0.84 * t)


def argmin_marker(ax, d, key, color, zorder):
    """The val-optimal point. IBProbit turns and the reference arms do not, and
    that difference is invisible if only the endpoint is scored."""
    i = int(np.argmin(d["val_nll"]))
    ax.plot(d["samples"][i], d[key][i], marker="o", ms=5, mfc="white",
            mec=color, mew=1.6, zorder=zorder + 2, ls="none")


def draw(ax, data, key, *, zoom=None, logy=False, marks=True):
    for nui in GRID:
        d = data.get(f"ibprobit_nui{nui}")
        if d is None:
            continue
        tuned = nui == TUNED
        ax.plot(d["samples"], d[key], color=nui_color(nui),
                lw=3.0 if tuned else 1.4,
                zorder=5 if tuned else 3,
                solid_capstyle="round",
                path_effects=None)
        if tuned:
            # A halo rather than a different colour: the line must stay on the
            # viridis scale so its POSITION on the axis is still readable.
            ax.plot(d["samples"], d[key], color="white", lw=5.0, zorder=4)
            ax.plot(d["samples"], d[key], color=nui_color(nui), lw=3.0, zorder=5)
        if marks:
            argmin_marker(ax, d, key, nui_color(nui), 5 if tuned else 3)
    for name, style in REF_STYLE.items():
        d = data.get(name)
        if d is None:
            continue
        ax.plot(d["samples"], d[key], zorder=6,
                **{k: v for k, v in style.items() if k != "label"})
        if marks and name != "ce_raw":
            argmin_marker(ax, d, key, style["color"], 6)
    if logy:
        ax.set_yscale("log")
    if zoom:
        ax.set_ylim(*zoom)
    ax.set_xlabel("samples seen (M)")
    ax.grid(alpha=0.25, lw=0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="nui_large_trajectories.csv")
    p.add_argument("--out", default="scripts/figures/nui_sweep_large.png")
    args = p.parse_args()

    data = read(args.csv)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    draw(axes[0], data, "test_nll", logy=True)
    axes[0].set_ylabel("test NLL")
    axes[0].set_title("test NLL, full range (log)", fontsize=10)

    draw(axes[1], data, "test_nll", zoom=(0.42, 0.62))
    axes[1].set_ylabel("test NLL")
    axes[1].set_title("test NLL, zoomed  (o = each run's val-optimal point)",
                      fontsize=10)

    draw(axes[2], data, "test_acc", zoom=(0.845, 0.885), marks=False)
    axes[2].set_ylabel("test accuracy")
    axes[2].set_title("test accuracy", fontsize=10)

    handles = [Line2D([], [], color=nui_color(n), lw=3.0 if n == TUNED else 1.4,
                      label=f"nui={n}" + ("  (tuned)" if n == TUNED else ""))
               for n in GRID]
    handles += [Line2D([], [], **s) for s in REF_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "dinov3_large, ImageNet-1k, 2 epochs, seed 137, constant LR — "
        "IBProbit num_update_iters sweep\n"
        "all coordinates but num_update_iters frozen at "
        "ft_ibprobit_noeps_large400k's best trial; CE is temperature-scaled "
        "(T refit on val at every eval)",
        fontsize=9.5, y=1.14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
