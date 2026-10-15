"""One visual style for every figure in the paper.

Figures are drawn at the size they are printed at. `TEXT_WIDTH` is the paper's
text block, so a figure made `TEXT_WIDTH` wide and included at `\\textwidth` is
not rescaled, and a font size set here is the font size on the page. That is
what keeps labels readable: a figure drawn 10-14 inches wide and shrunk to 5.5
takes a 9 pt label down to under 4 pt.

For the same reason nothing is saved with `bbox_inches="tight"` -- it changes
the size of the saved figure, so the scale on the page stops being 1. Layout is
left to matplotlib's constrained layout instead.

No titles and no explanatory text on the figures: both belong in the caption.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

TEXT_WIDTH = 5.5  # inches, the NeurIPS text block

UP = r"$\uparrow$"
DOWN = r"$\downarrow$"

# Shared by every script, so a method cannot change colour between figures.
# These are seaborn's "deep" blue, green, red and purple.
COLOURS = {
    "ce": "#4C72B0",
    "ce_ts": "#55A868",
    "vbll": "#C44E52",
    "ibprobit": "#8172B2",
}


def use(font_scale: float = 1.0) -> None:
    """seaborn's paper context on a white background, no grid, two spines.

    At `font_scale=1` the paper context gives 9.6 pt axis labels and 8.8 pt
    tick labels. Dense grids of small panels pass a smaller scale.
    """
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=font_scale,
        rc={
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 3.0,
            "axes.titlepad": 4.0,
            "legend.frameon": False,
            # Arial, Helvetica and Liberation Sans share metrics, so the layout
            # is the same on any machine that has one of them.
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                                "DejaVu Sans"],
            # TrueType rather than Type 3 fonts in the PDF.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def figure(nrows, ncols, *, width=TEXT_WIDTH, height, **kwargs):
    """`plt.subplots` at print size under constrained layout."""
    kwargs.setdefault("squeeze", False)
    return plt.subplots(nrows, ncols, figsize=(width, height),
                        layout="constrained", **kwargs)


def legend_below(fig, handles, labels, *, ncol=None, **kwargs):
    """One legend row under the panels, where it is not mistaken for a title."""
    return fig.legend(handles, labels, loc="outside lower center",
                      ncol=ncol or len(labels), handlelength=1.8,
                      columnspacing=1.4, handletextpad=0.5, borderaxespad=0.2,
                      **kwargs)


def few_ticks(ax, *, x=None, y=4):
    """Cap the number of major ticks, so labels do not collide in small panels.

    Steps of 1, 2 and 5 only: a step of 2.5 costs every label an extra digit.
    """
    steps = [1, 2, 5, 10]
    if x:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=x, steps=steps))
    if y:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=y, steps=steps))


# Candidate tick values within one decade, in the order they are thinned to.
_LOG_STEPS = [(1,), (1, 3), (1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 4, 5, 6, 8)]


def plain_log_ticks(ax, axis="y", *, max_ticks=4):
    """Label a log axis with plain decimals at a few round values.

    matplotlib labels a log axis that spans less than a decade with its minor
    ticks in `3 x 10^-1` form, which is wide and hard to read in a small panel.
    Call this after the data are drawn, when the axis limits are known.
    """
    import math

    which = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    chosen = []
    for steps in _LOG_STEPS:
        ticks = [s * 10.0 ** e
                 for e in range(math.floor(math.log10(lo)) - 1,
                                math.ceil(math.log10(hi)) + 1)
                 for s in steps]
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) > max_ticks:
            break
        chosen = ticks
        if len(chosen) >= 3:
            break
    if len(chosen) < 2:
        return
    which.set_major_locator(ticker.FixedLocator(chosen))
    which.set_major_formatter(ticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    which.set_minor_formatter(ticker.NullFormatter())


def save(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
