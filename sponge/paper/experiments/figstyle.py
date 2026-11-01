"""Shared figure style: LaTeX-rendered, publication-quality, colour-blind safe.

Figures are drawn at their final printed width (JCP text width 390 pt =
5.41 in) so fonts are true size (8--9 pt against 12 pt body text) and are
saved as vector PDF (for LaTeX) plus a 300-dpi PNG.
"""
import os

import matplotlib as mpl
from cycler import cycler

TEXTWIDTH = 390.0 / 72.27          # inches (elsarticle preprint, 12pt)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(ROOT, "figures")

# Okabe--Ito colour-blind-safe palette
BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY, BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
PALETTE = [BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY, BLACK]


def setup():
    mpl.use("Agg")
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
                               r"\usepackage{bm}\usepackage[Euler]{upgreek}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "axes.prop_cycle": cycler(color=PALETTE),
        "image.cmap": "RdBu_r",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def panel_label(ax, text, x=0.02, y=0.97, color="k"):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=color)


def save(fig, name):
    """Save figures/<name>.pdf and figures/<name>.png."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    print("wrote", os.path.join(FIGDIR, name + ".{pdf,png}"))


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 1.7\\times10^{-2}."""
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def wavefield_grid(rows, row_labels, col_titles, vmax, edges, notes=None,
                   cbar_label="", height=None, n=None):
    """Grid of wavefield snapshots sharing one symmetric colour scale.

    rows: list of lists of 2-D arrays (already oriented for imshow, origin lower).
    notes: optional list of lists of strings drawn at the bottom of each panel.
    edges: positions of the dashed layer lines (pixel coordinates).
    """
    import matplotlib.pyplot as plt
    nr, nc = len(rows), len(rows[0])
    w = TEXTWIDTH
    pw = (w - 0.55) / nc                      # panel width (in), room for row labels + colour bar
    h = height or nr * pw + 0.35 + ((0.22 if nc <= 5 else 0.34) * nr if notes else 0)
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(nr, nc + 1, width_ratios=[1] * nc + [0.05],
                          left=0.06, right=0.935, bottom=0.02, top=1 - 0.28 / h,
                          wspace=0.06, hspace=(0.15 if nc <= 5 else 0.30) if notes else 0.06)
    if notes:
        gs.update(bottom=0.2 / h)
    im = None
    axs = []
    for i in range(nr):
        rax = []
        for j in range(nc):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(rows[i][j], origin="lower", cmap="RdBu_r", vmin=-vmax,
                           vmax=vmax, interpolation="nearest")
            for e in edges:
                ax.axvline(e, color="0.25", ls=(0, (3, 2)), lw=0.5)
                ax.axhline(e, color="0.25", ls=(0, (3, 2)), lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if i == 0:
                ax.set_title(col_titles[j], pad=3)
            if j == 0:
                ax.set_ylabel(row_labels[i], labelpad=4)
            if notes and notes[i][j]:
                ax.text(0.5, -0.025, notes[i][j], transform=ax.transAxes, ha="center",
                        va="top", fontsize=7 if nc > 5 else 7.5, linespacing=1.1)
            rax.append(ax)
        axs.append(rax)
    cax = fig.add_subplot(gs[:, nc])
    fig.canvas.draw()                         # apply the equal aspect, then match the bar to the panels
    top = axs[0][-1].get_position()
    bot = axs[-1][-1].get_position()
    cpos = cax.get_position()
    cax.set_position([cpos.x0, bot.y0, cpos.width, top.y1 - bot.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, labelpad=2)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(direction="out", length=2.5, width=0.5, labelsize=7, right=True, left=False)
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.set_ticklabels([r"$-1$", r"$-\frac12$", r"$0$", r"$\frac12$", r"$1$"])
    return fig, axs
