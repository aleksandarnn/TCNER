"""Shared matplotlib style so every figure in the repo looks like one system."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import CLASSES, FIG_DIR

# Colour-blind-safe categorical palette (Okabe-Ito), one fixed colour per venue.
VENUE_COLORS = dict(zip(CLASSES, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]))
METHOD_COLORS = {
    "Exact Shapley": "#000000",
    "KernelSHAP": "#0072B2",
    "LIME": "#E69F00",
    "Occlusion": "#009E73",
    "Linear SHAP": "#CC79A7",
    "Coef x input": "#D55E00",
    "Anchors": "#56B4E9",
}
POS, NEG, NEUTRAL = "#0072B2", "#D55E00", "#8C8C8C"


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", facecolor="white")


def attribution_bar(ax, words, values, title: str = "", xlabel: str = "") -> None:
    """Horizontal bar chart of word attributions, largest |value| on top."""
    import numpy as np

    order = np.argsort(np.abs(values))
    w = [words[i] for i in order]
    v = np.asarray(values)[order]
    ax.barh(w, v, color=[POS if x > 0 else NEG for x in v])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def close(fig) -> None:
    plt.close(fig)
