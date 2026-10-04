"""Shared matplotlib style for report figures."""

import matplotlib as mpl

mpl.use("Agg")  # render to files; no display needed

PALETTE_MAIN = "#2D6A9F"
PALETTE_ACCENT = "#E07B39"
PALETTE_NEG = "#C0392B"
PALETTE_POS = "#27AE60"
PALETTE_NEUTRAL = "#7F8C8D"
PALETTE_CAT = ["#2D6A9F", "#E07B39", "#27AE60", "#8E44AD", "#C0392B", "#16A085", "#D4AC0D"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (11, 5),
            "figure.dpi": 120,
            "figure.constrained_layout.use": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE_CAT),
        }
    )
