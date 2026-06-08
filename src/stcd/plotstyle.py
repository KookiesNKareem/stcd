"""Shared figure style — a clean, consistent look for presentation figures.

Call ``apply_style()`` once after ``matplotlib.use("Agg")``. Use ``METHOD_COLORS``
so a method has the same colour in every figure (ours = blue throughout).
"""

from __future__ import annotations

# Canonical display name for our method (Spatio-Temporal Coincidence Denoiser).
OURS = "STCD"

# Consistent per-method palette (ours is always the same blue).
METHOD_COLORS = {
    "STCD": "#2b6cb0", "STCD (ours)": "#2b6cb0", "ours": "#2b6cb0",
    # legacy aliases so older JSON keys still colour correctly
    "front-end": "#2b6cb0", "front-end (ours)": "#2b6cb0", "Spiking front-end": "#2b6cb0",
    "EDnCNN": "#e53e3e", "EDnCNN-lite (learned)": "#e53e3e", "learned": "#e53e3e",
    "MLPF": "#319795",
    "time-surface": "#38a169",
    "BAF": "#dd6b20",
    "KNoise": "#805ad5",
    "event-based": "#2b6cb0", "event-driven": "#2b6cb0",
    "frame": "#e53e3e", "frame-based": "#e53e3e",
}
PALETTE = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#e53e3e", "#718096"]


def apply_style() -> None:
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
        "axes.prop_cycle": cycler(color=PALETTE),
    })


def color(name: str, default: str = "#718096") -> str:
    return METHOD_COLORS.get(name, default)
