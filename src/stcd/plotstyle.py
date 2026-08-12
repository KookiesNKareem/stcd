"""Shared figure style — a clean, consistent, print-quality look matching the
IEEE paper (Times/serif).

Call ``apply_style()`` once after ``matplotlib.use("Agg")``. Use ``color(name)``
so a method has the same colour in every figure (ours = blue throughout). Use
``save(fig, stem)`` to write a vector PDF (for LaTeX) plus a high-DPI PNG twin.
"""

from __future__ import annotations

import os

# Canonical display name for our method (Spatio-Temporal Coincidence Denoiser).
OURS = "STCD"

# Consistent per-method palette (ours is always the same blue).
METHOD_COLORS = {
    "STCD": "#1f5c99", "STCD (ours)": "#1f5c99", "ours": "#1f5c99",
    # legacy aliases so older JSON keys still colour correctly
    "front-end": "#1f5c99", "front-end (ours)": "#1f5c99", "Spiking front-end": "#1f5c99",
    "EDnCNN": "#d1495b", "EDnCNN-lite (learned)": "#d1495b", "learned": "#d1495b",
    "MLPF": "#2a9d8f",
    "time-surface": "#3a9d4f",
    "BAF": "#e07a1f",
    "KNoise": "#7048a8",
    "event-based": "#1f5c99", "event-driven": "#1f5c99",
    "frame": "#d1495b", "frame-based": "#d1495b",
}
# Family accents (used where bars are coloured by method type, e.g. E-MLB).
FAMILY_COLORS = {"ours": "#1f5c99", "learned": "#7048a8",
                 "classical": "#9aa7b4", "raw": "#cdd6df"}
PALETTE = ["#1f5c99", "#e07a1f", "#3a9d4f", "#7048a8", "#d1495b", "#5f6b78"]


def _serif_stack():
    """Prefer Times-like serif so figures match the IEEEtran body text; fall
    back gracefully if a given face is not installed."""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    pref = ["Times New Roman", "STIX Two Text", "Nimbus Roman",
            "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"]
    return [f for f in pref if f in have] or ["serif"]


def apply_style() -> None:
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,            # embed real (TrueType) fonts, not Type-3
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "font.family": "serif",
        "font.serif": _serif_stack(),
        "mathtext.fontset": "stix",
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.edgecolor": "#3a3a3a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#9aa7b4",
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "xtick.color": "#3a3a3a",
        "ytick.color": "#3a3a3a",
        "xtick.labelcolor": "#1a1a1a",
        "ytick.labelcolor": "#1a1a1a",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "axes.prop_cycle": cycler(color=PALETTE),
    })


def color(name: str, default: str = "#5f6b78") -> str:
    return METHOD_COLORS.get(name, default)


def save(fig, stem: str, formats=("pdf", "png"), dpi: int = 600):
    """Write ``fig`` to ``<stem>.<ext>`` for each requested format.

    PDF is vector (for ``\\includegraphics`` in the paper); PNG is a high-DPI
    raster twin for quick previewing. Photo-like figures (dense scatter / imshow)
    should pass ``formats=("png",)`` to avoid huge vector files.
    Returns the list of paths written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    written = []
    for ext in formats:
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    return written
