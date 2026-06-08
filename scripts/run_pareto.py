"""Accuracy-cost Pareto plot, the strongest framing of the efficiency result.

Real-data denoising AUC vs FLOPs/event for every method. STCD sits at the top-left,
matching the real pretrained EDnCNN's accuracy at ~22 million x fewer FLOPs.

All AUCs come from the SAME 16-recording eval as Table 1 (figures/data/edncnn_real.json),
so the figure and the table are numerically consistent.

Outputs:
  figures/pareto.png
  figures/pareto.json
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.plotstyle import apply_style, color  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA = os.path.join(FIG, "data")

# FLOPs/event and the edncnn_real.json key for each method (single source of truth).
METHODS = [
    ("STCD",         13,          "STCD"),
    ("EDnCNN",       282_581_248, "EDnCNN (real)"),
    ("MLPF",         1980,        "MLPF"),
    ("time-surface", 32,          "time-surface"),
    ("BAF",          16,          "BAF"),
    ("KNoise",       8,           "KNoise"),
]


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    ed_path = next(p for p in (os.path.join(DATA, "edncnn_real.json"),
                               os.path.join(FIG, "edncnn_real.json")) if os.path.isfile(p))
    auc = json.load(open(ed_path))["mean_auc"]

    pts = {name: (fl, auc[key], color(name)) for name, fl, key in METHODS if key in auc}
    for name, (fl, au, _) in pts.items():
        print(f"  {name:14s} FLOPs/ev={fl:>10}  AUC={au:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for name, (fl, au, c) in pts.items():
        ax.scatter(fl, au, s=130, color=c, edgecolor="k", linewidth=0.8, zorder=3)
        ax.annotate(name, (fl, au), textcoords="offset points", xytext=(8, 8), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per event  (compute cost →)")
    ax.set_ylabel("real-data denoising AUC")
    ax.set_title("Accuracy vs compute cost (real DVSNOISE20)")
    ax.set_ylim(0.5, 0.88)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pareto.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "pareto.json"), "w") as f:
        json.dump({n: {"flops_per_event": fl, "auc": au} for n, (fl, au, _) in pts.items()},
                  f, indent=2)
    print(f"\nFigures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
