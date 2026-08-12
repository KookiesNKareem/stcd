"""A real noisy event frame from DVSNOISE20 (DAVIS346), polarity-coloured, for a
slide. Just the picture — no method, minimal text. -> figures/event_picture.png"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.datasets import dvsnoise20 as dv  # noqa: E402
from stcd.plotstyle import apply_style, save  # noqa: E402

MAT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
SCENE = os.environ.get("EVPIC_SCENE", "bike")
WIN = float(os.environ.get("EVPIC_WIN", "0.1"))   # seconds


def main():
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    paths = sorted(glob.glob(os.path.join(MAT, f"{SCENE}*.mat"))) or sorted(glob.glob(os.path.join(MAT, "*.mat")))
    ev, _, _, _ = dv.load_full(paths[0])
    t0 = ev.ts.min(); best, bn = t0, -1                 # most-active WIN window
    for a in np.arange(t0, ev.ts.max() - WIN, WIN):
        n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
        if n > bn: best, bn = a, n
    w = ev.select((ev.ts >= best) & (ev.ts < best + WIN))
    on = w.ps > 0
    print(f"{SCENE}: {len(w)} events in {WIN*1e3:.0f} ms")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_facecolor("#0b0b0f"); fig.patch.set_facecolor("white")
    ax.scatter(w.xs[~on], w.ys[~on], s=3.0, c="#4aa3ff", linewidths=0, alpha=0.85)   # OFF
    ax.scatter(w.xs[on], w.ys[on], s=3.0, c="#ff5a5a", linewidths=0, alpha=0.85)      # ON
    ax.set_xlim(0, 346); ax.set_ylim(260, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    # tiny legend dots
    ax.scatter([], [], s=22, c="#ff5a5a", label="brighter (ON)")
    ax.scatter([], [], s=22, c="#4aa3ff", label="darker (OFF)")
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, labelcolor="#1a202c")
    fig.tight_layout()
    out = save(fig, os.path.join(FIG, "event_picture"), formats=("png",))[0]
    plt.close(fig)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
