"""Graphical abstract for IEEE Sensors Journal submission.

A single landscape panel: a real DVSNOISE20 slice with injected background noise
(left) is denoised by STCD, one spiking neuron on a \$50 FPGA inside the camera
(centre), yielding a clean, lower-rate stream (right); a banner states the
headline result. Built from real data + the real STCD filter so it is honest and
reproducible.

Output: figures/graphical_abstract.png  (landscape, high-res, large text)
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.events import Events                              # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig   # noqa: E402
from stcd.synth import inject_noise                         # noqa: E402
from stcd.plotstyle import apply_style, color, OURS         # noqa: E402
from stcd.datasets import dvsnoise20 as dv                  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
MAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
SCENE_PREF = ["bike", "soccer", "stairs", "toys", "conference", "classroom"]
WIN, NOISE_HZ = 0.03, 15.0


def accumulate(ev, mask=None):
    xs, ys = (ev.xs, ev.ys) if mask is None else (ev.xs[mask], ev.ys[mask])
    img = np.zeros((ev.H, ev.W)); np.add.at(img, (ys, xs), 1.0); return img


def main() -> None:
    apply_style()
    paths = {os.path.basename(p).split("-")[0]: p
             for p in sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))}
    if not paths:
        sys.exit("no DVSNOISE20 .mat present (see DATA.md).")
    scene = next((s for s in SCENE_PREF if s in paths), next(iter(paths)))
    ev, _, _, _ = dv.load_full(paths[scene])

    t0, t1 = float(ev.ts.min()), float(ev.ts.max())
    best, bn = t0, -1
    for a in np.arange(t0, max(t0, t1 - WIN), WIN):
        n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
        if n > bn:
            best, bn = a, n
    w = ev.select((ev.ts >= best) & (ev.ts < best + WIN))
    real = Events(w.xs, w.ys, w.ts, w.ps, labels=np.ones(len(w), bool), H=w.H, W=w.W)
    evn = inject_noise(real, NOISE_HZ, WIN, np.random.default_rng(0))
    lab = evn.labels
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    keep, _ = fe.filter(evn)
    nr = float((~keep & ~lab).sum() / max(int((~lab).sum()), 1))
    removed = 1.0 - keep.sum() / len(evn)

    noisy, den = accumulate(evn), accumulate(evn, keep)
    vmax = max(1.0, float(np.percentile(noisy[noisy > 0], 98.0)))
    shp = lambda im: np.clip(im, 0, vmax) / vmax
    ink, accent = "#1a202c", color(OURS)

    # ---- canvas: one background axis (0..1) + two image insets --------------
    fig = plt.figure(figsize=(11.0, 4.7))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    def panel(rect, img, title, sub, edge):
        ax = fig.add_axes(rect); ax.imshow(shp(img), cmap="magma", vmin=0, vmax=1,
                                            interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(edge); s.set_linewidth(2.4)
        x0, y0, ww, hh = rect
        bg.text(x0 + ww / 2, y0 + hh + 0.035, title, ha="center", va="bottom",
                fontsize=12.5, weight="bold", color=ink)
        bg.text(x0 + ww / 2, y0 - 0.04, sub, ha="center", va="top",
                fontsize=10, color=edge, weight="bold")

    panel([0.035, 0.30, 0.265, 0.50], noisy, "Raw event stream",
          "signal + background-activity noise", "#e53e3e")
    panel([0.700, 0.30, 0.265, 0.50], den, "STCD output",
          f"{nr*100:.0f}% of noise removed  ($-${removed*100:.0f}% events)", accent)

    # ---- centre: STCD-on-FPGA box -------------------------------------------
    bg.add_patch(FancyBboxPatch((0.365, 0.345), 0.27, 0.41,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 fc="#EBF4FF", ec=accent, lw=2.4, transform=bg.transData, zorder=2))
    cx = 0.5
    bg.text(cx, 0.705, "STCD", ha="center", va="center", fontsize=21, weight="bold", color=accent)
    bg.text(cx, 0.625, "one weight-shared", ha="center", va="center", fontsize=10.5, color=ink)
    bg.text(cx, 0.585, "LIF neuron per pixel", ha="center", va="center", fontsize=10.5, color=ink)
    bg.text(cx, 0.515, "13 integer ops/event", ha="center", va="center", fontsize=10, color=ink)
    bg.text(cx, 0.477, "no multipliers $\\cdot$ no training", ha="center", va="center", fontsize=10, color=ink)
    bg.text(cx, 0.405, r"\$50 iCE40 FPGA $\cdot$ $\sim$10 mW", ha="center", va="center",
            fontsize=10.5, weight="bold", color=ink)
    bg.text(cx, 0.368, "in-camera, 2.67 Mev/s", ha="center", va="center", fontsize=9.5, color="#4a5568")

    # ---- flow arrows --------------------------------------------------------
    for x0, x1 in [(0.305, 0.362), (0.638, 0.695)]:
        bg.add_patch(FancyArrowPatch((x0, 0.55), (x1, 0.55), transform=bg.transData,
                     arrowstyle="-|>", mutation_scale=20, lw=2.2, color="#4a5568", zorder=3))

    # ---- title + result banner ---------------------------------------------
    bg.text(0.5, 0.95, "Background-Activity Denoising for Event Cameras with a Single Spiking Neuron",
            ha="center", va="center", fontsize=14.5, weight="bold", color=ink)
    bg.add_patch(Rectangle((0.035, 0.045), 0.93, 0.115, transform=bg.transData,
                 fc=accent, ec="none", alpha=0.12, zorder=0))
    bg.text(0.5, 0.103,
            r"Ties learned EDnCNN $\cdot$ 3rd of 12 on E-MLB $\cdot$ $\sim$$2{\times}10^{7}$ fewer ops/event",
            ha="center", va="center", fontsize=10.5, weight="bold", color=ink)

    out = os.path.join(FIG, "graphical_abstract.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"scene={scene}  noise removed={nr*100:.0f}%  events removed={removed*100:.0f}%")
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
