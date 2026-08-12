"""Before/after denoising visualisation on REAL data with injected noise.

Standard controlled protocol: take a real DVSNOISE20 recording (real signal),
inject background-activity noise with *known* labels, run STCD, and accumulate the
three streams into frames on a common intensity scale:

  Real + injected noise   →   STCD denoised   →   Clean (real only)

Because we add the noise ourselves, the per-event ground truth is exact (real =
signal, injected = noise), so the SR / NR annotations are honest.

Outputs:
  figures/before_after.png
  figures/before_after.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stcd.events import Events                             # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig  # noqa: E402
from stcd.synth import inject_noise                        # noqa: E402
from stcd.plotstyle import apply_style, OURS, color, save  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
MAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
SCENE_PREF = ["bike", "soccer", "stairs", "toys", "conference", "classroom"]
WIN = 0.03          # short window → crisp structure + dark regions where noise shows
NOISE_HZ = 12.0     # injected background activity (events / pixel / second)


def accumulate(ev, mask=None):
    xs, ys = ev.xs, ev.ys
    if mask is not None:
        xs, ys = xs[mask], ys[mask]
    img = np.zeros((ev.H, ev.W), dtype=np.float64)
    np.add.at(img, (ys, xs), 1.0)
    return img


def pick_scene():
    paths = {os.path.basename(p).split("-")[0]: p
             for p in sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))}
    for name in SCENE_PREF:
        if name in paths:
            return name, paths[name]
    return (next(iter(paths.items())) if paths else (None, None))


def active_window(ev, win):
    t0, t1 = float(ev.ts.min()), float(ev.ts.max())
    best, bn = t0, -1
    for a in np.arange(t0, max(t0, t1 - win), win):
        n = int(((ev.ts >= a) & (ev.ts < a + win)).sum())
        if n > bn:
            best, bn = a, n
    return best


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()

    scene, path = pick_scene()
    if path is None:
        print("No DVSNOISE20 .mat scenes present; skipping.")
        return
    from stcd.datasets import dvsnoise20 as dv
    ev, _, _, _ = dv.load_full(path)
    a = active_window(ev, WIN)
    w = ev.select((ev.ts >= a) & (ev.ts < a + WIN))
    # label every REAL event as signal, then inject labelled BA noise
    real = Events(xs=w.xs, ys=w.ys, ts=w.ts, ps=w.ps,
                  labels=np.ones(len(w), dtype=bool), H=w.H, W=w.W)
    rng = np.random.default_rng(0)
    evn = inject_noise(real, NOISE_HZ, WIN, rng)
    lab = evn.labels                                  # True = real signal, False = injected

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    keep, _ = fe.filter(evn)

    sr = float((keep & lab).sum() / max(int(lab.sum()), 1))
    nr = float((~keep & ~lab).sum() / max(int((~lab).sum()), 1))
    print(f"scene={scene}  real={int(lab.sum())}  +noise={int((~lab).sum())}  "
          f"kept={int(keep.sum())}  SR={sr*100:.0f}%  NR={nr*100:.0f}%")

    # --- spatial frames (event count per pixel) ----------------------------- #
    noisy = accumulate(evn)
    den = accumulate(evn, keep)
    clean = accumulate(evn, lab)
    vmax = max(1.0, float(np.percentile(noisy[noisy > 0], 98.0)))
    shp = lambda im: np.clip(im, 0, vmax) / vmax

    # --- space-time strip (x vs t) to expose the TEMPORAL selectivity -------- #
    cy = evn.H // 2
    band = (evn.ys >= cy - 10) & (evn.ys <= cy + 10)        # central horizontal strip
    t_ms = (evn.ts - float(evn.ts.min())) * 1e3
    csig, cnoi = color("STCD"), "#e53e3e"

    def xt(ax, plot_mask):
        sig = plot_mask & band & lab
        noi = plot_mask & band & ~lab
        ax.scatter(evn.xs[noi], t_ms[noi], s=5, c=cnoi, alpha=0.55, lw=0)
        ax.scatter(evn.xs[sig], t_ms[sig], s=5, c=csig, alpha=0.75, lw=0)
        ax.set_xlim(0, evn.W); ax.set_ylim(WIN * 1e3, 0)
        ax.grid(alpha=0.15)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4),
                             gridspec_kw={"height_ratios": [1.12, 1.0]})
    spatial = [
        (noisy, "Real + injected noise", f"{len(evn):,} events  (+{int((~lab).sum()):,} noise)", cnoi),
        (den, f"{OURS} denoised", f"{nr*100:.0f}% noise removed\n{sr*100:.0f}% signal kept", csig),
        (clean, "Clean (real events only)", f"{int(lab.sum()):,} signal events", "#a0aec0"),
    ]
    for ax, (im, title, sub, edge) in zip(axes[0], spatial):
        ax.imshow(shp(im), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color="#2d3748")
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(edge); s.set_linewidth(2.2)

    masks = [np.ones(len(evn), bool), keep, lab]
    for ci, (ax, m) in enumerate(zip(axes[1], masks)):
        xt(ax, m)
        ax.set_xlabel("x (pixels)")
        if ci == 0:
            ax.set_ylabel("time (ms)")
    # one shared legend on the space-time row
    from matplotlib.lines import Line2D
    leg = axes[1, 0].legend(handles=[
        Line2D([0], [0], marker="o", ls="", color=csig, label="signal (real)", ms=6),
        Line2D([0], [0], marker="o", ls="", color=cnoi, label="noise (injected)", ms=6)],
        loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95,
        facecolor="white", edgecolor="#cbd5e0")
    leg.set_zorder(10)
    axes[1, 1].annotate("scatter removed · streaks kept",
                        xy=(0.5, 1.04), xycoords="axes fraction", ha="center",
                        fontsize=10, color="#2d3748")

    fig.tight_layout()
    out = save(fig, os.path.join(FIG, "before_after"), formats=("png",))[0]
    plt.close(fig)

    with open(os.path.join(FIG, "before_after.json"), "w") as f:
        json.dump({"scene": scene, "window_s": WIN, "injected_noise_hz": NOISE_HZ,
                   "n_real": int(lab.sum()), "n_injected": int((~lab).sum()),
                   "n_kept": int(keep.sum()), "sr": sr, "nr": nr}, f, indent=2)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
