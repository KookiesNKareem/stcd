"""Side-by-side QUALITATIVE comparison of all denoisers on identical real
sequences at identical operating points (reviewer 2, point 3).

Protocol: take a real DVSNOISE20 recording (real BA noise, nothing injected),
pick the most-active 1 s window, and display a 30 ms sub-window at its centre.
Every method scores the SAME events (the cheap stream filters see the full 1 s
history, so their state is warm; EDnCNN/MLPF featurize per event) and keeps the
SAME fraction of the displayed events (KEEP_FRAC, default 0.80 -- the ESR
protocol's matched operating point). Because sequence, events, and keep-count
are identical, the panels differ only in WHICH events each method keeps.

Figure layout (per scene): one column per method + a Raw reference column;
top row = accumulated frame of KEPT events, bottom row = frame of REMOVED
events (a good denoiser's bottom row is isolated speckle, not structure).

Outputs:
  figures/qualitative_compare.png / .pdf   (first scene -- for the paper)
  figures/qualitative_compare_<scene>.png  (any additional scenes)

Needs locally: data/dvsnoise20/2_mat/*.mat, data/edncnn/allData_v8_preTrained.mat,
data/mlpf/vendor/0316_soft_4bit_alpha1_sigmoid.h5 (see DATA.md).
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.frontend import SpikingFrontEnd, FrontEndConfig       # noqa: E402
from stcd.baselines import baf_scores, knoise_scores, time_surface_scores  # noqa: E402
from stcd.datasets import dvsnoise20 as dv                      # noqa: E402
from stcd.plotstyle import apply_style, OURS, save              # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
MAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
EDNCNN_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "edncnn",
                            "allData_v8_preTrained.mat")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
SCENES = os.environ.get("QUAL_SCENES", "bike,soccer").split(",")
WIN = 1.0            # context window (s); stream filters warm up over it
FRAME_WIN = 0.03     # displayed / scored sub-window (s)
KEEP_FRAC = float(os.environ.get("QUAL_KEEP", "0.80"))


def accumulate(w, idx):
    img = np.zeros((w.H, w.W), dtype=np.float64)
    np.add.at(img, (w.ys[idx], w.xs[idx]), 1.0)
    return img


def matched_keep(sc: np.ndarray, n_keep: int) -> np.ndarray:
    """Boolean keep mask holding exactly n_keep top-scoring events
    (threshold ties broken deterministically by index)."""
    order = np.argsort(-sc, kind="stable")
    k = np.zeros(len(sc), dtype=bool)
    k[order[:n_keep]] = True
    return k


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    paths = {os.path.basename(p).split("-")[0]: p
             for p in sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))}
    if not paths:
        print(f"No DVSNOISE20 .mat files under {os.path.abspath(MAT_DIR)}; "
              "see DATA.md. Skipping."); return

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3,
                                        theta=1.5, dt=5e-3))
    net = None
    if os.path.isfile(EDNCNN_MODEL):
        from stcd.downstream.edncnn_real import load_real_edncnn
        net = load_real_edncnn(EDNCNN_MODEL)
    else:
        print("pretrained EDnCNN missing -- its column will be omitted")
    mlpf_w = None
    if os.path.isfile(MLPF_MODEL):
        from stcd.downstream import mlpf as MLPF
        mlpf_w = MLPF.load_mlpf(MLPF_MODEL)
    else:
        print("MLPF weights missing -- its column will be omitted")

    first = True
    for scene in SCENES:
        scene = scene.strip()
        if scene not in paths:
            print(f"scene {scene!r} not present; skipping"); continue
        ev, _, _, _ = dv.load_full(paths[scene])
        t0 = ev.ts.min(); best, bn = t0, -1
        for a in np.arange(t0, ev.ts.max() - WIN, WIN):
            n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
            if n > bn: best, bn = a, n
        w = ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()

        # the displayed / scored sub-window (stream filters keep full-WIN state)
        fa = best + (WIN - FRAME_WIN) / 2
        q = np.where((w.ts >= fa) & (w.ts < fa + FRAME_WIN))[0]
        print(f"{scene}: {len(w):,} events in {WIN} s window; "
              f"{len(q):,} displayed in {FRAME_WIN*1e3:.0f} ms")

        scores = {OURS: fe.score_events(w)[q],
                  "time-surface": time_surface_scores(w, 5e-3)[q],
                  "BAF": baf_scores(w, 2e-3)[q],
                  "KNoise": knoise_scores(w, 2e-3)[q]}
        if net is not None:
            from stcd.downstream.edncnn_real import event_features_bulk
            scores["EDnCNN"] = net(event_features_bulk(w, q)).numpy()
        if mlpf_w is not None:
            from stcd.downstream import mlpf as MLPF
            scores["MLPF"] = MLPF.mlpf_scores(w, q, mlpf_w)

        # identical operating point: every method keeps the SAME event count
        n_keep = int(round(KEEP_FRAC * len(q)))
        keeps = {m: matched_keep(np.asarray(sc, dtype=np.float64), n_keep)
                 for m, sc in scores.items()}

        order = [m for m in [OURS, "EDnCNN", "MLPF", "time-surface", "BAF",
                             "KNoise"] if m in scores]
        raw = accumulate(w, q)
        vmax = max(1.0, float(np.percentile(raw[raw > 0], 98.0)))
        # gamma < 1 lifts the dark end so sparse removed-event maps stay
        # visible at print size; white background is unaffected
        shp = lambda im: (np.clip(im, 0, vmax) / vmax) ** 0.5

        # Two banks of columns so each panel prints ~2x wider than a single
        # 7-across row: bank 1 = Raw + first 3 methods, bank 2 = the rest.
        bank1 = ["Raw"] + order[:3]
        bank2 = order[3:]
        ncol = max(len(bank1), len(bank2) + 1)
        fig, axes = plt.subplots(4, ncol, figsize=(2.6 * ncol, 8.4))

        def panel(ax, img, title=None):
            ax.imshow(shp(img), cmap="magma", vmin=0, vmax=1,
                      interpolation="nearest")
            if title:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=4)

        for j, m in enumerate(bank1):
            if m == "Raw":
                panel(axes[0, j], raw, "Raw")
                axes[1, j].axis("off")
                axes[1, j].text(0.5, 0.55, f"scene: {scene}\n\n"
                                f"every method keeps {KEEP_FRAC*100:.0f}%\n"
                                f"({n_keep:,} of {len(q):,} events);\n"
                                f"they differ only in\n"
                                f"$\\it{{which}}$ events they keep",
                                ha="center", va="center", fontsize=11,
                                transform=axes[1, j].transAxes)
            else:
                panel(axes[0, j], accumulate(w, q[keeps[m]]), m)
                panel(axes[1, j], accumulate(w, q[~keeps[m]]))
        for j, m in enumerate(bank2):
            panel(axes[2, j], accumulate(w, q[keeps[m]]), m)
            panel(axes[3, j], accumulate(w, q[~keeps[m]]))
        for r in (2, 3):
            for j in range(len(bank2), ncol):
                axes[r, j].axis("off")
        axes[3, len(bank2)].text(0.5, 0.55,
            "upper: kept events\nlower: removed events\n\na good denoiser's\n"
            "removed map is\nspeckle, not structure", ha="center", va="center",
            fontsize=11, style="italic", transform=axes[3, len(bank2)].transAxes)
        for r, lab in ((0, "kept"), (1, "removed"), (2, "kept"), (3, "removed")):
            axc = axes[r, 0] if (r, 0) != (1, 0) else axes[r, 1]
            axc.set_ylabel(lab, fontsize=12)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        stem = "qualitative_compare" if first else f"qualitative_compare_{scene}"
        out = save(fig, os.path.join(FIG, stem))
        plt.close(fig)
        print("wrote", out)
        first = False


if __name__ == "__main__":
    main()
