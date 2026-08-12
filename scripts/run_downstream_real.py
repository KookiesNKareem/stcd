"""Downstream task on real data: does STCD-denoising help a real application?

Protocol: take a real DVSNOISE20 recording, inject background-activity noise,
denoise with each method at a **matched keep-fraction** (so every method removes
the same number of events — the only difference is *which* events), reconstruct an
intensity video with the pretrained FireNet, and measure SSIM vs the camera's own
APS frames. The method that removes noise most accurately reconstructs closest to
the clean reference.

Outputs:
  figures/downstream_real.png
  figures/downstream_real.json
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.datasets import dvsnoise20 as dv                       # noqa: E402
from stcd.synth import inject_noise                              # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig        # noqa: E402
from stcd.baselines import baf_scores, time_surface_scores       # noqa: E402
from stcd.downstream.firenet import load_firenet, firenet_available  # noqa: E402
from stcd.downstream import reconstruction as RC                 # noqa: E402
from stcd.downstream.edncnn_real import load_real_edncnn, predict_stream  # noqa: E402
from stcd.downstream import mlpf as MLPF                         # noqa: E402
from stcd.plotstyle import apply_style, color, OURS, save        # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20")
EDN_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "edncnn", "allData_v8_preTrained.mat")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
REC = "conference-2019_11_04_14_32_45"
WIN_T0 = float(os.environ.get("DS_T0", "6.2"))
WIN_T1 = float(os.environ.get("DS_T1", "6.8"))
RWIN, NUM_BINS, WARMUP = 0.02, 5, 5
NOISE_HZ = 10.0


def robust(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def ssim_mean(recon, ref):
    from skimage.metrics import structural_similarity as ssim
    return float(np.mean([ssim(g, robust(r), data_range=1.0) for r, g in zip(recon, ref)]))


def make_figure(ssim, kf, noise_hz):
    """Render the SSIM-bars figure from a dict of {method: SSIM}. Used both by the
    full compute path and the fast replot-from-cache path."""
    denoisers = sorted([k for k in ssim if k not in ("clean (real)", "noisy")],
                       key=lambda k: -ssim[k])
    order = (["clean (real)"] if "clean (real)" in ssim else []) + denoisers \
        + (["noisy"] if "noisy" in ssim else [])
    cols = {"clean (real)": "#5f6b78", OURS: color("STCD"), "EDnCNN": color("EDnCNN"),
            "MLPF": color("MLPF"), "time-surface": color("time-surface"), "BAF": color("BAF"),
            "noisy": color("EDnCNN")}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(order, [ssim[k] for k in order],
                  color=[cols.get(k, "#5f6b78") for k in order], zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.axhline(ssim["clean (real)"], ls="--", color="#5f6b78", alpha=0.6)
    ax.set_ylabel("FireNet reconstruction SSIM vs real APS")
    ax.set_ylim(0, max(ssim.values()) * 1.34)   # headroom so the inset clears the dashed line
    ax.text(0.985, 0.985,
            f"all methods remove the same {(1-kf)*100:.0f}% of events;\nonly which events differ",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#3a4655",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f8fb", ec="#cbd5e0"))
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "downstream_real"))
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    # Fast path: replot the styled figure from the cached numbers (avoids the
    # ~hour-long per-event EDnCNN scoring). Set DS_RECOMPUTE=1 to force recompute.
    cache = os.path.join(FIG, "downstream_real.json")
    if os.environ.get("DS_RECOMPUTE") != "1" and os.path.isfile(cache):
        d = json.load(open(cache))
        print(f"replot from cache {cache}: {d['ssim']}")
        make_figure(d["ssim"], d["matched_keep_frac"], d["injected_noise_hz"])
        print(f"wrote {os.path.abspath(os.path.join(FIG, 'downstream_real.pdf'))}")
        return
    ev_path = os.path.join(DATA, "2_mat", f"{REC}.mat")
    if not (os.path.isfile(ev_path) and firenet_available()):
        print("recording or FireNet missing; skip"); return

    ev, frame_ts, aps, _ = dv.load_full(ev_path)
    evw = ev.select((ev.ts >= WIN_T0) & (ev.ts < WIN_T1))
    rng = np.random.default_rng(0)
    evn = inject_noise(evw, NOISE_HZ, WIN_T1 - WIN_T0, rng).time_sorted()

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    keep_stcd, _ = fe.filter(evn)
    kf = float(keep_stcd.mean())                       # matched keep-fraction
    def topk(scores):
        return scores >= np.quantile(scores, 1.0 - kf)
    masks = {OURS: keep_stcd,
             "time-surface": topk(time_surface_scores(evn, 5e-3)),
             "BAF": topk(baf_scores(evn, 2e-3))}
    print(f"window {len(evw)} real +noise→{len(evn)}; matched keep-fraction {kf*100:.0f}%")
    if os.path.isfile(MLPF_MODEL):    # deployable learned baseline: published MLPF (cheap)
        print("scoring MLPF over the full stream...", flush=True)
        masks["MLPF"] = topk(MLPF.predict_stream(MLPF.load_mlpf(MLPF_MODEL), evn))
    if os.path.isfile(EDN_MODEL) and os.environ.get("SKIP_EDN") != "1":  # heavy: real EDnCNN
        print("scoring real EDnCNN over the full stream (slow; ~300 ev/s)...", flush=True)
        masks["EDnCNN"] = topk(predict_stream(load_real_edncnn(EDN_MODEL), evn))

    fr = load_firenet("cpu")
    rec = {"clean (real)": RC.reconstruct_video(fr, evw, NUM_BINS, RWIN),
           "noisy": RC.reconstruct_video(fr, evn, NUM_BINS, RWIN)}
    for name, mk in masks.items():
        rec[name] = RC.reconstruct_video(fr, evn.select(mk), NUM_BINS, RWIN)
    n = min(len(v) for v in rec.values())

    t0 = float(evw.ts.min())
    times = t0 + (np.arange(n) + 1) * RWIN
    fidx = np.clip(np.searchsorted(frame_ts, times), 0, aps.shape[0] - 1)
    ref = [robust(aps[i]) for i in fidx]
    ssim = {k: ssim_mean(v[WARMUP:n], ref[WARMUP:n]) for k, v in rec.items()}
    for k, s in ssim.items():
        print(f"  reconstruction SSIM | {k:14s} = {s:.3f}")

    # ---- figure: SSIM bars ------------------------------------------------- #
    make_figure(ssim, kf, NOISE_HZ)

    with open(os.path.join(FIG, "downstream_real.json"), "w") as f:
        json.dump({"recording": REC, "injected_noise_hz": NOISE_HZ, "matched_keep_frac": kf,
                   "ssim": ssim}, f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'downstream_real.pdf'))}")


if __name__ == "__main__":
    main()
