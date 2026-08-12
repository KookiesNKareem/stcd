"""Real-data downstream recognition: N-Cars (car vs background) under injected noise.

Loads real Prophesee N-Cars event recordings, injects background-activity noise to
create a low-SNR regime, and trains+tests a small spiking CNN on the events after
each denoiser. Every denoiser is applied as an event-level filter at a *matched
keep-fraction* (each removes the same per-clip count STCD removes), so the figure
isolates *which* events a method keeps, not how many — the same protocol as the
FireNet reconstruction experiment.

Conditions: clean (upper bound) / noisy (no denoising) / STCD / time-surface /
BAF / MLPF (deployable learned, if weights present).

Outputs:
  figures/ncars_recognition.{pdf,png}
  figures/ncars_recognition.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stcd.events import Events                               # noqa: E402
from stcd.datasets import ncars                              # noqa: E402
from stcd.synth import inject_noise                          # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig    # noqa: E402
from stcd.baselines import baf_scores, time_surface_scores   # noqa: E402
from stcd.downstream import recognition as R                 # noqa: E402
from stcd.downstream import mlpf as MLPF                      # noqa: E402
from stcd.plotstyle import apply_style, color, OURS, save     # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ncars")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
DS = 2
H, W = ncars.DEFAULT_H // DS, ncars.DEFAULT_W // DS
DUR, DT = 0.1, 1e-2
NOISE_HZ = 5.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def downscale(ev: Events) -> Events:
    return replace(ev, xs=ev.xs // DS, ys=ev.ys // DS, H=H, W=W)


def add_noise(streams, rate, seed):
    rng = np.random.default_rng(seed)
    return [inject_noise(s, rate, DUR, rng) for s in streams]


def topk_keep(scores, k):
    """Keep the k highest-scoring events (matched keep-count)."""
    n = len(scores)
    if k >= n:
        return np.ones(n, bool)
    if k <= 0:
        return np.zeros(n, bool)
    thr = np.partition(scores, n - k)[n - k]
    return scores >= thr


def make_filtered(streams, method, fe, mlpf_w):
    """Apply `method` to each noisy stream at STCD's per-clip keep-count."""
    out = []
    for s in streams:
        if len(s) == 0:
            out.append(s); continue
        keep_stcd, _ = fe.filter(s)
        k = int(keep_stcd.sum())
        if method == OURS:
            m = keep_stcd
        elif method == "time-surface":
            m = topk_keep(time_surface_scores(s, 5e-3), k)
        elif method == "BAF":
            m = topk_keep(baf_scores(s, 2e-3), k)
        elif method == "MLPF":
            m = topk_keep(MLPF.predict_stream(mlpf_w, s), k)
        else:
            raise ValueError(method)
        out.append(s.select(m))
    return out


def make_figure(results):
    import matplotlib.pyplot as plt
    order = ["clean (upper bound)"] + [m for m in [OURS, "MLPF", "time-surface", "BAF"]
                                       if m in results] + ["noisy"]
    cols = {"clean (upper bound)": "#5f6b78", OURS: color("STCD"), "MLPF": color("MLPF"),
            "time-surface": color("time-surface"), "BAF": color("BAF"), "noisy": color("EDnCNN")}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(order, [results[k] * 100 for k in order],
                  color=[cols[k] for k in order], zorder=3)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.axhline(results["clean (upper bound)"] * 100, ls="--", color="#5f6b78", alpha=0.6)
    ax.axhline(50, ls=":", color="#9aa7b4", alpha=0.7)
    ax.set_ylabel("N-Cars test accuracy (\\%)")
    ax.set_ylim(45, 100)
    ax.text(0.5, 0.97, "denoisers remove the same per-clip count; only which events differ",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#3a4655",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f8fb", ec="#cbd5e0"))
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "ncars_recognition"))
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    torch.manual_seed(0)
    cache = os.path.join(FIG, "ncars_recognition.json")
    if os.environ.get("NCARS_RECOMPUTE") != "1" and os.path.isfile(cache):
        d = json.load(open(cache))
        print(f"replot from cache {cache} (NCARS_RECOMPUTE=1 to retrain)")
        make_figure(d["test_accuracy"])
        print(f"wrote {os.path.abspath(os.path.join(FIG, 'ncars_recognition.pdf'))}")
        return
    print(f"device={DEVICE}, grid={H}x{W}, {int(DUR/DT)} time-bins")

    streams, labels, _ = ncars.load_split(ROOT, "test", limit_per_class=300,
                                          H=ncars.DEFAULT_H, W=ncars.DEFAULT_W, seed=0)
    streams = [downscale(s) for s in streams]
    labels = np.array(labels)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(streams))
    streams = [streams[i] for i in idx]; labels = labels[idx]
    n_test = 120
    te_streams, te_y = streams[:n_test], labels[:n_test]
    tr_streams, tr_y = streams[n_test:], labels[n_test:]
    tr_y, te_y = torch.from_numpy(tr_y).long(), torch.from_numpy(te_y).long()
    print(f"  {len(tr_streams)} train / {len(te_streams)} test")

    tr_noisy = add_noise(tr_streams, NOISE_HZ, seed=100)
    te_noisy = add_noise(te_streams, NOISE_HZ, seed=200)

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=DT))
    mlpf_w = MLPF.load_mlpf(MLPF_MODEL) if os.path.isfile(MLPF_MODEL) else None

    # tensorised inputs per condition (clean/noisy are unfiltered; denoisers matched-keep)
    conds = {"clean (upper bound)": (tr_streams, te_streams),
             OURS: None, "MLPF": None, "time-surface": None, "BAF": None,
             "noisy": (tr_noisy, te_noisy)}
    if mlpf_w is None:
        conds.pop("MLPF")
    for m in [OURS, "MLPF", "time-surface", "BAF"]:
        if m in conds:
            conds[m] = (make_filtered(tr_noisy, m, fe, mlpf_w),
                        make_filtered(te_noisy, m, fe, mlpf_w))

    cfg = R.RecogConfig(epochs=30, lr=2e-3, batch=32)
    results = {}
    for name, (xtr_s, xte_s) in conds.items():
        xtr = R.stack_tensors(xtr_s, DT, DUR, H, W)
        xte = R.stack_tensors(xte_s, DT, DUR, H, W)
        model = R.train_classifier(xtr, tr_y, cfg, device=DEVICE, seed=0)
        acc = R.accuracy(model, xte, te_y, device=DEVICE)
        results[name] = acc
        print(f"  {name:22s} test accuracy = {acc*100:.1f}%", flush=True)

    make_figure(results)

    with open(os.path.join(FIG, "ncars_recognition.json"), "w") as f:
        json.dump({"test_accuracy": results, "noise_rate_hz": NOISE_HZ,
                   "grid": [H, W], "n_train": len(tr_streams), "n_test": len(te_streams),
                   "protocol": "matched per-clip keep-count (= STCD's)"}, f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'ncars_recognition.pdf'))}")


if __name__ == "__main__":
    main()
