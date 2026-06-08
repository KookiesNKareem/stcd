"""Real-data downstream recognition: N-Cars (car vs background) under injected noise.

Loads real Prophesee N-Cars event recordings, injects background-activity noise to
create a low-SNR regime, and trains+tests a small spiking CNN on three input
conditions:
  * clean  (real events, no injected noise)        — upper bound
  * noisy  (real events + injected BA noise)        — degraded
  * filtered (noisy passed through the STCD)   — recovered

Shows the STCD recovering classification accuracy lost to noise, on real data.

Outputs:
  figures/ncars_recognition.png
  figures/ncars_recognition.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stcd.events import Events                            # noqa: E402
from stcd.datasets import ncars                           # noqa: E402
from stcd.synth import inject_noise                       # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig  # noqa: E402
from stcd.downstream import recognition as R              # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ncars")
DS = 2                                            # spatial downscale for speed
H, W = ncars.DEFAULT_H // DS, ncars.DEFAULT_W // DS
DUR, DT = 0.1, 1e-2                               # 10 time-bins of 10 ms
NOISE_HZ = 5.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def downscale(ev: Events) -> Events:
    return replace(ev, xs=ev.xs // DS, ys=ev.ys // DS, H=H, W=W)


def add_noise(streams, rate, seed):
    rng = np.random.default_rng(seed)
    return [inject_noise(s, rate, DUR, rng) for s in streams]


def filter_batch(fe, x):
    out = []
    with torch.no_grad():
        for b in range(x.shape[0]):
            out.append(fe.forward(x[b], fe.cfg.dt)[0])
    return torch.stack(out)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(0)
    print(f"device={DEVICE}, grid={H}x{W}, {int(DUR/DT)} time-bins")

    print("Loading N-Cars (real events)...")
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
    print(f"  {len(tr_streams)} train / {len(te_streams)} test  "
          f"(mean events/clip: {np.mean([len(s) for s in streams]):.0f})")
    tr_y, te_y = torch.from_numpy(tr_y).long(), torch.from_numpy(te_y).long()

    tr_noisy = add_noise(tr_streams, NOISE_HZ, seed=100)
    te_noisy = add_noise(te_streams, NOISE_HZ, seed=200)
    print(f"  injected ~{NOISE_HZ} Hz/px noise "
          f"(~{int(NOISE_HZ*H*W*DUR)} events/clip)")

    tr_clean = R.stack_tensors(tr_streams, DT, DUR, H, W)
    te_clean = R.stack_tensors(te_streams, DT, DUR, H, W)
    tr_noisy_t = R.stack_tensors(tr_noisy, DT, DUR, H, W)
    te_noisy_t = R.stack_tensors(te_noisy, DT, DUR, H, W)

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=DT))
    tr_filt = filter_batch(fe, tr_noisy_t)
    te_filt = filter_batch(fe, te_noisy_t)

    cfg = R.RecogConfig(epochs=30, lr=2e-3, batch=32)
    results = {}
    for name, (xtr, xte) in {
        "clean (upper bound)": (tr_clean, te_clean),
        "noisy": (tr_noisy_t, te_noisy_t),
        "STCD filtered": (tr_filt, te_filt),
    }.items():
        model = R.train_classifier(xtr, tr_y, cfg, device=DEVICE, seed=0)
        acc = R.accuracy(model, xte, te_y, device=DEVICE)
        results[name] = acc
        print(f"  {name:22s}  test accuracy = {acc*100:.1f}%")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    bars = ax.bar(names, accs, color=["#2ca02c", "#d62728", "#1f77b4"])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
    ax.axhline(50, ls="--", color="gray", alpha=0.6, label="chance")
    ax.set_title(f"N-Cars recognition (real events) under {NOISE_HZ} Hz/px noise")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ncars_recognition.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "ncars_recognition.json"), "w") as f:
        json.dump({"test_accuracy": results, "noise_rate_hz": NOISE_HZ,
                   "grid": [H, W], "n_train": len(tr_streams),
                   "n_test": len(te_streams)}, f, indent=2)
    print(f"\nFigures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
