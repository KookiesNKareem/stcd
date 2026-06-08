"""Unsupervised STDP learning of the denoising STCD (the novel contribution).

Regime: real-edge signal corrupted by **hot pixels** — the case where a temporal /
firing-rate view fails (a hot pixel fires constantly, so a 'blind' centre-only
filter keeps it) and only *spatial* neighbour support separates signal from noise.

We start the STCD blind (a centre-only receptive field) and learn its spatial
kernel with **Spike-Timing-Dependent Plasticity** — a local, biologically-plausible
rule using **no labels**. STDP discovers that genuine signal has neighbour support
and grows the receptive field, recovering denoising performance to match the
hand-tuned / supervised spatial filter.

Outputs:
  figures/stdp_learning.png   learning curve + learned-kernel evolution
  figures/stdp.json
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

from stcd import metrics                                      # noqa: E402
from stcd.synth import SynthConfig, generate                 # noqa: E402
from stcd.stdp import STDPDenoiser, STDPConfig                # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig     # noqa: E402
from stcd.train import train_frontend, TrainConfig           # noqa: E402
from stcd.plotstyle import apply_style, color, OURS           # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    cfg = dict(H=140, W=180, duration=0.3, scene="bars", num_objects=4,
               noise_rate_hz=1.0, n_hot_pixels=60, hot_pixel_rate_hz=500.0)
    train_ev = generate(SynthConfig(seed=1, **cfg))      # UNLABELLED for STDP
    eval_ev = generate(SynthConfig(seed=777, **cfg))     # held-out, labelled
    print(f"hot-pixel regime: {len(eval_ev)} events, signal={eval_ev.labels.mean()*100:.0f}%")

    # --- unsupervised STDP from a blind (centre-only) init --------------------
    stdp = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3, eta=0.04, epochs=50),
                        init="delta")
    auc_blind = stdp._auc(eval_ev)
    blind_kernel = stdp.kernel().copy()              # true blind init (centre-only)
    hist = stdp.train_unsupervised(train_ev, eval_ev=eval_ev)
    auc_stdp = hist["auc"][-1]
    # prepend the pre-training (blind) point so the curve shows the jump from blind
    curve_x = [0] + [e + 1 for e in hist["epoch"]]
    curve_auc = [auc_blind] + hist["auc"]

    # --- baselines on the same held-out stream -------------------------------
    box = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3), init="uniform")
    auc_box = box._auc(eval_ev)
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    sup_fe, _ = train_frontend(train_ev, fe=fe, tcfg=TrainConfig(epochs=120, lr=0.05, verbose_every=0))
    auc_sup = metrics.roc(sup_fe.score_events(eval_ev), eval_ev.labels)["auc"]

    print(f"blind centre-only init : AUC={auc_blind:.3f}")
    print(f"STDP (unsupervised)    : AUC={auc_stdp:.3f}   (NO labels used)")
    print(f"hand-tuned spatial box : AUC={auc_box:.3f}")
    print(f"supervised (labels)    : AUC={auc_sup:.3f}")

    # --- figure --------------------------------------------------------------
    cblue, cbox, csup, cred = color("STCD"), color("BAF"), "#6b46c1", "#e53e3e"
    fig = plt.figure(figsize=(12, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1, 1], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(curve_x, curve_auc, "-o", ms=3.5, color=cblue,
            label=f"{OURS} via STDP (no labels)")
    ax.scatter([0], [auc_blind], color=cred, zorder=6, s=55, ec="white", lw=1.0)
    ax.annotate("blind\n(centre-only)", (0, auc_blind), textcoords="offset points",
                xytext=(12, 2), fontsize=8.5, color=cred, va="center")
    ax.axhline(auc_sup, ls="-.", color=csup, lw=1.6, label=f"supervised, w/ labels: {auc_sup:.3f}")
    ax.axhline(auc_box, ls="--", color=cbox, lw=1.6, label=f"hand-tuned box: {auc_box:.3f}")
    ax.set_xlabel("STDP epoch"); ax.set_ylabel("held-out denoising AUC")
    ax.set_ylim(min(0.90, auc_blind - 0.02), 1.005)
    ax.set_title("Held-out AUC vs epoch")
    ax.legend(loc="lower right", fontsize=8.2)

    # receptive-field evolution (ON channel). Each panel auto-scales to its own
    # weight range so the learned field's *structure* is visible: the blind init
    # is a single centre point; STDP grows a distributed spatial receptive field.
    snaps = [(blind_kernel, "epoch 0 · blind init"),
             (hist["kernel"][-1], f"epoch {len(hist['kernel'])} · learned field")]
    for col, (kern, lab) in enumerate(snaps):
        axk = fig.add_subplot(gs[0, 1 + col])
        im = axk.imshow(kern[1], cmap="viridis")          # per-kernel auto-scale
        axk.set_title(lab, fontsize=9.5); axk.set_xticks([]); axk.set_yticks([])
        fig.colorbar(im, ax=axk, fraction=0.046, pad=0.04)
    fig.suptitle(f"Unsupervised STDP learning of the {OURS} spatial kernel (no labels)", y=1.0)
    fig.savefig(os.path.join(FIG, "stdp_learning.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "stdp.json"), "w") as f:
        json.dump({"auc": {"blind_center_only": auc_blind, "stdp_unsupervised": auc_stdp,
                           "hand_tuned_box": auc_box, "supervised": auc_sup},
                   "auc_curve": hist["auc"], "n_events": len(eval_ev)}, f, indent=2)
    print(f"\nFigures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
