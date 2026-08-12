"""Two evaluation axes beyond raw denoising quality.

(1) Sparsity ↔ retention: how much true signal survives as we drop more events.
    A filter that drops everything is trivially "sparse"; the useful question is
    signal retained at a given event-reduction. Front-end vs BAF.

(2) Noise-vs-latency (proposal §7): the Stage-3 leak τ sets both temporal
    coincidence quality and the confirmation latency (≈ a few τ). We sweep τ and
    plot denoising AUC / noise-removal against τ, with the added-latency axis.

Outputs:
  figures/sparsity_retention.png
  figures/latency_tradeoff.png
  figures/tradeoffs.json
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

from stcd import metrics, analysis                            # noqa: E402
from stcd.baselines import baf_min_dt, baf_filter            # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig     # noqa: E402
from stcd.synth import SynthConfig, generate                 # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
LATENCY_TAUS = 3.0   # confirmation latency ≈ this many τ


def sparsity_retention(ev):
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    red_fe, sr_fe = analysis.retention_curve_from_scores(ev, fe.score_events(ev))
    # BAF: sweep window -> (reduction, signal_retain)
    bdt = baf_min_dt(ev)
    red_baf, sr_baf = [], []
    for w in np.logspace(-4, -1.0, 50):
        kept, _ = baf_filter(ev, window=float(w), min_dt=bdt)
        m = metrics.evaluate_filter(ev.labels, kept)
        red_baf.append(m.reduction); sr_baf.append(m.signal_retain)
    order = np.argsort(red_baf)
    return (red_fe, sr_fe), (np.array(red_baf)[order], np.array(sr_baf)[order])


def latency_sweep(ev):
    taus = np.array([1, 2, 4, 8, 16, 32, 64]) * 1e-3
    aucs, nrs = [], []
    for tau in taus:
        fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=float(tau),
                                            theta=1.5, dt=5e-3))
        scores = fe.score_events(ev)
        aucs.append(metrics.roc(scores, ev.labels)["auc"])
        # NR at a fixed high signal-retain — fair across τ (no operating-point drift)
        nrs.append(analysis.nr_at_target_sr(ev, scores, 0.99).noise_removal)
    return taus, np.array(aucs), np.array(nrs)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    ev = generate(SynthConfig(H=160, W=200, duration=0.3, noise_rate_hz=3.0,
                              scene="bars", num_objects=4, seed=7))

    # ---- (1) sparsity-retention ------------------------------------------- #
    (red_fe, sr_fe), (red_baf, sr_baf) = sparsity_retention(ev)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(red_fe * 100, sr_fe * 100, "-o", ms=3, label="STCD")
    ax.plot(red_baf * 100, sr_baf * 100, "-s", ms=3, label="BAF")
    ax.set_xlabel("Events removed (%)  →  sparser")
    ax.set_ylabel("Signal retained (%)")
    ax.set_title("Sparsity ↔ retention tradeoff")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sparsity_retention.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- (2) latency tradeoff --------------------------------------------- #
    taus, aucs, nrs = latency_sweep(ev)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(taus * 1e3, aucs, "-o", color="#1f77b4", label="AUC")
    ax.plot(taus * 1e3, nrs, "-s", color="#2ca02c", label="Noise Removal @ SR≈0.99")
    ax.set_xscale("log")
    ax.set_xlabel("leak τ (ms)")
    ax.set_ylabel("denoising quality")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower center")
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(*(np.array(ax.get_xlim()) * LATENCY_TAUS))
    ax2.set_xlabel(f"added confirmation latency ≈ {LATENCY_TAUS:.0f}·τ (ms)")
    ax.set_title("Noise-vs-latency tradeoff (set by leak τ)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "latency_tradeoff.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    best = int(np.argmax(aucs))
    print(f"sparsity-retention & latency curves written.")
    print(f"  best AUC {aucs[best]:.3f} at τ={taus[best]*1e3:.0f} ms "
          f"(latency ≈ {taus[best]*LATENCY_TAUS*1e3:.0f} ms)")
    with open(os.path.join(FIG, "tradeoffs.json"), "w") as f:
        json.dump({"tau_ms": (taus * 1e3).tolist(), "auc": aucs.tolist(),
                   "noise_removal": nrs.tolist(),
                   "latency_taus": LATENCY_TAUS}, f, indent=2)
    print(f"Figures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
