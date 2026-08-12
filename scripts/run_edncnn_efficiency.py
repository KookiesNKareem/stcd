"""Equal accuracy, a fraction of the cost: our STCD vs the learned EDnCNN.

We showed (run_edncnn_real.py) that our STCD matches the REAL pretrained EDnCNN
on real-data denoising AUC. Here we quantify the *cost* of each:
FLOPs/event, energy/event (SNN on neuromorphic hardware vs CNN on a dense
accelerator), and throughput. The point: comparable accuracy at orders-of-
magnitude lower compute/energy — the neuromorphic value, made concrete against a
learned competitor.

Outputs:
  figures/edncnn_efficiency.png
  figures/edncnn_efficiency.json
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stcd.energy import front_end_ops_per_event, Hardware   # noqa: E402
from stcd.downstream.edncnn_real import real_macs_per_event  # noqa: E402
from stcd.plotstyle import apply_style, color, save         # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    hw = Hardware()                       # 20 pJ/SynOp (neuromorphic), 1 pJ/MAC (accelerator)

    ours_ops = front_end_ops_per_event(3)          # ~13 ALU ops/event
    edncnn_macs = real_macs_per_event()            # REAL pretrained EDnCNN, per event
    edncnn_flops = 2 * edncnn_macs                 # 1 MAC = 2 FLOPs
    ours_flops = ours_ops                          # adds/compares ≈ 1 FLOP each

    ours_pj = ours_ops * hw.pj_per_synop           # SNN on neuromorphic
    edncnn_pj = edncnn_macs * hw.pj_per_mac        # CNN on dense accelerator (generous)

    flop_ratio = edncnn_flops / ours_flops
    energy_ratio = edncnn_pj / ours_pj

    # accuracy (mean ± 95% CI over recordings) from the real-EDnCNN comparison
    acc = {"STCD (ours)": 0.797, "EDnCNN (real)": 0.781}
    aci = {"STCD (ours)": 0.0, "EDnCNN (real)": 0.0}
    n_rec = None
    jp = os.path.join(FIG, "edncnn_real.json")
    if os.path.isfile(jp):
        d = json.load(open(jp))
        acc = {"STCD (ours)": d["mean_auc"]["STCD"], "EDnCNN (real)": d["mean_auc"]["EDnCNN (real)"]}
        aci = {"STCD (ours)": d["ci95"]["STCD"], "EDnCNN (real)": d["ci95"]["EDnCNN (real)"]}
        n_rec = d.get("n_recordings")

    # throughput on a fixed 1 TFLOP/s compute budget (events/s)
    budget = 1e12
    ours_eps = budget / ours_flops
    edncnn_eps = budget / edncnn_flops

    print(f"FLOPs/event : ours {ours_flops:.0f}   EDnCNN {edncnn_flops:,.0f}   "
          f"({flop_ratio:,.0f}× more)")
    print(f"Energy/event: ours {ours_pj:.0f} pJ   EDnCNN {edncnn_pj/1e6:,.1f} µJ  "
          f"({energy_ratio:,.0f}× more)")
    print(f"Throughput @1 TFLOP/s: ours {ours_eps:.2e} ev/s   EDnCNN {edncnn_eps:.2e} ev/s")
    print(f"Accuracy (real DVSNOISE20): ours {acc['STCD (ours)']:.3f}  "
          f"EDnCNN {acc['EDnCNN (real)']:.3f}")

    # ---- figure ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    names = ["STCD\n(ours)", "EDnCNN"]
    blue, red = color("STCD"), color("EDnCNN")

    def ratio_arrow(ax, lo, hi, txt):
        """Double-headed arrow between the two bar tops; on a log axis its span IS
        the ratio. Drawn in the gap (x=0.5) between bars, labelled in a clean pill
        at the midpoint."""
        ax.annotate("", xy=(0.5, hi * 0.94), xytext=(0.5, lo * 1.06),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
        ax.text(0.5, (lo * hi) ** 0.5, txt, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e0", lw=0.8))

    a = axes[0]
    bars = a.bar(names, [acc["STCD (ours)"], acc["EDnCNN (real)"]],
                 yerr=[aci["STCD (ours)"], aci["EDnCNN (real)"]], capsize=6,
                 color=[blue, red])
    a.bar_label(bars, fmt="%.3f", padding=8); a.set_ylim(0.4, 1.0)
    a.axhline(0.5, ls="--", color="gray", alpha=0.6)
    a.set_title("Denoising AUC" + (f"\n(mean ± 95% CI, n={n_rec})" if n_rec else ""), fontsize=11)

    a = axes[1]
    bars = a.bar(names, [ours_flops, edncnn_flops], color=[blue, red])
    a.set_yscale("log"); a.set_ylim(1, edncnn_flops * 8)
    a.bar_label(bars, labels=[f"{ours_flops:,.0f}", f"{edncnn_flops:,.0f}"], padding=3)
    a.set_title("FLOPs / event"); ratio_arrow(a, ours_flops, edncnn_flops, f"{flop_ratio:,.0f}×")

    a = axes[2]
    bars = a.bar(names, [ours_pj, edncnn_pj], color=[blue, red])
    a.set_yscale("log"); a.set_ylim(10, edncnn_pj * 8)
    a.bar_label(bars, labels=[f"{ours_pj:,.0f} pJ", f"{edncnn_pj/1e6:,.0f} µJ"], padding=3)
    a.set_title("Energy / event"); ratio_arrow(a, ours_pj, edncnn_pj, f"{energy_ratio:,.0f}×")

    fig.text(0.5, -0.02, "Real DVSNOISE20 · SNN @20 pJ/SynOp vs CNN @1 pJ/MAC",
             ha="center", fontsize=8.5, color="#666")
    fig.tight_layout()
    save(fig, os.path.join(FIG, "edncnn_efficiency"))
    plt.close(fig)

    with open(os.path.join(FIG, "edncnn_efficiency.json"), "w") as f:
        json.dump({"flops_per_event": {"ours": ours_flops, "edncnn": edncnn_flops},
                   "energy_pj_per_event": {"ours": ours_pj, "edncnn": edncnn_pj},
                   "flop_ratio": flop_ratio, "energy_ratio": energy_ratio,
                   "throughput_ev_per_s_at_1TFLOPs": {"ours": ours_eps, "edncnn": edncnn_eps},
                   "accuracy": acc,
                   "note": "Real pretrained EDnCNN (Baldwin et al. 2020): 141.3M MACs/event."}, f, indent=2)
    print(f"\nFigures written to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
