"""Fair head-to-head: STCD vs the REAL pretrained EDnCNN (Baldwin et al., CVPR 2020).

Loads the authors' published network (`allData_v8_preTrained.mat`) and runs it on
real DVSNOISE20 events — replacing our EDnCNN-lite proxy. For each scene we take an
active window, label events by APS brightness-change (our decodable real-data
ground truth), draw a balanced sample, and score every method on the SAME events:

  ROC-AUC vs APS-motion labels  +  the objective FLOPs/event each method costs.

The real net is 141.3 M MACs/event (282.6 M FLOPs) — 770x our lite proxy and
~21.7 million x more than STCD's 13 FLOPs/event.

Outputs:
  figures/denoising_auc.png   (all methods; STCD vs learned + classical baselines)
  figures/edncnn_real.json    (data; consumed by conclusion/pareto figures)
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
from stcd import metrics                                       # noqa: E402
from stcd.datasets import dvsnoise20 as dv                     # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig      # noqa: E402
from stcd.baselines import baf_scores, knoise_scores, time_surface_scores  # noqa: E402
from stcd.downstream.edncnn_real import load_real_edncnn, event_features, NEIGH  # noqa: E402
from stcd.downstream import mlpf as MLPF                       # noqa: E402
from stcd.plotstyle import apply_style, color, OURS            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "edncnn", "allData_v8_preTrained.mat")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
WIN = 1.0
PER_CLASS = 400
REAL_EDN_FLOPS = 282_581_248      # measured from the loaded net (141.3M MACs x2)
MLPF_FLOPS = 1980                 # 990 MACs x2 (published 98->10->1 MLPF)
STCD_FLOPS = 13


def scenes():
    want = [s for s in os.environ.get("EDN_SCENES", "").split(",") if s]   # empty = all takes
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*.mat"))):
        name = os.path.basename(p).split("-")[0]
        if not want or name in want:
            out.append((os.path.basename(p)[:-4], name, p))
    return out


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    apply_style()
    if not os.path.isfile(MODEL):
        print("Pretrained EDnCNN model missing:", MODEL); return
    net = load_real_edncnn(MODEL)
    mlpf_w = MLPF.load_mlpf(MLPF_MODEL) if os.path.isfile(MLPF_MODEL) else None
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    methods = [OURS, "EDnCNN (real)"] + (["MLPF"] if mlpf_w else []) + ["time-surface", "BAF", "KNoise"]
    records = []                       # one dict of AUCs per recording

    for label, name, path in scenes():
        ev, fts, aps, _ = dv.load_full(path)
        t0 = ev.ts.min(); best, bn = t0, -1
        for a in np.arange(t0, ev.ts.max() - WIN, WIN):
            n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
            if n > bn: best, bn = a, n
        w = ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()
        sig, val = dv.aps_motion_labels(w, fts, aps)
        notedge = (w.xs >= NEIGH) & (w.xs < w.W - NEIGH) & (w.ys >= NEIGH) & (w.ys < w.H - NEIGH)
        late = w.ts >= (best + 0.2)
        cand = np.where(val & notedge & late)[0]
        rng = np.random.default_rng(0)
        pos, neg = cand[sig[cand]], cand[~sig[cand]]
        m = min(len(pos), len(neg), PER_CLASS)
        if m < 50:
            print(f"  {label}: too few samples; skip"); continue
        q = np.sort(np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)]))
        lab = sig[q]
        sc = {OURS: fe.score_events(w)[q],
              "EDnCNN (real)": net(event_features(w, q)).numpy(),
              "time-surface": time_surface_scores(w, 5e-3)[q],
              "BAF": baf_scores(w, 2e-3)[q],
              "KNoise": knoise_scores(w, 2e-3)[q]}
        if mlpf_w:
            sc["MLPF"] = MLPF.mlpf_scores(w, q, mlpf_w)
        aucs = {k: float(metrics.roc(v, lab)["auc"]) for k, v in sc.items()}
        aucs["_scene"] = name
        records.append(aucs)
        print(f"  {label:32s} " + "  ".join(f"{k}={aucs[k]:.3f}" for k in methods))

    if not records:
        print("No recordings scored."); return
    n = len(records)
    A = {k: np.array([r[k] for r in records]) for k in methods}
    mean = {k: float(A[k].mean()) for k in methods}
    ci = {k: float(1.96 * A[k].std(ddof=1) / np.sqrt(n)) for k in methods}   # 95% CI
    from scipy.stats import wilcoxon, ttest_rel
    def paired(ref):
        d = A[OURS] - A[ref]
        try:
            pw = float(wilcoxon(A[OURS], A[ref]).pvalue)
        except ValueError:
            pw = float("nan")
        return {"delta": float(d.mean()), "wilcoxon_p": pw, "ttest_p": float(ttest_rel(A[OURS], A[ref]).pvalue)}
    pe = paired("EDnCNN (real)")
    diff = A[OURS] - A["EDnCNN (real)"]; p_w, p_t = pe["wilcoxon_p"], pe["ttest_p"]
    pm = paired("MLPF") if "MLPF" in methods else None
    print(f"\nn={n} recordings | STCD {mean[OURS]:.3f}±{ci[OURS]:.3f}  "
          f"EDnCNN {mean['EDnCNN (real)']:.3f}±{ci['EDnCNN (real)']:.3f}  "
          f"(Δ={diff.mean():+.3f}) | paired wilcoxon p={p_w:.3f}, t-test p={p_t:.3f}")
    if pm:
        print(f"             | MLPF {mean['MLPF']:.3f}±{ci['MLPF']:.3f}  "
              f"(Δ={pm['delta']:+.3f}) | paired wilcoxon p={pm['wilcoxon_p']:.3f}, t-test p={pm['ttest_p']:.3f}")

    # ---- figure: AUC bars (95% CI) + significance ------------------------- #
    fig, ax = plt.subplots(figsize=(9, 5.4))
    cols = [color(m.replace(" (real)", "")) for m in methods]
    xpos = np.arange(len(methods))
    bars = ax.bar(xpos, [mean[k] for k in methods], yerr=[ci[k] for k in methods],
                  capsize=5, color=cols, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=8, fontsize=9.5)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([m.replace(" (real)", "") for m in methods], rotation=10, ha="right")
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("ROC-AUC vs APS-motion labels")
    ax.set_title(f"Real-data denoising AUC (n={n})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "denoising_auc.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "edncnn_real.json"), "w") as f:
        flops = {"STCD": STCD_FLOPS, "EDnCNN_real": REAL_EDN_FLOPS}
        if pm:
            flops["MLPF"] = MLPF_FLOPS
        json.dump({"records": records, "mean_auc": mean, "ci95": ci, "n_recordings": n,
                   "paired_stcd_vs_edncnn": pe,
                   "paired_stcd_vs_mlpf": pm,
                   "flops_per_event": flops,
                   "label_source": "APS brightness-change (real-data proxy)",
                   "note": "EDnCNN trained on EPM labels, MLPF on DND21; both evaluated "
                           "on APS-motion labels (cross-label). MLPF = published 98->10->1 weights."},
                  f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'denoising_auc.png'))}")


if __name__ == "__main__":
    main()
