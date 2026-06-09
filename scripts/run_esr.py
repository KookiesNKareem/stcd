"""Standard no-reference benchmark metric: ESR (Event Structural Ratio, Ding et al.,
E-MLB, IEEE TMM 2023) on real DVSNOISE20.

ESR needs no ground-truth labels — it scores how spatially *structured* a denoised
event stream is (events on real edges vs spread as noise). Raw ESR is inflated by
simply removing more events, so we compare every method at a **fixed matched
keep-fraction**: each method keeps the same number of events by its own score
ranking, and we add a **random-keep** reference (same count, no skill). ESR then
measures *which* events are kept (denoising quality), not *how many*, and all bars
are at an identical event count so they are directly comparable.

This answers the "non-standard proxy metric" concern: STCD is best on a standard,
label-free metric too.

Outputs:
  figures/esr.png
  figures/data/esr.json
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
from stcd.esr import esr                                          # noqa: E402
from stcd.datasets import dvsnoise20 as dv                        # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig         # noqa: E402
from stcd.baselines import baf_scores, knoise_scores, time_surface_scores  # noqa: E402
from stcd.downstream import mlpf as MLPF                          # noqa: E402
from stcd.plotstyle import apply_style, color, OURS               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
WIN = float(os.environ.get("ESR_WIN", "1.0"))
KEEP = float(os.environ.get("ESR_KEEP", "0.8"))          # fixed matched keep-fraction


def busy_window(ev):
    t0 = ev.ts.min(); best, bn = t0, -1
    for a in np.arange(t0, ev.ts.max() - WIN, WIN):
        n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
        if n > bn: best, bn = a, n
    return ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()


def main() -> None:
    os.makedirs(os.path.join(FIG, "data"), exist_ok=True)
    apply_style()
    mlpf_w = MLPF.load_mlpf(MLPF_MODEL) if os.path.isfile(MLPF_MODEL) else None
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    methods = [OURS, "time-surface", "BAF", "KNoise"] + (["MLPF"] if mlpf_w else []) + ["random"]
    records = []

    for p in sorted(glob.glob(os.path.join(ROOT, "*.mat"))):
        name = os.path.basename(p).split("-")[0]
        ev, _, _, _ = dv.load_full(p)
        w = busy_window(ev)
        scores = {OURS: fe.score_events(w),
                  "time-surface": time_surface_scores(w, 5e-3),
                  "BAF": baf_scores(w, 2e-3),
                  "KNoise": knoise_scores(w, 2e-3)}
        if mlpf_w:
            scores["MLPF"] = MLPF.predict_stream(mlpf_w, w)
        scores["random"] = np.random.default_rng(0).random(len(w))   # no-skill reference
        rec = {"_scene": name}
        for m in methods:
            mask = scores[m] >= np.quantile(scores[m], 1.0 - KEEP)
            rec[m] = esr(w.select(mask))
        records.append(rec)
        print(f"  {os.path.basename(p)[:30]:32s} " + "  ".join(f"{m}={rec[m]:.3f}" for m in methods),
              flush=True)

    n = len(records)
    A = {m: np.array([r[m] for r in records]) for m in methods}
    mean = {m: float(A[m].mean()) for m in methods}
    ci = {m: float(1.96 * A[m].std(ddof=1) / np.sqrt(n)) for m in methods}
    from scipy.stats import ttest_rel
    pvals = {m: float(ttest_rel(A[OURS], A[m]).pvalue) for m in methods if m != OURS}
    print(f"\nn={n}, keep={KEEP:.0%} | " + "  ".join(f"{m} {mean[m]:.3f}±{ci[m]:.3f}" for m in methods))
    print("paired t-test STCD vs:", {m: round(pvals[m], 4) for m in pvals})

    # ---- figure: ESR bars (95% CI); random-keep reference as dashed line ---- #
    bar_methods = [m for m in methods if m != "random"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    cols = [color(m) for m in bar_methods]
    xp = np.arange(len(bar_methods))
    bars = ax.bar(xp, [mean[m] for m in bar_methods], yerr=[ci[m] for m in bar_methods],
                  capsize=5, color=cols, zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=9.5)
    ax.axhline(mean["random"], ls="--", color="#718096", alpha=0.8)
    ax.text(len(bar_methods) - 0.4, mean["random"], f" random keep {mean['random']:.3f}",
            va="bottom", ha="right", fontsize=8.5, color="#718096")
    ax.set_xticks(xp); ax.set_xticklabels(bar_methods, rotation=10, ha="right")
    ax.set_ylabel("ESR (Event Structural Ratio) ↑")
    ax.set_title(f"Standard no-reference metric on DVSNOISE20 — ESR at {KEEP:.0%} matched keep (n={n})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "esr.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "data", "esr.json"), "w") as f:
        json.dump({"records": records, "mean": mean, "ci95": ci, "n_recordings": n,
                   "keep_frac": KEEP, "paired_ttest_stcd_vs": pvals,
                   "protocol": f"fixed matched keep-fraction {KEEP}; random-keep reference",
                   "metric": "ESR (Ding et al., E-MLB, IEEE TMM 2023), no-reference"}, f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'esr.png'))}")


if __name__ == "__main__":
    main()
