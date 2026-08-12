"""8-bit precision study: does STCD's denoising survive the 8-bit/pixel packing
needed to fit full DAVIS346 on the iCE40 UP5K, and how should the 8 bits split
between count and tick?

We mirror the RTL's integer arithmetic exactly (stcd_frontend.v): per-pixel
saturating count, lazy shift-leak ``leak(m,dt)=m>>min(dt,MEMW)``, support = sum of
the 8 neighbours' leaked counts (centre excluded), keep iff support>=THETA. The
support value is the per-event score for ROC-AUC. One tick = the front-end's
5 ms grid (dt=5e-3), so the integer leak (>>1 per tick) matches the float leak
(alpha=exp(-5/8)≈0.54).

Configs compared (count_bits, tick_bits) → bits/pixel:
  (8,8)=16 b/px  current 16-bit design (baseline)
  (4,4)= 8 b/px  NAIVE split (ΔAUC≈-0.013) — shown only for contrast
  (2,6)= 8 b/px  CHOSEN packed design (ΔAUC≈-0.003); the RTL uses 2-count/6-tick
                 because tick precision dominates and count precision is near-free
plus (3,5)/(5,3)/(4,8)/(8,4) to map the count-vs-tick trade-off, and the float
front-end as the upstream reference.

Outue: per-recording + mean ROC-AUC for each config (figures/precision_study.json).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics                                       # noqa: E402
from stcd.datasets import dvsnoise20 as dv                     # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig      # noqa: E402
from stcd.downstream.edncnn_real import NEIGH                  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
WIN = float(os.environ.get("PREC_WIN", "1.0"))
TICK_S = 5e-3            # 1 tick = 5 ms (the front-end's dt grid)
PER_CLASS = 400
THETA = 2
INC = 1
# Float front-end is the upstream reference but builds a dense [P,H,W,T] tensor
# per 1 s window (slow); enable with PREC_FLOAT=1. The int16-vs-int8 decision does
# not need it.
CONFIGS = ([("float", None, None)] if os.environ.get("PREC_FLOAT") == "1" else []) + [
    ("int16 (8c,8t)", 8, 8), ("int8 (4c,4t)", 4, 4),
    ("8b 3c,5t", 3, 5), ("8b 2c,6t", 2, 6), ("8b 5c,3t", 5, 3)]


@njit(cache=True)
def int_scores(xs, ys, tks, H, W, memw, tbits, inc):
    """Faithful integer STCD: returns per-event support (the ROC score)."""
    mask = (1 << tbits) - 1
    maxc = (1 << memw) - 1
    cnt = np.zeros((H, W), np.int32)
    tick = np.zeros((H, W), np.int32)
    n = xs.shape[0]
    out = np.zeros(n, np.int32)
    dxs = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
    dys = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
    for i in range(n):
        cx = xs[i]; cy = ys[i]; T = tks[i] & mask
        support = 0
        for k in range(8):
            nx = cx + dxs[k]; ny = cy + dys[k]
            if 0 <= nx < W and 0 <= ny < H:
                dt = (T - tick[ny, nx]) & mask
                s = dt if dt < memw else memw
                support += cnt[ny, nx] >> s
        out[i] = support
        # own read-modify-write (centre pixel updates its own count/tick)
        dt0 = (T - tick[cy, cx]) & mask
        s0 = dt0 if dt0 < memw else memw
        newc = (cnt[cy, cx] >> s0) + inc
        cnt[cy, cx] = maxc if newc > maxc else newc
        tick[cy, cx] = T
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    paths = sorted(glob.glob(os.path.join(ROOT, "*.mat")))
    recs = []
    for path in paths:
        name = os.path.basename(path)[:-4]
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
            print(f"  {name}: too few samples; skip"); continue
        q = np.sort(np.concatenate([rng.choice(pos, m, replace=False),
                                    rng.choice(neg, m, replace=False)]))
        lab = sig[q]
        xs = w.xs.astype(np.int64); ys = w.ys.astype(np.int64)
        tks = np.floor((w.ts - best) / TICK_S).astype(np.int64)

        row = {"_scene": name.split("-")[0]}
        for label, memw, tbits in CONFIGS:
            if memw is None:
                sc = fe.score_events(w)[q]
            else:
                sc = int_scores(xs, ys, tks, int(w.H), int(w.W), memw, tbits, INC)[q]
            row[label] = float(metrics.roc(sc, lab)["auc"])
        recs.append(row)
        print("  " + f"{name:34s} " + "  ".join(
            f"{lab}={row[lab]:.3f}" for lab, _, _ in CONFIGS), flush=True)

    labels = [c[0] for c in CONFIGS]
    mean = {lab: float(np.mean([r[lab] for r in recs])) for lab in labels}
    ci = {lab: float(1.96 * np.std([r[lab] for r in recs], ddof=1) / np.sqrt(len(recs)))
          for lab in labels}
    print(f"\n=== mean ROC-AUC over {len(recs)} recordings (WIN={WIN}s) ===")
    for lab in labels:
        print(f"  {lab:16s} {mean[lab]:.3f} ± {ci[lab]:.3f}")
    d16 = mean["int16 (8c,8t)"]
    chosen = mean.get("8b 2c,6t")     # the design the RTL/paper uses
    if chosen is not None:
        print(f"\n16-bit -> 8-bit CHOSEN packing (2c+6t) ΔAUC = {chosen - d16:+.3f}")
    print(f"(naive 4c+4t split for contrast: {mean['int8 (4c,4t)'] - d16:+.3f})")
    with open(os.path.join(FIG, "precision_study.json"), "w") as f:
        json.dump({"records": recs, "mean_auc": mean, "ci95": ci,
                   "n_recordings": len(recs), "win_s": WIN, "tick_s": TICK_S}, f, indent=2)
    print(f"wrote {os.path.abspath(os.path.join(FIG, 'precision_study.json'))}")


if __name__ == "__main__":
    main()
