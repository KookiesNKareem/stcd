"""Headroom probe: can a still-cheap STCD config beat EDnCNN on per-event AUC?
Sweeps receptive-field size + a 2-scale variant (no training), evaluates on the
DVSNOISE20 proxy AUC, and compares to the cached pretrained-EDnCNN per-recording
AUCs. Tells us whether *capacity* (not training) opens a path to superiority.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import dvsnoise20 as dv
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.events import Events
from stcd.downstream.edncnn_real import NEIGH

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
CACHE = os.path.join(os.path.dirname(__file__), "..", "figures", "data", "edncnn_real.json")
WIN, PER_CLASS = 1.0, 400


def prep(path):
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
    if m < 50: return None
    q = np.sort(np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)]))
    return Events(xs=w.xs, ys=w.ys, ts=w.ts, ps=w.ps, labels=sig, H=w.H, W=w.W), q, sig[q]


def main():
    edn = json.load(open(CACHE))["records"]
    paths = sorted(glob.glob(os.path.join(ROOT, "*.mat")))
    names = [os.path.basename(p)[:-4] for p in paths]
    fes = {
        "k3 (paper)": SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3)),
        "k5":         SpikingFrontEnd(FrontEndConfig(neighbor_k=5, pool=1, tau=8e-3, theta=1.5, dt=5e-3)),
        "k7":         SpikingFrontEnd(FrontEndConfig(neighbor_k=7, pool=1, tau=8e-3, theta=1.5, dt=5e-3)),
    }
    # 2-scale: fast/tight (k3, short tau) + slow/wide (k5, long tau), summed (z-scored)
    s_fast = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=4e-3, theta=1.5, dt=5e-3))
    s_slow = SpikingFrontEnd(FrontEndConfig(neighbor_k=5, pool=1, tau=16e-3, theta=1.5, dt=5e-3))

    cols = list(fes) + ["2-scale", "EDnCNN"]
    acc = {c: [] for c in cols}
    for i, p in enumerate(paths):
        r = prep(p)
        if r is None: continue
        w, q, lab = r
        row = {}
        for name, fe in fes.items():
            row[name] = metrics.roc(fe.score_events(w)[q], lab)["auc"]
        def z(s):
            s = s.astype(float); return (s - s.mean()) / (s.std() + 1e-9)
        sc2 = z(s_fast.score_events(w)) + z(s_slow.score_events(w))
        row["2-scale"] = metrics.roc(sc2[q], lab)["auc"]
        row["EDnCNN"] = edn[i]["EDnCNN (real)"]
        for c in cols: acc[c].append(row[c])
        print("  " + f"{names[i]:32s} " + "  ".join(f"{c}={row[c]:.3f}" for c in cols), flush=True)

    from scipy import stats
    print(f"\n=== mean AUC over {len(acc['EDnCNN'])} recordings ===")
    E = np.array(acc["EDnCNN"])
    for c in list(fes) + ["2-scale"]:
        A = np.array(acc[c]); d = A - E
        t, pp = stats.ttest_rel(A, E)
        flag = "BEATS EDnCNN (p<.05)" if d.mean() > 0 and pp < 0.05 else ("ahead" if d.mean() > 0 else "behind")
        print(f"  {c:12s} {A.mean():.3f}   vs EDnCNN dAUC={d.mean():+.3f} p={pp:.3f}  [{flag}]")
    print(f"  {'EDnCNN':12s} {E.mean():.3f}")


if __name__ == "__main__":
    main()
