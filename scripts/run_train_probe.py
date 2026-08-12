"""Feasibility probe: can an end-to-end TRAINED STCD beat EDnCNN (not just tie)?

Trains the front-end (W, tau, theta) by surrogate gradients on a couple of
DVSNOISE20 recordings, then evaluates on HELD-OUT recordings (proxy AUC), vs the
untrained STCD and the pretrained EDnCNN (cached per-recording AUCs). Clean
cross-recording split, so a lift is not overfitting.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import dvsnoise20 as dv
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.train import train_frontend, TrainConfig
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
    w = Events(xs=w.xs, ys=w.ys, ts=w.ts, ps=w.ps, labels=sig, H=w.H, W=w.W)
    notedge = (w.xs >= NEIGH) & (w.xs < w.W - NEIGH) & (w.ys >= NEIGH) & (w.ys < w.H - NEIGH)
    late = w.ts >= (best + 0.2)
    cand = np.where(val & notedge & late)[0]
    rng = np.random.default_rng(0)
    pos, neg = cand[sig[cand]], cand[~sig[cand]]
    m = min(len(pos), len(neg), PER_CLASS)
    q = np.sort(np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)])) if m >= 50 else None
    return w, val, q


def train_window(path, dur=0.4):
    """Short valid-labelled window for training (keeps the dense tensor small)."""
    ev, fts, aps, _ = dv.load_full(path)
    t0 = ev.ts.min(); best, bn = t0, -1
    for a in np.arange(t0, ev.ts.max() - dur, dur):
        n = int(((ev.ts >= a) & (ev.ts < a + dur)).sum())
        if n > bn: best, bn = a, n
    w = ev.select((ev.ts >= best) & (ev.ts < best + dur)).time_sorted()
    sig, val = dv.aps_motion_labels(w, fts, aps)
    w = w.select(val)
    sig = sig[val]
    return Events(xs=w.xs, ys=w.ys, ts=w.ts - w.ts.min(), ps=w.ps, labels=sig, H=w.H, W=w.W)


def main():
    torch.manual_seed(0)
    cache = {r["_scene"]: r for r in json.load(open(CACHE))["records"]} if os.path.isfile(CACHE) else {}
    edn = json.load(open(CACHE))["records"] if os.path.isfile(CACHE) else []
    paths = sorted(glob.glob(os.path.join(ROOT, "*.mat")))
    names = [os.path.basename(p)[:-4] for p in paths]
    scenes = [n.split("-")[0] for n in names]
    # train on 2 recordings from 2 scenes; eval on the rest (held-out)
    train_idx = [i for i, s in enumerate(scenes) if s in ("conference",)][:1] + \
                [i for i, s in enumerate(scenes) if s in ("stairs",)][:1]
    eval_idx = [i for i in range(len(paths)) if i not in train_idx]
    print(f"train on: {[names[i] for i in train_idx]}")

    cfg = FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3)
    tev = [train_window(paths[i]) for i in train_idx]
    fe_u = SpikingFrontEnd(cfg)               # untrained baseline (hand-tuned)
    fe_t = SpikingFrontEnd(cfg)               # to be trained
    for k, ev in enumerate(tev):
        print(f"  training on window {k} ({len(ev)} ev)...", flush=True)
        fe_t, _ = train_frontend(ev, fe=fe_t, tcfg=TrainConfig(epochs=120, lr=0.03, verbose_every=40))

    rows = []
    edn_by_name = {r["_scene"]: r for r in edn}  # _scene holds the scene name
    # build per-recording edncnn auc lookup by NAME via re-eval order: edncnn_real stored per-record but keyed by scene; use list order
    edn_list = edn
    for j, i in enumerate(eval_idx):
        w, val, q = prep(paths[i])
        if q is None: continue
        lab = w.labels[q]
        au = metrics.roc(fe_u.score_events(w)[q], lab)["auc"]
        at = metrics.roc(fe_t.score_events(w)[q], lab)["auc"]
        ae = edn_list[i]["EDnCNN (real)"] if i < len(edn_list) else float("nan")
        rows.append((names[i], au, at, ae))
        print(f"  {names[i]:34s} untrained={au:.3f}  TRAINED={at:.3f}  EDnCNN={ae:.3f}", flush=True)

    U = np.array([r[1] for r in rows]); T = np.array([r[2] for r in rows]); E = np.array([r[3] for r in rows])
    from scipy import stats
    print(f"\n=== held-out mean AUC (n={len(rows)}) ===")
    print(f"  untrained STCD : {U.mean():.3f}")
    print(f"  TRAINED STCD   : {T.mean():.3f}")
    print(f"  EDnCNN         : {np.nanmean(E):.3f}")
    d = T - E
    t, p = stats.ttest_rel(T, E, nan_policy="omit")
    print(f"\nTRAINED STCD vs EDnCNN: dAUC={np.nanmean(d):+.3f}  paired p={p:.3f}  "
          f"({'TRAINED BEATS EDnCNN' if np.nanmean(d) > 0 and p < 0.05 else 'not significant / not beating'})")
    print(f"train vs untrained dAUC = {(T-U).mean():+.3f}")


if __name__ == "__main__":
    main()
