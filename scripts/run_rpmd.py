"""Field-standard RPMD evaluation (Baldwin et al., CVPR 2020) on the DVSNOISE20
recordings for which the EPM volume is available locally (data/dvsnoise20/5_epm).

RPMD (Relative Plausibility Measure of Denoising) scores a denoiser against the
EDnCNN Event Probability Mask — the field-standard *labelled* metric. Lower is
better; 0 = optimal MAP decision. We report the threshold-free minimum RPMD per
method (analogous to AUC), plus the raw (keep-all) RPMD as the no-denoising
reference. This validates whether STCD is competitive on the standard metric,
not only on our APS-motion proxy AUC.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.datasets import dvsnoise20 as dv                       # noqa: E402
from stcd import rpmd as R                                       # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig        # noqa: E402
from stcd.baselines import baf_scores, knoise_scores, time_surface_scores  # noqa: E402

MAT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
EPM = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "5_epm")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
WIN = float(os.environ.get("RPMD_WIN", "2.0"))   # active window scored (s)


def exposure_windows(events_path):
    d = sio.loadmat(events_path, squeeze_me=True, struct_as_record=False)
    pol = d["aedat"].data.polarity
    t0 = np.asarray(pol.timeStamp, dtype=np.float64).min()
    fr = d["aedat"].data.frame
    es = (np.asarray(fr.expStart, dtype=np.float64) - t0) / 1e6
    ee = (np.asarray(fr.expEnd, dtype=np.float64) - t0) / 1e6
    return es, ee


def main():
    os.makedirs(FIG, exist_ok=True)
    pairs = []
    for ep in sorted(glob.glob(os.path.join(EPM, "*_epm_array.mat"))):
        name = os.path.basename(ep).replace("_epm_array.mat", "")
        mp = os.path.join(MAT, name + ".mat")
        if os.path.isfile(mp):
            pairs.append((name, mp, ep))
    if not pairs:
        print("No (events, EPM) pairs found locally."); return
    print(f"recordings with EPM available: {[p[0] for p in pairs]}")

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    recs = []
    for name, mp, ep in pairs:
        ev, frame_ts, aps, epm = dv.load_full(mp, ep)
        es, ee = exposure_windows(mp)
        # score on the most-active WIN window (keeps STCD's dense tensor tractable)
        t0 = ev.ts.min(); best, bn = t0, -1
        for a in np.arange(t0, ev.ts.max() - WIN, WIN):
            n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
            if n > bn: best, bn = a, n
        w = ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()
        prob, during, aps_good, _ = R.epm_event_prob(
            w.xs, w.ys, w.ts, w.ps, frame_ts, es, ee, aps, epm)
        nslot = int(((during > 0) & aps_good).sum())
        print(f"\n{name}: {len(w)} events in {WIN}s window, {nslot} scored slots")

        scores = {"STCD": fe.score_events(w),
                  "time-surface": time_surface_scores(w, 5e-3),
                  "BAF": baf_scores(w, 2e-3),
                  "KNoise": knoise_scores(w, 2e-3)}
        row = {"_scene": name}
        raw, _ = R.rpmd_keep_all(prob, during, aps_good, w.xs, w.ys, w.ps)
        row["raw (no denoising)"] = raw
        for m, sc in scores.items():
            out = R.rpmd(sc, prob, during, aps_good, w.xs, w.ys, w.ps)
            row[m] = out["rpmd_min"]
            print(f"  RPMD {m:14s} = {out['rpmd_min']:.3f}  (keep {out['keep_frac']*100:.0f}%)")
        print(f"  RPMD {'raw':14s} = {raw:.3f}")
        recs.append(row)

    methods = ["STCD", "time-surface", "BAF", "KNoise", "raw (no denoising)"]
    print(f"\n=== mean RPMD over {len(recs)} recording(s) (lower is better) ===")
    for m in methods:
        vals = [r[m] for r in recs if m in r]
        print(f"  {m:20s} {np.mean(vals):.3f}")
    with open(os.path.join(FIG, "data", "rpmd.json"), "w") as f:
        json.dump({"records": recs, "win_s": WIN}, f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'data', 'rpmd.json'))}")


if __name__ == "__main__":
    main()
