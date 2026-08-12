"""STCD zero-shot on DND21 (Guo & Delbruck) -- the home dataset of MLPF and
SNNF -- under the benchmark's exact-label shot-noise protocol.

Why: reviewers noted (R1-1) that our MLPF comparison is out-of-domain for
MLPF, and (R2-1) that the SNNF accuracy comparison is cross-dataset. Running
STCD zero-shot on DND21 puts STCD next to MLPF's *in-domain* operating regime
(and SNNF's reported AUC 0.89) on their own dataset, with no training and no
parameter changes (k=3, tau=8 ms, theta=1.5 -- the paper's fixed config).

Protocol (per recording): take the most-active CLEAN window (the DND21
recordings are captured to be near-noise-free), label its events as signal,
inject uncorrelated Poisson shot noise at NOISE_HZ Hz/px (exact labels by
construction -- the same mixing idea as DND21's NoiseTesterFilter), then score
every method on the identical stream and report per-event ROC-AUC.

Calibration: MLPF runs with its published 4-bit weights, *in-domain* here
(trained on DND21 sequences with this style of noise). If our pipeline's MLPF
number lands near its published in-domain AUC (~0.87), the protocol is
validated, and STCD's number on the identical events is directly comparable.

Caveat (same as the original protocol): the clean recordings still contain
some real BA noise, which is labelled signal; this biases AUC downward
equally for all methods.

Usage: put DND21 files under data/dnd21/ (aedat2 or DVS text), then
  DND21_NOISE_HZ="1,3,5" python scripts/run_dnd21.py
Outputs figures/data/dnd21.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics                                         # noqa: E402
from stcd.datasets import dnd21 as dd                            # noqa: E402
from stcd.synth import inject_noise                              # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig        # noqa: E402
from stcd.baselines import baf_scores, knoise_scores, time_surface_scores  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dnd21")
MLPF_MODEL = os.path.join(os.path.dirname(__file__), "..", "data", "mlpf",
                          "vendor", "0316_soft_4bit_alpha1_sigmoid.h5")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "data",
                   "dnd21.json")
WIN = float(os.environ.get("DND21_WIN", "2.0"))
RATES = [float(r) for r in os.environ.get("DND21_NOISE_HZ", "1,3,5").split(",")
         if r.strip()]
# Optional: mix REAL recorded noise (DND21 measured-noise recordings) instead
# of / in addition to synthetic injection. Path to a noise-only recording.
NOISE_FILE = os.environ.get("DND21_NOISE_FILE", "")
SEED = 0


def mix_real_noise(sig, noise_ev, rng):
    """Overlay a random same-length window of a noise-only recording onto the
    signal window (labels False), shifting its timestamps into the signal
    span. Exact labels: every event from the noise recording is noise."""
    from stcd.events import Events
    t0, t1 = float(sig.ts.min()), float(sig.ts.max())
    span = t1 - t0
    n0, n1 = float(noise_ev.ts.min()), float(noise_ev.ts.max())
    if n1 - n0 <= span:
        a = n0
    else:
        a = rng.uniform(n0, n1 - span)
    m = (noise_ev.ts >= a) & (noise_ev.ts < a + span)
    nw = noise_ev.select(m)
    noise = Events(xs=nw.xs, ys=nw.ys, ts=nw.ts - a + t0, ps=nw.ps,
                   H=sig.H, W=sig.W,
                   labels=np.zeros(len(nw), dtype=bool))
    return Events.concat(sig, noise).time_sorted(), len(nw) / span / (sig.H * sig.W)


def active_window(ev, win):
    t0, t1 = float(ev.ts.min()), float(ev.ts.max())
    best, bn = t0, -1
    for a in np.arange(t0, max(t0, t1 - win), win):
        n = int(((ev.ts >= a) & (ev.ts < a + win)).sum())
        if n > bn:
            best, bn = a, n
    return best


def main() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "*.aedat"))
                   + glob.glob(os.path.join(ROOT, "*.txt")))
    if not files:
        print(f"No DND21 recordings under {os.path.abspath(ROOT)}; skipping.")
        return
    mlpf_w = None
    if os.path.isfile(MLPF_MODEL):
        from stcd.downstream import mlpf as MLPF
        mlpf_w = MLPF.load_mlpf(MLPF_MODEL)
    else:
        print("MLPF weights missing -- calibration column omitted")

    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3,
                                        theta=1.5, dt=5e-3))
    records = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            ev = dd.load(path)
        except Exception as e:
            print(f"{name}: load failed ({e}); skipping")
            continue
        dur = float(ev.ts.max() - ev.ts.min())
        rate = len(ev) / max(dur, 1e-9) / (ev.H * ev.W)
        print(f"{name}: {len(ev):,} events, {dur:.1f} s, "
              f"{rate:.2f} ev/px/s mean rate")
        a = active_window(ev, WIN)
        w = ev.select((ev.ts >= a) & (ev.ts < a + WIN)).time_sorted()
        from stcd.events import Events
        sig = Events(xs=w.xs, ys=w.ys, ts=w.ts, ps=w.ps,
                     labels=np.ones(len(w), dtype=bool), H=w.H, W=w.W)
        variants = [("synthetic", hz) for hz in RATES]
        noise_ev = None
        if NOISE_FILE:
            noise_ev = dd.load(NOISE_FILE)
            nr = len(noise_ev) / max(float(noise_ev.ts.max()
                                           - noise_ev.ts.min()), 1e-9) \
                / (noise_ev.H * noise_ev.W)
            print(f"  real-noise file: {os.path.basename(NOISE_FILE)} "
                  f"({len(noise_ev):,} ev, {nr:.2f} ev/px/s)")
            variants.append(("real", nr))
        for kind, hz in variants:
            rng = np.random.default_rng(SEED)
            if kind == "synthetic":
                evn = inject_noise(sig, hz, WIN, rng)
            else:
                evn, hz = mix_real_noise(sig, noise_ev, rng)
            lab = evn.labels
            q = np.arange(len(evn))
            sc = {"STCD": fe.score_events(evn),
                  "time-surface": time_surface_scores(evn, 5e-3),
                  "BAF": baf_scores(evn, 2e-3),
                  "KNoise": knoise_scores(evn, 2e-3)}
            if mlpf_w is not None:
                from stcd.downstream import mlpf as MLPF
                sc["MLPF"] = MLPF.mlpf_scores(evn, q, mlpf_w)
            row = {"_file": name, "noise_kind": kind, "noise_hz": float(hz),
                   "n_signal": int(lab.sum()), "n_noise": int((~lab).sum())}
            for m, v in sc.items():
                row[m] = float(metrics.roc(np.asarray(v, dtype=np.float64),
                                           lab)["auc"])
            records.append(row)
            cells = "  ".join(f"{m}={row[m]:.3f}" for m in sc)
            print(f"  {kind:9s} {hz:>5.2f} Hz/px  (+{row['n_noise']:,} noise)"
                  f"  {cells}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"records": records, "win_s": WIN, "seed": SEED,
                   "config": "k=3, tau=8ms, theta=1.5 (paper-fixed, zero-shot)"},
                  f, indent=2)
    print(f"\nwrote {os.path.abspath(OUT)}")

    methods = [m for m in ["STCD", "MLPF", "time-surface", "BAF", "KNoise"]
               if any(m in r for r in records)]
    print("\n=== mean AUC over recordings, per noise variant ===")
    seen = []
    for r in records:
        key = (r["noise_kind"], round(r["noise_hz"], 2))
        if key not in seen:
            seen.append(key)
    for kind, hz in seen:
        rows = [r for r in records
                if r["noise_kind"] == kind and round(r["noise_hz"], 2) == hz]
        cells = "  ".join(f"{m}={np.mean([r[m] for r in rows if m in r]):.3f}"
                          for m in methods)
        print(f"  {kind:9s} {hz:>5.2f} Hz/px: {cells}")


if __name__ == "__main__":
    main()
