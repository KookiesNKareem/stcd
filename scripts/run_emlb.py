"""Standard benchmark: STCD on **E-MLB** (Ding et al., *E-MLB: Multilevel Benchmark
for Event-Based Camera Denoising*, IEEE TMM 2023) with the benchmark's own
no-reference **ESR** metric.

This addresses the "evaluate on the standard benchmark, not just the standard
metric" concern: we run STCD + classical baselines on the actual E-MLB dataset and
score each denoised stream with ESR V1 (the version ``eval_benchmark.py`` uses),
sliced per 30,000 events as in the benchmark.

Two safeguards make the result trustworthy:
  * **Pipeline validation.** We recompute the *Raw* (undenoised) ESR per noise
    level and compare it to E-MLB's *published* Raw ESR. Raw has no operating-point
    ambiguity, so a match validates our loader + ESR implementation against their
    exact data and metric.
  * **Published baselines.** E-MLB already evaluated EDnCNN/MLPF/EventZoom/GEF on
    this data; we print those alongside, so STCD is placed against numbers produced
    by the benchmark authors themselves (not only our re-implementations).

Data (≈ tens of GB; D-END daytime + N-END night, 100 scenes x 4 ND noise levels):
    pip install gdown
    mkdir -p data/emlb
    gdown 1ZatTSewmb-j6RsrJxMWEQIE3Sm1yraK- -O data/emlb/D-END.zip
    gdown 17ZDhuYdtHui9nqJAfiYYX27omPY7Rpl9 -O data/emlb/N-END.zip
    (cd data/emlb && unzip -q D-END.zip && unzip -q N-END.zip)
If gdown hits a Drive quota, download the two files in a browser (links in
src/stcd/datasets/emlb.py). You need NOT unzip: this script reads .aedat4 members
straight from data/emlb/{D-END,N-END}.zip one at a time (disk-frugal), and uses an
extracted tree automatically if you do unzip.

Env:
  EMLB_ROOT     dataset root (default data/emlb)
  EMLB_LIMIT    process at most N recordings per (subset,level) — quick check
  EMLB_PROC_S   processing-window length in seconds (memory bound, default 2.0)
  EMLB_SELFTEST set to 1 to validate the ESR path on synthetic data (no dataset)
  SKIP_BASE     set to 1 to score only Raw + STCD (fastest)

Outputs: figures/emlb.png, figures/data/emlb.json
"""

from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.esr import esr                                                  # noqa: E402
from stcd.events import Events                                            # noqa: E402
from stcd.datasets import emlb                                            # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig                 # noqa: E402
from stcd.baselines import baf_filter, knoise_filter, time_surface_filter  # noqa: E402
from stcd.plotstyle import apply_style, color, OURS                       # noqa: E402

try:
    from stcd.downstream import mlpf as MLPF
except Exception:
    MLPF = None

HERE = os.path.dirname(__file__)
ROOT = os.environ.get("EMLB_ROOT", os.path.join(HERE, "..", "data", "emlb"))
MLPF_MODEL = os.path.join(HERE, "..", "data", "mlpf", "vendor",
                          "0316_soft_4bit_alpha1_sigmoid.h5")
FIG = os.path.join(HERE, "..", "figures")
LIMIT = int(os.environ.get("EMLB_LIMIT", "0"))
PROC_S = float(os.environ.get("EMLB_PROC_S", "2.0"))
SKIP_BASE = os.environ.get("SKIP_BASE", "0") == "1"
SKIP_MLPF = os.environ.get("SKIP_MLPF", "0") == "1"   # MLPF (numpy MLP) is slow per event;
                                                      # E-MLB publishes its number, so skip by default in full runs
ESR_VERSION = "v1"   # eval_benchmark.py default; matches E-MLB's published table

# E-MLB published ESR (V1), Table II, by noise level [ND1, ND4, ND16, ND64].
PUBLISHED = {
    "D-END": {"Raw": [0.821, 0.824, 0.815, 0.786], "BAF": [0.861, 0.869, 0.876, 0.890],
              "KNoise": [0.846, 0.837, 0.830, 0.807], "EDnCNN": [0.887, 0.908, 0.903, 0.912],
              "EventZoom": [0.996, 0.988, 0.996, 0.970], "GEF": [1.051, 0.938, 0.935, 0.927]},
    "N-END": {"Raw": [0.890, 0.824, 0.786, 0.768], "BAF": [0.946, 0.973, 0.992, 0.942],
              "KNoise": [0.954, 0.956, 0.871, 0.817], "EDnCNN": [1.001, 1.024, 1.079, 1.086],
              "EventZoom": [1.055, 1.007, 1.010, 0.988], "GEF": [1.027, 0.955, 0.946, 0.935]},
}
LEVEL_ORDER = ["ND1", "ND4", "ND16", "ND64"]


def _level_key(level_dir: str) -> str:
    """Map a level folder name (e.g. 'nd00','nd04','ND16') to ND1/ND4/ND16/ND64."""
    m = re.search(r"(\d+)", level_dir)
    if not m:
        return level_dir
    return {0: "ND1", 4: "ND4", 16: "ND16", 64: "ND64"}.get(int(m.group(1)),
                                                            f"ND{int(m.group(1))}")


def _windows(ev: Events, proc_s: float):
    t = ev.ts
    if len(t) == 0:
        return
    a, tend = t[0], t[-1]
    while a <= tend:
        m = (t >= a) & (t < a + proc_s)
        if m.any():
            yield ev.select(m)
        a += proc_s


def score_recording(ev: Events, fe: SpikingFrontEnd, mlpf_w, methods) -> dict:
    """ESR (V1) per method on one recording. Raw is scored directly; every filtered
    method accumulates its kept events across processing windows (bounds memory and
    keeps STCD's dense tensor small) then is scored over the full kept stream."""
    out = {"Raw": esr(ev, version=ESR_VERSION)}
    if methods == ["Raw"]:
        return out
    kept = {m: [] for m in methods if m != "Raw"}
    for w in _windows(ev, PROC_S):
        if OURS in kept:
            _, k = fe.filter(w);                         kept[OURS].append(k)
        if "time-surface" in kept:
            _, k = time_surface_filter(w, 5e-3, 0.5);    kept["time-surface"].append(k)
        if "BAF" in kept:
            _, k = baf_filter(w, 2e-3);                  kept["BAF"].append(k)
        if "KNoise" in kept:
            _, k = knoise_filter(w, 2e-3);               kept["KNoise"].append(k)
        if "MLPF" in kept and mlpf_w is not None:
            s = MLPF.predict_stream(mlpf_w, w);          kept["MLPF"].append(w.select(s >= 0.5))
    for m, parts in kept.items():
        parts = [p for p in parts if len(p) > 0]
        out[m] = esr(Events.concat(*parts), version=ESR_VERSION) if parts else float("nan")
    return out


def selftest() -> None:
    """No dataset needed: a structured stream must score far above a random one."""
    rng = np.random.default_rng(0); H, W, N = 260, 346, 60000
    ys = rng.integers(0, H, N); xs = (ys * 1.2 % W).astype(int)         # a diagonal edge
    t = np.sort(rng.random(N)); st = Events(xs, ys, t, rng.integers(0, 2, N), H=H, W=W)
    rx, ry = rng.integers(0, W, N), rng.integers(0, H, N)
    rd = Events(rx, ry, t, rng.integers(0, 2, N), H=H, W=W)
    a, b = esr(st, version=ESR_VERSION), esr(rd, version=ESR_VERSION)
    print(f"[selftest] ESR structured={a:.3f}  random={b:.3f}  -> {'OK' if a > b else 'FAIL'}")


def main() -> None:
    if os.environ.get("EMLB_SELFTEST") == "1":
        selftest(); return
    if not emlb.available(ROOT):
        sys.exit(f"no E-MLB recordings under {os.path.abspath(ROOT)} — see this file's header to download.")

    os.makedirs(os.path.join(FIG, "data"), exist_ok=True)
    apply_style()
    mlpf_w = None if SKIP_MLPF else (MLPF.load_mlpf(MLPF_MODEL)
                                     if (MLPF and os.path.isfile(MLPF_MODEL)) else None)
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    methods = ["Raw", OURS] + ([] if SKIP_BASE else
                               ["time-surface", "BAF", "KNoise"] + (["MLPF"] if mlpf_w else []))

    recs = emlb.find_recordings(ROOT)
    if LIMIT:
        seen: dict = {}
        kept_recs = []
        for r in recs:
            key = (r["subset"], r["level"]); seen[key] = seen.get(key, 0) + 1
            if seen[key] <= LIMIT:
                kept_recs.append(r)
        recs = kept_recs
    print(f"E-MLB: {len(recs)} recordings under {os.path.abspath(ROOT)}; methods={methods}", flush=True)

    records = []
    for i, r in enumerate(recs):
        try:
            ev = emlb.load_rec(r)
        except Exception as e:
            print(f"  [skip] {os.path.basename(r['path'])}: {type(e).__name__}: {e}", flush=True)
            continue
        sc = score_recording(ev, fe, mlpf_w, methods)
        rec = {"subset": r["subset"], "level": _level_key(r["level"]), "scene": r["scene"],
               "n_events": int(len(ev)), **sc}
        records.append(rec)
        print(f"  [{i+1}/{len(recs)}] {r['subset']}/{r['level']:>5} {r['scene'][:16]:16s} "
              f"N={len(ev):>9,} | " + "  ".join(f"{m}={sc.get(m, float('nan')):.3f}" for m in methods),
              flush=True)
        if (i + 1) % 20 == 0:
            with open(os.path.join(FIG, "data", "emlb.json"), "w") as f:
                json.dump({"records": records, "_partial": True}, f, indent=2)

    if not records:
        sys.exit("no recordings scored.")

    # ---- aggregate: per (subset, level) mean, and overall mean per method ----
    levels = [l for l in LEVEL_ORDER if any(r["level"] == l for r in records)] or \
             sorted({r["level"] for r in records})
    by_sl: dict = {}
    for sub in sorted({r["subset"] for r in records}):
        for lv in levels:
            grp = [r for r in records if r["subset"] == sub and r["level"] == lv]
            if grp:
                by_sl[f"{sub}/{lv}"] = {m: float(np.nanmean([g[m] for g in grp if m in g]))
                                        for m in methods}
    overall = {m: float(np.nanmean([r[m] for r in records if m in r and np.isfinite(r[m])]))
               for m in methods}
    n = len(records)
    ci = {m: float(1.96 * np.nanstd([r[m] for r in records if m in r], ddof=1) / np.sqrt(n))
          for m in methods}

    # ---- pipeline validation: our Raw vs E-MLB published Raw, per (subset,level) -
    print("\n=== pipeline validation: Raw ESR (ours vs E-MLB published) ===")
    val = []
    for sub in sorted({r["subset"] for r in records}):
        if sub not in PUBLISHED:
            continue
        for j, lv in enumerate(LEVEL_ORDER):
            key = f"{sub}/{lv}"
            if key in by_sl:
                pub = PUBLISHED[sub]["Raw"][j]
                ours = by_sl[key]["Raw"]
                print(f"  {key:>12}: ours={ours:.3f}  published={pub:.3f}  Δ={ours-pub:+.3f}")
                val.append({"key": key, "ours_raw": ours, "pub_raw": pub})

    print(f"\n=== overall mean ESR (n={n}) ===")
    for m in methods:
        print(f"  {m:>14}: {overall[m]:.3f} ± {ci[m]:.3f}")
    print("  (E-MLB published baselines for reference are in emlb.json)")

    # ---- figure: overall mean ESR per method (95% CI) ----
    bar = [m for m in methods if m != "Raw"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xp = np.arange(len(bar))
    bars = ax.bar(xp, [overall[m] for m in bar], yerr=[ci[m] for m in bar],
                  capsize=5, color=[color(m) for m in bar], zorder=3, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=9.5)
    ax.axhline(overall["Raw"], ls="--", color="#718096", alpha=0.8)
    ax.text(len(bar) - 0.4, overall["Raw"], f" raw {overall['Raw']:.3f}",
            va="bottom", ha="right", fontsize=8.5, color="#718096")
    ax.set_xticks(xp); ax.set_xticklabels(bar, rotation=10, ha="right")
    ax.set_ylabel("ESR (Event Structural Ratio) ↑")
    ax.set_title(f"Standard benchmark: ESR on E-MLB (n={n} recordings, V1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "emlb.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIG, "data", "emlb.json"), "w") as f:
        json.dump({"records": records, "by_subset_level": by_sl, "overall_mean": overall,
                   "ci95": ci, "n_recordings": n, "proc_window_s": PROC_S,
                   "esr_version": ESR_VERSION, "published_V1": PUBLISHED,
                   "raw_validation": val,
                   "metric": "ESR V1 (Ding et al., E-MLB, IEEE TMM 2023), no-reference, 30k-event slices",
                   "protocol": "each method at its native operating point; ESR over the kept stream"},
                  f, indent=2)
    print(f"\nwrote {os.path.abspath(os.path.join(FIG, 'emlb.png'))}")


if __name__ == "__main__":
    main()
