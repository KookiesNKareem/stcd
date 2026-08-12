"""Multi-seed robustness study for the unsupervised-STDP result (reviewer 2,
point 4): re-run the run_stdp_demo.py protocol over several independent
synthetic-scene seeds and report mean +/- std for each operating point.

Protocol per seed pair (identical to run_stdp_demo.py):
  * generate an UNLABELLED training stream and a LABELLED held-out eval stream
    (hot-pixel regime: 1 Hz/px BA noise + 60 hot pixels at 500 Hz);
  * STDP (k=5, tau=8 ms, dt=5 ms, eta=0.04, 50 epochs, unit-sum weight
    normalization + threshold homeostasis theta += 0.5*(r-0.15) from theta0=0.5)
    from a blind centre-only init -- the plasticity rule adapts ONLY the spatial
    kernel W; tau stays fixed; theta is moved by homeostasis, not by STDP;
  * baselines on the same held-out stream: blind centre-only (no learning),
    hand-tuned uniform box, supervised surrogate-gradient training (labels).

Output: figures/stdp_multiseed.json with per-seed AUCs and mean/std summary.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics                                      # noqa: E402
from stcd.synth import SynthConfig, generate                  # noqa: E402
from stcd.stdp import STDPDenoiser, STDPConfig                # noqa: E402
from stcd.frontend import SpikingFrontEnd, FrontEndConfig     # noqa: E402
from stcd.train import train_frontend, TrainConfig            # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
N_SEEDS = int(os.environ.get("STDP_SEEDS", "8"))


def one_seed(train_seed: int, eval_seed: int) -> dict:
    cfg = dict(H=140, W=180, duration=0.3, scene="bars", num_objects=4,
               noise_rate_hz=1.0, n_hot_pixels=60, hot_pixel_rate_hz=500.0)
    train_ev = generate(SynthConfig(seed=train_seed, **cfg))   # unlabelled use
    eval_ev = generate(SynthConfig(seed=eval_seed, **cfg))     # held-out

    stdp = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3, eta=0.04, epochs=50),
                        init="delta")
    auc_blind = stdp._auc(eval_ev)
    hist = stdp.train_unsupervised(train_ev, eval_ev=eval_ev)
    auc_stdp = hist["auc"][-1]

    box = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3), init="uniform")
    auc_box = box._auc(eval_ev)
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3,
                                        theta=1.5, dt=5e-3))
    sup_fe, _ = train_frontend(train_ev, fe=fe,
                               tcfg=TrainConfig(epochs=120, lr=0.05,
                                                verbose_every=0))
    auc_sup = metrics.roc(sup_fe.score_events(eval_ev), eval_ev.labels)["auc"]
    return {"train_seed": train_seed, "eval_seed": eval_seed,
            "blind": float(auc_blind), "stdp": float(auc_stdp),
            "box": float(auc_box), "supervised": float(auc_sup)}


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for i in range(N_SEEDS):
        r = one_seed(train_seed=1 + i, eval_seed=701 + i)
        rows.append(r)
        print(f"seed {i + 1}/{N_SEEDS}: blind={r['blind']:.3f} "
              f"stdp={r['stdp']:.3f} box={r['box']:.3f} sup={r['supervised']:.3f}",
              flush=True)

    summary = {}
    for key in ("blind", "stdp", "box", "supervised"):
        vals = np.array([r[key] for r in rows])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                        "min": float(vals.min()), "max": float(vals.max())}
    print("\n=== mean +/- std over", N_SEEDS, "seeds ===")
    for key, s in summary.items():
        print(f"  {key:11s} {s['mean']:.3f} +/- {s['std']:.3f} "
              f"[{s['min']:.3f}, {s['max']:.3f}]")

    out = os.path.join(FIG, "stdp_multiseed.json")
    with open(out, "w") as f:
        json.dump({"n_seeds": N_SEEDS, "runs": rows, "summary": summary}, f,
                  indent=2)
    print(f"\nwrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
