"""Figure + LaTeX table for the E-MLB result, regenerated from the cached
``figures/data/emlb.json`` (no need to rerun the multi-hour benchmark).

Places STCD (our pipeline, Raw-validated against E-MLB to within 0.007) against
E-MLB's *published* Table II ESR (V1) for every method. Emits:
  figures/emlb.png         sorted overall-ESR bar, STCD highlighted
  prints a LaTeX table (paste into the paper)
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd.plotstyle import apply_style, color, OURS   # noqa: E402

HERE = os.path.dirname(__file__)
JSON = os.path.join(HERE, "..", "figures", "data", "emlb.json")
FIG = os.path.join(HERE, "..", "figures", "emlb.png")

# E-MLB Table II, ESR V1, columns [D-ND1,D-ND4,D-ND16,D-ND64, N-ND1,N-ND4,N-ND16,N-ND64].
# "kind": classical (no training) / learned (trained network).
PUB = {
    "Raw":       ([0.821, 0.824, 0.815, 0.786, 0.890, 0.824, 0.786, 0.768], "raw"),
    "KNoise":    ([0.846, 0.837, 0.830, 0.807, 0.954, 0.956, 0.871, 0.817], "classical"),
    "YNoise":    ([0.866, 0.863, 0.857, 0.821, 1.009, 0.943, 0.875, 0.792], "classical"),
    "EvFlow":    ([0.848, 0.878, 0.868, 0.833, 0.969, 0.983, 0.889, 0.797], "classical"),
    "TS":        ([0.877, 0.887, 0.870, 0.837, 1.033, 0.944, 0.886, 0.797], "classical"),
    "DWF":       ([0.878, 0.876, 0.866, 0.865, 0.923, 0.962, 0.988, 0.932], "classical"),
    "BAF":       ([0.861, 0.869, 0.876, 0.890, 0.946, 0.973, 0.992, 0.942], "classical"),
    "GEF":       ([1.051, 0.938, 0.935, 0.927, 1.027, 0.955, 0.946, 0.935], "classical"),
    "IETS":      ([0.772, 0.785, 0.777, 0.753, 0.950, 0.823, 0.804, 0.711], "classical"),
    "MLPF":      ([0.851, 0.855, 0.846, 0.840, 0.926, 0.928, 0.910, 0.906], "learned"),
    "EDnCNN":    ([0.887, 0.908, 0.903, 0.912, 1.001, 1.024, 1.079, 1.086], "learned"),
    "EventZoom": ([0.996, 0.988, 0.996, 0.970, 1.055, 1.007, 1.010, 0.988], "learned"),
}
LV = ["ND1", "ND4", "ND16", "ND64"]


def _lk(x):
    return {"ND00": "ND1", "ND04": "ND4", "ND16": "ND16", "ND64": "ND64"}.get(x, x)


def main() -> None:
    apply_style()
    d = json.load(open(JSON))
    recs = d["records"]

    # --- STCD per cell (subset, level) and overall, from the cached run ---
    cells = {(s, lv): [] for s in ("D-END", "N-END") for lv in LV}
    for r in recs:
        cells[(r["subset"], _lk(r["level"]))].append(r["STCD"])
    stcd_vec = [float(np.mean(cells[("D-END", lv)])) for lv in LV] + \
               [float(np.mean(cells[("N-END", lv)])) for lv in LV]
    stcd_all = [r["STCD"] for r in recs]
    stcd_mean = float(np.mean(stcd_all))
    stcd_ci = float(1.96 * np.std(stcd_all, ddof=1) / np.sqrt(len(stcd_all)))

    rows = {"STCD": (stcd_vec, "ours")}
    rows.update(PUB)
    overall = {m: float(np.mean(v)) for m, (v, _) in rows.items()}
    d_mean = {m: float(np.mean(v[:4])) for m, (v, _) in rows.items()}
    n_mean = {m: float(np.mean(v[4:])) for m, (v, _) in rows.items()}
    order = sorted(rows, key=lambda m: -overall[m])
    rank = order.index("STCD") + 1
    n_methods = len([m for m in rows if m != "Raw"])

    # --- figure: sorted overall ESR, STCD highlighted -------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    kcol = {"ours": color(OURS), "learned": "#8B5CF6", "classical": "#94A3B8", "raw": "#CBD5E1"}
    y = np.arange(len(order))[::-1]
    cols = [kcol[rows[m][1]] for m in order]
    ax.barh(y, [overall[m] for m in order], color=cols, zorder=3,
            xerr=[stcd_ci if m == "STCD" else 0 for m in order], capsize=3)
    for yi, m in zip(y, order):
        ax.text(overall[m] + 0.004, yi, f"{overall[m]:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(
        [(m + "  (ours)" if m == "STCD" else m) for m in order])
    ax.set_xlabel("mean ESR over E-MLB (V1, n=1152) ↑")
    ax.set_xlim(0.75, max(overall.values()) + 0.05)
    ax.set_title("STCD on the E-MLB benchmark (standard label-free metric)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=kcol["ours"], label="STCD (ours, training-free)"),
                       Patch(color=kcol["learned"], label="learned (trained net)"),
                       Patch(color=kcol["classical"], label="classical / guided"),
                       Patch(color=kcol["raw"], label="raw (no denoising)")],
              loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- console summary + LaTeX table ---------------------------------------
    print(f"STCD overall ESR = {stcd_mean:.3f} ± {stcd_ci:.3f}  (rank {rank}/{n_methods} methods)")
    best_tf = [m for m in order if rows[m][1] in ("ours", "classical")][0]
    print(f"best training-free method: {best_tf}")
    print("\n% --- E-MLB ESR table (V1); STCD = ours (Raw-validated), rest = E-MLB Table II ---")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Method & Type & D-END & N-END & Overall \\\\\n\\midrule")
    for m in order:
        kind = {"ours": "spiking (ours)", "learned": "learned", "classical": "classical",
                "raw": "--"}[rows[m][1]]
        star = "\\textbf{" if m == "STCD" else ""
        end = "}" if m == "STCD" else ""
        name = "STCD" if m == "STCD" else m
        print(f"{star}{name}{end} & {kind} & {star}{d_mean[m]:.3f}{end} & "
              f"{star}{n_mean[m]:.3f}{end} & {star}{overall[m]:.3f}{end} \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print(f"\nwrote {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
