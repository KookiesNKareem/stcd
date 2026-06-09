"""ESR — Event Structural Ratio (Ding et al., E-MLB, IEEE TMM 2023).

Faithful NumPy reimplementation of the *no-reference* denoising metric from the
E-MLB benchmark (``KugaMaxx/cuke-emlb``). Higher ESR = events more spatially
structured (on real edges) vs spread as noise; needs no ground-truth labels.

Per slice of ``N`` consecutive events on a ``W*H`` sensor, ``n[x,y]`` = per-pixel
event count (polarity ignored), ``K = W*H``:

  V1 ``EventStructuralRatio``:   ntss = Σ n(n-1)/(N(N-1));  ln = K - Σ(1-M/N)^n,
     M=⌊2N/3⌋;  ESR = sqrt(ntss·ln).
  V2 ``EventStructuralRatioV2``: median-filter n (3x3) first, then
     ntss = Σ n²/N²;  ln = (K - Σ 0.5^n)/K;  ESR = 1000·sqrt(ntss·ln).

We use **V2 as the default**: V1 (no spatial smoothing) is dominated by isolated
hot pixels — DVSNOISE20 has pixels firing thousands of times per second, which
inflate Σn(n-1) and make V1 unreliable (denoisers can score below a random
keep). V2's 3x3 median filter damps these, matching the metric's intent.
The benchmark slices every 30,000 events and reports the mean.
"""

from __future__ import annotations

import numpy as np

_EPS = np.spacing(1)


def _calc_v1(xs, ys, H, W):
    N = xs.size
    if N < 2:
        return float("nan")
    n = np.bincount(ys.astype(np.int64) * W + xs.astype(np.int64), minlength=H * W).astype(np.float64)
    K = float(H * W); M = int(N * 2 // 3)
    ntss = (n * (n - 1.0)).sum() / (N + _EPS) / (N - 1.0 + _EPS)
    ln = K - np.power(1.0 - M / N, n).sum()
    return float(np.sqrt(ntss * ln))


def _calc_v2(xs, ys, H, W):
    from scipy.ndimage import median_filter
    N = xs.size
    if N < 2:
        return float("nan")
    n = np.bincount(ys.astype(np.int64) * W + xs.astype(np.int64),
                    minlength=H * W).astype(np.float64).reshape(H, W)
    n = median_filter(n, size=3)
    K = float(H * W)
    ntss = (n * n).sum() / (N * N)
    ln = (K - np.power(0.5, n).sum()) / K
    return float(1000.0 * np.sqrt(ntss * ln))


def esr(ev, H: int | None = None, W: int | None = None, chunk: int = 30000,
        version: str = "v2") -> float:
    """Mean ESR over 30k-event slices of a (time-sorted) event set.

    ``ev`` is an Events object (uses ev.xs/ys/H/W) or a tuple ``(xs, ys, H, W)``.
    ``version`` = 'v2' (default, hot-pixel-robust) or 'v1' (as in eval_denoisor.py).
    """
    if hasattr(ev, "xs"):
        xs, ys = np.asarray(ev.xs), np.asarray(ev.ys)
        H = ev.H if H is None else H
        W = ev.W if W is None else W
    else:
        xs, ys, H, W = ev
        xs, ys = np.asarray(xs), np.asarray(ys)
    calc = _calc_v2 if version == "v2" else _calc_v1
    scores = []
    for s in range(0, xs.size, chunk):
        sc = calc(xs[s:s + chunk], ys[s:s + chunk], H, W)
        if not np.isnan(sc):
            scores.append(sc)
    return float(np.mean(scores)) if scores else float("nan")
