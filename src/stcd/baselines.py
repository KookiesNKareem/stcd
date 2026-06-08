"""Baseline denoisers — primarily the classic Background Activity Filter (BAF).

BAF (Lichtsteiner/Delbrück): an event is real if a *spatial neighbour* fired
recently. We compute, for each event, the minimum time-gap to the most recent
event in its 8-neighbourhood (centre excluded, so hot pixels can't self-support).
From that single quantity we derive both:

* a binary filter at a chosen window ``W`` (keep iff ``min_dt ≤ W``), and
* a smooth per-event score ``exp(-min_dt / W)`` for a parameter-free ROC,

so BAF and the spiking front-end can be compared on the same ROC axes.

BAF is intentionally polarity-agnostic (any neighbour supports), the standard
formulation.
"""

from __future__ import annotations

import numpy as np

from .events import Events


def baf_min_dt(ev: Events, neighborhood: int = 1) -> np.ndarray:
    """Per-event minimum time-gap (s) to the most recent event in the spatial
    neighbourhood (radius ``neighborhood``, centre excluded). ``inf`` if none.

    Events are processed in time order; the returned array is aligned to the
    *input* ordering of ``ev``.
    """
    n = len(ev)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(ev.ts, kind="stable")
    xs, ys, ts = ev.xs[order], ev.ys[order], ev.ts[order]
    H, W = ev.H, ev.W

    last = np.full((H, W), -np.inf, dtype=np.float64)
    offs = [
        (dy, dx)
        for dy in range(-neighborhood, neighborhood + 1)
        for dx in range(-neighborhood, neighborhood + 1)
        if not (dy == 0 and dx == 0)
    ]
    min_dt_sorted = np.empty(n, dtype=np.float64)
    for i in range(n):
        y, x, t = ys[i], xs[i], ts[i]
        best = -np.inf
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                lv = last[ny, nx]
                if lv > best:
                    best = lv
        min_dt_sorted[i] = t - best  # inf when best == -inf
        last[y, x] = t

    # Undo the time-sort so output aligns with input event ordering.
    out = np.empty(n, dtype=np.float64)
    out[order] = min_dt_sorted
    return out


def baf_scores(ev: Events, window: float, neighborhood: int = 1,
               min_dt: np.ndarray | None = None) -> np.ndarray:
    """Smooth support score in [0,1]: ``exp(-min_dt / window)`` (0 when no
    neighbour). Pass a precomputed ``min_dt`` to sweep ``window`` cheaply."""
    if min_dt is None:
        min_dt = baf_min_dt(ev, neighborhood)
    with np.errstate(over="ignore"):
        s = np.exp(-min_dt / window)
    return np.where(np.isfinite(min_dt), s, 0.0)


def baf_filter(ev: Events, window: float, neighborhood: int = 1,
               min_dt: np.ndarray | None = None) -> tuple[np.ndarray, Events]:
    """Classic binary BAF: keep an event iff a neighbour fired within ``window``."""
    if min_dt is None:
        min_dt = baf_min_dt(ev, neighborhood)
    kept = min_dt <= window
    return kept, ev.select(kept)


# --------------------------------------------------------------------------- #
# KNoise — O(N)-memory spatiotemporal filter (Khodamoradi & Kastner, 2018)
# --------------------------------------------------------------------------- #
def knoise_min_dt(ev: Events) -> np.ndarray:
    """Per-event min time-gap to a recent row/column neighbour, KNoise-style.

    Instead of a full per-pixel grid, KNoise keeps only the most recent event per
    row and per column (O(H+W) memory). An event is supported if the last event
    in its column was in an adjacent row, or the last event in its row was in an
    adjacent column — an O(N) approximation of the 8-neighbour BAF check.
    """
    n = len(ev)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(ev.ts, kind="stable")
    xs, ys, ts = ev.xs[order], ev.ys[order], ev.ts[order]
    H, W = ev.H, ev.W

    colT = np.full(W, -np.inf); colY = np.full(W, -10)
    rowT = np.full(H, -np.inf); rowX = np.full(H, -10)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y, t = xs[i], ys[i], ts[i]
        best = -np.inf
        if abs(y - colY[x]) <= 1 and colT[x] > best:   # vertical neighbour in column
            best = colT[x]
        if abs(x - rowX[y]) <= 1 and rowT[y] > best:    # horizontal neighbour in row
            best = rowT[y]
        out[i] = t - best
        colT[x] = t; colY[x] = y
        rowT[y] = t; rowX[y] = x
    res = np.empty(n, dtype=np.float64)
    res[order] = out
    return res


def knoise_scores(ev: Events, window: float,
                  min_dt: np.ndarray | None = None) -> np.ndarray:
    if min_dt is None:
        min_dt = knoise_min_dt(ev)
    with np.errstate(over="ignore"):
        s = np.exp(-min_dt / window)
    return np.where(np.isfinite(min_dt), s, 0.0)


def knoise_filter(ev: Events, window: float,
                  min_dt: np.ndarray | None = None) -> tuple[np.ndarray, Events]:
    if min_dt is None:
        min_dt = knoise_min_dt(ev)
    kept = min_dt <= window
    return kept, ev.select(kept)


# --------------------------------------------------------------------------- #
# Time-surface filter — a stronger, standard baseline (HOTS/HATS-style)
# --------------------------------------------------------------------------- #
def time_surface_scores(ev: Events, tau: float = 5e-3, neighborhood: int = 1) -> np.ndarray:
    """Per-event support from the local *time surface*: the exponentially-decayed
    recent activity of the spatial neighbourhood, ``Σ_nbr exp(-(t − t_last_nbr)/τ)``.

    A principled, widely-used denoiser/representation (Lagorce et al. HOTS, Sironi
    et al. HATS): real edge/motion events sit in a "hot" neighbourhood, noise does
    not. Stronger than binary BAF — it is graded and rate-adaptive — so it is a
    fair, non-strawman classical baseline. Centre excluded (hot-pixel-robust)."""
    n = len(ev)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(ev.ts, kind="stable")
    xs, ys, ts = ev.xs[order], ev.ys[order], ev.ts[order]
    H, W = ev.H, ev.W
    last = np.full((H, W), -np.inf, dtype=np.float64)
    offs = [(dy, dx) for dy in range(-neighborhood, neighborhood + 1)
            for dx in range(-neighborhood, neighborhood + 1) if not (dy == 0 and dx == 0)]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y, x, t = ys[i], xs[i], ts[i]
        s = 0.0
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and np.isfinite(last[ny, nx]):
                s += np.exp(-(t - last[ny, nx]) / tau)
        out[i] = s
        last[y, x] = t
    res = np.empty(n, dtype=np.float64)
    res[order] = out
    return res


def time_surface_filter(ev: Events, tau: float = 5e-3, threshold: float = 0.5,
                        neighborhood: int = 1, scores: np.ndarray | None = None):
    if scores is None:
        scores = time_surface_scores(ev, tau, neighborhood)
    kept = scores >= threshold
    return kept, ev.select(kept)


# --------------------------------------------------------------------------- #
# Per-pixel rate cap — hot-pixel mitigation (proposal §9)
# --------------------------------------------------------------------------- #
def rate_cap(ev: Events, max_rate_hz: float) -> tuple[np.ndarray, Events]:
    """Drop events from pixels whose firing rate exceeds ``max_rate_hz`` over the
    recording — removes stuck/leaky hot pixels that survive correlation filters
    because their repeated firing is temporally self-correlated."""
    if len(ev) == 0:
        return np.ones(0, dtype=bool), ev
    dur = ev.duration or 1.0
    counts = np.zeros((ev.H, ev.W), dtype=np.int64)
    np.add.at(counts, (ev.ys, ev.xs), 1)
    hot = counts > (max_rate_hz * dur)
    kept = ~hot[ev.ys, ev.xs]
    return kept, ev.select(kept)
