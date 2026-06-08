"""Shared analysis helpers for the sweep/error-bar experiments."""

from __future__ import annotations

import numpy as np

from . import metrics
from .events import Events

_WINDOWS = np.logspace(-4, -1.3, 40)   # 0.1 ms .. ~50 ms


def match_sr_window(ev: Events, min_dt: np.ndarray, target_sr: float, filter_fn):
    """Pick the window whose signal-retain best matches ``target_sr`` for a
    BAF/KNoise-style ``filter_fn(ev, window=..., min_dt=...) -> (kept, _)``."""
    best_w, best_gap, best_m = float(_WINDOWS[0]), 1e9, None
    for w in _WINDOWS:
        kept, _ = filter_fn(ev, window=float(w), min_dt=min_dt)
        m = metrics.evaluate_filter(ev.labels, kept)
        gap = abs(m.signal_retain - target_sr)
        if gap < best_gap:
            best_w, best_gap, best_m = float(w), gap, m
    return best_w, best_m


def retention_curve_from_scores(ev: Events, scores: np.ndarray,
                                n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a score threshold and return (event_reduction, signal_retain) arrays
    — the sparsity↔retention tradeoff. ``reduction`` = fraction of all events
    dropped; ``signal_retain`` = fraction of true signal kept."""
    thr = np.quantile(scores, np.linspace(0, 1, n))
    red, sr = [], []
    for t in thr:
        m = metrics.evaluate_filter(ev.labels, scores >= t)
        red.append(m.reduction)
        sr.append(m.signal_retain)
    order = np.argsort(red)
    return np.array(red)[order], np.array(sr)[order]


def nr_at_target_sr(ev: Events, scores: np.ndarray, target_sr: float, n: int = 200):
    """Sweep a score threshold (higher score ⇒ keep) and return the metrics at the
    operating point whose signal-retain is closest to ``target_sr``. Works for any
    per-event signal-support score (front-end membrane, baf_scores, knoise_scores)."""
    thr = np.quantile(scores, np.linspace(0, 1, n))
    best = None
    for t in thr:
        m = metrics.evaluate_filter(ev.labels, scores >= t)
        if best is None or abs(m.signal_retain - target_sr) < abs(best.signal_retain - target_sr):
            best = m
    return best


def max_signal_retain(ev: Events, scores: np.ndarray) -> float:
    """Highest signal-retain a score-based filter can reach while dropping any
    noise — i.e. keeping every event with positive support. For KNoise this is
    < 1 (some signal events never get a row/column neighbour); for the front-end
    it is ~1 (it can operate losslessly)."""
    return metrics.evaluate_filter(ev.labels, scores > 0).signal_retain


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std())
