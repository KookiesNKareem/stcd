"""RPMD — Relative Plausibility Measure of Denoising (Baldwin et al., CVPR 2020).

Faithful reimplementation of the EDnCNN evaluation so our numbers are on the same
scale as the paper. Ported from the authors' MATLAB (bald6354/edncnn):
``mainScript.m`` (per-event EPM probability) and ``scoreDenoise.m`` (RPMD).

Per event, the EPM gives ``Prob`` = P(event is a real/plausible event) in [0,1]:
the stored ``epm`` array is ``int16(Jt/gamma * 32767)`` (signed; sign = expected
polarity), so ``Prob = clamp( (+1 if ON else -1) * epm[frame,x,y]/32767, 0, 1 )``.

Only events that occur **during an APS exposure window** (``duringAPS>0``) at a
pixel with **valid APS intensity** (5..250, median-filtered + 3x3 dilated) are
scored; one event is kept per ``(x, y, polarity, window)`` (the highest-scoring).

RPMD = logOptimal - logDenoise, where for the kept slots
  logOptimal = mean( Prob>0.5 ? log Prob : log(1-Prob) )           (MAP decision)
  logDenoise = mean( keep    ? log Prob : log(1-Prob) )            (method decision)
Lower is better; 0 = optimal. Keeping a Prob~=0 event or dropping a Prob~=1 event
each costs ``log(realmin)`` ~= -708 (matching the paper's flooring).
"""

from __future__ import annotations

import numpy as np

_TINY = np.finfo(np.float64).tiny   # ~2.2e-308, matches MATLAB realmin


def epm_event_prob(x, y, t_s, p, frame_ts_s, exp_start_s, exp_end_s, aps_fhw, epm_fxy):
    """Per-event EPM probability + scoring validity.

    Args (all 1-D per-event unless noted):
      x, y      : pixel coords (x in [0,W), y in [0,H)), int
      t_s       : event time (s)
      p         : polarity (1 = ON / >0, 0 = OFF)
      frame_ts_s, exp_start_s, exp_end_s : per-APS-frame time / exposure window (s)
      aps_fhw   : APS frames, shape (F, H, W)
      epm_fxy   : EPM int16 volume, shape (F, W, H)
    Returns prob[N] float64, during[N] int (1-indexed frame, 0 = none),
            aps_good[N] bool, closest[N] int (0-indexed nearest frame).
    """
    from scipy.ndimage import median_filter, binary_dilation

    F = frame_ts_s.size
    t = np.asarray(t_s, np.float64)
    x = np.asarray(x, np.int64); y = np.asarray(y, np.int64); p = np.asarray(p)

    # nearest APS frame (closestFrame), 0-indexed
    cf = np.clip(np.searchsorted(frame_ts_s, t), 1, F - 1)
    cf = np.where(np.abs(t - frame_ts_s[cf - 1]) <= np.abs(t - frame_ts_s[cf]), cf - 1, cf)

    # duringAPS: event within an exposure window [expStart, expEnd] (1-indexed; 0 = none)
    j = np.clip(np.searchsorted(exp_start_s, t, side="right") - 1, 0, F - 1)
    during = np.where((t >= exp_start_s[j]) & (t <= exp_end_s[j]), j + 1, 0)

    # apsIntGood: APS in [5,250], per-frame median-filtered + 3x3 dilated
    good = (aps_fhw >= 5) & (aps_fhw <= 250)
    for fi in range(F):
        good[fi] = binary_dilation(median_filter(good[fi], size=3), structure=np.ones((3, 3)))
    aps_good = good[cf, y, x]

    # Prob from the signed EPM, conditioned on polarity
    g = epm_fxy[cf, x, y].astype(np.float64) / 32767.0
    prob = np.clip(np.where(np.asarray(p) > 0, g, -g), 0.0, 1.0)
    return prob, during, aps_good, cf


def _slots(score, prob, during, aps_good, x, y, p):
    """Reduce to one scored event per (x,y,polarity,window): the highest-scoring.
    Returns (prob_slot, score_slot) for valid slots only."""
    valid = (during > 0) & aps_good & ~np.isnan(score)
    s = score[valid]; pr = prob[valid]
    key = (((during[valid].astype(np.int64) * 400 + x[valid]) * 300 + y[valid]) * 2
           + (p[valid] > 0).astype(np.int64))
    # keep argmax score per key: sort by (key, score) then take last of each key
    order = np.lexsort((s, key))
    ks = key[order]
    last = np.ones(ks.size, bool); last[:-1] = ks[1:] != ks[:-1]
    sel = order[last]
    return pr[sel], s[sel]


def rpmd(score, prob, during, aps_good, x, y, p, thresholds=200):
    """Faithful RPMD for a denoiser whose per-event signal score is ``score``
    (higher = more likely real). Sweeps the keep-threshold and returns the best
    (minimum) RPMD — a threshold-free, per-method measure analogous to AUC.

    Returns dict: rpmd_min, rpmd_raw (keep-all), keep_frac (at min), n_slots,
    and (thr, rpmd) curve arrays.
    """
    pr, s = _slots(score, prob, during, aps_good, x, y, p)
    N = pr.size
    if N == 0:
        return {"rpmd_min": float("nan"), "rpmd_raw": float("nan"),
                "keep_frac": float("nan"), "n_slots": 0}
    log_pr = np.log(np.maximum(pr, _TINY))
    log_1mpr = np.log(np.maximum(1.0 - pr, _TINY))
    log_opt = (np.where(pr > 0.5, log_pr, log_1mpr)).mean()

    # sweep keep-threshold T (keep iff s >= T) over score quantiles
    qs = np.unique(np.quantile(s, np.linspace(0, 1, int(thresholds))))
    thr = np.concatenate([[s.min() - 1.0], qs, [s.max() + 1.0]])
    rp = np.empty(thr.size)
    for i, T in enumerate(thr):
        keep = s >= T
        log_den = (np.where(keep, log_pr, log_1mpr)).mean()
        rp[i] = log_opt - log_den
    imin = int(np.argmin(rp))
    return {"rpmd_min": float(rp[imin]), "rpmd_raw": float(log_opt - log_pr.mean()),
            "keep_frac": float((s >= thr[imin]).mean()), "n_slots": int(N),
            "thr": thr, "rpmd": rp}


def rpmd_keep_all(prob, during, aps_good, x, y, p):
    """RPMD of the raw stream (no denoising = keep every event)."""
    score = np.ones(prob.shape)               # keep all at any threshold < 1
    out = rpmd(score, prob, during, aps_good, x, y, p, thresholds=2)
    return out["rpmd_raw"], out["n_slots"]
