"""DVSNOISE20 loader (Baldwin et al., EPM/EDnCNN, CVPR 2020).

Real DAVIS346 recordings with background-activity noise and per-(frame,x,y) EPM
(Event Probability Mask) plausibility labels, for real-data denoising evaluation.

Format (validated against the actual files):
* Events are nested in a MATLAB struct: ``aedat.data.polarity.{x, y, timeStamp,
  polarity}`` (x∈[0,345], y∈[0,259], µs timestamps, polarity {0,1}). Per-frame
  timestamps are at ``aedat.data.frame.timeStamp`` (used to map events → frames).
* EPM is a *separate* v7.3/HDF5 ``*_epm_array.mat`` with one key ``epm``, an
  ``int16`` volume of shape ``(n_frames, W=346, H=260)``. The sign is the
  ground-truth label: ``> 0`` plausible (signal), ``< 0`` implausible (noise);
  ``-32768`` is a no-info sentinel. This is exactly EDnCNN's labelling scheme.

Acquire the data with ``scripts/download_data.py`` (large/gated Google Drive).
"""

from __future__ import annotations

import os

import numpy as np

from ..events import Events

H, W = 260, 346
EPM_SENTINEL = -32768


def available(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "2_mat"))


def load_recording(events_path: str, epm_path: str | None = None,
                   H: int = H, W: int = W):
    """Load one recording. Returns ``(events, frame_ts_s, epm)`` where
    ``frame_ts_s`` are APS-frame timestamps (s, same zero as the events) and
    ``epm`` is the ``(n_frames, W, H)`` int16 volume (or ``None``)."""
    import scipy.io as sio

    d = sio.loadmat(events_path, squeeze_me=True, struct_as_record=False)
    pol = d["aedat"].data.polarity
    ts_us = np.asarray(pol.timeStamp, dtype=np.float64)
    t0 = ts_us.min()
    ev = Events(np.asarray(pol.x), np.asarray(pol.y), (ts_us - t0) / 1e6,
                np.asarray(pol.polarity), H=H, W=W)
    frame_ts = (np.asarray(d["aedat"].data.frame.timeStamp, dtype=np.float64) - t0) / 1e6

    epm = None
    if epm_path:
        import h5py
        with h5py.File(epm_path, "r") as f:
            epm = np.array(f["epm"])   # (n_frames, W, H), int16
    return ev, frame_ts, epm


def load_full(events_path: str, epm_path: str | None = None, H: int = H, W: int = W):
    """Load events + APS frames (real intensity reference) + optional EPM in one
    pass. Returns ``(events, frame_ts_s, aps, epm)`` where ``aps`` is ``[F,H,W]``
    float APS frames (the ground-truth intensity used as a reconstruction
    reference) and ``frame_ts_s`` their timestamps (s, same zero as events)."""
    import scipy.io as sio

    d = sio.loadmat(events_path, squeeze_me=True, struct_as_record=False)
    pol = d["aedat"].data.polarity
    ts_us = np.asarray(pol.timeStamp, dtype=np.float64)
    t0 = ts_us.min()
    ev = Events(np.asarray(pol.x), np.asarray(pol.y), (ts_us - t0) / 1e6,
                np.asarray(pol.polarity), H=H, W=W)
    fr = d["aedat"].data.frame
    frame_ts = (np.asarray(fr.timeStamp, dtype=np.float64) - t0) / 1e6
    aps = np.transpose(np.asarray(fr.samples, dtype=np.float32), (2, 0, 1))  # (F,H,W)
    epm = None
    if epm_path:
        import h5py
        with h5py.File(epm_path, "r") as f:
            epm = np.array(f["epm"])
    return ev, frame_ts, aps, epm


def aps_motion_labels(ev: Events, frame_ts: np.ndarray, aps: np.ndarray,
                      hi_pct: float = 66.0, lo_pct: float = 33.0):
    """Per-event ground truth from the APS brightness change (the physical basis of
    EPM: a real event occurs where log-intensity is changing, ``dL = -∇L·v``).

    For each event we read ``|dL|`` = |APS log-intensity change| at its pixel over
    the surrounding frame interval. **Signal** = large ``|dL|`` *and* the event
    polarity matches the sign of the change (a genuine motion/edge event);
    **noise** = small ``|dL|`` (a flat region with no real brightness change).
    The ambiguous middle band is excluded. Returns ``(labels, valid)`` (signal=True).

    Validated on DVSNOISE20: 92% of high-|dL| events have matching polarity, and
    the labels are separable by our front-end (AUC ≈ 0.84) — unlike the raw EPM
    volume, which needs EDnCNN's full IMU-motion pipeline to decode.
    """
    L = np.log(np.clip(aps, 1.0, None))
    dL = np.diff(L, axis=0)                                   # [F-1,H,W]
    f = np.clip(np.searchsorted(frame_ts, ev.ts) - 1, 0, dL.shape[0] - 1)
    x = np.clip(ev.xs, 0, dL.shape[2] - 1)
    y = np.clip(ev.ys, 0, dL.shape[1] - 1)
    dl_at = dL[f, y, x]
    mag = np.abs(dl_at)
    hi, lo = np.percentile(mag, hi_pct), np.percentile(mag, lo_pct)
    pol_match = (ev.ps == 1) == (dl_at > 0)
    signal = (mag >= hi) & pol_match
    noise = mag <= lo
    valid = signal | noise
    return signal, valid


def epm_labels(ev: Events, frame_ts: np.ndarray, epm: np.ndarray):
    """Per-event ground-truth from the EPM volume.

    Returns ``(labels, valid)``: ``labels[i]`` True = signal (EPM > 0), False =
    noise (EPM < 0); ``valid[i]`` False where the EPM is the no-info sentinel (or
    out of range) and the event should be excluded from labelled metrics.
    """
    nf = epm.shape[0]
    f = np.clip(np.searchsorted(frame_ts, ev.ts), 0, nf - 1)
    x = np.clip(ev.xs, 0, epm.shape[1] - 1)
    y = np.clip(ev.ys, 0, epm.shape[2] - 1)
    v = epm[f, x, y].astype(np.int32)
    valid = (v != EPM_SENTINEL) & (v != 0)
    labels = v > 0
    return labels, valid


def labelled_events(events_path: str, epm_path: str, H: int = H, W: int = W) -> Events:
    """Convenience: load a recording and attach EPM signal/noise labels, keeping
    only the labelled (valid) events."""
    ev, frame_ts, epm = load_recording(events_path, epm_path, H, W)
    labels, valid = epm_labels(ev, frame_ts, epm)
    ev = ev.select(valid)
    ev.labels = labels[valid]
    return ev
