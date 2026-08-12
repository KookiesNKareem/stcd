"""DND21 (Guo & Delbruck, T-PAMI 2022) recordings: hotel-bar and driving.

The DND21 benchmark distributes *clean* DAVIS346 recordings as AEDAT-2.0
binaries (jAER format) and, for some sequences, as DVS text files. Its
denoising protocol mixes a clean recording with known noise and scores the
per-event signal/noise discrimination -- the same exact-label protocol as our
``stcd.synth.inject_noise``.

This module only loads the recordings; noise injection and scoring live in
``scripts/run_dnd21.py``.

AEDAT-2.0 DAVIS decoding (jAER convention):
  * file starts with '#'-prefixed ASCII header lines;
  * then int32-BE (address, timestamp_us) pairs;
  * ADC/APS samples have bit 31 set and are skipped;
  * DVS events: y = (addr >> 22) & 0x1FF, x = (addr >> 12) & 0x3FF,
    polarity = (addr >> 11) & 1; out-of-range addresses (IMU/special) are
    dropped by the x < W, y < H check.
"""
from __future__ import annotations

import os

import numpy as np

from ..events import Events

H, W = 260, 346


def _load_aedat2(path: str, H: int = H, W: int = W) -> Events:
    with open(path, "rb") as f:
        # skip ASCII header lines starting with '#'
        pos = 0
        while True:
            line = f.readline()
            if not line.startswith(b"#"):
                break
            pos = f.tell()
        f.seek(pos)
        raw = np.frombuffer(f.read(), dtype=">u4")
    if len(raw) % 2:
        raw = raw[:-1]
    addr, ts = raw[0::2], raw[1::2].astype(np.int64)
    dvs = (addr & 0x80000000) == 0            # drop APS/ADC samples
    addr, ts = addr[dvs], ts[dvs]
    y = (addr >> 22) & 0x1FF
    x = (addr >> 12) & 0x3FF
    p = (addr >> 11) & 1
    ok = (x < W) & (y < H)                    # drops IMU/special/trigger
    x, y, p, ts = x[ok], y[ok], p[ok], ts[ok]
    # timestamps are 32-bit us and may wrap on long files; unwrap monotonically
    dt = np.diff(ts)
    for i in np.where(dt < -2**30)[0]:
        ts[i + 1:] += 2**32
    t = (ts - ts.min()) / 1e6
    return Events(xs=x.astype(np.int64), ys=(H - 1 - y).astype(np.int64),
                  ts=t.astype(np.float64), ps=p.astype(np.int64), H=H, W=W)


def _load_txt(path: str, H: int = H, W: int = W) -> Events:
    """DVS text file: one event per line, ``t x y p`` (t in seconds)."""
    a = np.loadtxt(path, comments="#")
    t, x, y, p = a[:, 0], a[:, 1].astype(np.int64), a[:, 2].astype(np.int64), \
        a[:, 3].astype(np.int64)
    p = (p > 0).astype(np.int64)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return Events(xs=x[ok], ys=y[ok], ts=(t[ok] - t[ok].min()),
                  ps=p[ok], H=H, W=W)


def load(path: str, H: int = H, W: int = W) -> Events:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv", ".dat"):
        ev = _load_txt(path, H, W)
    else:
        ev = _load_aedat2(path, H, W)
    return ev.time_sorted()
