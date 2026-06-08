"""N-Cars loader (Prophesee, Sironi et al. HATS, CVPR 2018).

Real event-camera recordings of cars / background, 2-class recognition. Files are
Prophesee ATIS ``.dat``: a ``%``-prefixed ASCII header, then 1-byte evType +
1-byte evSize, then ``evSize``-byte records of ``(uint32 ts_µs, uint32 addr)``.
Address bits (per the dataset's ``load_atis_data.m``): x = bits[0:14],
y = bits[14:28], polarity = bit 28.
"""

from __future__ import annotations

import os
import glob

import numpy as np

from ..events import Events

XMASK = 0x00003FFF
YMASK = 0x0FFFC000
POLMASK = 0x10000000

# N-Cars ROIs are small (~120×100); we pad/clip onto a fixed grid for batching.
DEFAULT_H = 100
DEFAULT_W = 120


def parse_dat(path: str, H: int = DEFAULT_H, W: int = DEFAULT_W) -> Events:
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line or not line.startswith(b"%"):
                f.seek(pos)
                break
        ev_type = f.read(1)
        ev_size = f.read(1)
        if not ev_size:
            return Events([], [], [], [], H=H, W=W)
        ev_size = ev_size[0]
        buf = f.read()
    n = len(buf) // ev_size
    rec = np.frombuffer(buf[: n * ev_size],
                        dtype=np.dtype([("ts", "<u4"), ("addr", "<u4")]))
    ts = rec["ts"].astype(np.float64) * 1e-6   # µs -> s
    addr = rec["addr"].astype(np.uint32)
    xs = (addr & XMASK).astype(np.int64)
    ys = ((addr & YMASK) >> 14).astype(np.int64)
    ps = ((addr & POLMASK) >> 28).astype(np.int64)
    # clip to grid (a few stray addresses can exceed the nominal ROI)
    keep = (xs < W) & (ys < H)
    return Events(xs[keep], ys[keep], ts[keep] - ts.min() if len(ts) else ts,
                  ps[keep], H=H, W=W)


def load_split(root: str, split: str = "test", limit_per_class: int | None = None,
               H: int = DEFAULT_H, W: int = DEFAULT_W,
               seed: int = 0) -> tuple[list[Events], np.ndarray, list[str]]:
    """Load an N-Cars split. Returns (streams, labels[0=background,1=cars], paths).

    ``root`` should contain ``n-cars_<split>/<split>/{cars,background}/*.dat``
    (the layout the bundle extracts to)."""
    base = _find_split_dir(root, split)
    streams, labels, paths = [], [], []
    rng = np.random.default_rng(seed)
    for cls, name in enumerate(("background", "cars")):
        files = sorted(glob.glob(os.path.join(base, name, "*.dat")))
        if not files:
            raise FileNotFoundError(f"no .dat files under {os.path.join(base, name)}")
        if limit_per_class is not None and len(files) > limit_per_class:
            idx = rng.choice(len(files), size=limit_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        for fp in files:
            ev = parse_dat(fp, H, W)
            if len(ev) == 0:
                continue
            streams.append(ev)
            labels.append(cls)
            paths.append(fp)
    return streams, np.array(labels, dtype=np.int64), paths


def _find_split_dir(root: str, split: str) -> str:
    candidates = [
        os.path.join(root, f"n-cars_{split}", split),
        os.path.join(root, split),
        os.path.join(root, f"n-cars_{split}"),
        root,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "cars")):
            return c
    raise FileNotFoundError(
        f"could not locate a '{split}' split with cars/ + background/ under {root}")
