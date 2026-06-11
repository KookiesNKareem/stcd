"""E-MLB loader (Ding et al., *E-MLB: Multilevel Benchmark for Event-Based Camera
Denoising*, IEEE TMM 2023; ``KugaMaxx/cuke-emlb``).

E-MLB ships DAVIS346 recordings as ``.aedat4`` under
``<root>/{D-END,N-END}/<level>/<Scene>-<ND>-<n>.aedat4`` (100 scenes x 4 ND noise
levels, daytime + night). We read events with the numpy-native ``aedat`` package
and return our :class:`Events`. Orientation (y-flip) is irrelevant to ESR and to
the coincidence filters, all of which are spatially symmetric.

Download (see ``scripts/run_emlb.py`` header for the exact commands):
  D-END  https://drive.google.com/file/d/1ZatTSewmb-j6RsrJxMWEQIE3Sm1yraK-/view
  N-END  https://drive.google.com/file/d/17ZDhuYdtHui9nqJAfiYYX27omPY7Rpl9/view
unzip into ``data/emlb/`` so that e.g. ``data/emlb/D-END/nd00/`` exists.
"""

from __future__ import annotations

import glob
import os
import re
import tempfile
import zipfile

import numpy as np

from ..events import Events

DAVIS346 = (260, 346)   # (H, W) fallback if the stream header omits resolution


def _events_stream_res(dec) -> tuple[int, int]:
    for meta in dec.id_to_stream().values():
        if meta.get("type") == "events" and "width" in meta and "height" in meta:
            return int(meta["height"]), int(meta["width"])
    return DAVIS346


def load_events(path: str, H: int | None = None, W: int | None = None) -> Events:
    """Read one ``.aedat4`` into a time-sorted :class:`Events` (timestamps in s,
    zeroed to the first event). Resolution is taken from the file unless given."""
    import aedat

    dec = aedat.Decoder(path)
    if H is None or W is None:
        H, W = _events_stream_res(dec)

    xs, ys, ts, ps = [], [], [], []
    for pkt in dec:
        e = pkt.get("events") if isinstance(pkt, dict) else None
        if e is None or len(e) == 0:
            continue
        xs.append(np.asarray(e["x"])); ys.append(np.asarray(e["y"]))
        ts.append(np.asarray(e["t"])); ps.append(np.asarray(e["on"]))
    if not xs:
        z = np.zeros(0)
        return Events(z, z, z, z, H=H, W=W)

    xs = np.concatenate(xs); ys = np.concatenate(ys)
    ts = np.concatenate(ts).astype(np.float64); ps = np.concatenate(ps)
    t0 = ts.min()
    return Events(xs, ys, (ts - t0) / 1e6, ps.astype(np.int64), H=H, W=W).time_sorted()


def _parse(fname: str) -> str:
    m = re.search(r"(ND\d+)", fname, re.IGNORECASE)
    return m.group(1).upper() if m else "NA"


def _find_extracted(root: str) -> list[dict]:
    """Recordings from an extracted tree ``<root>/<subset>/<Scene>/<file>.aedat4``."""
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "*.aedat4"))):
        recs.append({"path": p, "_zip": None,
                     "subset": os.path.basename(os.path.dirname(os.path.dirname(p))),
                     "scene": os.path.basename(os.path.dirname(p)),
                     "level": _parse(os.path.basename(p))})
    return recs


def _find_in_zips(root: str) -> list[dict]:
    """Recordings read in place from ``<root>/{D-END,N-END}.zip`` (disk-frugal).

    The subset is the zip's basename; inside, members are ``<Scene>/<file>.aedat4``."""
    recs = []
    for zname in sorted(glob.glob(os.path.join(root, "*.zip"))):
        subset = os.path.splitext(os.path.basename(zname))[0]
        with zipfile.ZipFile(zname) as zf:
            for m in zf.namelist():
                if not m.lower().endswith(".aedat4"):
                    continue
                recs.append({"path": m, "_zip": zname, "subset": subset,
                             "scene": m.split("/")[0], "level": _parse(os.path.basename(m))})
    return sorted(recs, key=lambda r: (r["subset"], r["scene"], r["path"]))


def find_recordings(root: str) -> list[dict]:
    """Enumerate E-MLB recordings under ``root``. Prefers an extracted tree; falls
    back to reading members straight from ``D-END.zip``/``N-END.zip`` when the disk
    has no room to unzip. Each rec is ``{path, _zip, subset, level, scene}`` where
    ``level`` is the raw ND token (ND00/ND04/ND16/ND64) parsed from the filename."""
    return _find_extracted(root) or _find_in_zips(root)


def load_rec(rec: dict) -> Events:
    """Load one recording dict from :func:`find_recordings`, transparently handling
    both extracted files and in-zip members (extracted to a temp file, then removed)."""
    if not rec.get("_zip"):
        return load_events(rec["path"])
    with zipfile.ZipFile(rec["_zip"]) as zf:
        data = zf.read(rec["path"])
    fd, tmp = tempfile.mkstemp(suffix=".aedat4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return load_events(tmp)
    finally:
        os.remove(tmp)


def available(root: str) -> bool:
    return bool(find_recordings(root))
