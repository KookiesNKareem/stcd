"""MLPF — the within-camera Multilayer Perceptron DVS denoising Filter
(Navaro, Guo, Gnaneswaran, ... Delbruck; CVPR-W 2023, building on Guo & Delbruck
T-PAMI 2022). A tiny MLP over a small timestamp-image patch around each event —
the *deployable* learned competitor to STCD (it too is meant to run in-camera).

We load the authors' published trained weights *exactly* (no retraining): the
deployed ``0316_soft_4bit_alpha1_sigmoid.h5`` — the 10-hidden, ~1k-weight model
hls4ml synthesised to hardware (weights already sit on the 4-bit 1/8 grid). We
run their exact architecture and feature with float activations, i.e. the paper's
"floating-point network", whose AUC the paper reports as "nearly indistinguishable"
from the 4-bit hardware.

Feature (ground truth: dnd_hls C++ ``dnd_create_mlp_activation.cpp``):
  - one timestamp image: per pixel, the most-recent event's (timestamp, polarity).
  - for each incoming event, read the 7x7 patch of that image around it.
  - per patch pixel: age = max(0, 1 - dt/TAU), TAU = 64 ms (linear leaky decay,
    0 if the last event there is older than TAU or absent); polarity = +-1 of that
    last event (0 if none); the centre pixel's polarity = the incoming event's.
  - MLP input = [49 ages, 49 polarities] = 98, ordered col-major (x outer, y inner).
  - MLP: Dense(98->10, ReLU) -> Dense(10->1, sigmoid) = P(signal). ~1k weights.
"""

from __future__ import annotations

import numpy as np

PATCH = 7              # 7x7 neighbourhood
BORDER = PATCH // 2     # 3
TAU_MS = 64.0           # 2^6 ms, the hardware TAU
CENTER = (PATCH * PATCH) // 2   # flat index of the current event's pixel (=24)


def mlpf_macs_per_event(weights=None) -> int:
    """MACs/event of the deployed MLPF (98->10->1). = 990 MACs (1980 FLOPs)."""
    n_in, n_hid = 98, (weights["W1"].shape[1] if weights else 10)
    return n_in * n_hid + n_hid * 1


def load_mlpf(h5path: str) -> dict:
    """Load the published trained weights from the qKeras .h5 (read directly via
    h5py; we run the latent float weights — per the paper, equal AUC to 4-bit)."""
    import h5py
    with h5py.File(h5path, "r") as h:
        g = h["model_weights"] if "model_weights" in h else h
        W1 = np.array(g["fc1"]["fc1"]["kernel:0"], dtype=np.float32)   # (98, H)
        b1 = np.array(g["fc1"]["fc1"]["bias:0"], dtype=np.float32)     # (H,)
        W2 = np.array(g["fc2"]["fc2"]["kernel:0"], dtype=np.float32)   # (H, 1)
        b2 = np.array(g["fc2"]["fc2"]["bias:0"], dtype=np.float32)     # (1,)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    p = z >= 0
    out[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[~p]); out[~p] = e / (1.0 + e)
    return out


def _forward(feat: np.ndarray, w: dict) -> np.ndarray:
    """feat [N,98] -> P(signal) [N]."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):   # macOS BLAS spurious
        h = np.maximum(feat @ w["W1"] + w["b1"], 0.0)    # ReLU
        z = h @ w["W2"] + w["b2"]                        # (N,1)
    return _sigmoid(z[:, 0])                             # sigmoid


# ---- vectorized timestamp-image-patch feature ------------------------------- #
def _build_store(ev):
    """Single timestamp image sorted by composite (pixel, time) key for grouped
    searchsorted lookups (same trick as the EDnCNN scorer)."""
    H, W = ev.H, ev.W
    xs = ev.xs.astype(np.int64); ys = ev.ys.astype(np.int64)
    ts = (ev.ts * 1e6).astype(np.float64)                # microseconds
    pol = np.where(ev.ps > 0, 1.0, -1.0).astype(np.float64)
    pix = ys * W + xs
    big = float(ts.max()) + 10.0
    comp = pix.astype(np.float64) * big + ts
    o = np.argsort(comp, kind="stable")
    store = (comp[o], ts[o], pix[o], pol[o])
    return store, big, xs, ys, ts, pol, H, W


def _feat_chunk(store, big, xs, ys, ts, pol, idx, H, W) -> np.ndarray:
    """Vectorized 98-vector feature for events `idx` (matches the HLS builder)."""
    comp_s, ts_s, pix_s, pol_s = store
    qx = xs[idx]; qy = ys[idx]; qt = ts[idx]; qpol = pol[idx]
    m = len(idx)
    ages = np.zeros((m, PATCH * PATCH), np.float64)
    pols = np.zeros((m, PATCH * PATCH), np.float64)
    n = len(ts_s)
    for px in range(PATCH):
        nx = qx + (px - BORDER); vx = (nx >= 0) & (nx < W)
        for py in range(PATCH):
            ny = qy + (py - BORDER); vb = vx & (ny >= 0) & (ny < H)
            if not vb.any():
                continue
            flat = px * PATCH + py                        # HLS PATCH_addr ordering
            tp = ny * W + nx
            k = np.searchsorted(comp_s, tp.astype(np.float64) * big + qt, side="left")
            j = k - 1                                     # last event strictly before qt
            jc = np.clip(j, 0, n - 1)
            ok = vb & (j >= 0) & (pix_s[jc] == tp)
            rows = np.where(ok)[0]
            dt_ms = (qt[rows] - ts_s[jc[rows]]) / 1000.0
            ages[rows, flat] = np.maximum(1.0 - dt_ms / TAU_MS, 0.0)
            pols[rows, flat] = pol_s[jc[rows]]
    pols[:, CENTER] = qpol                                # centre = incoming polarity
    return np.concatenate([ages, pols], axis=1)           # [m, 98]


def mlpf_scores(ev, query_idx, w: dict) -> np.ndarray:
    """MLPF P(signal) for the events at `query_idx` (ev must be time-sorted)."""
    store, big, xs, ys, ts, pol, H, W = _build_store(ev)
    feat = _feat_chunk(store, big, xs, ys, ts, pol, np.asarray(query_idx), H, W)
    return _forward(feat.astype(np.float32), w)


def predict_stream(w: dict, ev, chunk: int = 60000) -> np.ndarray:
    """MLPF P(signal) for EVERY event in `ev` (time-sorted), chunked for memory."""
    store, big, xs, ys, ts, pol, H, W = _build_store(ev)
    N = len(ev); out = np.empty(N, np.float32)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(s + chunk, N))
        feat = _feat_chunk(store, big, xs, ys, ts, pol, idx, H, W)
        out[idx] = _forward(feat.astype(np.float32), w)
    return out
