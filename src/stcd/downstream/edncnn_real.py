"""The REAL pretrained EDnCNN (Baldwin et al., CVPR 2020), loaded from the authors'
``allData_v8_preTrained.mat`` for a fair head-to-head vs STCD on identical
events/labels — replacing our ``EDnCNNLite`` proxy.

Architecture (read from the saved net):
  input 25x25x4 (zerocenter: subtract AverageImage)
  -> conv 4->32 (3x3 same) -> BN -> ReLU -> dropout
  -> conv 32->64           -> BN -> ReLU -> dropout
  -> conv 64->128          -> BN -> ReLU -> dropout
  -> FC 80000->1024 -> FC 1024->1024 -> FC 1024->2 -> softmax

Feature (events2FeatML.m): per event, a k=25 (neighborhood=12) spatial window over
``2*depth`` channels = [pos time-surface(depth), neg time-surface(depth)], polarity-
ordered (the event's own polarity first). Surfaces hold time-since-recent events per
pixel; values are clamped to [minTime, maxTime] and log-scaled.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEIGH = 12            # -> 25x25 window
DEPTH = 2             # surface slots per polarity -> 2*DEPTH = 4 channels
MIN_TIME = 150.0      # us
MAX_TIME = 5e6        # us


def real_macs_per_event() -> int:
    """Analytic MACs/event of the *real* pretrained EDnCNN (25x25 kept via 'same'
    padding; the FC head dominates). = 141,290,624 MACs (282,581,248 FLOPs)."""
    s = 25 * 25
    conv1 = s * 32 * (4 * 9)
    conv2 = s * 64 * (32 * 9)
    conv3 = s * 128 * (64 * 9)
    fc1 = (128 * 25 * 25) * 1024
    fc2 = 1024 * 1024
    fc3 = 1024 * 2
    return conv1 + conv2 + conv3 + fc1 + fc2 + fc3


def _refarr(refs, *path):
    g = refs[path[0]]
    for p in path[1:]:
        g = g[p]
    return np.array(g)


def load_real_edncnn(matpath: str) -> nn.Module:
    """Build the EDnCNN and load the pretrained MATLAB weights with correct
    MATLAB->PyTorch conventions (dim reversal from HDF5, conv kh/kw swap, the
    conv->FC (H,W,C)->(C,H,W) flatten remap)."""
    import h5py
    h = h5py.File(matpath, "r"); refs = h["#refs#"]

    def conv_w(key):                      # h5py (out,in,kw,kh) -> torch (out,in,kh,kw)
        return _refarr(refs, key, "Weights", "Value").transpose(0, 1, 3, 2)

    net = EDnCNNReal()
    sd = net.state_dict()
    with torch.no_grad():
        net.avg_image.copy_(torch.tensor(_refarr(refs, "c", "Normalization", "AverageImage"),
                                         dtype=torch.float32))           # (4,25,25)=(C,H,W)
        for i, lk in enumerate(["d", "j", "p"]):                          # conv_1/2/3
            net.convs[i*4].weight.copy_(torch.tensor(conv_w(lk), dtype=torch.float32))
            net.convs[i*4].bias.copy_(torch.tensor(_refarr(refs, lk, "Bias", "Value").ravel(), dtype=torch.float32))
        for i, bk in enumerate(["g", "m", "s"]):                          # batchnorm_1/2/3
            bn = net.convs[i*4 + 1]
            bn.weight.copy_(torch.tensor(_refarr(refs, bk, "Scale", "Value").ravel(), dtype=torch.float32))
            bn.bias.copy_(torch.tensor(_refarr(refs, bk, "Offset", "Value").ravel(), dtype=torch.float32))
            bn.running_mean.copy_(torch.tensor(_refarr(refs, bk, "TrainedMean").ravel(), dtype=torch.float32))
            bn.running_var.copy_(torch.tensor(_refarr(refs, bk, "TrainedVariance").ravel(), dtype=torch.float32))
            bn.eps = float(_refarr(refs, bk, "Epsilon").ravel()[0])
        # fc_1: h5py (1024,128,25,25)=(out,C,W,H) -> (out,C,H,W) -> flatten (C,H,W)
        w1 = _refarr(refs, "v", "Weights", "Value").transpose(0, 1, 3, 2).reshape(1024, -1)
        net.fc1.weight.copy_(torch.tensor(w1, dtype=torch.float32))
        net.fc1.bias.copy_(torch.tensor(_refarr(refs, "v", "Bias", "Value").ravel(), dtype=torch.float32))
        net.fc2.weight.copy_(torch.tensor(_refarr(refs, "y", "Weights", "Value").reshape(1024, 1024), dtype=torch.float32))
        net.fc2.bias.copy_(torch.tensor(_refarr(refs, "y", "Bias", "Value").ravel(), dtype=torch.float32))
        net.fc3.weight.copy_(torch.tensor(_refarr(refs, "B", "Weights", "Value").reshape(2, 1024), dtype=torch.float32))
        net.fc3.bias.copy_(torch.tensor(_refarr(refs, "B", "Bias", "Value").ravel(), dtype=torch.float32))
    net.eval()
    return net


class EDnCNNReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("avg_image", torch.zeros(4, 25, 25))
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(), nn.Identity(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Identity(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Identity())
        self.fc1 = nn.Linear(128 * 25 * 25, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    @torch.no_grad()
    def forward(self, x):                       # x: [N,4,25,25] (log time-surface)
        x = x - self.avg_image
        x = self.convs(x)
        x = x.flatten(1)
        x = self.fc3(self.fc2(self.fc1(x)))     # no activations between FCs (per saved net)
        return torch.softmax(x, dim=1)[:, 1]    # P(signal)


def event_features(ev, query_idx: np.ndarray) -> torch.Tensor:
    """Build EDnCNN's 25x25x4 log-time-surface feature for the events at
    ``query_idx``. ``ev`` must be time-sorted. Returns [len(query), 4, 25, 25]."""
    H, W = ev.H, ev.W
    xs, ys, ts, ps = ev.xs, ev.ys, ev.ts * 1e6, ev.ps    # ts -> microseconds
    # per-pixel, per-polarity sorted event-time lists
    pix = ys.astype(np.int64) * W + xs.astype(np.int64)
    times = {0: {}, 1: {}}
    for pol in (0, 1):
        m = (ps > 0) if pol == 1 else (ps <= 0)
        order = np.argsort(pix[m], kind="stable")
        pp = pix[m][order]; tt = ts[m][order]
        # split into per-pixel arrays
        uniq, start = np.unique(pp, return_index=True)
        for u, s, e in zip(uniq, start, list(start[1:]) + [len(pp)]):
            times[pol][int(u)] = tt[s:e]            # already time-sorted within pixel

    qset = np.asarray(query_idx)
    feat = np.full((len(qset), 4, 25, 25), MAX_TIME, dtype=np.float32)
    for qi, ei in enumerate(qset):
        te = ts[ei]; cx, cy = int(xs[ei]), int(ys[ei]); own = int(ps[ei] > 0)
        for dy in range(-NEIGH, NEIGH + 1):
            py = cy + dy
            if py < 0 or py >= H: continue
            for dx in range(-NEIGH, NEIGH + 1):
                px = cx + dx
                if px < 0 or px >= W: continue
                pidx = py * W + px
                for pol in (0, 1):
                    arr = times[pol].get(pidx)
                    if arr is None: continue
                    k = np.searchsorted(arr, te, side="left")   # events strictly before te
                    n = min(DEPTH, k)
                    # channel block: own polarity first
                    ch0 = 0 if pol == own else DEPTH
                    for d in range(n):
                        age = te - arr[k - 1 - d]
                        feat[qi, ch0 + d, NEIGH + dy, NEIGH + dx] = age
    # scale: clamp, log, subtract log(minTime+1), floor at 0
    feat = np.clip(feat, None, MAX_TIME)
    feat = np.log(feat + 1.0) - np.log(MIN_TIME + 1.0)
    feat[feat < 0] = 0.0
    return torch.tensor(feat, dtype=torch.float32)


# ---- vectorized scoring (for scoring whole streams, e.g. downstream) -------- #
def _build_store(ev):
    """Per-polarity event-time structures sorted by (pixel, time) for fast lookup."""
    H, W = ev.H, ev.W
    xs = ev.xs.astype(np.int64); ys = ev.ys.astype(np.int64)
    ts = (ev.ts * 1e6).astype(np.float64); on = ev.ps > 0
    pix = ys * W + xs
    big = float(ts.max()) + 10.0
    store = {}
    for pol in (0, 1):
        m = on if pol == 1 else ~on
        comp = pix[m].astype(np.float64) * big + ts[m]
        o = np.argsort(comp, kind="stable")
        store[pol] = (comp[o], ts[m][o], pix[m][o])
    return store, big, xs, ys, ts, on, H, W


def _feat_chunk(store, big, xs, ys, ts, on, idx, H, W):
    """Vectorized 25x25x4 log-time-surface features for events `idx` (grouped
    searchsorted via composite (pixel,time) key; identical to event_features)."""
    qx = xs[idx]; qy = ys[idx]; qt = ts[idx]; qon = on[idx].astype(np.int64)
    m = len(idx)
    feat = np.full((m, 4, 25, 25), MAX_TIME, np.float64)
    for dy in range(-NEIGH, NEIGH + 1):
        ny = qy + dy; vy = (ny >= 0) & (ny < H)
        for dx in range(-NEIGH, NEIGH + 1):
            nx = qx + dx; vb = vy & (nx >= 0) & (nx < W)
            if not vb.any():
                continue
            tp = ny * W + nx
            for pol in (0, 1):
                comp_s, ts_s, pix_s = store[pol]
                k = np.searchsorted(comp_s, tp.astype(np.float64) * big + qt, side="left")
                base = np.where(pol == qon, 0, DEPTH)
                for d in range(DEPTH):
                    j = k - 1 - d
                    jc = np.clip(j, 0, len(ts_s) - 1)
                    ok = vb & (j >= 0) & (pix_s[jc] == tp)
                    rows = np.where(ok)[0]
                    feat[rows, (base + d)[rows], NEIGH + dy, NEIGH + dx] = (qt - ts_s[jc])[rows]
    feat = np.clip(feat, None, MAX_TIME)
    feat = np.log(feat + 1.0) - np.log(MIN_TIME + 1.0)
    feat[feat < 0] = 0.0
    return feat


def event_features_bulk(ev, query_idx) -> torch.Tensor:
    """Vectorized equivalent of event_features (build-once); for validation/small sets."""
    store, big, xs, ys, ts, on, H, W = _build_store(ev)
    return torch.tensor(_feat_chunk(store, big, xs, ys, ts, on, np.asarray(query_idx), H, W),
                        dtype=torch.float32)


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"          # Apple-silicon GPU: ~10x faster than CPU on the fc1 bottleneck
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def predict_stream(net, ev, chunk: int = 2000, device: str | None = None) -> np.ndarray:
    """Real EDnCNN P(signal) for EVERY event in `ev` (time-sorted), chunked to bound
    memory. Each chunk materializes a (chunk, 128, 25, 25) conv tensor and a
    (chunk, 80000) fc1 input, so chunk must stay small: 2000 -> ~1.5 GB peak;
    40000 would be ~25 GB. The fc1 weight (328 MB) is memory-bandwidth bound, so we
    auto-use the GPU (MPS/CUDA) when present (~3 kev/s vs ~0.3 kev/s on CPU)."""
    device = device or _auto_device()
    net = net.to(device)
    store, big, xs, ys, ts, on, H, W = _build_store(ev)
    N = len(ev); out = np.empty(N, np.float32)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(s + chunk, N))
        feat = _feat_chunk(store, big, xs, ys, ts, on, idx, H, W)
        out[idx] = net(torch.tensor(feat, dtype=torch.float32, device=device)).cpu().numpy()
        del feat
    return out
