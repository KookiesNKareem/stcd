"""Downstream reconstruction head (E2VID/FireNet-style).

Turns an event stream into intensity video so we can show the qualitative
"cleaner events -> cleaner video" result of the front-end. The representation and
windowing follow rpg_e2vid: events in a temporal window become a ``num_bins``
voxel grid (bilinear in time), normalised, and fed to a recurrent reconstructor
that emits one frame per window.

Two reconstructors:
  * ``FireNetReconstructor`` — the pretrained Scheerlinck et al. (WACV 2020) net
    (loaded from the fetched checkpoint; see ``firenet.py``).
  * ``IntegratorReconstructor`` — a weights-free leaky event-integration baseline,
    so a reconstruction figure is always available even without the checkpoint.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..events import Events

Tensor = torch.Tensor


def events_to_voxel_grid(ev: Events, num_bins: int, t0: float, t1: float,
                         device: str = "cpu") -> Tensor:
    """E2VID voxel grid ``[num_bins, H, W]`` over window ``[t0, t1)`` with
    bilinear temporal interpolation. Polarity is mapped to ±1."""
    H, W = ev.H, ev.W
    voxel = np.zeros(num_bins * H * W, dtype=np.float32)
    m = (ev.ts >= t0) & (ev.ts < t1)
    if m.any():
        xs, ys = ev.xs[m].astype(np.int64), ev.ys[m].astype(np.int64)
        ts, ps = ev.ts[m].astype(np.float64), ev.ps[m]
        pol = np.where(ps > 0, 1.0, -1.0).astype(np.float32)
        dt = (t1 - t0) or 1.0
        tn = (num_bins - 1) * (ts - t0) / dt          # in [0, num_bins-1]
        ti = np.floor(tn).astype(np.int64)
        frac = (tn - ti).astype(np.float32)
        base = ys * W + xs
        for tb, val in ((ti, pol * (1 - frac)), (ti + 1, pol * frac)):
            valid = (tb >= 0) & (tb < num_bins)
            np.add.at(voxel, (tb[valid] * H * W + base[valid]), val[valid])
    grid = torch.from_numpy(voxel).view(num_bins, H, W)
    return grid.to(device)


def normalize_voxel(voxel: Tensor) -> Tensor:
    """Zero-mean / unit-std normalisation over non-zero cells (rpg_e2vid)."""
    nz = voxel != 0
    if nz.any():
        mean = voxel[nz].mean()
        std = voxel[nz].std()
        if std > 0:
            voxel = torch.where(nz, (voxel - mean) / std, voxel)
    return voxel


def reconstruct_video(
    model: Callable,
    ev: Events,
    num_bins: int = 5,
    window_dt: float = 0.02,
    device: str = "cpu",
    reset_state: bool = True,
) -> list[np.ndarray]:
    """Run a recurrent reconstructor over sliding windows; return a list of
    intensity frames (each ``[H, W]`` float in ~[0,1])."""
    if len(ev) == 0:
        return []
    t_start, t_end = float(ev.ts.min()), float(ev.ts.max())
    frames, state = [], None
    t = t_start
    if hasattr(model, "reset") and reset_state:
        model.reset()
    while t < t_end:
        voxel = normalize_voxel(events_to_voxel_grid(ev, num_bins, t, t + window_dt, device))
        out = model(voxel.unsqueeze(0))   # [1, num_bins, H, W] -> [1,1,H,W] or [H,W]
        frame = _to_frame(out)
        frames.append(frame)
        t += window_dt
    return frames


def _to_frame(out) -> np.ndarray:
    if isinstance(out, torch.Tensor):
        out = out.detach().float().cpu().squeeze()
        return out.numpy()
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Weights-free baseline reconstructor
# --------------------------------------------------------------------------- #
class IntegratorReconstructor:
    """Leaky integration of signed events into an intensity estimate. No training;
    a sensible fallback so we always have a reconstruction to show."""

    def __init__(self, H: int, W: int, leak: float = 0.92, gain: float = 0.25,
                 device: str = "cpu"):
        self.H, self.W, self.leak, self.gain = H, W, leak, gain
        self.device = device
        self.state = torch.zeros(H, W, device=device)

    def reset(self) -> None:
        self.state = torch.zeros(self.H, self.W, device=self.device)

    def __call__(self, voxel: Tensor) -> Tensor:
        # voxel: [1, num_bins, H, W]; collapse polarity-signed bins to an increment
        inc = voxel.squeeze(0).sum(dim=0)
        self.state = self.leak * self.state + self.gain * inc
        return torch.sigmoid(self.state)


def reference_frames_at(video: np.ndarray, fps: float, times: np.ndarray) -> np.ndarray:
    """Sample a rendered reference video ``[F,H,W]`` at the given times (s)."""
    idx = np.clip((times * fps).astype(np.int64), 0, video.shape[0] - 1)
    return video[idx]
