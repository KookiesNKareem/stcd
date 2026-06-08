"""Direction-selective motion-coincidence front-end (the novel contribution).

A spiking realization of the **Hassenstein–Reichardt correlator** / **Barlow–Levick**
motion detector — the canonical neuromorphic motion-sensing circuit found in
insect optic lobe and vertebrate retina (direction-selective ganglion cells).

Where the Background Activity Filter (and our plain coincidence front-end) keep an
event if it merely *has neighbours*, this layer keeps an event only if it is part
of **coherent motion**. For each of 8 directions it forms an *opponent* response:

    motion_d = ReLU( leak-trace at the PREFERRED-side neighbour
                     − leak-trace at the NULL-side neighbour )

so a real moving edge (asymmetric: upstream fired just before, downstream hasn't)
gives a large response in its direction of travel, while a **static flickering
cluster** or **isolated noise** — which has symmetric or no neighbour activity —
cancels to ~0. Per-event motion support = max over directions. This rejects the
spatially-correlated noise (bursts, hot blobs, flicker) that defeats pure
neighbour-counting, while still rejecting uncorrelated noise.

The "leak trace" is the same LIF membrane leak used elsewhere (the delay line of
the correlator); thresholds and τ are learnable, so the whole thing trains by
surrogate gradient like the rest of the front-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .events import Events, TimeGrid, events_to_tensor
from .frontend import TemporalLeak

Tensor = torch.Tensor

# 8 compass directions (dy, dx); each is a preferred direction of motion.
DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _shift(t: Tensor, dy: int, dx: int) -> Tensor:
    """Zero-filled spatial shift: ``out[:, y, x, :] = t[:, y-dy, x-dx, :]`` (H,W are
    dims 1,2). Brings a neighbour's value to the centre cell."""
    t = torch.roll(t, shifts=(dy, dx), dims=(1, 2))
    if dy > 0:
        t[:, :dy] = 0
    elif dy < 0:
        t[:, dy:] = 0
    if dx > 0:
        t[:, :, :dx] = 0
    elif dx < 0:
        t[:, :, dx:] = 0
    return t


@dataclass
class MotionConfig:
    tau: float = 8e-3        # leak-trace time-constant (the correlator delay line)
    theta: float = 0.5       # motion-energy threshold to forward an event
    dt: float = 5e-3         # time-bin width
    dirs: tuple = tuple(DIRS8)


class DirectionSelectiveFrontEnd(nn.Module):
    """Reichardt/Barlow–Levick opponent motion detector used as an event denoiser.

    Mirrors ``SpikingFrontEnd``'s interface (``score_events`` / ``filter``) so it
    drops into the same evaluation. Polarity channels are kept separate (motion of
    ON and OFF edges is detected independently)."""

    def __init__(self, cfg: MotionConfig | None = None):
        super().__init__()
        self.cfg = cfg or MotionConfig()
        self.leak = TemporalLeak(tau=self.cfg.tau)
        # learnable per-direction gain (starts uniform) and threshold
        self.dir_gain = nn.Parameter(torch.ones(len(self.cfg.dirs)))
        self.raw_theta = nn.Parameter(torch.tensor(float(self.cfg.theta)))

    @property
    def theta(self) -> Tensor:
        return torch.relu(self.raw_theta)

    def motion_support(self, tensor: Tensor, dt: float) -> Tensor:
        """Per-cell motion energy ``[P,H,W,T]`` = max over directions of the
        opponent (preferred − null) leak-trace correlation."""
        trace = self.leak(tensor, dt)                 # decaying activity per cell
        resp = []
        for g, (dy, dx) in zip(self.dir_gain, self.cfg.dirs):
            pref = _shift(trace, dy, dx)              # upstream (preferred) neighbour
            null = _shift(trace, -dy, -dx)            # downstream (null) neighbour
            resp.append(torch.relu(g * (pref - null)))
        return torch.stack(resp, dim=0).amax(dim=0)   # [P,H,W,T]

    def forward(self, tensor: Tensor, dt: float | None = None) -> Tensor:
        dt = self.cfg.dt if dt is None else dt
        return self.motion_support(tensor, dt)

    # -- per-event interface (matches SpikingFrontEnd) ----------------------- #
    def _cells(self, ev: Events, grid: TimeGrid):
        p = torch.from_numpy(ev.ps).long()
        oy = torch.from_numpy(ev.ys).long()
        ox = torch.from_numpy(ev.xs).long()
        tb = torch.from_numpy(grid.bin_index(ev.ts)).long()
        return p, oy, ox, tb

    @torch.no_grad()
    def score_events(self, ev: Events, grid: TimeGrid | None = None) -> np.ndarray:
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        support = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._cells(ev, grid)
        return support[p, oy, ox, tb].cpu().numpy()

    def event_support(self, tensor: Tensor, grid: TimeGrid, ev: Events) -> Tensor:
        """Differentiable per-event motion support (for surrogate-gradient training)."""
        support = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._cells(ev, grid)
        return support[p, oy, ox, tb]

    @torch.no_grad()
    def filter(self, ev: Events, grid: TimeGrid | None = None):
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        support = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._cells(ev, grid)
        kept = (support[p, oy, ox, tb] >= float(self.theta)).cpu().numpy()
        return kept, ev.select(kept)
