"""Downstream recognition head: a small spiking CNN classifier.

Used to answer the proposal's downstream question for the *recognition* task: does
running the spiking front-end first improve classification under noise? We build
a synthetic 2-class task (vertical-bar vs horizontal-bar motion), heavily
corrupted by background-activity noise, and compare a classifier trained/tested
on **raw** event tensors vs on **front-end-filtered** tensors.

The classifier is a 2-layer conv SNN: each layer is a leaky integrate-and-fire
neuron unrolled over the time bins (shared surrogate gradient), with a
spike-count linear readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..events import Events, TimeGrid, events_to_tensor
from ..frontend import spike
from ..synth import SynthConfig, generate

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Synthetic recognition dataset
# --------------------------------------------------------------------------- #
def make_dataset(
    n_per_class: int,
    H: int = 64,
    W: int = 64,
    duration: float = 0.12,
    noise_rate_hz: float = 5.0,
    seed: int = 0,
) -> tuple[list[Events], np.ndarray]:
    """Two classes by the motion *axis* of a single small disk:
    0 = horizontal motion, 1 = vertical motion.

    The disk is small, so its real events are *sparse* and easily buried by
    background activity — the regime where denoising actually helps a classifier.
    The class is the trajectory orientation (a horizontal vs vertical streak)."""
    # class 0: horizontal motion (vy≈0); class 1: vertical motion (vx≈0)
    axes = [
        dict(vx_range=(-0.7, 0.7), vy_range=(-0.03, 0.03)),
        dict(vx_range=(-0.03, 0.03), vy_range=(-0.7, 0.7)),
    ]
    streams, labels = [], []
    for cls, ax in enumerate(axes):
        for i in range(n_per_class):
            cfg = SynthConfig(H=H, W=W, duration=duration, fps=2000,
                              contrast_threshold=0.15, noise_rate_hz=noise_rate_hz,
                              scene="disks", num_objects=1,
                              size_range=(0.04, 0.07),
                              seed=seed + cls * 10000 + i, **ax)
            streams.append(generate(cfg))
            labels.append(cls)
    return streams, np.array(labels, dtype=np.int64)


def stack_tensors(streams: list[Events], dt: float, duration: float,
                  H: int, W: int) -> Tensor:
    """Bin every stream onto a *shared* time grid -> batch tensor [B,P,H,W,T]."""
    T = max(1, int(round(duration / dt)))
    grid = TimeGrid(t0=0.0, dt=dt, T=T)
    tens = [events_to_tensor(ev, grid=grid)[0] for ev in streams]
    return torch.stack(tens, dim=0)


# --------------------------------------------------------------------------- #
# Spiking CNN classifier
# --------------------------------------------------------------------------- #
class _ConvLIF(nn.Module):
    def __init__(self, c_in, c_out, stride, alpha=0.9, theta=1.0, beta=10.0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.alpha, self.theta, self.beta = alpha, theta, beta

    def forward(self, x_t, v):  # x_t: [B,c_in,H,W]
        cur = self.conv(x_t)
        v = self.alpha * v + cur if v is not None else cur
        s = spike(v - self.theta, self.beta)
        v = v - s * self.theta
        return s, v


class SNNClassifier(nn.Module):
    """2-layer conv-LIF SNN. The readout keeps coarse *spatial* structure
    (adaptive-pool to 4×4 then flatten) rather than global-pooling it away —
    essential for distinguishing classes by *where* activity is (car ROI vs
    spread-out background, or trajectory orientation)."""

    def __init__(self, n_classes=2, c1=16, c2=32):
        super().__init__()
        self.l1 = _ConvLIF(2, c1, stride=2)
        self.l2 = _ConvLIF(c1, c2, stride=2)
        # LazyLinear infers the flattened feature size on first forward, so the
        # full spatial layout of accumulated spikes feeds the readout (MPS-safe,
        # no adaptive pooling).
        self.readout = nn.LazyLinear(n_classes)

    def forward(self, x: Tensor) -> Tensor:  # x: [B,P,H,W,T]
        B, P, H, W, T = x.shape
        v1 = v2 = None
        acc = None
        for t in range(T):
            s1, v1 = self.l1(x[..., t], v1)
            s2, v2 = self.l2(s1, v2)
            acc = s2 if acc is None else acc + s2   # accumulate spikes, keep [B,C,h,w]
        return self.readout((acc / T).flatten(1))


@dataclass
class RecogConfig:
    epochs: int = 25
    lr: float = 1e-3
    batch: int = 16


def train_classifier(x: Tensor, y: Tensor, cfg: RecogConfig,
                     device: str = "cpu", seed: int = 0) -> SNNClassifier:
    torch.manual_seed(seed)
    model = SNNClassifier(n_classes=int(y.max().item()) + 1).to(device)
    with torch.no_grad():                       # materialise LazyLinear params
        model(x[:2].to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = x.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            logits = model(x[idx].to(device))
            loss = F.cross_entropy(logits, y[idx].to(device))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model: SNNClassifier, x: Tensor, y: Tensor, device: str = "cpu") -> float:
    model.eval()
    preds = model(x.to(device)).argmax(dim=1).cpu()
    return float((preds == y).float().mean())
