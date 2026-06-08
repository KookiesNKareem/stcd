"""Unsupervised STDP learning of the denoising front-end (the novel contribution).

Instead of hand-tuning the spatial filter (BAF) or training it with supervised
surrogate-gradient backprop, we learn it with **Spike-Timing-Dependent Plasticity**
— the canonical neuromorphic, *local*, *label-free* learning rule.

Setup: each output cell is a spiking (LIF-trace) neuron whose ``k×k`` spatial
receptive field ``W`` (depthwise, ON/OFF separate) is plastic. Pre-synaptic
eligibility traces (leaky-integrated input events) carry recent neighbour
activity. On each post-synaptic spike, STDP **potentiates** the weights of inputs
whose trace is high *at the moment of firing* (they fired just before → causal,
pre-before-post) and the per-kernel weight is renormalised (homeostasis), so
uncorrelated (noise) inputs lose relative weight. Threshold adapts to hold a
target firing rate.

The trace-based potentiation ``Σ_t post(t)·pre_trace_patch(t)`` is exactly the
spatial cross-correlation of post-spikes with pre-traces — computed here as the
gradient of ``Σ post·membrane`` w.r.t. ``W`` (a vectorised STDP update, **not** a
supervised loss: no labels are used).

Result: from an uninformative init, STDP converges — with **no labels** — to a
sensible centre-weighted correlation kernel that denoises about as well as the
hand-tuned / supervised filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import Events, TimeGrid, events_to_tensor
from .frontend import TemporalLeak
from . import metrics

Tensor = torch.Tensor


@dataclass
class STDPConfig:
    k: int = 5               # receptive-field size
    tau: float = 8e-3        # pre-synaptic trace time-constant
    dt: float = 5e-3
    theta: float = 0.5       # firing threshold (homeostatically adapted)
    eta: float = 0.02        # STDP learning rate
    target_rate: float = 0.15  # target post-spike fraction (homeostasis)
    epochs: int = 40


class STDPDenoiser(nn.Module):
    """A plastic spatial denoising kernel learned by trace-based STDP."""

    def __init__(self, cfg: STDPConfig | None = None, P: int = 2, seed: int = 0,
                 init: str = "delta"):
        super().__init__()
        self.cfg = cfg or STDPConfig()
        k = self.cfg.k
        self.leak = TemporalLeak(tau=self.cfg.tau)
        if init == "delta":
            # "blind" start: weight only on the centre pixel (no spatial context).
            # STDP must DISCOVER that neighbours carry signal and grow their weights.
            W = torch.full((P, 1, k, k), 1e-3)
            W[:, :, k // 2, k // 2] = 1.0
        elif init == "uniform":
            W = torch.ones(P, 1, k, k)
        else:  # random
            g = torch.Generator().manual_seed(seed)
            W = 0.5 + torch.rand(P, 1, k, k, generator=g)
        self.W = self._normalize(W)
        self.theta = float(self.cfg.theta)

    @staticmethod
    def _normalize(W: Tensor) -> Tensor:
        W = W.clamp(min=0)
        return W / (W.sum(dim=(2, 3), keepdim=True) + 1e-8)   # unit-sum per kernel

    def _traces(self, ev: Events, grid=None):
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        return self.leak(tensor, grid.dt).detach(), grid     # [P,H,W,T]

    def membrane(self, trace: Tensor) -> Tensor:
        P = trace.shape[0]
        z = trace.permute(3, 0, 1, 2).contiguous()           # [T,P,H,W]
        v = F.conv2d(z, self.W, padding=self.cfg.k // 2, groups=P)
        return v.permute(1, 2, 3, 0).contiguous()            # [P,H,W,T]

    # -- unsupervised STDP training ----------------------------------------- #
    def train_unsupervised(self, ev: Events, eval_ev: Events | None = None):
        """Run STDP over an unlabelled stream. If ``eval_ev`` (labelled) is given,
        records held-out AUC each epoch for monitoring (labels NOT used to learn)."""
        trace, _ = self._traces(ev)
        z = trace.permute(3, 0, 1, 2).contiguous()           # [T,P,H,W]
        P = trace.shape[0]
        history = {"epoch": [], "auc": [], "rate": [], "kernel": []}
        for epoch in range(self.cfg.epochs):
            Wv = self.W.clone().requires_grad_(True)
            v = F.conv2d(z, Wv, padding=self.cfg.k // 2, groups=P)
            post = (v >= self.theta).float()
            rate = float(post.mean())
            # STDP potentiation = d(Σ post·membrane)/dW = Σ_t post·pre_trace_patch
            (post.detach() * v).sum().backward()
            dW = Wv.grad
            self.W = self._normalize(self.W + self.cfg.eta * dW)
            # homeostatic threshold to hold the target firing rate
            self.theta += 0.5 * (rate - self.cfg.target_rate)
            history["epoch"].append(epoch)
            history["rate"].append(rate)
            history["kernel"].append(self.kernel().copy())
            history["auc"].append(self._auc(eval_ev) if eval_ev is not None else float("nan"))
        return history

    @torch.no_grad()
    def _auc(self, ev: Events) -> float:
        return metrics.roc(self.score_events(ev), ev.labels)["auc"]

    # -- per-event interface (matches the other front-ends) ----------------- #
    @torch.no_grad()
    def score_events(self, ev: Events, grid: TimeGrid | None = None) -> np.ndarray:
        trace, grid = self._traces(ev, grid)
        m = self.membrane(trace)
        p = torch.from_numpy(ev.ps).long()
        oy = torch.from_numpy(ev.ys).long()
        ox = torch.from_numpy(ev.xs).long()
        tb = torch.from_numpy(grid.bin_index(ev.ts)).long()
        return m[p, oy, ox, tb].cpu().numpy()

    @torch.no_grad()
    def filter(self, ev: Events, grid: TimeGrid | None = None):
        s = self.score_events(ev, grid)
        kept = s >= self.theta
        return kept, ev.select(kept)

    def kernel(self) -> np.ndarray:
        """Learned receptive field(s), shape [P, k, k]."""
        return self.W.squeeze(1).cpu().numpy()


@dataclass
class CompetitiveSTDPConfig:
    n_features: int = 8
    k: int = 7
    tau: float = 8e-3
    dt: float = 5e-3
    theta: float = 0.4
    eta: float = 0.04
    target_rate: float = 0.08      # total post-fire fraction (shared across features)
    theta_lr: float = 0.2
    epochs: int = 60


class CompetitiveSTDPDenoiser(nn.Module):
    """A bank of STDP neurons with **winner-take-all lateral inhibition**.

    At each location/time only the highest-responding feature may fire and learn,
    so the neurons are forced to *specialise* — partitioning the input and
    self-organising into a set of **diverse oriented edge detectors** (the classic
    Masquelier / Diehl–Cook emergent-receptive-field result), with no labels.
    Per-feature homeostatic thresholds keep every neuron in use (encourages
    diversity). As a denoiser, an event's score is the *best-matching* feature's
    response — i.e. "does this event fit any learned edge pattern?".
    """

    def __init__(self, cfg: CompetitiveSTDPConfig | None = None, P: int = 2, seed: int = 0):
        super().__init__()
        self.cfg = cfg or CompetitiveSTDPConfig()
        N, k = self.cfg.n_features, self.cfg.k
        self.leak = TemporalLeak(tau=self.cfg.tau)
        g = torch.Generator().manual_seed(seed)
        W = torch.rand(N, P, k, k, generator=g)      # random → breaks symmetry
        self.W = self._normalize(W)
        self.theta = torch.full((N,), float(self.cfg.theta))

    @staticmethod
    def _normalize(W: Tensor) -> Tensor:
        W = W.clamp(min=0)
        return W / (W.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)

    def _ztrace(self, ev: Events, grid=None):
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        trace = self.leak(tensor, grid.dt).detach()
        return trace.permute(3, 0, 1, 2).contiguous(), grid     # [T,P,H,W]

    def _responses(self, z: Tensor) -> Tensor:
        return F.conv2d(z, self.W, padding=self.cfg.k // 2)      # [T,N,H,W]

    def train_unsupervised(self, ev: Events, eval_ev: Events | None = None):
        z, _ = self._ztrace(ev)
        N = self.cfg.n_features
        history = {"epoch": [], "auc": []}
        for epoch in range(self.cfg.epochs):
            Wv = self.W.clone().requires_grad_(True)
            m = F.conv2d(z, Wv, padding=self.cfg.k // 2)          # [T,N,H,W]
            winner = m.argmax(dim=1, keepdim=True)                # WTA over features
            is_win = torch.zeros_like(m).scatter_(1, winner, 1.0)
            post = is_win * (m >= self.theta.view(1, N, 1, 1)).float()
            rate_n = post.mean(dim=(0, 2, 3))                     # per-feature win rate
            (post.detach() * m).sum().backward()                 # STDP potentiation
            self.W = self._normalize(self.W + self.cfg.eta * Wv.grad)
            # homeostasis: balance feature usage so all neurons stay in play
            self.theta += self.cfg.theta_lr * (rate_n - self.cfg.target_rate / N)
            history["epoch"].append(epoch)
            history["auc"].append(self._auc(eval_ev) if eval_ev is not None else float("nan"))
        return history

    @torch.no_grad()
    def _auc(self, ev: Events) -> float:
        return metrics.roc(self.score_events(ev), ev.labels)["auc"]

    @torch.no_grad()
    def score_events(self, ev: Events, grid: TimeGrid | None = None) -> np.ndarray:
        z, grid = self._ztrace(ev, grid)
        m = self._responses(z).amax(dim=1)                       # [T,H,W] best feature
        m = m.permute(1, 2, 0).contiguous()                      # [H,W,T]
        oy = torch.from_numpy(ev.ys).long()
        ox = torch.from_numpy(ev.xs).long()
        tb = torch.from_numpy(grid.bin_index(ev.ts)).long()
        return m[oy, ox, tb].cpu().numpy()

    @torch.no_grad()
    def filter(self, ev: Events, grid: TimeGrid | None = None):
        s = self.score_events(ev, grid)
        kept = s >= float(self.theta.min())
        return kept, ev.select(kept)

    @torch.no_grad()
    def encode(self, ev: Events, pool: int = 4) -> np.ndarray:
        """Encode a clip as a fixed feature vector: each neuron's mean response,
        coarse-pooled to ``pool×pool`` → length ``N·pool·pool``. The unsupervised
        feature representation handed to a downstream classifier."""
        z, _ = self._ztrace(ev)
        m = self._responses(z).clamp(min=0).mean(dim=0)      # [N,H,W]
        m = F.adaptive_avg_pool2d(m.unsqueeze(0), (pool, pool))[0]
        return m.flatten().cpu().numpy()

    def kernels(self) -> np.ndarray:
        """Learned feature kernels, shape [N, P, k, k]."""
        return self.W.cpu().numpy()


@dataclass
class SpatioTemporalSTDPConfig:
    n_features: int = 8
    k: int = 5                 # spatial receptive field
    n_lags: int = 5            # temporal taps (learnable synaptic delays)
    dt: float = 5e-3
    theta: float = 0.35
    eta: float = 0.04
    target_rate: float = 0.06
    theta_lr: float = 0.2
    epochs: int = 50


class SpatioTemporalSTDP(nn.Module):
    """Competitive STDP over **space *and* time** (learnable delay taps).

    Each neuron's weight ``W[n, p, lag, dy, dx]`` is a small spatio-temporal kernel
    (a 3-D conv): it can learn that an input at neighbour ``(dy,dx)`` ``lag`` bins
    *ago* should precede the cell's spike — i.e. the diagonal space-time signature
    of an edge moving in a particular direction. With winner-take-all competition,
    neurons specialise to **different motion directions** — *direction selectivity
    emerges from STDP*, the way direction-selective cells are thought to develop.
    Input is the raw event tensor (the lags carry the timing; no leak needed).
    """

    def __init__(self, cfg: SpatioTemporalSTDPConfig | None = None, P: int = 2, seed: int = 0):
        super().__init__()
        self.cfg = cfg or SpatioTemporalSTDPConfig()
        N, L, k = self.cfg.n_features, self.cfg.n_lags, self.cfg.k
        g = torch.Generator().manual_seed(seed)
        W = torch.rand(N, P, L, k, k, generator=g)     # [N,P,lag,ky,kx]
        self.W = self._normalize(W)
        self.theta = torch.full((N,), float(self.cfg.theta))

    @staticmethod
    def _normalize(W: Tensor) -> Tensor:
        W = W.clamp(min=0)
        return W / (W.sum(dim=(1, 2, 3, 4), keepdim=True) + 1e-8)

    def _input(self, ev: Events, grid=None):
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)  # [P,H,W,T]
        z = tensor.permute(0, 3, 1, 2).unsqueeze(0)     # [1, P, T, H, W]
        return z, grid

    def _responses(self, z: Tensor) -> Tensor:
        L, k = self.cfg.n_lags, self.cfg.k
        zt = F.pad(z, (k // 2, k // 2, k // 2, k // 2, L - 1, 0))   # causal in T
        return F.conv3d(zt, self.W)                      # [1, N, T, H, W]

    def train_unsupervised(self, ev: Events, eval_ev: Events | None = None):
        z, _ = self._input(ev)
        N, L, k = self.cfg.n_features, self.cfg.n_lags, self.cfg.k
        history = {"epoch": [], "auc": []}
        for epoch in range(self.cfg.epochs):
            Wv = self.W.clone().requires_grad_(True)
            zt = F.pad(z, (k // 2, k // 2, k // 2, k // 2, L - 1, 0))
            m = F.conv3d(zt, Wv)                         # [1,N,T,H,W]
            winner = m.argmax(dim=1, keepdim=True)
            is_win = torch.zeros_like(m).scatter_(1, winner, 1.0)
            post = is_win * (m >= self.theta.view(1, N, 1, 1, 1)).float()
            rate_n = post.mean(dim=(0, 2, 3, 4))
            (post.detach() * m).sum().backward()
            self.W = self._normalize(self.W + self.cfg.eta * Wv.grad)
            self.theta += self.cfg.theta_lr * (rate_n - self.cfg.target_rate / N)
            history["epoch"].append(epoch)
            history["auc"].append(self._auc(eval_ev) if eval_ev is not None else float("nan"))
        return history

    @torch.no_grad()
    def _auc(self, ev: Events) -> float:
        return metrics.roc(self.score_events(ev), ev.labels)["auc"]

    @torch.no_grad()
    def feature_responses(self, ev: Events, grid=None) -> Tensor:
        """Per-feature response volume ``[N,H,W,T]`` (max-pooled is the denoise score)."""
        z, grid = self._input(ev, grid)
        m = self._responses(z)[0]                        # [N,T,H,W]
        return m.permute(0, 2, 3, 1).contiguous(), grid  # [N,H,W,T]

    @torch.no_grad()
    def score_events(self, ev: Events, grid: TimeGrid | None = None) -> np.ndarray:
        m, grid = self.feature_responses(ev, grid)
        best = m.amax(dim=0)                             # [H,W,T]
        oy = torch.from_numpy(ev.ys).long()
        ox = torch.from_numpy(ev.xs).long()
        tb = torch.from_numpy(grid.bin_index(ev.ts)).long()
        return best[oy, ox, tb].cpu().numpy()

    @torch.no_grad()
    def total_response_per_feature(self, ev: Events) -> np.ndarray:
        """Total activation of each neuron over a clip — used for direction tuning."""
        z, _ = self._input(ev)
        m = self._responses(z)[0]                        # [N,T,H,W]
        return m.clamp(min=0).sum(dim=(1, 2, 3)).cpu().numpy()

    def kernels(self) -> np.ndarray:
        """Learned spatio-temporal kernels, shape [N, P, n_lags, k, k]."""
        return self.W.cpu().numpy()
