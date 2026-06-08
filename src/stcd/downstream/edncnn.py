"""EDnCNN-lite — a learned (supervised) event denoiser, the SOTA-style baseline.

Following Baldwin et al. (CVPR 2020): classify each event as signal/noise from a
small **local spatiotemporal patch** of event activity around it, with a CNN
trained on labels. This is the strongest, fairest baseline — it is *supervised on
the same labels we test against*, so it has a home-field advantage over our
fixed/unsupervised filter and the classical baselines. We report it honestly as
the learned upper-ish bound.

`extract_patches` builds a `[N, P·Tw, k, k]` tensor (a `k×k` spatial window over
`Tw` time bins, both polarities) per event; `EDnCNNLite` is a small CNN trained
with BCE; `train_eval` does a stratified split and returns test-set scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..events import Events, events_to_tensor

Tensor = torch.Tensor


def extract_patches(ev: Events, k: int = 7, Tw: int = 5, dt: float = 5e-3) -> Tensor:
    """Per-event local spatiotemporal patch → ``[N, P*Tw, k, k]`` (P·Tw channels)."""
    tensor, grid = events_to_tensor(ev, dt=dt)            # [P,H,W,T]
    P = tensor.shape[0]
    r, rt = k // 2, Tw // 2
    pad = F.pad(tensor, (rt, rt, r, r, r, r))             # pad T, W, H
    ys, xs = ev.ys, ev.xs
    tb = grid.bin_index(ev.ts)
    patches = torch.empty(len(ev), P, k, k, Tw)
    for i in range(len(ev)):
        patches[i] = pad[:, ys[i]:ys[i] + k, xs[i]:xs[i] + k, tb[i]:tb[i] + Tw]
    return patches.permute(0, 1, 4, 2, 3).reshape(len(ev), P * Tw, k, k)


def macs_per_event(in_ch: int = 10, c: int = 16, k: int = 7) -> int:
    """Analytic multiply-accumulates per event for EDnCNNLite (the CNN runs once
    *per event* on its ``[in_ch, k, k]`` patch). conv1 + conv2 + the 1-unit readout."""
    conv1 = k * k * c * (in_ch * 3 * 3)
    conv2 = k * k * c * (c * 3 * 3)
    fc = c * 1
    return conv1 + conv2 + fc


class EDnCNNLite(nn.Module):
    def __init__(self, in_ch: int, c: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@dataclass
class EDnCNNConfig:
    k: int = 7
    Tw: int = 5
    dt: float = 5e-3
    epochs: int = 20
    lr: float = 2e-3
    batch: int = 256
    max_events: int = 20000      # balanced subsample per scene (speed)


def train_eval(ev: Events, labels: np.ndarray, cfg: EDnCNNConfig | None = None,
               device: str = "cpu", seed: int = 0):
    """Balanced subsample → stratified 50/50 split → train EDnCNN-lite → return
    ``(test_scores, test_labels, test_index)`` for AUC + fair comparison."""
    cfg = cfg or EDnCNNConfig()
    rng = np.random.default_rng(seed)
    labels = labels.astype(bool)
    pos, neg = np.where(labels)[0], np.where(~labels)[0]
    m = min(len(pos), len(neg), cfg.max_events // 2)
    sel = np.concatenate([rng.choice(pos, m, replace=False),
                          rng.choice(neg, m, replace=False)])
    sel = np.sort(sel)
    mask = np.zeros(len(ev), dtype=bool); mask[sel] = True
    sub = ev.select(mask)                 # sub-events in ascending-index (= sel) order
    sub_labels = labels[sel]

    X = extract_patches(sub, cfg.k, cfg.Tw, cfg.dt)
    y = torch.from_numpy(sub_labels.astype(np.float32))
    n = len(y)
    perm = torch.from_numpy(rng.permutation(n))
    tr, te = perm[: n // 2], perm[n // 2:]

    torch.manual_seed(seed)
    model = EDnCNNLite(in_ch=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    for _ in range(cfg.epochs):
        idx = torch.randperm(len(tr))
        for b in range(0, len(tr), cfg.batch):
            j = idx[b:b + cfg.batch]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(Xtr[j]), ytr[j])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X[te].to(device))).cpu().numpy()
    return scores, sub_labels[te.numpy()], sel[te.numpy()]
