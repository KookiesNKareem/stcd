"""Joint surrogate-gradient optimisation of the front-end parameters.

This is the contribution over a hand-tuned BAF: instead of picking a single time
window by hand, we *optimise* {leak τ, threshold θ, (learned spatial/temporal
weights)} against a downstream-aligned denoising objective.

Loss: a per-event soft classification. Each event's support is the membrane
potential ``v_i`` at its pooled cell; we pass ``(v_i − θ)/temp`` through a sigmoid
to get a differentiable keep-probability and apply BCE against the ground-truth
label (signal=1, noise=0). Gradients flow — via the SuperSpike surrogate inside
the LIF — to τ, θ and any learned kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .events import Events, events_to_tensor
from .frontend import SpikingFrontEnd, FrontEndConfig
from . import metrics


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 0.05
    temp: float = 0.5            # logit temperature for the soft keep-probability
    weight_decay: float = 0.0
    verbose_every: int = 25


def train_frontend(
    ev: Events,
    fe: SpikingFrontEnd | None = None,
    init: FrontEndConfig | None = None,
    tcfg: TrainConfig | None = None,
) -> tuple[SpikingFrontEnd, dict]:
    """Optimise a front-end on a labelled stream. Returns the trained model and a
    history dict (loss / AUC / F1 per logged epoch)."""
    if ev.labels is None:
        raise ValueError("training requires ground-truth labels")
    tcfg = tcfg or TrainConfig()
    fe = fe or SpikingFrontEnd(init or FrontEndConfig())

    tensor, grid = events_to_tensor(ev, dt=fe.cfg.dt)
    target = torch.from_numpy(ev.labels.astype(np.float32))
    opt = torch.optim.Adam(fe.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    history = {"epoch": [], "loss": [], "auc": [], "f1": [], "tau": [], "theta": []}
    for epoch in range(tcfg.epochs):
        opt.zero_grad()
        v = fe.event_membrane(tensor, grid, ev)            # differentiable
        logits = (v - fe.lif.theta) / tcfg.temp
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        opt.step()

        if (tcfg.verbose_every and epoch % tcfg.verbose_every == 0) or epoch == tcfg.epochs - 1:
            with torch.no_grad():
                scores = v.detach().cpu().numpy()
                auc = metrics.roc(scores, ev.labels)["auc"]
                kept = scores >= float(fe.lif.theta)
                f1 = metrics.evaluate_filter(ev.labels, kept).f1
            history["epoch"].append(epoch)
            history["loss"].append(float(loss.detach()))
            history["auc"].append(float(auc))
            history["f1"].append(float(f1))
            history["tau"].append(float(fe.lif.tau.detach()))
            history["theta"].append(float(fe.lif.theta.detach()))
            if tcfg.verbose_every:
                print(f"  epoch {epoch:3d}  loss={float(loss):.4f}  "
                      f"AUC={auc:.4f}  F1={f1:.4f}  "
                      f"tau={float(fe.lif.tau)*1e3:.2f}ms  theta={float(fe.lif.theta):.3f}")
    return fe, history
