"""Rendering and plotting helpers for figures (matplotlib, headless Agg)."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .events import Events  # noqa: E402


def render_event_image(ev: Events, t_range: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Accumulate events into an RGB image: ON→red, OFF→blue, on a white field.

    Pixels are coloured by which polarity dominates, with intensity scaled by
    event count, so cleaner streams look visibly cleaner.
    """
    if t_range is not None and len(ev):
        m = (ev.ts >= t_range[0]) & (ev.ts < t_range[1])
        ev = ev.select(m)
    on = np.zeros((ev.H, ev.W), np.float64)
    off = np.zeros((ev.H, ev.W), np.float64)
    if len(ev):
        np.add.at(on, (ev.ys[ev.ps == 1], ev.xs[ev.ps == 1]), 1.0)
        np.add.at(off, (ev.ys[ev.ps == 0], ev.xs[ev.ps == 0]), 1.0)
    img = np.ones((ev.H, ev.W, 3), np.float64)
    scale = np.percentile(np.concatenate([on.ravel(), off.ravel()]), 99) or 1.0
    on_n = np.clip(on / scale, 0, 1)
    off_n = np.clip(off / scale, 0, 1)
    on_dom = on >= off
    # ON-dominant -> fade green+blue toward 0 (=> red); OFF-dominant -> fade red+green
    img[..., 1] -= np.where(on_dom, on_n, off_n)        # green always fades
    img[..., 2] -= np.where(on_dom, on_n, 0.0)          # blue fades for ON
    img[..., 0] -= np.where(on_dom, 0.0, off_n)         # red fades for OFF
    return np.clip(img, 0, 1)


def plot_before_after(panels: list[tuple[str, Events]], path: str,
                      t_range: Optional[tuple[float, float]] = None,
                      title: str = "") -> None:
    """Save a row of event-image panels: ``[(label, Events), ...]``."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, ev) in zip(axes, panels):
        ax.imshow(render_event_image(ev, t_range))
        ax.set_title(f"{label}\n({len(ev)} events)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_roc(curves: dict[str, dict], path: str, title: str = "Denoising ROC") -> None:
    """Overlay ROC curves. ``curves`` maps name -> {'fpr','tpr','auc'}."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], lw=2, label=f"{name} (AUC={c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("Noise kept  (FPR = 1 − Noise-Removal)")
    ax.set_ylabel("Signal kept  (TPR = Signal-Retain)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_operating_points(points: dict[str, tuple[float, float]], path: str,
                          title: str = "Signal-Retain vs Noise-Removal") -> None:
    """Scatter of (noise_removal, signal_retain) operating points per method."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (nr, sr) in points.items():
        ax.scatter(nr, sr, s=90, label=name, zorder=3)
        ax.annotate(name, (nr, sr), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Noise Removal (↑ better)")
    ax.set_ylabel("Signal Retain (↑ better)")
    ax.set_title(title)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
