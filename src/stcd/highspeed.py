"""High-speed tracking: event-based vs frame-based localisation of a fast object.

Core of the motion-blur benchmark (used by ``scripts/run_highspeed_demo.py`` and
tests). A small disk orbits at frequency ``omega``; a frame camera integrates over
its exposure (blur) and samples at ``cam_fps`` (Nyquist = cam_fps/2), while the
event stream resolves the motion finely. We localise the object both ways and
compare RMSE to the known trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import Events
from .synth import SynthConfig, _events_from_video


@dataclass
class HighSpeedConfig:
    H: int = 128
    W: int = 128
    duration: float = 0.3
    fps_render: int = 2000       # fine GT / event-generation rate
    cam_fps: int = 30            # frame camera under test
    dt_event: float = 2e-3       # event localisation window
    radius: int = 4              # object radius (px)
    contrast: float = 0.15

    @property
    def R(self) -> float:
        return 0.28 * self.H     # orbit radius


def render_orbit(omega: float, cfg: HighSpeedConfig):
    """Return (video[F,H,W], gt_centers[F,2], times[F]) for an orbiting disk."""
    F = int(cfg.duration * cfg.fps_render)
    video = np.full((F, cfg.H, cfg.W), 0.5, np.float32)
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    cx0, cy0 = cfg.W / 2, cfg.H / 2
    centers = np.zeros((F, 2))
    for f in range(F):
        t = f / cfg.fps_render
        cx = cx0 + cfg.R * np.cos(2 * np.pi * omega * t)
        cy = cy0 + cfg.R * np.sin(2 * np.pi * omega * t)
        video[f][(xx - cx) ** 2 + (yy - cy) ** 2 <= cfg.radius ** 2] = 1.0
        centers[f] = (cx, cy)
    return video, centers, np.arange(F) / cfg.fps_render


def event_localize(ev: Events, cfg: HighSpeedConfig) -> np.ndarray:
    """Centroid of events per ``dt_event`` window → (t, x, y) estimates."""
    est, t = [], 0.0
    while t < cfg.duration:
        m = (ev.ts >= t) & (ev.ts < t + cfg.dt_event)
        if m.sum() >= 3:
            est.append((t + cfg.dt_event / 2, ev.xs[m].mean(), ev.ys[m].mean()))
        t += cfg.dt_event
    return np.array(est) if est else np.zeros((0, 3))


def frame_localize(video: np.ndarray, times: np.ndarray, cfg: HighSpeedConfig) -> np.ndarray:
    """Integrate over each exposure (blur), centroid the object → per-frame (t,x,y)."""
    step = int(cfg.fps_render / cfg.cam_fps)
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    est = []
    for k in range(0, video.shape[0] - step, step):
        w = np.abs(video[k:k + step].mean(0) - 0.5)
        if w.sum() > 1e-6:
            est.append((times[k + step // 2],
                        (xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()))
    return np.array(est) if est else np.zeros((0, 3))


def localization_rmse(est: np.ndarray, omega: float, cfg: HighSpeedConfig) -> float:
    if len(est) == 0:
        return float("nan")
    cx0, cy0 = cfg.W / 2, cfg.H / 2
    gx = cx0 + cfg.R * np.cos(2 * np.pi * omega * est[:, 0])
    gy = cy0 + cfg.R * np.sin(2 * np.pi * omega * est[:, 0])
    return float(np.sqrt(((est[:, 1] - gx) ** 2 + (est[:, 2] - gy) ** 2).mean()))


def run_speed(omega: float, cfg: HighSpeedConfig, rng):
    """Render at one orbit frequency, localise both ways. Returns a dict of results."""
    video, _, times = render_orbit(omega, cfg)
    ev = _events_from_video(video, SynthConfig(H=cfg.H, W=cfg.W, fps=cfg.fps_render,
                                               contrast_threshold=cfg.contrast), rng)
    return {
        "omega": omega,
        "event_rmse": localization_rmse(event_localize(ev, cfg), omega, cfg),
        "frame_rmse": localization_rmse(frame_localize(video, times, cfg), omega, cfg),
        "events": ev,
        "blurred_frame": np.abs(video[:int(cfg.fps_render / cfg.cam_fps)].mean(0) - 0.5),
    }
