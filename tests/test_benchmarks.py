"""Tests for the high-speed (motion-blur) benchmark and real-data denoising."""

import os

import numpy as np
import pytest

from stcd.highspeed import HighSpeedConfig, run_speed

_REC = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20",
                    "2_mat", "conference-2019_11_04_14_32_45.mat")


def test_frame_fails_event_wins_at_high_speed():
    cfg = HighSpeedConfig(H=64, W=64, duration=0.2, fps_render=1000, cam_fps=30)
    rng = np.random.default_rng(0)
    slow = run_speed(2.0, cfg, rng)
    fast = run_speed(30.0, cfg, rng)        # well past the 15 Hz frame Nyquist
    # event-based stays accurate; frame-based degrades badly with speed
    assert fast["event_rmse"] < 0.15 * cfg.H
    assert fast["frame_rmse"] > 3 * fast["event_rmse"]
    assert fast["frame_rmse"] > slow["frame_rmse"]


@pytest.mark.skipif(not os.path.isfile(_REC), reason="DVSNOISE20 recording not present")
def test_real_denoising_frontend_beats_baselines():
    from stcd.datasets import dvsnoise20 as dv
    from stcd.frontend import SpikingFrontEnd, FrontEndConfig
    from stcd.baselines import baf_scores, knoise_scores
    from stcd import metrics
    ev, fts, aps, _ = dv.load_full(_REC)
    w = ev.select((ev.ts >= 6.0) & (ev.ts < 6.5))      # an active window
    sig, val = dv.aps_motion_labels(w, fts, aps)
    lw, lab = w.select(val), sig[val]
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    auc_fe = metrics.roc(fe.score_events(lw), lab)["auc"]
    auc_baf = metrics.roc(baf_scores(lw, 2e-3), lab)["auc"]
    auc_kn = metrics.roc(knoise_scores(lw, 2e-3), lab)["auc"]
    assert auc_fe > 0.80              # real-data: front-end is a strong denoiser
    assert auc_fe > auc_baf > auc_kn  # and beats both classical baselines
