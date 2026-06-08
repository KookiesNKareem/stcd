"""Integration tests for downstream heads, datasets, and energy model.

Tests needing the fetched assets (FireNet checkpoint, N-Cars files) are skipped
when those assets are absent, so the suite still passes on a clean checkout.
"""

import glob
import os

import numpy as np
import pytest
import torch

from stcd.synth import SynthConfig, generate
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.energy import CostInputs, estimate, front_end_ops_per_event
from stcd.downstream import reconstruction as RC
from stcd.downstream.recognition import make_dataset, stack_tensors, SNNClassifier
from stcd.datasets import ncars
from stcd import metrics

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


# ---- energy --------------------------------------------------------------- #
def test_front_end_ops_per_event():
    assert front_end_ops_per_event(3) == 3 * 3 + 4   # k^2 + leak(2) + threshold(2)


def test_energy_pays_for_itself_when_downstream_expensive():
    c = CostInputs(neighbor_k=3, tau=8e-3, n_in=10000, n_out=6000, duration=0.3,
                   downstream_ops_per_event=1e4)
    r = estimate(c)
    assert r.events_saved == 4000
    assert r.pays_for_itself           # 4000*1e4 saved >> 10000*13 added
    assert r.added_latency_s > 0


def test_energy_overhead_when_downstream_cheap():
    c = CostInputs(neighbor_k=3, tau=8e-3, n_in=10000, n_out=9999, duration=0.3,
                   downstream_ops_per_event=1.0)
    assert not estimate(c).pays_for_itself


def test_efficiency_event_driven_beats_dense_at_low_activity():
    from stcd.energy import Scene, Hardware, efficiency
    rep = efficiency(Scene(event_rate_hz=3.0), Hardware())
    # sparse event-driven does far fewer ops than the same filter applied densely
    assert rep.event_driven_ops < rep.dense_same_filter_ops
    assert rep.ops_speedup_vs_dense > 10
    # and far less energy than a dense frame CNN even at higher per-op energy
    assert rep.energy_win_vs_frame_cnn > 5
    assert rep.latency_event_ms > 0


def test_efficiency_crossover_at_full_activity():
    from stcd.energy import Scene, efficiency
    # when every pixel-time cell has an event, event-driven ≈ dense (no free lunch)
    busy = efficiency(Scene(event_rate_hz=200.0))   # ~1/dt ⇒ sparsity ~1
    assert busy.ops_speedup_vs_dense < 2


# ---- reconstruction representation --------------------------------------- #
def test_voxel_grid_shape_and_bilinear():
    ev = generate(SynthConfig(H=40, W=50, duration=0.1, noise_rate_hz=1.0, seed=0))
    v = RC.events_to_voxel_grid(ev, num_bins=5, t0=0.0, t1=0.02)
    assert v.shape == (5, 40, 50)
    vn = RC.normalize_voxel(v.clone())
    nz = vn != 0
    if nz.any():
        assert abs(float(vn[nz].mean())) < 1e-4   # zero-mean over non-zero cells


def test_integrator_reconstructor_runs():
    ev = generate(SynthConfig(H=48, W=64, duration=0.1, noise_rate_hz=1.0, seed=0))
    model = RC.IntegratorReconstructor(48, 64)
    frames = RC.reconstruct_video(model, ev, num_bins=5, window_dt=0.02)
    assert len(frames) >= 3 and frames[0].shape == (48, 64)


# ---- recognition model ---------------------------------------------------- #
def test_edncnn_lite_patches_and_learns():
    from stcd.downstream import edncnn
    ev = generate(SynthConfig(H=48, W=64, duration=0.15, noise_rate_hz=6.0, seed=0))
    patches = edncnn.extract_patches(ev, k=7, Tw=5)
    assert patches.shape == (len(ev), 2 * 5, 7, 7)       # [N, P*Tw, k, k]
    scores, lab, idx = edncnn.train_eval(
        ev, ev.labels, edncnn.EDnCNNConfig(epochs=8, max_events=2000), seed=0)
    assert scores.shape == lab.shape
    assert metrics.roc(scores, lab)["auc"] > 0.65        # learns on easy synthetic


def test_edncnn_far_more_expensive_than_frontend():
    from stcd.downstream.edncnn import macs_per_event
    from stcd.energy import front_end_ops_per_event
    cnn = macs_per_event(in_ch=10, c=16, k=7)
    ours = front_end_ops_per_event(3)
    assert cnn == 183472                      # analytic conv1+conv2+fc
    assert cnn / ours > 1000                  # the CNN is orders of magnitude costlier


def test_snn_classifier_forward_shape():
    streams, y = make_dataset(n_per_class=2, H=32, W=32, duration=0.06, seed=0)
    x = stack_tensors(streams, dt=5e-3, duration=0.06, H=32, W=32)
    out = SNNClassifier(n_classes=2)(x)
    assert out.shape == (len(streams), 2)


# ---- FireNet (needs checkpoint) ------------------------------------------- #
@pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "firenet", "firenet_1000.pth.tar")),
                    reason="FireNet checkpoint not present")
def test_firenet_loads_and_runs():
    from stcd.downstream.firenet import load_firenet
    fr = load_firenet("cpu")
    assert sum(p.numel() for p in fr.model.parameters()) == 37777
    v = torch.zeros(1, 5, 64, 64)
    img = fr(v)
    assert img.shape == (1, 1, 64, 64)


# ---- N-Cars loader (needs data) ------------------------------------------- #
@pytest.mark.skipif(not glob.glob(os.path.join(ROOT, "ncars", "**", "*.dat"), recursive=True),
                    reason="N-Cars data not present")
def test_ncars_parse_real_file():
    f = glob.glob(os.path.join(ROOT, "ncars", "**", "*.dat"), recursive=True)[0]
    ev = ncars.parse_dat(f, H=ncars.DEFAULT_H, W=ncars.DEFAULT_W)
    assert len(ev) > 0
    assert ev.xs.max() < ncars.DEFAULT_W and ev.ys.max() < ncars.DEFAULT_H
    assert set(np.unique(ev.ps).tolist()).issubset({0, 1})
    assert 0.0 <= ev.duration <= 1.0   # N-Cars clips are ~100 ms
