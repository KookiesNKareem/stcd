"""Correctness tests for the core simulator pieces."""

import math

import numpy as np
import torch

from stcd.events import Events, TimeGrid, events_to_tensor, tensor_to_events, labels_to_tensor
from stcd.synth import SynthConfig, generate
from stcd.frontend import SpatialPool, TemporalLeak, LIFCoincidence, SpikingFrontEnd, FrontEndConfig, spike
from stcd.baselines import baf_min_dt, baf_filter
from stcd import metrics


# ---- events <-> tensor ---------------------------------------------------- #
def test_event_tensor_roundtrip_counts():
    ev = Events(xs=[1, 1, 3], ys=[0, 0, 2], ts=[0.001, 0.002, 0.05], ps=[1, 1, 0], H=4, W=5)
    tensor, grid = events_to_tensor(ev, dt=0.01)
    assert tensor.shape[:3] == (2, 4, 5)
    # total count preserved
    assert tensor.sum().item() == 3
    # the two ON events at (y=0,x=1) land in the same early bin
    assert tensor[1, 0, 1, 0].item() == 2


def test_timegrid_bin_index_bounds():
    ev = Events(xs=[0], ys=[0], ts=[0.0], ps=[1], H=2, W=2)
    g = TimeGrid(t0=0.0, dt=0.01, T=5)
    assert g.bin_index(np.array([-1.0, 0.0, 0.005, 0.049, 100.0])).tolist() == [0, 0, 0, 4, 4]


def test_labels_to_tensor_signal_fraction():
    ev = Events(xs=[0, 0], ys=[0, 0], ts=[0.0, 0.001], ps=[1, 1], H=1, W=1,
                labels=np.array([True, False]))
    _, grid = events_to_tensor(ev, dt=0.01)
    frac = labels_to_tensor(ev, grid)
    assert math.isclose(frac[1, 0, 0, 0].item(), 0.5, abs_tol=1e-6)


# ---- synthetic generator -------------------------------------------------- #
def test_synth_has_both_classes_and_labels():
    ev = generate(SynthConfig(H=40, W=50, duration=0.2, noise_rate_hz=5.0, seed=1))
    assert ev.labels is not None and len(ev) == len(ev.labels)
    assert ev.labels.sum() > 0          # some signal
    assert (~ev.labels).sum() > 0       # some noise
    # timestamps sorted
    assert np.all(np.diff(ev.ts) >= 0)


def test_synth_noise_rate_in_ballpark():
    cfg = SynthConfig(H=60, W=60, duration=0.5, noise_rate_hz=10.0, num_objects=0, seed=2)
    ev = generate(cfg)
    n_noise = int((~ev.labels).sum())
    expected = cfg.noise_rate_hz * cfg.H * cfg.W * cfg.duration
    assert 0.7 * expected < n_noise < 1.3 * expected


# ---- frontend stages ------------------------------------------------------ #
def test_temporal_leak_is_ema():
    leak = TemporalLeak(tau=0.01)
    x = torch.zeros(1, 1, 1, 4)
    x[0, 0, 0, 0] = 1.0
    dt = 0.01
    out = leak(x, dt)
    a = math.exp(-dt / 0.01)
    expected = [1.0, a, a**2, a**3]
    assert torch.allclose(out[0, 0, 0], torch.tensor(expected), atol=1e-5)


def test_lif_fires_on_coincidence_not_isolated():
    # threshold 1.5: a single unit input never crosses; two coincident inputs do.
    lif = LIFCoincidence(tau=0.02, theta=1.5)
    dt = 0.001
    isolated = torch.zeros(1, 1, 1, 3); isolated[0, 0, 0, 0] = 1.0
    coincident = torch.zeros(1, 1, 1, 3); coincident[0, 0, 0, 0] = 2.0
    s_iso, _ = lif(isolated, dt)
    s_co, _ = lif(coincident, dt)
    assert s_iso.sum().item() == 0.0
    assert s_co.sum().item() >= 1.0


def test_spike_surrogate_gradient_flows():
    u = torch.tensor([-0.2, 0.0, 0.3], requires_grad=True)
    s = spike(u, beta=10.0)
    s.sum().backward()
    assert u.grad is not None and torch.all(u.grad >= 0) and u.grad.abs().sum() > 0


def test_lif_threshold_is_trainable():
    lif = LIFCoincidence(tau=0.02, theta=1.0)
    x = torch.zeros(1, 1, 1, 5, requires_grad=False); x[0, 0, 0, 0] = 1.2
    s, _ = lif(x, 0.001)
    s.sum().backward()
    assert lif.raw_theta.grad is not None and lif.raw_theta.grad.abs().item() > 0


def test_spatial_pool_sum_and_or():
    x = torch.zeros(2, 2, 2, 1)
    x[1, 0, 0, 0] = 1.0
    x[1, 1, 1, 0] = 3.0
    s = SpatialPool(pool=2, mode="sum")(x)
    assert s.shape == (2, 1, 1, 1)
    assert math.isclose(s[1, 0, 0, 0].item(), 4.0, abs_tol=1e-5)
    o = SpatialPool(pool=2, mode="or")(x)
    assert math.isclose(o[1, 0, 0, 0].item(), 3.0, abs_tol=1e-5)


# ---- BAF ------------------------------------------------------------------ #
def test_baf_drops_isolated_keeps_clustered():
    # Two neighbouring events close in time (should support each other) + one
    # far-away isolated event (no neighbour) -> isolated dropped.
    ev = Events(
        xs=[10, 11, 40], ys=[10, 10, 40], ts=[0.000, 0.001, 0.002], ps=[1, 1, 1],
        H=64, W=64,
    )
    mdt = baf_min_dt(ev)
    assert np.isinf(mdt[2])               # isolated has no neighbour ever
    kept, _ = baf_filter(ev, window=0.005, min_dt=mdt)
    assert kept[1] and not kept[2]


# ---- metrics -------------------------------------------------------------- #
def test_evaluate_filter_known_counts():
    labels = np.array([True, True, False, False])
    kept = np.array([True, False, False, False])  # keep 1 signal, drop 1 noise correctly
    m = metrics.evaluate_filter(labels, kept)
    assert math.isclose(m.signal_retain, 0.5)
    assert math.isclose(m.noise_removal, 1.0)
    assert math.isclose(m.denoise_accuracy, 0.75)


def test_roc_auc_perfect_separation():
    labels = np.array([True, True, False, False])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    out = metrics.roc(scores, labels)
    assert math.isclose(out["auc"], 1.0, abs_tol=1e-9)


# ---- end-to-end smoke ----------------------------------------------------- #
def test_frontend_filter_runs_and_reduces_events():
    ev = generate(SynthConfig(H=48, W=64, duration=0.2, noise_rate_hz=8.0, seed=3))
    fe = SpikingFrontEnd(FrontEndConfig(pool=1, tau=8e-3, theta=1.0, dt=5e-3))
    scores = fe.score_events(ev)
    assert scores.shape == (len(ev),)
    kept, filtered = fe.filter(ev)
    assert len(filtered) == int(kept.sum())
    # ROC should be better than chance at separating signal from noise
    auc = metrics.roc(scores, ev.labels)["auc"]
    assert auc > 0.6
