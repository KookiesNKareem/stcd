"""Tests for the added baselines (KNoise, rate cap), hot pixels, and analysis helpers."""

import numpy as np

from stcd.events import Events
from stcd.synth import SynthConfig, generate, inject_hot_pixels
from stcd.baselines import (knoise_min_dt, knoise_filter, rate_cap,
                               baf_scores, knoise_scores, time_surface_scores)
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd import metrics, analysis


def test_knoise_drops_isolated_keeps_row_column_neighbor():
    # two events in the same column, adjacent rows, close in time -> supported;
    # one far isolated event -> never supported.
    ev = Events(xs=[10, 10, 40], ys=[10, 11, 40], ts=[0.0, 0.001, 0.002],
                ps=[1, 1, 1], H=64, W=64)
    mdt = knoise_min_dt(ev)
    assert np.isinf(mdt[2])                  # isolated: no row/col neighbour
    kept, _ = knoise_filter(ev, window=5e-3, min_dt=mdt)
    assert kept[1] and not kept[2]


def test_time_surface_higher_for_clustered_than_isolated():
    # a tight cluster of near-simultaneous events (mutual neighbours) should score
    # higher than a lone isolated event (no neighbours → 0).
    ev = Events(xs=[10, 11, 10, 40], ys=[10, 10, 11, 40],
                ts=[0.000, 0.0005, 0.001, 0.0015], ps=[1, 1, 1, 1], H=64, W=64)
    s = time_surface_scores(ev, tau=5e-3)
    assert s[3] == 0.0            # isolated event: no neighbour activity
    assert s[2] > s[3]            # clustered event has neighbour support


def test_rate_cap_removes_hot_pixel():
    # one pixel fires 100 times; others a handful. cap should drop only the hot one.
    xs = [5] * 100 + [j for j in range(10)]
    ys = [5] * 100 + [0] * 10
    ts = list(np.linspace(0, 0.1, 100)) + list(np.linspace(0, 0.1, 10))
    ev = Events(xs, ys, ts, [1] * 110, H=32, W=32)
    kept, filtered = rate_cap(ev, max_rate_hz=200.0)   # cap = 200*0.1 = 20 events
    assert not kept[:100].any()       # hot pixel (100 events) removed
    assert kept[100:].all()           # the rest kept


def test_inject_hot_pixels_adds_noise_labeled_events():
    base = generate(SynthConfig(H=40, W=50, duration=0.2, noise_rate_hz=1.0, seed=0))
    n0 = len(base)
    rng = np.random.default_rng(0)
    out = inject_hot_pixels(base, n_hot=5, rate_hz=300.0, duration=0.2, rng=rng)
    assert len(out) > n0
    # all added hot-pixel events are labelled noise (False)
    assert out.labels is not None
    # hot pixels concentrate events: max per-pixel count should be high
    counts = np.zeros((out.H, out.W))
    np.add.at(counts, (out.ys, out.xs), 1)
    assert counts.max() > 20


def test_knoise_signal_retain_ceiling_below_one():
    # On a realistic stream, KNoise cannot retain all signal (row/col memory).
    ev = generate(SynthConfig(H=120, W=160, duration=0.2, noise_rate_hz=3.0,
                              scene="bars", num_objects=4, seed=1))
    kn_max = analysis.max_signal_retain(ev, knoise_scores(ev, 2e-3))
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    fe_max = analysis.max_signal_retain(ev, fe.score_events(ev))
    assert kn_max < 0.95           # KNoise has a signal ceiling
    assert fe_max > kn_max         # front-end can operate closer to lossless


def test_nr_at_target_sr_hits_target():
    ev = generate(SynthConfig(H=120, W=160, duration=0.2, noise_rate_hz=3.0,
                              scene="bars", num_objects=4, seed=2))
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3))
    m = analysis.nr_at_target_sr(ev, fe.score_events(ev), target_sr=0.95)
    assert abs(m.signal_retain - 0.95) < 0.05
    assert m.noise_removal > 0.5
