"""Tests for the unsupervised STDP denoiser (the novel contribution)."""

import numpy as np

from stcd.synth import SynthConfig, generate
from stcd.stdp import STDPDenoiser, STDPConfig
from stcd import metrics


def _hotpixel_stream(seed):
    return generate(SynthConfig(H=100, W=120, duration=0.2, scene="bars",
                                num_objects=4, noise_rate_hz=1.0,
                                n_hot_pixels=40, hot_pixel_rate_hz=500.0, seed=seed))


def test_stdp_improves_auc_unsupervised():
    # On hot-pixel noise the blind centre-only filter fails; STDP (no labels)
    # should learn spatial structure and improve held-out AUC.
    train_ev, eval_ev = _hotpixel_stream(1), _hotpixel_stream(777)
    st = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3, eta=0.06, epochs=25), init="delta")
    auc_blind = st._auc(eval_ev)
    hist = st.train_unsupervised(train_ev, eval_ev=eval_ev)  # no labels used to learn
    assert auc_blind < 0.96                      # blind centre-only is poor here
    assert max(hist["auc"]) > auc_blind + 0.03   # STDP measurably improves it
    assert hist["auc"][-1] > 0.96


def test_stdp_grows_receptive_field():
    st = STDPDenoiser(STDPConfig(k=5, tau=8e-3, dt=5e-3, eta=0.06, epochs=20), init="delta")
    k0 = st.kernel()[1]
    c0 = k0[2, 2]; n0 = (k0.sum() - k0[2, 2]) / 24
    assert c0 > 10 * n0                          # blind: weight concentrated at centre
    st.train_unsupervised(_hotpixel_stream(2))
    k1 = st.kernel()[1]
    c1 = k1[2, 2]; n1 = (k1.sum() - k1[2, 2]) / 24
    assert n1 > 5 * n0                           # neighbours gained weight (field grew)


def test_stdp_filter_interface():
    ev = _hotpixel_stream(3)
    st = STDPDenoiser(STDPConfig(k=5, epochs=10), init="uniform")
    scores = st.score_events(ev)
    assert scores.shape == (len(ev),)
    kept, filtered = st.filter(ev)
    assert len(filtered) == int(kept.sum())


def _oriented_stream(seed):
    return generate(SynthConfig(H=100, W=120, duration=0.2, scene="disks",
                                num_objects=5, size_range=(0.06, 0.12),
                                noise_rate_hz=2.0, seed=seed))


def test_competitive_stdp_features_diversify():
    from stcd.stdp import CompetitiveSTDPDenoiser, CompetitiveSTDPConfig
    cs = CompetitiveSTDPDenoiser(
        CompetitiveSTDPConfig(n_features=8, k=7, epochs=30, eta=0.05), seed=0)
    cs.train_unsupervised(_oriented_stream(1))         # no labels
    K = cs.kernels().sum(axis=1).reshape(8, -1)        # [N, k*k] (ON+OFF)
    K = K - K.mean(axis=1, keepdims=True)
    C = np.corrcoef(K)
    mean_off_diag = C[np.triu_indices(8, 1)].mean()
    assert mean_off_diag < 0.5                          # neurons specialised (diverse)


def test_competitive_stdp_interface_and_auc():
    from stcd.stdp import CompetitiveSTDPDenoiser, CompetitiveSTDPConfig
    cs = CompetitiveSTDPDenoiser(
        CompetitiveSTDPConfig(n_features=6, k=7, epochs=25, eta=0.05), seed=0)
    ev = _oriented_stream(7)
    cs.train_unsupervised(_oriented_stream(1))
    scores = cs.score_events(ev)
    assert scores.shape == (len(ev),)
    assert metrics.roc(scores, ev.labels)["auc"] > 0.85   # still a usable denoiser


def test_competitive_stdp_encode_shape():
    from stcd.stdp import CompetitiveSTDPDenoiser, CompetitiveSTDPConfig
    cs = CompetitiveSTDPDenoiser(CompetitiveSTDPConfig(n_features=10, k=7, epochs=5), seed=0)
    vec = cs.encode(_oriented_stream(1), pool=4)
    assert vec.shape == (10 * 4 * 4,)        # N · pool · pool feature vector
    assert np.isfinite(vec).all()


def test_spatiotemporal_stdp_learns_temporal_structure():
    from stcd.stdp import SpatioTemporalSTDP, SpatioTemporalSTDPConfig
    st = SpatioTemporalSTDP(
        SpatioTemporalSTDPConfig(n_features=6, k=5, n_lags=5, epochs=20, eta=0.06), seed=0)
    st.train_unsupervised(_oriented_stream(1))
    K = st.kernels()                                   # [N,P,L,k,k]
    assert K.shape[2] == 5
    lag_energy = K.sum(axis=(1, 3, 4)).mean(axis=0)    # mean energy per lag
    # weight is spread across time taps, not collapsed onto a single (spatial) lag
    assert (lag_energy > 0.05).sum() >= 3
    ev = _oriented_stream(7)
    assert st.score_events(ev).shape == (len(ev),)
