# STCD — A Single Spiking Neuron for Event-Camera Denoising

**STCD (Spatio-Temporal Coincidence Denoiser)** is a denoiser for event cameras
that is a *single* weight-shared leaky-integrate-and-fire (LIF) neuron, replicated
per pixel. It keeps an event only when its spatial neighbourhood has recently been
co-active — real edges fire neighbouring pixels in quick succession, background
-activity (BA) noise is isolated in space and time. STCD costs **13 integer
operations per event, uses no multipliers**, and exposes three interpretable
parameters (`k, τ, θ`).

On all 16 real DVSNOISE20 recordings, STCD is **statistically tied with the
authors' pretrained EDnCNN** at **~2.2×10⁷ fewer FLOPs/event**, beats the classical
filters and the deployable learned MLPF, synthesizes to a **$50 FPGA** (10% logic,
zero DSPs), and its spatial kernel can be learned **without labels** by STDP.

> The thesis is *parity at a tiny fraction of the cost* — STCD does **not** beat the
> learned CNN on accuracy; it matches it on the cheapest hardware on the frontier.

## Results

Real-data denoising, all 16 DVSNOISE20 recordings (per-event ROC-AUC vs an
APS-motion proxy label; mean ± 95% CI) versus per-event compute:

| Method | Type | ROC-AUC | FLOPs/event |
|---|---|---|---|
| **STCD (ours)** | spiking coincidence | **0.797 ± 0.017** | **13** |
| EDnCNN (pretrained) | learned CNN | 0.781 ± 0.032 | 2.83×10⁸ |
| time-surface | strong classical | 0.763 ± 0.032 | 32 |
| MLPF (as published) | deployable learned MLP | 0.702 ± 0.047 | 1.98×10³ |
| BAF | classical | 0.693 ± 0.031 | 16 |
| KNoise | memory-light | 0.516 ± 0.018 | 8 |

STCD vs pretrained EDnCNN: Δ = +0.016, paired *t*-test *p* = 0.39 (not significant).

- **FPGA (iCE40 UP5K, post-PnR):** 535 logic cells (10%), 2 SPRAM, **0 BRAM, 0 DSP**,
  ~24–26 MHz, ~9 cycles/event ⇒ ~2.9 Mev/s, ~10 mW (estimate).
- **Unsupervised STDP:** grows the kernel from a blind centre-only start (AUC 0.927)
  to 0.989, matching the hand-tuned box and approaching the supervised filter (0.991).
- **Downstream (real data):** FireNet reconstruction SSIM — STCD recovers the most
  (0.201 vs 0.139 noisy); N-Cars recognition 82.5% (noisy) → 88.3% (STCD), ≈ clean 87.5%.

## Install

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## Reproduce

Datasets/weights are not included (~34 GB); see **[DATA.md](DATA.md)** to obtain
them. The one expensive result (the 16-recording eval) is cached in
`figures/data/edncnn_real.json`, so the cost figures regenerate without the raw data.

```bash
# --- regenerate from the cached eval (no raw data needed) ---
python scripts/run_pareto.py             # -> figures/pareto.png        (accuracy vs cost)
python scripts/run_edncnn_efficiency.py  # -> figures/edncnn_efficiency.png

# --- full pipeline (requires the datasets/weights from DATA.md) ---
python scripts/run_edncnn_real.py        # 16-recording AUC -> figures/data/edncnn_real.json
python scripts/run_esr.py                # standard label-free ESR -> figures/data/esr.json
python scripts/run_event_picture.py      # -> figures/event_picture.png
python scripts/run_before_after.py       # -> figures/before_after.png
python scripts/run_stdp_demo.py          # -> figures/stdp_learning.png
python scripts/run_downstream_real.py    # FireNet SSIM -> figures/downstream_real.png
python scripts/run_ncars_recognition.py  # N-Cars recognition under noise

# --- FPGA (open toolchain: yosys, nextpnr-ice40, icepack, iverilog) ---
cd hw && make test     # Icarus Verilog self-checking testbench
        make           # synth + place-and-route (resource report)

# --- paper ---
cd paper && tectonic main.tex     # -> main.pdf

# --- tests ---
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q     # 41 tests
```

`scripts/run_downstream_real.py` auto-uses the GPU (Apple MPS / CUDA) to score the
pretrained EDnCNN over the reconstruction stream; set `SKIP_EDN=1` to skip it.

## Layout

```
src/stcd/
  events.py        # event representation & dense-tensor conversion
  frontend.py      # the single weight-shared LIF coincidence neuron (STCD)
  baselines.py     # BAF, KNoise, time-surface
  stdp.py          # unsupervised STDP learning of the spatial kernel
  metrics.py       # ROC-AUC / SR / NR / RPMD
  energy.py        # ops/event + energy accounting
  synth.py         # synthetic generator with ground-truth labels
  datasets/        # dvsnoise20.py, ncars.py, epm.py (partial EPM/Jt port)
  downstream/
    edncnn_real.py # the real pretrained EDnCNN (h5py, weights-only)
    mlpf.py        # the published MLPF (h5py, weights-only)
    firenet.py     # pretrained FireNet reconstruction (weights-only load)
    recognition.py # N-Cars classifier
    reconstruction.py
hw/                # synthesizable Verilog (stcd_frontend.v) + testbench + Makefile
scripts/           # paper-figure reproduction + download_data.py
tests/             # correctness tests (41)
paper/             # main.tex (+ built main.pdf)
figures/           # the 6 paper figures + figures/data/edncnn_real.json (cached eval)
data/              # datasets + weights (see DATA.md)
attic/             # archived exploratory scripts/figures
```

## Honest scope

- The accuracy comparison uses an **APS-motion proxy** label on DVSNOISE20; we could
  not reproduce the field-standard EPM/RPMD metric without the authors' full MATLAB
  pipeline, and have not yet evaluated on the larger E-MLB/LED benchmarks.
- FPGA throughput and power are **post-PnR / datasheet estimates**, not on-board
  measurements.
- The MLPF comparison runs the authors' published weights **as-is** (trained on
  DND21); its score reflects cross-dataset transfer, not its in-domain ceiling.

## License

STCD source code: MIT (see [LICENSE](LICENSE)). Third-party datasets and pretrained
weights retain their own licenses and are not redistributed (see [DATA.md](DATA.md)).
