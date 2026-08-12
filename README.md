# STCD — A Single Spiking Neuron for Event-Camera Denoising

**STCD (Spatio-Temporal Coincidence Denoiser)** is a denoiser for event cameras
that is a *single* weight-shared leaky-integrate-and-fire (LIF) neuron, replicated
per pixel. It keeps an event only when its spatial neighbourhood has recently been
co-active — real edges fire neighbouring pixels in quick succession, background
-activity (BA) noise is isolated in space and time. STCD costs **13 integer
operations per event, uses no multipliers**, and exposes three interpretable
parameters (`k, τ, θ`).

On all 16 real DVSNOISE20 recordings, STCD is **statistically equivalent to the
authors' pretrained EDnCNN (TOST, within a 0.05-AUC margin)** at **~2.2×10⁷ fewer
FLOPs/event**, beats the classical filters and the deployable learned MLPF,
synthesizes to a **$50 FPGA** (10% logic, zero DSPs), and its spatial kernel can be
learned **without labels** by STDP. On the standard label-free **E-MLB** benchmark
(1152 recordings) it ranks **3rd of 12**, the best training-free method.

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

STCD vs pretrained EDnCNN: Δ = +0.016; the difference is **statistically equivalent**
by TOST within a 0.05-AUC margin (*p* = 0.04, 90% CI [−0.016, +0.048]) — i.e. not a
meaningful accuracy gap in either direction.

- **FPGA (iCE40 UP5K):** the **full DAVIS346 (346×260)** sensor fits with state packed
  to 8 bits/pixel — 823 logic cells (15%), **4 SPRAM, 0 BRAM, 0 DSP**, ~24–26 MHz
  post-PnR. On the board (24 MHz PLL) the packed core **measures** 9.0 cycles/event,
  2.67 Mev/s, and ~10 mW of event-processing power (9–10 mW active–idle board delta,
  0.616–0.617 W active vs 0.607 W idle; ≈3.4–3.7 nJ/event), matching the same-datapath
  128×128 core (0.505 vs 0.495 W). A minimal 128×128, 16-bit demonstrator core is
  535 LC / 2 SPRAM for reference.
- **Unsupervised STDP:** grows the kernel from a blind centre-only start (AUC 0.927)
  to 0.989, matching the hand-tuned box and approaching the supervised filter (0.991).
- **Downstream (real data):** FireNet reconstruction SSIM — STCD recovers the most
  (0.201 vs 0.139 noisy); N-Cars recognition 82.5% (noisy) → 87.5% (STCD), restoring the
  clean baseline (87.5%) in this protocol (120 test clips).
- **Zero-shot cross-dataset (DND21, MLPF/SNNF's home dataset):** exact-label noise
  mixing (synthetic 1-5 Hz/px + DND21's recorded noise, incl. MLPF's training-noise
  file); MLPF calibration reproduces its published in-domain level. On the dynamic
  *driving* sequence STCD (zero-shot) beats in-domain MLPF at every noise level
  (0.941 vs 0.899 at 1 Hz; 0.908 vs 0.896 at real 5.4 Hz dark noise), above SNNF's
  reported 0.89; in-domain MLPF wins on static *hotel-bar* at high noise
  (`scripts/run_dnd21.py`, `figures/data/dnd21_*.json`).
- **Standard E-MLB benchmark (label-free ESR, 1152 recordings):** STCD ranks **3rd of
  12** denoisers (0.971), the **best training-free method**, within 0.004 ESR of the
  learned EDnCNN (0.975) and behind only EventZoom; it is the **single best of all 12**
  in high noise (night set 1.063; noisiest night ND64 1.231 vs EDnCNN 1.086, EventZoom
  0.988). Run zero-shot at the same `k,τ,θ`.

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
python scripts/run_emlb_figure.py        # -> figures/emlb.png + LaTeX table (from emlb.json)
python scripts/make_graphical_abstract.py # -> figures/graphical_abstract.png (IEEE submission asset)

# --- full pipeline (requires the datasets/weights from DATA.md) ---
python scripts/run_edncnn_real.py        # 16-recording AUC -> figures/data/edncnn_real.json
python scripts/run_esr.py                # label-free ESR on DVSNOISE20 -> figures/data/esr.json
python scripts/run_emlb.py               # ESR on the standard E-MLB benchmark -> figures/data/emlb.json
python scripts/run_event_picture.py      # -> figures/event_picture.png
python scripts/run_before_after.py       # -> figures/before_after.png
python scripts/run_stdp_demo.py          # -> figures/stdp_learning.png
python scripts/run_downstream_real.py    # FireNet SSIM -> figures/downstream_real.png
python scripts/run_ncars_recognition.py  # N-Cars recognition under noise

# --- FPGA (open toolchain: yosys, nextpnr-ice40, icepack, iverilog) ---
# 128x128 demonstrator core (16-bit state):
cd hw && make test          # Icarus Verilog self-checking testbench
        make                # synth + place-and-route (resource report)
        make meas-prog      # flash throughput harness, then: python read_uart.py
# Full-resolution DAVIS346 packed core (8-bit state, the headline result):
        make packed-test    # co-sim the packed RTL vs the fixed-point reference
        make packed         # synth + place-and-route (4 SPRAM, ~26 MHz)
        make packed-meas-prog   # flash full-res measurement bitstream, then: python read_uart.py
        make packed-power-prog  # flash full-res power bitstream; button toggles idle/active, read USB meter

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
  not reproduce the field-standard *labelled* EPM/RPMD metric without the authors'
  full MATLAB pipeline. We do evaluate on the standard **E-MLB** benchmark (1152
  recordings) with its label-free ESR metric — our ESR is Raw-validated to within
  0.008 of E-MLB's published values; the higher-resolution LED benchmark is untested.
- FPGA cycles/event, throughput, and the ~10 mW power are all **on-board measurements**
  of the full-resolution packed core (24 MHz PLL; power also corroborated on the
  same-datapath 128×128 core). The power is a whole-board active–idle delta at the
  5 V input, so it bounds the 1.2 V core power rather than isolating it.
- The MLPF comparison runs the authors' published weights **as-is** (trained on
  DND21); its score reflects cross-dataset transfer, not its in-domain ceiling.

## License

STCD source code: MIT (see [LICENSE](LICENSE)). Third-party datasets and pretrained
weights retain their own licenses and are not redistributed (see [DATA.md](DATA.md)).
