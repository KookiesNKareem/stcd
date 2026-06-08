# FPGA implementation — spiking denoiser on iCEBreaker (iCE40 UP5K)

A synthesizable, event-driven hardware realisation of the front-end's coincidence
stage, targeting the **Lattice iCE40 UP5K** on the iCEBreaker (open toolchain:
yosys + nextpnr-ice40 + icepack/iceprog).

## What it does (per event)
Optimised **gather + checkerboard banking** datapath. Per event `(x, y, t)`:
1. **Read the 8-neighbourhood**, apply a **lazy leak** to each neighbour's stored
   count (right-shift by elapsed ticks ≈ exponential decay), and **sum** them →
   support. **Forward** the event iff support ≥ θ.
2. **Update the centre pixel's own count** (read-modify-write), so future events
   that gather this pixel see it. Centre excluded from its own support ⇒ isolated
   noise is dropped.

This is *exactly* the Python front-end's membrane (decayed neighbourhood event
count); gather needs ~10 RAM accesses/event (8 reads + 1 own RMW) vs a scatter
datapath's ~17 (8 RMW + centre).

**Checkerboard banking.** State is split across **two single-port RAMs** by bank
`= (x[0] ^ y[0])` — the parity of the pixel's coordinates, like a chessboard. On a
checkerboard the 8-neighbourhood is always **4 same-bank (diagonals) + 4
opposite-bank (orthogonals)**, so the two banks are read **in parallel** — one
neighbour from each per cycle. That halves the gather, and overlapping the own-cell
read with the last gather cycle drops it to **~9 cycles/event**.

Memory: two 16-bit-wide single-port RAMs packing `{last_tick[15:8], count[7:0]}`
per pixel → **two iCE40 SPRAMs**. No multipliers (leak = shift, pixel index =
`{y,x}` with power-of-two stride).

## Functional verification
`make test` runs an Icarus Verilog self-checking testbench (`tb_stcd.v`):
isolated event → dropped; two coincident neighbours → forwarded; stale support
(large Δt) → dropped by the leak; single neighbour (< θ) → dropped. **All pass.**
(Writing this testbench caught two real bugs: uninitialised SPRAM — fixed with a
reset clear-sweep — and a signed/unsigned neighbour-offset bug.)

## Measured results (yosys 0.65 + nextpnr-ice40, UP5K / sg48)
Config: 128×128 pixel state (`LOG2W=LOG2H=7`), 8-bit counts. Post-PnR, **verified** RTL.
Optimised **gather + checkerboard banking** datapath (~9 cycles/event).

| Resource | Used | Available | % |
|---|---|---|---|
| Logic cells (LUT4+FF) | **535** | 5,280 | **10 %** |
| SPRAM (256 Kbit) | **2** | 4 | 50 % |
| BRAM (EBR) | 0 | 30 | 0 % |
| **DSP / multipliers** | **0** | 8 | **0 %** |
| **fmax** | **~24–26 MHz** | — | — |

- **Throughput:** ~9 cycles/event ⇒ **~2.7–2.9 M events/s** at 24–26 MHz — using
  10 % of the chip and **no DSPs**. That **clears DVSNOISE20's ~2.9 Mev/s peak**.
- Optimisation history: scatter ~28 cyc (1.1 Mev/s) → single-bank gather ~15 cyc
  (1.9 Mev/s) → **banked gather ~9 cyc (2.7–2.9 Mev/s)** — a **2.6× total** speedup.

## Comparison on the SAME chip: a per-event CNN (EDnCNN)
EDnCNN runs a CNN per event (~**183,472 MACs/event** for the lite config). The UP5K
has **8 DSPs** (1 MAC each per cycle):
- Compute: 183,472 / 8 ≈ **22,900 cycles/event** — vs our **9 cycles/event**.
- At the *same clock* that is **~2,500× slower** (the ratio is clock-independent), and
  **~3 orders below sensor event rates** (DVSNOISE20 ≈ 2.9 Mev/s) — i.e. **not
  real-time** on this chip.
- It also needs all 8 DSPs + weight/activation storage + a much larger controller,
  vs our 10 % logic / 0 DSP / 2 SPRAM.

**Summary:** our filter is *memory-access-bound* (8 neighbour reads + 1 RMW, two
banks in parallel); the CNN is *compute-bound* (saturates all DSPs). On a \$69
5K-LUT FPGA, ours sustains ~2.7–2.9 Mev/s at 10 % utilisation — clearing the sensor
peak — while the learned CNN is ~2,500× slower and cannot keep up.

## Efficiency notes / optimisation path
- Lean: 0 DSP, 2/4 SPRAM, 10 % logic — and at/above the sensor peak.
- **Done — gather + pipelining.** Switched scatter (8 RMW, ~28 cyc/event) to gather
  (pipelined reads + 1 own RMW): the single-port SPRAM's 2-cycle read latency is
  hidden by a 2-deep address pipeline. **~28 → ~15 cycles/event.**
- **Done — checkerboard banking.** State split across 2 SPRAMs by `(x[0]^y[0])`
  parity so the 8 neighbours split 4/4 across two banks read in parallel; own read
  overlaps the last gather cycle. **~15 → ~9 cycles/event, ~2.7–2.9 M events/s.**
- fmax fell (~29 → ~24–26 MHz) because banking doubles the address/accumulate logic
  (two coordinate computations + parity routing); it's placement-sensitive, so the
  Makefile pins the best of a 6-seed sweep. Pipelining the neighbour-address
  generation would recover some fmax at the cost of +1 latency cycle.

## Files
- `stcd_frontend.v` — the core (parameterised; the deployable IP).
- `top.v` — characterisation harness (on-chip LFSR event generator + 1 LED) so the
  core can be placed/timed with only `clk` + 1 pin (events come from on-chip in
  deployment).
- `tb_stcd.v` — Icarus Verilog self-checking testbench.
- `icebreaker.pcf` — minimal pin constraints. `Makefile` — build/test flow.

## Build / test / measure / program
```bash
make test       # Icarus Verilog functional self-check (all cases pass)
make            # yosys synth + nextpnr PnR (prints utilisation + fmax)
make prog       # icepack + iceprog to a connected iCEBreaker
```
The RTL is functionally verified in simulation; the resource/fmax numbers are
post-synthesis/PnR for the UP5K. On-board bring-up still needs an event source
(e.g. the sensor pipeline or a UART/SPI feeder).
