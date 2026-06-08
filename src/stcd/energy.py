"""Energy & latency accounting for the front-end (proposal §7).

We count the *added* operations the front-end performs per input event, against
the *saved* downstream + transmission cost from emitting fewer events, and report
whether the filtering "pays for itself" and at what operating point.

On event-driven hardware, work scales with events, not pixels×time. Per input
event the front-end performs:

* Stage 2 — neighbourhood combination: ``k²`` accumulate ops (the event adds
  support to its ``k×k`` neighbourhood).
* Stage 3 — capacitive/LIF leak update: 1 multiply + 1 add.
* Stage 4 — threshold compare + reset-by-subtraction: ~2 ops.

Latency is dominated by the Stage-3 support window: an event can only be
confirmed after its coincidence window has elapsed, ``≈ τ`` (we report a small
multiple of τ as the confirmation delay).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostInputs:
    neighbor_k: int          # Stage-2 spatial kernel size
    tau: float               # Stage-3 leak time-constant (s)
    n_in: int                # input event count
    n_out: int               # output (surviving) event count
    duration: float          # stream length (s)
    downstream_ops_per_event: float = 1.0e4   # cost to process one event downstream
    confirm_window_taus: float = 3.0           # latency ≈ this many τ


@dataclass
class CostReport:
    ops_per_event_added: float
    total_added_ops: float
    events_saved: int
    ops_saved_downstream: float
    net_ops: float                  # added − saved (negative ⇒ net win)
    pays_for_itself: bool
    breakeven_downstream_ops: float  # downstream cost/event above which it pays
    added_latency_s: float
    retention: float
    added_ops_per_sec: float

    def summary(self) -> str:
        verdict = "PAYS FOR ITSELF" if self.pays_for_itself else "net overhead"
        return (
            f"added {self.ops_per_event_added:.0f} ops/in-event; "
            f"saved {self.events_saved} events × {self.ops_saved_downstream/max(self.events_saved,1):.0f} "
            f"downstream ops ⇒ net {self.net_ops:+.3e} ops ({verdict}); "
            f"break-even at {self.breakeven_downstream_ops:.0f} downstream ops/event; "
            f"added latency {self.added_latency_s*1e3:.1f} ms"
        )


def front_end_ops_per_event(neighbor_k: int) -> float:
    """Added ops per input event: Stage 2 (k²) + Stage 3 (2) + Stage 4 (2)."""
    return float(neighbor_k * neighbor_k + 2 + 2)


def estimate(c: CostInputs) -> CostReport:
    ope = front_end_ops_per_event(c.neighbor_k)
    total_added = ope * c.n_in
    saved_events = max(c.n_in - c.n_out, 0)
    saved_ops = saved_events * c.downstream_ops_per_event
    net = total_added - saved_ops
    # break-even downstream cost: added_total = saved_events * X  =>  X = added_total / saved_events
    breakeven = (total_added / saved_events) if saved_events else float("inf")
    latency = c.confirm_window_taus * c.tau
    return CostReport(
        ops_per_event_added=ope,
        total_added_ops=total_added,
        events_saved=saved_events,
        ops_saved_downstream=saved_ops,
        net_ops=net,
        pays_for_itself=net < 0,
        breakeven_downstream_ops=breakeven,
        added_latency_s=latency,
        retention=(c.n_out / c.n_in) if c.n_in else 0.0,
        added_ops_per_sec=(total_added / c.duration) if c.duration else 0.0,
    )


def pays_for_itself_table(c: CostInputs,
                          downstream_costs=(1e2, 1e3, 1e4, 1e5)) -> list[dict]:
    """Sweep downstream cost/event and report the net verdict at each."""
    rows = []
    for dc in downstream_costs:
        r = estimate(CostInputs(**{**c.__dict__, "downstream_ops_per_event": dc}))
        rows.append({
            "downstream_ops_per_event": dc,
            "added_ops_per_event": r.ops_per_event_added,
            "net_ops": r.net_ops,
            "pays_for_itself": r.pays_for_itself,
        })
    return rows


# --------------------------------------------------------------------------- #
# Event-driven (sparse) vs frame-based (dense) efficiency — the neuromorphic win
# --------------------------------------------------------------------------- #
@dataclass
class Hardware:
    """Per-operation energy (pJ). Defaults: Loihi-class neuromorphic synaptic op
    (Davies et al. 2018, ~23.6 pJ/SynOp) and a generous edge GPU/accelerator MAC."""
    pj_per_synop: float = 20.0      # neuromorphic, event-driven
    pj_per_mac: float = 1.0         # dense accelerator (optimistic — favours baseline)


@dataclass
class Scene:
    H: int = 346                    # DAVIS346
    W: int = 260
    duration: float = 0.1           # s
    event_rate_hz: float = 3.0      # events / pixel / second (scene activity)
    neighbor_k: int = 3
    dt: float = 5e-3                # time-bin (event-driven confirmation granularity)
    confirm_taus: float = 3.0
    tau: float = 8e-3
    # frame-based baseline:
    fps: int = 30                   # frame rate of the dense baseline
    cnn_macs_per_pixel: float = 1.0e3   # a small frame denoising CNN

    @property
    def n_events(self) -> int:
        return int(self.event_rate_hz * self.H * self.W * self.duration)

    @property
    def n_cells(self) -> int:        # pixels × time-bins (the dense volume)
        return int(self.H * self.W * max(1, round(self.duration / self.dt)))

    @property
    def sparsity(self) -> float:     # fraction of pixel-time cells that have an event
        return self.n_events / max(self.n_cells, 1)


@dataclass
class EfficiencyReport:
    event_driven_ops: float          # our spiking front-end (sparse)
    dense_same_filter_ops: float     # the SAME filter applied densely (every cell)
    frame_cnn_macs: float            # a dense frame-based CNN denoiser
    event_driven_uj: float           # energy (µJ) on neuromorphic hardware
    frame_cnn_uj: float              # energy (µJ) of the frame CNN on a dense accelerator
    ops_speedup_vs_dense: float
    energy_win_vs_frame_cnn: float
    latency_event_ms: float          # event-driven confirmation latency
    latency_frame_ms: float          # frame-based: frame period (+ CNN, ignored here)


def efficiency(scene: Scene, hw: Hardware | None = None) -> EfficiencyReport:
    hw = hw or Hardware()
    ope = front_end_ops_per_event(scene.neighbor_k)
    ed_ops = scene.n_events * ope
    dense_ops = scene.n_cells * ope
    frame_macs = scene.H * scene.W * scene.fps * scene.duration * scene.cnn_macs_per_pixel
    ed_uj = ed_ops * hw.pj_per_synop / 1e6
    frame_uj = frame_macs * hw.pj_per_mac / 1e6
    return EfficiencyReport(
        event_driven_ops=ed_ops,
        dense_same_filter_ops=dense_ops,
        frame_cnn_macs=frame_macs,
        event_driven_uj=ed_uj,
        frame_cnn_uj=frame_uj,
        ops_speedup_vs_dense=dense_ops / max(ed_ops, 1),
        energy_win_vs_frame_cnn=frame_uj / max(ed_uj, 1e-12),
        latency_event_ms=scene.confirm_taus * scene.tau * 1e3,
        latency_frame_ms=1e3 / scene.fps,
    )
