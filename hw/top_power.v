// top_power.v — steady-state power-measurement harness for stcd_frontend.
//
// An inline USB power meter reads WHOLE-BOARD power (FPGA core + FT2232H + the
// 3.3 V/1.2 V regulators + oscillator + LED). To isolate STCD's core power we
// read power in two otherwise-identical steady states and subtract:
//   idle   : ev_valid held low, the coincidence FSM sits quiescent
//   active : events stream back-to-back, SRAM + datapath fully exercised
// active_power - idle_power cancels the always-on board overhead, leaving STCD's
// event-processing dynamic power.
//
// IMPORTANT for a clean delta: nothing visible may differ between the two states
// except the core's own switching, else (e.g.) an indicator LED turning on would
// inflate the difference. So the indicator (green) is held SOLID in both states
// (constant current, cancels in the subtraction); you track which state you are
// in from the button presses and the wattage itself (active draws more). A
// free-running LFSR drives the DUT inputs in both states so the stimulus's own
// switching also cancels. Same 12 -> 24 MHz PLL as top_meas so the core runs at
// its operating clock. Define SIM to bypass the PLL.
module top_power (
    input  wire CLK,
    input  wire BTN_N,    // active-low user button (pin 10): each press toggles active
    output wire TX,        // carries a reduction of the DUT outputs so the datapath is
                           // not optimised away; not read (negligible, equal toggling)
    output wire LEDR_N,    // off
    output wire LEDG_N     // solid on = powered; identical in both states -> cancels
);
    // ---- core clock: 12 MHz osc -> 24 MHz via PLL (bypassed in sim) ----------
`ifdef SIM
    wire clk_core = CLK;
    wire pll_lock = 1'b1;
`else
    wire clk_core, pll_lock;
    SB_PLL40_PAD #(
        .FEEDBACK_PATH("SIMPLE"),
        .DIVR(4'd0), .DIVF(7'd63), .DIVQ(3'd5), .FILTER_RANGE(3'b001)   // 12 -> 24 MHz
    ) pll (
        .PACKAGEPIN(CLK), .PLLOUTGLOBAL(clk_core), .LOCK(pll_lock),
        .RESETB(1'b1), .BYPASS(1'b0)
    );
`endif

    // ---- power-on reset (held until PLL locked) ----------------------------
    reg [3:0] por = 4'hF;
    always @(posedge clk_core) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0) | ~pll_lock;

    // ---- button: synchronise, debounce (~44 ms), toggle `active` on each press
    reg [1:0]  bsync   = 2'b11;
    reg [19:0] dbnc    = 0;
    reg        bstable = 1'b1;   // debounced level (1 = released, active-low button)
    reg        bprev   = 1'b1;
    reg        active  = 1'b0;
    always @(posedge clk_core) begin
        bsync <= {bsync[0], BTN_N};
        if (rst) begin
            dbnc <= 0; bstable <= 1'b1; bprev <= 1'b1; active <= 1'b0;
        end else begin
            if (bsync[1] == bstable)            dbnc <= 0;
            else if (dbnc == 20'hFFFFF) begin   bstable <= bsync[1]; dbnc <= 0; end
            else                                dbnc <= dbnc + 1'b1;
            bprev <= bstable;
            if (bprev == 1'b1 && bstable == 1'b0) active <= ~active;   // press = falling edge
        end
    end

    // ---- free-running LFSR event source (drives DUT inputs in BOTH states) ---
    reg  [30:0] lfsr = 31'h1;
    wire        fb = lfsr[30] ^ lfsr[27];
    wire        ready, ov;
    wire [6:0]  ox, oy;
    wire        ev_valid = active & ready;
    always @(posedge clk_core)
        if (rst) lfsr <= 31'h1;
        else     lfsr <= {lfsr[29:0], fb};   // advances every cycle in both states

    stcd_frontend dut (
        .clk(clk_core), .rst(rst), .ev_valid(ev_valid),
        .ev_x(lfsr[6:0]), .ev_y(lfsr[13:7]), .ev_t(lfsr[21:14]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy));

    // ---- state-independent indicator (cancels in the delta) ----------------
    assign LEDR_N = 1'b1;   // off
    assign LEDG_N = 1'b0;   // solid on = powered

    // ---- retain the DUT datapath with minimal logic (else yosys trims it) ----
    // ov gates the support compare -> memory reads -> address gen, so observing
    // it alone keeps the whole power-relevant datapath. Kept tiny to ease placement.
    assign TX = ov;
endmodule
