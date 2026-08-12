// top_packed_power.v — steady-state power-measurement harness for the full-res
// PACKED core (stcd_frontend_packed @ 346x260, 2c+6t). Identical protocol to
// top_power.v (see there for the measurement rationale): the user button
// toggles between two otherwise-identical steady states,
//   idle   : ev_valid held low, the coincidence FSM sits quiescent
//   active : events stream back-to-back, all 4 SPRAMs + datapath exercised
// and active_power - idle_power at the inline USB meter is the packed core's
// event-processing power. Indicator LED solid in both states; free-running
// LFSR stimulus in both states; 12 -> 24 MHz PLL as in top_packed_meas.
module top_packed_power (
    input  wire CLK,
    input  wire BTN_N,    // active-low user button: each press toggles active
    output wire TX,        // reduction of DUT outputs so the datapath survives synthesis
    output wire LEDR_N,    // off
    output wire LEDG_N     // solid on = powered; identical in both states -> cancels
);
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

    reg [3:0] por = 4'hF;
    always @(posedge clk_core) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0) | ~pll_lock;

    // ---- button: synchronise, debounce (~44 ms), toggle `active` on each press
    reg [1:0]  bsync   = 2'b11;
    reg [19:0] dbnc    = 0;
    reg        bstable = 1'b1;
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
    wire [8:0]  ox, oy;
    wire        ev_valid = active & ready;
    always @(posedge clk_core)
        if (rst) lfsr <= 31'h1;
        else     lfsr <= {lfsr[29:0], fb};   // advances every cycle in both states

    stcd_frontend_packed dut (
        .clk(clk_core), .rst(rst), .ev_valid(ev_valid),
        .ev_x(lfsr[8:0]), .ev_y(lfsr[17:9]), .ev_t(lfsr[23:18]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy));

    assign LEDR_N = 1'b1;   // off
    assign LEDG_N = 1'b0;   // solid on = powered

    assign TX = ov;
endmodule
