// top_meas.v — on-board throughput/latency measurement for stcd_frontend.
//
// Streams events back-to-back from an on-chip LFSR (the per-event cycle count is
// data-independent — the FSM is fixed-latency — so event content does not affect
// timing), counts total clock cycles, events accepted, and events kept over a
// window of NWIN events, then reports {cycles, n_events, n_kept} over UART
// (115200 8N1) as a repeating 14-byte packet: 0xAA 0x55 then three big-endian
// u32s. Host (read_uart.py) computes cycles/event = cycles/n_events and
// throughput = f_core / (cycles/event).
//
// A PLL multiplies the 12 MHz board oscillator to F_CORE (default 24 MHz, the
// place-and-route f_max), so the measured throughput is taken at the core's
// operating clock, not the bare oscillator. Define SIM to bypass the PLL for
// Icarus Verilog (which has no SB_PLL40 model).
module top_meas #(
    parameter [31:0]  NWIN = 32'd1000000,    // events per measurement window
    parameter integer DIV  = 208             // 24 MHz / 115200 baud
)(
    input  wire CLK,
    output wire TX,
    output wire LEDR_N
);
    // ---- core clock: 12 MHz osc -> 24 MHz via PLL (bypassed in sim) ---------
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

    // ---- LFSR pseudo-event source (back-to-back) ---------------------------
    reg  [30:0] lfsr = 31'h1;
    wire        fb = lfsr[30] ^ lfsr[27];
    wire        ready, ov;
    wire [6:0]  ox, oy;
    reg         running = 1'b1, started = 1'b0;
    wire        ev_valid = running & ready;

    stcd_frontend dut (
        .clk(clk_core), .rst(rst), .ev_valid(ev_valid),
        .ev_x(lfsr[6:0]), .ev_y(lfsr[13:7]), .ev_t(lfsr[21:14]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy));

    // ---- counters ----------------------------------------------------------
    reg [31:0] cyc = 0, nev = 0, kept = 0;
    always @(posedge clk_core) begin
        if (rst) begin
            cyc <= 0; nev <= 0; kept <= 0; running <= 1'b1; started <= 1'b0; lfsr <= 31'h1;
        end else if (running) begin
            if (ev_valid) started <= 1'b1;
            if (started)  cyc <= cyc + 1'b1;
            if (ev_valid) begin
                lfsr <= {lfsr[29:0], fb};
                nev  <= nev + 1'b1;
                if (nev == NWIN - 1) running <= 1'b0;
            end
            if (ov) kept <= kept + 1'b1;
        end
    end

    // ---- UART report: repeating 14-byte packet while !running --------------
    reg  [3:0]  bidx = 0;
    reg         tstb = 0;
    reg  [7:0]  tdata;
    reg  [16:0] gap = 0;
    wire        tx_busy;

    function [7:0] rbyte(input [3:0] i, input [31:0] c, input [31:0] n, input [31:0] k);
        case (i)
            4'd0: rbyte = 8'hAA;       4'd1: rbyte = 8'h55;
            4'd2: rbyte = c[31:24];    4'd3: rbyte = c[23:16];
            4'd4: rbyte = c[15:8];     4'd5: rbyte = c[7:0];
            4'd6: rbyte = n[31:24];    4'd7: rbyte = n[23:16];
            4'd8: rbyte = n[15:8];     4'd9: rbyte = n[7:0];
            4'd10: rbyte = k[31:24];   4'd11: rbyte = k[23:16];
            4'd12: rbyte = k[15:8];    default: rbyte = k[7:0];
        endcase
    endfunction

    always @(posedge clk_core) begin
        tstb <= 1'b0;
        if (rst) begin bidx <= 0; gap <= 0; end
        else if (!running) begin
            if (gap != 0) gap <= gap - 1'b1;
            else if (!tx_busy && !tstb) begin
                tdata <= rbyte(bidx, cyc, nev, kept);
                tstb  <= 1'b1;
                if (bidx == 4'd13) begin bidx <= 0; gap <= 17'd120000; end
                else bidx <= bidx + 1'b1;
            end
        end
    end

    uart_tx #(.DIV(DIV)) u_tx (.clk(clk_core), .rst(rst), .stb(tstb), .data(tdata),
                               .tx(TX), .busy(tx_busy));

    assign LEDR_N = running ? 1'b0 : 1'b1;   // LED on while measuring
endmodule
