// top_meas.v — on-board throughput/latency measurement for stcd_frontend.
//
// Streams events back-to-back from an on-chip LFSR (the per-event cycle count is
// data-independent — the FSM is fixed-latency — so event content does not affect
// timing), counts total clock cycles, events accepted, and events kept over a
// window of NWIN events, then reports {cycles, n_events, n_kept} over UART
// (115200 8N1) as a repeating 14-byte packet: 0xAA 0x55 then three big-endian
// u32s. Host (read_uart.py) computes cycles/event = cycles/n_events and
// throughput = f_clk / (cycles/event). Measures the core on real silicon.
module top_meas #(
    parameter [31:0]  NWIN = 32'd1000000,    // events per measurement window
    parameter integer DIV  = 104             // 12 MHz / 115200 baud
)(
    input  wire CLK,
    output wire TX,
    output wire LEDR_N
);
    // ---- power-on reset ----------------------------------------------------
    reg [3:0] por = 4'hF;
    always @(posedge CLK) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0);

    // ---- LFSR pseudo-event source (back-to-back) ---------------------------
    reg  [30:0] lfsr = 31'h1;
    wire        fb = lfsr[30] ^ lfsr[27];
    wire        ready, ov;
    wire [6:0]  ox, oy;
    reg         running = 1'b1, started = 1'b0;
    wire        ev_valid = running & ready;

    stcd_frontend dut (
        .clk(CLK), .rst(rst), .ev_valid(ev_valid),
        .ev_x(lfsr[6:0]), .ev_y(lfsr[13:7]), .ev_t(lfsr[21:14]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy));

    // ---- counters ----------------------------------------------------------
    reg [31:0] cyc = 0, nev = 0, kept = 0;
    always @(posedge CLK) begin
        if (rst) begin
            cyc <= 0; nev <= 0; kept <= 0; running <= 1'b1; started <= 1'b0; lfsr <= 31'h1;
        end else if (running) begin
            if (ev_valid) started <= 1'b1;
            if (started)  cyc <= cyc + 1'b1;           // count from the first accept
            if (ev_valid) begin
                lfsr <= {lfsr[29:0], fb};
                nev  <= nev + 1'b1;
                if (nev == NWIN - 1) running <= 1'b0;  // this is the NWIN-th event
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

    always @(posedge CLK) begin
        tstb <= 1'b0;
        if (rst) begin bidx <= 0; gap <= 0; end
        else if (!running) begin
            if (gap != 0) gap <= gap - 1'b1;
            else if (!tx_busy && !tstb) begin
                tdata <= rbyte(bidx, cyc, nev, kept);
                tstb  <= 1'b1;
                if (bidx == 4'd13) begin bidx <= 0; gap <= 17'd120000; end  // pause between packets
                else bidx <= bidx + 1'b1;
            end
        end
    end

    uart_tx #(.DIV(DIV)) u_tx (.clk(CLK), .rst(rst), .stb(tstb), .data(tdata),
                               .tx(TX), .busy(tx_busy));

    assign LEDR_N = running ? 1'b0 : 1'b1;   // LED on while measuring
endmodule
