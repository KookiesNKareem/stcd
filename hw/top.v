// top.v — characterisation harness for the iCEBreaker (iCE40 UP5K, sg48).
// Drives the denoiser from an on-chip LFSR event generator and folds the output
// into one LED, so the design needs only clk + 1 pin — letting nextpnr report
// real utilisation and fmax for the core (events come from on-chip in deployment).

module top (input wire CLK, output wire LEDR_N);
    // power-on reset
    reg [3:0] por = 4'hF;
    always @(posedge CLK) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0);

    // LFSR pseudo-event generator
    reg [30:0] lfsr = 31'h1;
    wire fb = lfsr[30] ^ lfsr[27];
    wire ready;
    always @(posedge CLK) if (!rst && ready) lfsr <= {lfsr[29:0], fb};

    wire ov; wire [6:0] ox, oy;
    stcd_frontend dut (
        .clk(CLK), .rst(rst),
        .ev_valid(ready), .ev_x(lfsr[6:0]), .ev_y(lfsr[13:7]), .ev_t(lfsr[21:14]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy)
    );

    // fold all outputs into one LED so they aren't optimised away
    reg led;
    always @(posedge CLK) if (ov) led <= led ^ (^{ox, oy});
    assign LEDR_N = ~led;
endmodule
