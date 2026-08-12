// top_packed.v — characterisation harness for the full-resolution packed core.
// LFSR pseudo-events drive the denoiser; outputs fold into one LED so nothing is
// optimised away, letting nextpnr report real LC/SPRAM/fmax for stcd_frontend_packed
// at the full DAVIS346 (346x260) state. (Coord values are don't-care for timing.)
module top_packed (input wire CLK, output wire LEDR_N);
    reg [3:0] por = 4'hF;
    always @(posedge CLK) if (por != 0) por <= por - 1'b1;
    wire rst = (por != 0);

    reg [30:0] lfsr = 31'h1;
    wire fb = lfsr[30] ^ lfsr[27];
    wire ready;
    always @(posedge CLK) if (!rst && ready) lfsr <= {lfsr[29:0], fb};

    wire ov; wire [8:0] ox, oy;
    stcd_frontend_packed dut (
        .clk(CLK), .rst(rst),
        .ev_valid(ready), .ev_x(lfsr[8:0]), .ev_y(lfsr[17:9]), .ev_t(lfsr[23:18]),
        .ev_ready(ready), .out_valid(ov), .out_x(ox), .out_y(oy)
    );

    reg led;
    always @(posedge CLK) if (ov) led <= led ^ (^{ox, oy});
    assign LEDR_N = ~led;
endmodule
