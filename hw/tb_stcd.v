`timescale 1ns/1ps
// Functional self-check for stcd_frontend (Icarus Verilog).
// Verifies the coincidence logic: isolated events are dropped; events with
// enough recent neighbour support are forwarded; the leak drops stale support.
module tb;
    localparam LOG2W = 4, LOG2H = 4;   // 16×16 grid (small, fast sim)
    reg clk = 0, rst = 1, ev_valid = 0;
    reg [LOG2W-1:0] ev_x; reg [LOG2H-1:0] ev_y; reg [7:0] ev_t;
    wire ev_ready, out_valid; wire [LOG2W-1:0] out_x; wire [LOG2H-1:0] out_y;

    stcd_frontend #(.LOG2W(LOG2W), .LOG2H(LOG2H), .THETA(8'd2), .INC(8'd1)) dut (
        .clk(clk), .rst(rst), .ev_valid(ev_valid), .ev_x(ev_x), .ev_y(ev_y),
        .ev_t(ev_t), .ev_ready(ev_ready), .out_valid(out_valid),
        .out_x(out_x), .out_y(out_y));

    always #5 clk = ~clk;   // 100 MHz

    integer errors = 0;
    reg kept;
    task send(input [7:0] x, input [7:0] y, input [7:0] t);
        integer i;
        begin
            @(posedge clk); while (!ev_ready) @(posedge clk);
            ev_x <= x[LOG2W-1:0]; ev_y <= y[LOG2H-1:0]; ev_t <= t; ev_valid <= 1;
            @(posedge clk); ev_valid <= 0;
            kept = 0;
            for (i = 0; i < 40; i = i + 1) begin
                @(posedge clk);
                if (out_valid && out_x == x[LOG2W-1:0] && out_y == y[LOG2H-1:0]) kept = 1;
            end
        end
    endtask
    task chk(input [7:0] x, input [7:0] y, input want);
        begin
            if (kept !== want) begin
                $display("FAIL: (%0d,%0d) kept=%0b expected=%0b", x, y, kept, want);
                errors = errors + 1;
            end else
                $display("ok:   (%0d,%0d) kept=%0b", x, y, kept);
        end
    endtask

    initial begin
        repeat (6) @(posedge clk); rst <= 0; repeat (2) @(posedge clk);

        // 1) isolated event, no neighbours → dropped
        send(10, 10, 0); chk(10, 10, 1'b0);

        // 2) two coincident neighbours of (6,6), then (6,6) → kept
        send(5, 5, 0);   // deposits into (6,6)
        send(7, 7, 0);   // deposits into (6,6)  → membrane(6,6)=2
        send(6, 6, 0); chk(6, 6, 1'b1);

        // 3) same spatial support but stale in time → leak drops it
        send(2, 2, 0); send(4, 4, 0);   // membrane(3,3)=2 at tick 0
        send(3, 3, 9); chk(3, 3, 1'b0);  // dt=9 → 2>>8 = 0 < θ

        // 4) one neighbour only (membrane=1 < θ=2) → dropped
        send(12, 12, 0); send(13, 13, 0); chk(13, 13, 1'b0);

        if (errors == 0) $display("\nALL TESTS PASSED");
        else $display("\n%0d TEST(S) FAILED", errors);
        $finish;
    end
endmodule
