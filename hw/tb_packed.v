`timescale 1ns/1ps
// Co-simulation: drive stcd_frontend_packed from hw/cosim_vectors.txt and check
// every keep/drop decision against the faithful integer reference (gen_cosim.py).
module tb;
    parameter W=16, H=16, LOG2W=5, LOG2H=5, CB=2, TB=6;   // overridable via iverilog -P
    reg clk=0, rst=1, ev_valid=0;
    reg [LOG2W-1:0] ev_x; reg [LOG2H-1:0] ev_y; reg [TB-1:0] ev_t;
    wire ev_ready, out_valid; wire [LOG2W-1:0] out_x; wire [LOG2H-1:0] out_y;

    stcd_frontend_packed #(.W(W),.H(H),.LOG2W(LOG2W),.LOG2H(LOG2H),
                           .CB(CB),.TB(TB),.INC(1)) dut (
        .clk(clk),.rst(rst),.ev_valid(ev_valid),.ev_x(ev_x),.ev_y(ev_y),.ev_t(ev_t),
        .ev_ready(ev_ready),.out_valid(out_valid),.out_x(out_x),.out_y(out_y));

    always #5 clk = ~clk;

    integer fd, n, i, j, code, errors, hx, hy, ht, hkeep, tmp;
    reg kept;
    initial begin
        fd = $fopen("cosim_vectors.txt", "r");
        if (fd == 0) begin $display("cannot open cosim_vectors.txt"); $finish; end
        code = $fscanf(fd, "%d %d %d %d %d %d %d %d %d\n",
                       n, tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp);
        repeat (6) @(posedge clk); rst <= 0; repeat (2) @(posedge clk);
        while (!ev_ready) @(posedge clk);          // wait out the SPRAM clear
        errors = 0;
        for (i = 0; i < n; i = i + 1) begin
            code = $fscanf(fd, "%d %d %d %d\n", hx, hy, ht, hkeep);
            @(posedge clk); while (!ev_ready) @(posedge clk);
            ev_x <= hx[LOG2W-1:0]; ev_y <= hy[LOG2H-1:0]; ev_t <= ht[TB-1:0]; ev_valid <= 1;
            @(posedge clk); ev_valid <= 0;
            kept = 0;
            for (j = 0; j < 14; j = j + 1) begin
                @(posedge clk);
                if (out_valid && out_x == hx[LOG2W-1:0] && out_y == hy[LOG2H-1:0]) kept = 1;
            end
            if (kept !== hkeep[0]) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("MISMATCH ev %0d (%0d,%0d,t=%0d): rtl=%0b exp=%0d",
                             i, hx, hy, ht, kept, hkeep);
            end
        end
        if (errors == 0) $display("\nCOSIM PASSED: all %0d events match the reference", n);
        else             $display("\nCOSIM FAILED: %0d / %0d mismatches", errors, n);
        $fclose(fd); $finish;
    end
endmodule
