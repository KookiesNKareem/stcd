`timescale 1ns/1ps
// Verifies top_meas in simulation: runs a small measurement window, checks the
// measured cycles/event is ~9 (fixed-latency FSM), and decodes the UART packet
// to confirm framing matches the internal counters.
module tb_meas;
    localparam integer CLKP = 10;        // 100 MHz sim clock (period ns)
    localparam integer DIV  = 16;        // UART bit = DIV clk cycles
    localparam integer BIT  = CLKP * DIV;
    localparam integer NWIN = 200;

    reg clk = 0;
    wire tx, led;
    always #(CLKP/2) clk = ~clk;

    top_meas #(.NWIN(NWIN[31:0]), .DIV(DIV)) dut (.CLK(clk), .TX(tx), .LEDR_N(led));

    reg [7:0] b, prev;
    reg [31:0] cyc, nev, kept;
    integer i, errors = 0;

    task uart_get(output [7:0] ob);
        integer k;
        begin
            @(negedge tx); #(BIT*1.5);              // center of bit0
            for (k = 0; k < 8; k = k + 1) begin ob[k] = tx; #(BIT); end
        end
    endtask

    initial begin
        wait (dut.running == 1'b0);                 // measurement finished
        $display("peek : cyc=%0d nev=%0d kept=%0d  cyc/ev=%0.3f",
                 dut.cyc, dut.nev, dut.kept, dut.cyc * 1.0 / dut.nev);
        if (dut.nev !== NWIN) begin $display("FAIL: nev=%0d != %0d", dut.nev, NWIN); errors = errors + 1; end
        if (dut.cyc < 9*NWIN - 60 || dut.cyc > 9*NWIN + 60) begin
            $display("FAIL: cyc=%0d not ~%0d", dut.cyc, 9*NWIN); errors = errors + 1; end

        // sync on 0xAA 0x55, then read three big-endian u32
        prev = 8'h00; uart_get(b);
        while (!(prev == 8'hAA && b == 8'h55)) begin prev = b; uart_get(b); end
        cyc = 0; nev = 0; kept = 0;
        for (i = 0; i < 4; i = i + 1) begin uart_get(b); cyc  = (cyc  << 8) | b; end
        for (i = 0; i < 4; i = i + 1) begin uart_get(b); nev  = (nev  << 8) | b; end
        for (i = 0; i < 4; i = i + 1) begin uart_get(b); kept = (kept << 8) | b; end
        $display("uart : cyc=%0d nev=%0d kept=%0d", cyc, nev, kept);
        if (cyc !== dut.cyc || nev !== dut.nev || kept !== dut.kept) begin
            $display("FAIL: UART packet != counters"); errors = errors + 1; end

        if (errors == 0) $display("\nMEAS HARNESS OK  (cycles/event = %0.3f)", cyc * 1.0 / nev);
        else $display("\n%0d TEST(S) FAILED", errors);
        $finish;
    end

    initial begin #5000000 $display("TIMEOUT"); $finish; end
endmodule
