// uart_tx.v — minimal 8N1 UART transmitter (idle high, LSB first).
// DIV = clk/baud (e.g. 12e6 / 115200 ~= 104). Pulse `stb` for one cycle when
// `busy` is low to send `data`.
module uart_tx #(parameter integer DIV = 104) (
    input  wire       clk,
    input  wire       rst,
    input  wire       stb,
    input  wire [7:0] data,
    output reg        tx,
    output reg        busy
);
    reg [9:0]  sh;                       // {stop, data[7:0], start}
    reg [3:0]  nbits;
    reg [15:0] cnt;
    always @(posedge clk) begin
        if (rst) begin
            tx <= 1'b1; busy <= 1'b0; sh <= 10'h3FF; nbits <= 0; cnt <= 0;
        end else if (!busy) begin
            tx <= 1'b1;
            if (stb) begin sh <= {1'b1, data, 1'b0}; nbits <= 4'd10; cnt <= 0; busy <= 1'b1; end
        end else begin
            tx <= sh[0];
            if (cnt == DIV-1) begin
                cnt   <= 0;
                sh    <= {1'b1, sh[9:1]};
                nbits <= nbits - 1'b1;
                if (nbits == 4'd1) busy <= 1'b0;     // stop bit sent
            end else cnt <= cnt + 1'b1;
        end
    end
endmodule
