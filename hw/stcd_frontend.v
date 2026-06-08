// stcd_frontend.v — event-driven spiking coincidence denoiser (iCE40 target)
//
// GATHER + CHECKERBOARD BANKING (optimised). Pixel state is split across TWO
// single-port RAMs by bank = (x[0]^y[0]) → two iCE40 SPRAMs. On a checkerboard the
// 8-neighbourhood is exactly 4 same-bank (diagonals) + 4 opposite-bank
// (orthogonals), so the two banks are read in PARALLEL — one neighbour from each
// per cycle → ~6-cycle gather (vs ~10 single-bank).
//
// Per event (x,y,t): sum the 8 neighbours' leaked counts → support; forward iff
// support>=THETA; then RMW the centre pixel's own count (in its bank). This is
// exactly the Python front-end's membrane. No multipliers (leak = shift).
// ~9 cycles/event (own read overlaps the last gather cycle).

module stcd_frontend #(
    parameter LOG2W = 7,
    parameter LOG2H = 7,
    parameter MEMW  = 8,
    parameter TBITS = 8,
    parameter [MEMW-1:0] THETA = 8'd2,
    parameter [MEMW-1:0] INC   = 8'd1
)(
    input  wire             clk,
    input  wire             rst,
    input  wire             ev_valid,
    input  wire [LOG2W-1:0] ev_x,
    input  wire [LOG2H-1:0] ev_y,
    input  wire [TBITS-1:0] ev_t,
    output wire             ev_ready,
    output reg              out_valid,
    output reg  [LOG2W-1:0] out_x,
    output reg  [LOG2H-1:0] out_y
);
    localparam AW = LOG2W + LOG2H;
    localparam NPIX = (1 << AW);
    localparam S_CLR=0, S_IDLE=1, S_GATH=2, S_OWAIT=3, S_OWR=4;

    // ---- FSM state ---------------------------------------------------------
    reg [2:0]       state;
    reg [LOG2W-1:0] x; reg [LOG2H-1:0] y; reg [TBITS-1:0] t;
    reg [2:0]       gc;                                 // gather counter 0..5
    reg             m0inb_d1, m0inb_d2, m1inb_d1, m1inb_d2;  // bounds pipeline
    reg [MEMW+2:0]  sup0, sup1;          // per-bank running sums (parallel adders)
    reg             keep;
    reg [AW-1:0]    clr;

    // ---- per-cycle neighbour pair: orthogonal (opposite bank) + diagonal ----
    reg signed [1:0] oox, ooy, dox, doy;
    always @(*) case (gc)
        3'd0: begin oox= 0; ooy=-1; dox=-1; doy=-1; end
        3'd1: begin oox=-1; ooy= 0; dox= 1; doy=-1; end
        3'd2: begin oox= 1; ooy= 0; dox=-1; doy= 1; end
        default: begin oox= 0; ooy= 1; dox= 1; doy= 1; end
    endcase
    wire signed [LOG2W+1:0] axs = $signed({2'b00,x}) + oox;  // orthogonal coord
    wire signed [LOG2H+1:0] ays = $signed({2'b00,y}) + ooy;
    wire signed [LOG2W+1:0] bxs = $signed({2'b00,x}) + dox;  // diagonal coord
    wire signed [LOG2H+1:0] bys = $signed({2'b00,y}) + doy;
    wire [LOG2W-1:0] anx=axs[LOG2W-1:0], bnx=bxs[LOG2W-1:0];
    wire [LOG2H-1:0] any=ays[LOG2H-1:0], bny=bys[LOG2H-1:0];
    wire ainb=(axs>=0)&&(axs<(1<<LOG2W))&&(ays>=0)&&(ays<(1<<LOG2H));
    wire binb=(bxs>=0)&&(bxs<(1<<LOG2W))&&(bys>=0)&&(bys<(1<<LOG2H));
    // route the parity-0 member to bank0, parity-1 member to bank1 (always splits)
    wire a_is0 = ~(anx[0]^any[0]);                      // orthogonal has parity 0?
    wire [AW-1:0] m0a   = a_is0 ? {any,anx} : {bny,bnx};
    wire [AW-1:0] m1a   = a_is0 ? {bny,bnx} : {any,anx};
    wire          m0inb = a_is0 ? ainb : binb;
    wire          m1inb = a_is0 ? binb : ainb;

    // ---- two single-port RAMs → two SPRAMs ---------------------------------
    reg [15:0] mem0 [0:NPIX-1];
    reg [15:0] mem1 [0:NPIX-1];
    reg [AW-1:0] a0,a1; reg we0,we1; reg [15:0] wd0,wd1,rd0,rd1;
    always @(posedge clk) begin
        if (we0) mem0[a0]<=wd0; else rd0<=mem0[a0];
        if (we1) mem1[a1]<=wd1; else rd1<=mem1[a1];
    end

    // lazy leak: count >> min(dt, MEMW)
    function [MEMW-1:0] leak;
        input [MEMW-1:0] m; input [TBITS-1:0] dt; reg [TBITS-1:0] s;
        begin s = (dt > MEMW) ? MEMW[TBITS-1:0] : dt; leak = m >> s; end
    endfunction
    wire [MEMW-1:0] c0 = leak(rd0[MEMW-1:0], t - rd0[8 +: TBITS]);  // bank0 neighbour
    wire [MEMW-1:0] c1 = leak(rd1[MEMW-1:0], t - rd1[8 +: TBITS]);  // bank1 neighbour

    wire        ownp     = x[0] ^ y[0];                 // centre pixel's bank
    wire [15:0] own_word = ownp ? rd1 : rd0;
    wire [MEMW-1:0] own_lk = leak(own_word[MEMW-1:0], t - own_word[8 +: TBITS]);
    wire [MEMW:0]   own_sum = {1'b0, own_lk} + INC;
    wire [MEMW-1:0] own_new = own_sum[MEMW] ? {MEMW{1'b1}} : own_sum[MEMW-1:0];

    assign ev_ready = (state == S_IDLE);

    always @(posedge clk) begin
        if (rst) begin
            state <= S_CLR; clr <= 0; we0 <= 0; we1 <= 0; out_valid <= 0;
        end else begin
            we0 <= 0; we1 <= 0; out_valid <= 0;
            case (state)
                S_CLR: begin                            // clear both banks in parallel
                    we0<=1; we1<=1; a0<=clr; a1<=clr; wd0<=16'd0; wd1<=16'd0;
                    if (clr == {AW{1'b1}}) state <= S_IDLE; else clr <= clr + 1'b1;
                end
                S_IDLE: if (ev_valid) begin
                    x<=ev_x; y<=ev_y; t<=ev_t; gc<=0; sup0<=0; sup1<=0;
                    m0inb_d1<=0; m0inb_d2<=0; m1inb_d1<=0; m1inb_d2<=0;
                    state<=S_GATH;
                end
                S_GATH: begin
                    // 2-deep pipeline: pair issued 2 cycles ago lands now (both banks);
                    // each bank accumulates into its own sum (two independent adders).
                    if (gc >= 3'd2) begin
                        if (m0inb_d2) sup0 <= sup0 + {{3{1'b0}}, c0};
                        if (m1inb_d2) sup1 <= sup1 + {{3{1'b0}}, c1};
                    end
                    m0inb_d2<=m0inb_d1; m1inb_d2<=m1inb_d1;
                    m0inb_d1<=(gc<3'd4)?m0inb:1'b0;
                    m1inb_d1<=(gc<3'd4)?m1inb:1'b0;
                    if (gc < 3'd4) begin a0<=m0a; a1<=m1a; end  // parallel pair read
                    if (gc == 3'd5) begin
                        // last pair read already committed → reuse ownp bank for own read
                        if (ownp) a1<={y,x}; else a0<={y,x};
                        state<=S_OWAIT;
                    end else gc<=gc+1'b1;
                end
                S_OWAIT: begin                          // own read latency; finalise decision
                    keep <= (({1'b0,sup0} + {1'b0,sup1}) >= {{4{1'b0}}, THETA});
                    state<=S_OWR;
                end
                S_OWR: begin                            // own RMW write + emit
                    if (ownp) begin we1<=1; a1<={y,x}; wd1<={t,own_new}; end
                    else      begin we0<=1; a0<={y,x}; wd0<={t,own_new}; end
                    if (keep) begin out_valid<=1; out_x<=x; out_y<=y; end
                    state<=S_IDLE;
                end
            endcase
        end
    end
endmodule
