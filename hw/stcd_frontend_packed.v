// stcd_frontend_packed.v — full-resolution (DAVIS346) packed variant.
//
// Same algorithm and parity-banked 9-cycle datapath as stcd_frontend.v, but the
// per-pixel state is packed to 8 bits (CB-bit saturating count + TB-bit tick) and
// TWO pixels share one 16-bit SPRAM word, with linear (non-power-of-2) addressing.
// This fits the full W×H sensor in 4 SPRAM on the iCE40 UP5K, where the
// power-of-2 16-bit design needs ~32 (see paper). Bit allocation 2c+6t is chosen
// from the precision study (ΔAUC ≈ −0.003 vs the 16-bit design).
//
// Bank-local layout: bank = x[0]^y[0]; within a bank a pixel is (row=y, col=x>>1)
// so each bank holds H×(W/2) pixels; packing 2/word → W·H/4 words/bank.
//   bli   = y*(W/2) + (x>>1)         bank-local pixel index
//   addr  = bli >> 1                 SPRAM word address
//   half  = bli[0]                   which byte of the 16-bit word
// The row base y*(W/2) is computed once per event (one constant multiply → adders,
// no DSP); neighbour rows reuse base±(W/2).

module stcd_frontend_packed #(
    parameter W     = 346,
    parameter H     = 260,
    parameter LOG2W = 9,         // ceil(log2(W))
    parameter LOG2H = 9,         // ceil(log2(H))
    parameter CB    = 2,         // count bits  (max count 2^CB-1; also max leak shift)
    parameter TB    = 6,         // tick  bits
    parameter [CB-1:0] THETA_C = 2'd2,   // placeholder width; real THETA below
    parameter INC   = 1
)(
    input  wire             clk,
    input  wire             rst,
    input  wire             ev_valid,
    input  wire [LOG2W-1:0] ev_x,
    input  wire [LOG2H-1:0] ev_y,
    input  wire [TB-1:0]    ev_t,
    output wire             ev_ready,
    output reg              out_valid,
    output reg  [LOG2W-1:0] out_x,
    output reg  [LOG2H-1:0] out_y
);
    localparam HALFW = W/2;                       // columns per bank-local row
    localparam NPIXB = H*HALFW;                   // pixels per bank
    localparam NWORD = (NPIXB+1)/2;               // 16-bit words per bank
    localparam AW    = 15;                        // ceil(log2(NWORD)) ; 22490 -> 15
    localparam BW    = AW+1;                       // bank-local pixel index width
    localparam PIXW  = CB+TB;                      // = 8
    localparam [4:0] THETA = 5'd2;
    localparam S_CLR=0, S_IDLE=1, S_GATH=2, S_OWAIT=3, S_OWR=4;

    reg [2:0]        state;
    reg [LOG2W-1:0]  x; reg [LOG2H-1:0] y; reg [TB-1:0] t;
    reg [2:0]        gc;
    reg              m0inb_d1,m0inb_d2,m1inb_d1,m1inb_d2;
    reg              m0h_d1,m0h_d2,m1h_d1,m1h_d2;   // half-select pipeline
    reg [4:0]        sup0, sup1;
    reg              keep;
    reg [AW-1:0]     clr;
    reg [BW-1:0]     rb0, rbm, rbp;                 // y*HALFW and ±HALFW (neighbour rows)

    // ---- per-cycle neighbour pair: orthogonal (opposite bank) + diagonal ----
    reg signed [1:0] oox, ooy, dox, doy;
    always @(*) case (gc)
        3'd0: begin oox= 0; ooy=-1; dox=-1; doy=-1; end
        3'd1: begin oox=-1; ooy= 0; dox= 1; doy=-1; end
        3'd2: begin oox= 1; ooy= 0; dox=-1; doy= 1; end
        default: begin oox= 0; ooy= 1; dox= 1; doy= 1; end
    endcase
    wire signed [LOG2W+1:0] axs = $signed({2'b00,x}) + oox;   // orthogonal x
    wire signed [LOG2H+1:0] ays = $signed({2'b00,y}) + ooy;   // orthogonal y
    wire signed [LOG2W+1:0] bxs = $signed({2'b00,x}) + dox;   // diagonal x
    wire signed [LOG2H+1:0] bys = $signed({2'b00,y}) + doy;   // diagonal y
    wire [LOG2W-1:0] anx=axs[LOG2W-1:0], bnx=bxs[LOG2W-1:0];
    wire ainb=(axs>=0)&&(axs<W)&&(ays>=0)&&(ays<H);
    wire binb=(bxs>=0)&&(bxs<W)&&(bys>=0)&&(bys<H);

    // bank-local pixel index for each neighbour (row base reused; no per-cycle mult)
    function [BW-1:0] rowbase; input signed [1:0] dy; begin
        rowbase = (dy<0) ? rbm : (dy>0) ? rbp : rb0; end
    endfunction
    wire [BW-1:0] a_bli = rowbase(ooy) + {{(BW-LOG2W+1){1'b0}}, anx[LOG2W-1:1]};
    wire [BW-1:0] b_bli = rowbase(doy) + {{(BW-LOG2W+1){1'b0}}, bnx[LOG2W-1:1]};
    wire [AW-1:0] a_wa = a_bli[BW-1:1];  wire a_hf = a_bli[0];
    wire [AW-1:0] b_wa = b_bli[BW-1:1];  wire b_hf = b_bli[0];

    // route parity-0 member to bank0, parity-1 to bank1 (orthogonal vs diagonal)
    wire a_is0 = ~(anx[0]^ays[0]);
    wire [AW-1:0] m0a = a_is0 ? a_wa : b_wa;
    wire [AW-1:0] m1a = a_is0 ? b_wa : a_wa;
    wire          m0h = a_is0 ? a_hf : b_hf;
    wire          m1h = a_is0 ? b_hf : a_hf;
    wire          m0inb = a_is0 ? ainb : binb;
    wire          m1inb = a_is0 ? binb : ainb;

    // ---- two SPRAMs (packed words) -----------------------------------------
    reg [15:0] mem0 [0:NWORD-1];
    reg [15:0] mem1 [0:NWORD-1];
    reg [AW-1:0] a0,a1; reg we0,we1; reg [15:0] wd0,wd1,rd0,rd1;
    always @(posedge clk) begin
        if (we0) mem0[a0]<=wd0; else rd0<=mem0[a0];
        if (we1) mem1[a1]<=wd1; else rd1<=mem1[a1];
    end

    // select the byte (pixel) for the read that lands this cycle, then leak it
    function [CB-1:0] leak; input [CB-1:0] m; input [TB-1:0] dt; reg [TB-1:0] s; begin
        s = (dt > CB) ? CB[TB-1:0] : dt; leak = m >> s; end
    endfunction
    wire [PIXW-1:0] pix0 = m0h_d2 ? rd0[15:8] : rd0[7:0];
    wire [PIXW-1:0] pix1 = m1h_d2 ? rd1[15:8] : rd1[7:0];
    wire [CB-1:0] c0 = leak(pix0[CB-1:0], t - pix0[PIXW-1:CB]);
    wire [CB-1:0] c1 = leak(pix1[CB-1:0], t - pix1[PIXW-1:CB]);

    // own pixel
    wire          ownp  = x[0]^y[0];
    wire [BW-1:0] o_bli = rb0 + {{(BW-LOG2W+1){1'b0}}, x[LOG2W-1:1]};
    wire [AW-1:0] o_wa  = o_bli[BW-1:1];  wire o_hf = o_bli[0];
    wire [15:0]   own_word = ownp ? rd1 : rd0;
    wire [PIXW-1:0] own_pix = o_hf ? own_word[15:8] : own_word[7:0];
    wire [CB-1:0] own_lk  = leak(own_pix[CB-1:0], t - own_pix[PIXW-1:CB]);
    wire [CB:0]   own_sum = {1'b0, own_lk} + INC[CB:0];
    wire [CB-1:0] own_new = own_sum[CB] ? {CB{1'b1}} : own_sum[CB-1:0];
    wire [PIXW-1:0] own_newpix = {t, own_new};
    // RMW: keep the paired pixel, replace own half
    wire [15:0]   own_wr = o_hf ? {own_newpix, own_word[7:0]} : {own_word[15:8], own_newpix};

    assign ev_ready = (state == S_IDLE);

    always @(posedge clk) begin
        if (rst) begin
            state<=S_CLR; clr<=0; we0<=0; we1<=0; out_valid<=0;
        end else begin
            we0<=0; we1<=0; out_valid<=0;
            case (state)
                S_CLR: begin
                    we0<=1; we1<=1; a0<=clr; a1<=clr; wd0<=16'd0; wd1<=16'd0;
                    if (clr == (NWORD-1)) state<=S_IDLE; else clr<=clr+1'b1;
                end
                S_IDLE: if (ev_valid) begin
                    x<=ev_x; y<=ev_y; t<=ev_t; gc<=0; sup0<=0; sup1<=0;
                    rb0 <= ev_y*HALFW; rbm <= ev_y*HALFW - HALFW; rbp <= ev_y*HALFW + HALFW;
                    m0inb_d1<=0; m0inb_d2<=0; m1inb_d1<=0; m1inb_d2<=0;
                    state<=S_GATH;
                end
                S_GATH: begin
                    if (gc >= 3'd2) begin
                        if (m0inb_d2) sup0 <= sup0 + {3'b0, c0};
                        if (m1inb_d2) sup1 <= sup1 + {3'b0, c1};
                    end
                    m0inb_d2<=m0inb_d1; m1inb_d2<=m1inb_d1;
                    m0h_d2<=m0h_d1;     m1h_d2<=m1h_d1;
                    m0inb_d1<=(gc<3'd4)?m0inb:1'b0;
                    m1inb_d1<=(gc<3'd4)?m1inb:1'b0;
                    m0h_d1<=m0h; m1h_d1<=m1h;
                    if (gc < 3'd4) begin a0<=m0a; a1<=m1a; end
                    if (gc == 3'd5) begin
                        if (ownp) a1<=o_wa; else a0<=o_wa;   // own read into its bank
                        state<=S_OWAIT;
                    end else gc<=gc+1'b1;
                end
                S_OWAIT: begin
                    keep <= (({1'b0,sup0}+{1'b0,sup1}) >= {1'b0,THETA});
                    state<=S_OWR;
                end
                S_OWR: begin
                    if (ownp) begin we1<=1; a1<=o_wa; wd1<=own_wr; end
                    else      begin we0<=1; a0<=o_wa; wd0<=own_wr; end
                    if (keep) begin out_valid<=1; out_x<=x; out_y<=y; end
                    state<=S_IDLE;
                end
            endcase
        end
    end
endmodule
