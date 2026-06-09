# iCEBreaker (iCE40 UP5K, sg48) build flow — open toolchain.
TOP    = top
SRCS   = top.v stcd_frontend.v
PCF    = icebreaker.pcf
FREQ   = 20     # PnR target; design closes ~24-26 MHz (seed-dependent). Board osc 12 MHz.
SEED   = 2      # placement seed giving best fmax (swept 1..6 → ~26 MHz)

$(TOP).json: $(SRCS)
	yosys -p "read_verilog $(SRCS); synth_ice40 -spram -top $(TOP) -json $@"

$(TOP).asc: $(TOP).json $(PCF)
	nextpnr-ice40 --up5k --package sg48 --json $(TOP).json --pcf $(PCF) \
		--freq $(FREQ) --seed $(SEED) --asc $@

$(TOP).bin: $(TOP).asc
	icepack $< $@

# ---- on-board throughput/latency measurement harness -----------------------
MSRCS = top_meas.v stcd_frontend.v uart_tx.v
MPCF  = icebreaker_meas.pcf

top_meas.json: $(MSRCS)
	yosys -p "read_verilog $(MSRCS); synth_ice40 -spram -top top_meas -json $@"
top_meas.asc: top_meas.json $(MPCF)
	nextpnr-ice40 --up5k --package sg48 --json top_meas.json --pcf $(MPCF) \
		--freq $(FREQ) --seed $(SEED) --asc $@
top_meas.bin: top_meas.asc
	icepack $< $@

.PHONY: all prog timing test clean meas meas-sim meas-prog
all: $(TOP).bin
test:                                  # functional sim (Icarus Verilog)
	iverilog -g2012 -o tb.vvp tb_stcd.v stcd_frontend.v && vvp tb.vvp; rm -f tb.vvp
prog: $(TOP).bin
	iceprog $<
timing: $(TOP).asc
	icetime -d up5k -mtr $(TOP).rpt $(TOP).asc

meas-sim:                              # verify the measurement harness in sim
	iverilog -g2012 -o meas.vvp tb_meas.v top_meas.v uart_tx.v stcd_frontend.v && vvp meas.vvp; rm -f meas.vvp
meas: top_meas.bin                     # build the on-board measurement bitstream
meas-prog: top_meas.bin                # flash it, then run:  python read_uart.py
	iceprog $<

clean:
	rm -f $(TOP).json $(TOP).asc $(TOP).bin $(TOP).rpt top_meas.json top_meas.asc top_meas.bin *.log
