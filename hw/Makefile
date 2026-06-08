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

.PHONY: all prog timing test clean
all: $(TOP).bin
test:                                  # functional sim (Icarus Verilog)
	iverilog -g2012 -o tb.vvp tb_stcd.v stcd_frontend.v && vvp tb.vvp; rm -f tb.vvp
prog: $(TOP).bin
	iceprog $<
timing: $(TOP).asc
	icetime -d up5k -mtr $(TOP).rpt $(TOP).asc
clean:
	rm -f $(TOP).json $(TOP).asc $(TOP).bin $(TOP).rpt *.log
