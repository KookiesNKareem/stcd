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
		--freq 24 --seed $(SEED) --asc $@
top_meas.bin: top_meas.asc
	icepack $< $@

# ---- on-board power measurement harness (idle vs active via button) --------
PSRCS = top_power.v stcd_frontend.v
PPCF  = icebreaker_power.pcf

top_power.json: $(PSRCS)
	yosys -p "read_verilog $(PSRCS); synth_ice40 -spram -top top_power -json $@"
top_power.asc: top_power.json $(PPCF)
	nextpnr-ice40 --up5k --package sg48 --json top_power.json --pcf $(PPCF) \
		--freq 24 --seed 1 --asc $@      # seed 1 closes at 24.7 MHz (2,4,6,8 also pass)
top_power.bin: top_power.asc
	icepack $< $@

.PHONY: all prog timing test clean meas meas-sim meas-prog power power-prog
all: $(TOP).bin
test:                                  # functional sim (Icarus Verilog)
	iverilog -g2012 -o tb.vvp tb_stcd.v stcd_frontend.v && vvp tb.vvp; rm -f tb.vvp
prog: $(TOP).bin
	iceprog $<
timing: $(TOP).asc
	icetime -d up5k -mtr $(TOP).rpt $(TOP).asc

meas-sim:                              # verify the measurement harness in sim (PLL bypassed)
	iverilog -g2012 -DSIM -o meas.vvp tb_meas.v top_meas.v uart_tx.v stcd_frontend.v && vvp meas.vvp; rm -f meas.vvp
meas: top_meas.bin                     # build the on-board measurement bitstream
meas-prog: top_meas.bin                # flash it, then run:  python read_uart.py
	iceprog $<

power: top_power.bin                   # build the power-measurement bitstream
power-prog: top_power.bin              # flash it, then toggle the button and read the USB meter
	iceprog $<

clean:
	rm -f $(TOP).json $(TOP).asc $(TOP).bin $(TOP).rpt \
	      top_meas.json top_meas.asc top_meas.bin \
	      top_power.json top_power.asc top_power.bin *.log
