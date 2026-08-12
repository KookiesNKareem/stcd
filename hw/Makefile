# iCEBreaker (iCE40 UP5K, sg48) build flow — open toolchain.
TOP    = top
SRCS   = top.v stcd_frontend.v
PCF    = icebreaker.pcf
FREQ   = 20     # PnR target; design closes ~24-26 MHz (seed-dependent). Board osc 12 MHz.
SEED   = 2      # placement seed giving best fmax (swept 1..6 → ~26 MHz)

# Bare `make` runs the full flow (synth → place-and-route → bitstream) so the
# nextpnr resource report is produced, as the README describes.
.DEFAULT_GOAL := all

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

# ---- full-resolution PACKED design (DAVIS346, 8 b/px = 2-bit count + 6-bit tick)
# The headline full-sensor result: co-sim verified (packed-test), characterised
# (packed), measured on-board (packed-meas, then python read_uart.py).
KSRCS  = top_packed.v stcd_frontend_packed.v
KMSRCS = top_packed_meas.v stcd_frontend_packed.v uart_tx.v

top_packed.json: $(KSRCS)
	yosys -p "read_verilog $(KSRCS); synth_ice40 -spram -top top_packed -json $@"
top_packed.asc: top_packed.json $(PCF)
	nextpnr-ice40 --up5k --package sg48 --json top_packed.json --pcf $(PCF) \
		--freq 24 --seed $(SEED) --asc $@
top_packed.bin: top_packed.asc
	icepack $< $@

KPSRCS = top_packed_power.v stcd_frontend_packed.v

top_packed_power.json: $(KPSRCS)
	yosys -p "read_verilog $(KPSRCS); synth_ice40 -spram -top top_packed_power -json $@"
top_packed_power.asc: top_packed_power.json $(PPCF)
	nextpnr-ice40 --up5k --package sg48 --json top_packed_power.json --pcf $(PPCF) \
		--freq 24 --seed $(SEED) --asc $@
top_packed_power.bin: top_packed_power.asc
	icepack $< $@

top_packed_meas.json: $(KMSRCS)
	yosys -p "read_verilog $(KMSRCS); synth_ice40 -spram -top top_packed_meas -json $@"
top_packed_meas.asc: top_packed_meas.json $(MPCF)
	nextpnr-ice40 --up5k --package sg48 --json top_packed_meas.json --pcf $(MPCF) \
		--freq 24 --seed $(SEED) --asc $@
top_packed_meas.bin: top_packed_meas.asc
	icepack $< $@

.PHONY: all prog timing test clean meas meas-sim meas-prog power power-prog \
        packed packed-prog packed-meas packed-meas-prog packed-test \
        packed-power packed-power-prog
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

packed-test:                           # co-sim packed RTL vs the fixed-point reference (DAVIS346)
	python3 gen_cosim.py && iverilog -g2012 -o ptb.vvp tb_packed.v stcd_frontend_packed.v && vvp ptb.vvp; rm -f ptb.vvp
packed: top_packed.bin                 # synth+PnR the full DAVIS346 packed core (resource report)
packed-prog: top_packed.bin
	iceprog $<
packed-meas: top_packed_meas.bin       # build the full-res on-board measurement bitstream
packed-meas-prog: top_packed_meas.bin  # flash it, then run:  python read_uart.py
	iceprog $<
packed-power: top_packed_power.bin     # build the full-res power-measurement bitstream
packed-power-prog: top_packed_power.bin # flash it, then toggle the button and read the USB meter
	iceprog $<

clean:
	rm -f $(TOP).json $(TOP).asc $(TOP).bin $(TOP).rpt \
	      top_meas.json top_meas.asc top_meas.bin \
	      top_power.json top_power.asc top_power.bin \
	      top_packed.json top_packed.asc top_packed.bin \
	      top_packed_meas.json top_packed_meas.asc top_packed_meas.bin \
	      top_packed_power.json top_packed_power.asc top_packed_power.bin *.log
