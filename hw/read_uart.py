#!/usr/bin/env python3
"""Read STCD's on-board throughput measurement over UART (115200 8N1).

The iCEBreaker (running top_meas.bin) repeatedly sends a 14-byte packet:
  0xAA 0x55 | cycles(u32 BE) | n_events(u32 BE) | n_kept(u32 BE)
This computes measured cycles/event = cycles / n_events and
throughput = f_clk / (cycles/event). cycles/event is clock-independent (the FSM
is fixed-latency), so it is the silicon-validated efficiency number; throughput
scales with whatever clock the design runs at on the board.

Usage:
  pip install pyserial
  python read_uart.py [--port /dev/cu.usbserial-XXXX] [--clk 12e6] [--n 5]

The iCEBreaker exposes TWO serial interfaces (FT2232H); the UART is the second
one. If --port is omitted this lists candidates and uses the last.
"""
import argparse
import glob
import struct
import sys

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed:  pip install pyserial")


def candidates():
    return (sorted(glob.glob("/dev/cu.usbserial*"))
            + sorted(glob.glob("/dev/ttyUSB*"))
            + sorted(glob.glob("/dev/cu.usbmodem*")))


def read_packet(s):
    prev = -1
    while True:
        b = s.read(1)
        if not b:
            sys.exit("UART timeout — check the board is running top_meas and the port is right")
        b = b[0]
        if prev == 0xAA and b == 0x55:
            break
        prev = b
    cyc, nev, kept = struct.unpack(">III", s.read(12))
    return cyc, nev, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--clk", type=float, default=12e6, help="board clock in Hz (12e6 = iCEBreaker osc)")
    ap.add_argument("--n", type=int, default=5, help="packets to read")
    a = ap.parse_args()

    port = a.port
    if not port:
        c = candidates()
        if not c:
            sys.exit("no serial port found; pass --port (iCEBreaker's 2nd FTDI interface)")
        port = c[-1]
        print(f"using {port}  (candidates: {c})")
    s = serial.Serial(port, a.baud, timeout=3)
    print(f"reading {a.n} packets, assuming f_clk = {a.clk/1e6:.1f} MHz ...")
    for _ in range(a.n):
        cyc, nev, kept = read_packet(s)
        cpe = cyc / nev if nev else float("nan")
        thr = a.clk / cpe / 1e6 if cpe else float("nan")
        print(f"  cycles={cyc:,}  events={nev:,}  kept={kept:,} ({100*kept/nev:.1f}%)  "
              f"| cycles/event={cpe:.3f}  throughput={thr:.3f} Mev/s")


if __name__ == "__main__":
    main()
