"""Generate co-simulation vectors for stcd_frontend_packed.v.

Emits a small event stream (mixed isolated + coincident bursts so both keep and
drop paths are exercised) on an even-width grid, computes the EXPECTED keep/drop
with a faithful integer mirror of the RTL (count/tick/shift-leak/threshold, with
TB-bit tick wrap and CB-bit saturating count), and writes:
  hw/cosim_vectors.txt :  "<x> <y> <t_masked> <expected_keep>" per line
The header line is "<N> <W> <H> <LOG2W> <LOG2H> <CB> <TB> <THETA> <INC>".
"""
import os
import sys
import numpy as np

# argv: W H LOG2W LOG2H [CB TB N SEED]
_a = sys.argv[1:]
W = int(_a[0]) if len(_a) > 0 else 16          # even width
H = int(_a[1]) if len(_a) > 1 else 16
LOG2W = int(_a[2]) if len(_a) > 2 else 5
LOG2H = int(_a[3]) if len(_a) > 3 else 5
CB = int(_a[4]) if len(_a) > 4 else 2           # 2-bit count + 6-bit tick (chosen split)
TB = int(_a[5]) if len(_a) > 5 else 6
N = int(_a[6]) if len(_a) > 6 else 800
SEED = int(_a[7]) if len(_a) > 7 else 7
THETA, INC = 2, 1
OUT = os.path.join(os.path.dirname(__file__), "cosim_vectors.txt")


def leak(m, dt, cb):
    s = dt if dt < cb else cb
    return m >> s


def main():
    rng = np.random.default_rng(SEED)
    mask = (1 << TB) - 1
    maxc = (1 << CB) - 1
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # build an event stream: bursts of neighbours at close ticks + isolated noise
    evs = []          # (x, y, tick)
    tick = 0
    for _ in range(N):
        tick += int(rng.integers(0, 3))           # 0..2 tick gaps
        if rng.random() < 0.45:                    # coincident burst around a centre
            cx = int(rng.integers(2, W - 2)); cy = int(rng.integers(2, H - 2))
            for _ in range(int(rng.integers(2, 5))):
                dx, dy = NB[int(rng.integers(0, 8))]
                evs.append((cx + dx, cy + dy, tick))
            evs.append((cx, cy, tick))             # the centre event (should be kept)
        else:                                      # isolated noise event
            evs.append((int(rng.integers(0, W)), int(rng.integers(0, H)), tick))
    evs = evs[:N]

    cnt = np.zeros((H, W), np.int64)
    tk = np.zeros((H, W), np.int64)
    lines = []
    for (x, y, t) in evs:
        tm = t & mask
        support = 0
        for dx, dy in NB:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                dt = (tm - tk[ny, nx]) & mask
                support += leak(int(cnt[ny, nx]), dt, CB)
        keep = 1 if support >= THETA else 0
        dt0 = (tm - tk[y, x]) & mask
        newc = leak(int(cnt[y, x]), dt0, CB) + INC
        cnt[y, x] = maxc if newc > maxc else newc
        tk[y, x] = tm
        lines.append(f"{x} {y} {tm} {keep}")

    with open(OUT, "w") as f:
        f.write(f"{len(lines)} {W} {H} {LOG2W} {LOG2H} {CB} {TB} {THETA} {INC}\n")
        f.write("\n".join(lines) + "\n")
    kept = sum(int(l.split()[3]) for l in lines)
    print(f"wrote {OUT}: {len(lines)} events, {kept} kept ({100*kept/len(lines):.0f}%)")


if __name__ == "__main__":
    main()
