#!/usr/bin/env python3
"""rssi_distance.py -- link loss against distance and orientation, from an rssi_monitor.py CSV.

Run the monitor, and at each position type a label of the form  <distance>cm-<orientation>  (for example  30cm-base,  30cm-r90,
30cm-r270 ), then hold still for about 12 s. This script takes the 10 s after each label (see rssi_segments.py) and prints, per
distance, the loss in every orientation, the swing between orientations, the free-space loss for two ideal (0 dBi) antennas, and what
that implies for the antennas.

  python rssi_distance.py results/rssi_monitor_<time>.csv --txpwr 0

How to read it:
  * swing between orientations: if it stays above about 10 to 15 dB at 30 cm and beyond, the antennas themselves are directional; if it
    shrinks with distance, the large swing at 5 cm was near-field coupling.
  * implied gain = free-space loss minus measured loss, averaged over orientations: roughly the two antennas' gains together, minus
    mismatch and cable losses, plus the RSSI scale error. Only meaningful in the far field (30 cm or more; the wavelength is 12 cm).
  * fitted exponent n in loss = L0 + 10 n log10(d): free space is 2. A value much above 2 means obstacles, nulls or ground effects.
Indoor multipath moves single readings by several dB, so look at the trend over distances, not one point.
"""
import argparse
import math
import re
from collections import defaultdict

import numpy as np

from rssi_segments import stable_positions

C = 299792458.0
F_HZ = 2.44e9                       # middle of the BLE band
LAMBDA_M = C / F_HZ
FAR_FIELD_CM = 30.0
LABEL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*cm[-_ ]+([A-Za-z0-9]+)\s*$", re.I)


def fspl_db(d_m):
    return 20.0 * math.log10(4.0 * math.pi * d_m / LAMBDA_M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--txpwr", type=float, default=0.0, help="transmit power of the run in dBm (default 0)")
    ap.add_argument("--from", dest="t_from", type=float, default=1.0)
    ap.add_argument("--to", dest="t_to", type=float, default=11.0)
    ap.add_argument("--relabel", default="", help="fix labels typed wrongly, applied together and without touching the CSV: "
                    "'60cm-r90=60cm-r270,60cm-r270=60cm-r90' swaps those two")
    a = ap.parse_args()
    fix = dict(pair.split("=", 1) for pair in a.relabel.split(",") if "=" in pair)

    grid = defaultdict(dict)        # distance cm -> orientation -> (rx median, tx median)
    seen = defaultdict(int)
    for p in stable_positions(a.csv, a.t_from, a.t_to):
        label = fix.get(p["label"], p["label"])
        if label != p["label"]:
            print(f"relabelled '{p['label']}' (at {p['t0']:.0f} s) as '{label}'")
        m = LABEL.match(label)
        if m:
            key = (float(m.group(1)), m.group(2).lower())
            seen[key] += 1
            if seen[key] > 1:
                print(f"note: '{label}' occurs {seen[key]} times; the LAST one (at {p['t0']:.0f} s) is used")
            grid[key[0]][key[1]] = (p["rx_med"], p["tx_med"])
        elif p["label"] != "start":
            print(f"ignored label '{p['label']}' (not '<distance>cm-<orientation>')")
    if not grid:
        raise SystemExit("no labels of the form '<distance>cm-<orientation>' (for example 30cm-base) found in this CSV")
    dists = sorted(grid)
    orients = sorted({o for d in grid.values() for o in d}, key=lambda o: (o != "base", o))

    print(f"window {a.t_from:g}-{a.t_to:g} s after each label; loss = {a.txpwr:g} dBm - receiver reading; "
          f"free space for 0 dBi antennas at {F_HZ / 1e9:.2f} GHz (wavelength {LAMBDA_M * 100:.1f} cm)\n")
    head = f"{'cm':>5} | " + " ".join(f"{o:>7}" for o in orients) + f" | {'swing':>5} {'mean':>5} | {'free space':>10} {'implied gain':>12}"
    print("link loss in dB per orientation")
    print(head)
    loss = {}
    for d in dists:
        row = {o: a.txpwr - grid[d][o][0] for o in orients if o in grid[d]}
        loss[d] = row
        vals = list(row.values())
        swing, mean = (max(vals) - min(vals)) if len(vals) > 1 else float("nan"), float(np.mean(vals))
        fs = fspl_db(d / 100.0)
        far = d >= FAR_FIELD_CM
        print(f"{d:5g} | " + " ".join(f"{row[o]:7.1f}" if o in row else f"{'-':>7}" for o in orients)
              + f" | {swing:5.1f} {mean:5.1f} | {fs:10.1f} {(f'{fs - mean:+12.1f}' if far else '   (near field)'):>12}")

    far_d = [d for d in dists if d >= 15]
    for o in orients:
        pts = [(d, loss[d][o]) for d in far_d if o in loss[d]]
        if len(pts) >= 3:
            x = np.array([10.0 * math.log10(d / 100.0) for d, _ in pts]); y = np.array([l for _, l in pts])
            n, l0 = np.polyfit(x, y, 1)
            print(f"\n{o:>6}: loss = {l0:.1f} dB + 10 x {n:.2f} x log10(d / 1 m), fitted over {len(pts)} distances of 15 cm or more "
                  f"(free space: exponent 2.00)")
    sw = [(d, max(loss[d].values()) - min(loss[d].values())) for d in dists if len(loss[d]) > 1]
    if sw:
        print("\nswing between orientations by distance: " + ", ".join(f"{d:g} cm {s:.0f} dB" for d, s in sw))


if __name__ == "__main__":
    main()
