#!/usr/bin/env python3
"""rssi_segments.py -- stable reading per labelled position from an rssi_monitor.py CSV.

rssi_monitor.py averages each segment over its whole length. If you type a label after you have moved a board (the usual way), the
START of the segment is the new position and the END contains the move to the next one, so the whole-segment mean is dragged by the
transitions. This script takes only a window just after each label (default 1 s to 11 s after it) and reports the median and range of
the RX and TX readings there, plus the link loss at the transmit power you give.

  python rssi_segments.py results/rssi_monitor_20261104_123406.csv
  python rssi_segments.py results/rssi_monitor_20261104_123406.csv --from 2 --to 12 --txpwr 0
  python rssi_segments.py results/rssi_monitor_20261104_123406.csv --series      # median per 4 s window, to see the moves

Check the --series output before trusting a window: a position held for less than the window, or labelled late, shows up as a step.
"""
import argparse
import csv

import numpy as np


def read_runs(path):
    """Consecutive runs of one label, in time order (a label typed twice gives two runs): [{seg, t0, t1, pts=[(t, node, rssi)]}]."""
    runs = []
    for r in csv.DictReader(open(path)):
        t, node, v, seg = float(r["t_s"]), r["node"], int(r["rssi_dbm"]), r["segment"]
        if not runs or runs[-1]["seg"] != seg:
            runs.append({"seg": seg, "t0": t, "t1": t, "pts": []})
        runs[-1]["t1"] = t
        runs[-1]["pts"].append((t, node, v))
    return runs


def stable_positions(path, t_from=1.0, t_to=11.0, min_samples=5):
    """One row per run with enough samples in the window: label, held seconds, RX/TX median, min, max, sample count."""
    out = []
    for g in read_runs(path):
        w = [(n, v) for t, n, v in g["pts"] if t_from <= t - g["t0"] < t_to]
        rx = [v for n, v in w if n == "RX"]
        tx = [v for n, v in w if n == "TX"]
        if len(rx) < min_samples:
            continue
        out.append(dict(label=g["seg"], held_s=g["t1"] - g["t0"], rx_med=float(np.median(rx)), rx_min=min(rx), rx_max=max(rx),
                        tx_med=float(np.median(tx)) if tx else None, tx_min=min(tx) if tx else None, tx_max=max(tx) if tx else None,
                        n=len(rx), t0=g["t0"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--from", dest="t_from", type=float, default=1.0, help="window start, seconds after the label (default 1)")
    ap.add_argument("--to", dest="t_to", type=float, default=11.0, help="window end, seconds after the label (default 11)")
    ap.add_argument("--txpwr", type=float, default=0.0, help="transmit power of the run, for the loss column (default 0 dBm)")
    ap.add_argument("--series", action="store_true", help="also print the RX median per 4 s window of every segment run")
    a = ap.parse_args()

    print(f"window: {a.t_from:g} s to {a.t_to:g} s after each label; loss = {a.txpwr:g} dBm - RX median\n")
    print(f"{'position':>14} {'held s':>6} | {'RX med':>6} {'min':>4} {'max':>4} | {'TX med':>6} {'min':>4} {'max':>4} | {'loss dB':>7}   n")
    pos = stable_positions(a.csv, a.t_from, a.t_to)
    for p in pos:
        txm = f"{p['tx_med']:6.1f} {p['tx_min']:4d} {p['tx_max']:4d}" if p["tx_med"] is not None else "   n/a"
        print(f"{p['label']:>14} {p['held_s']:6.0f} | {p['rx_med']:6.1f} {p['rx_min']:4d} {p['rx_max']:4d} | {txm} | "
              f"{a.txpwr - p['rx_med']:7.1f}   {p['n']}")
    if len(pos) > 1:
        lo, hi = min(pos, key=lambda p: p["rx_med"]), max(pos, key=lambda p: p["rx_med"])
        print(f"\nrange of the stable readings: {lo['label']} {lo['rx_med']:.0f} dBm (weakest) to {hi['label']} {hi['rx_med']:.0f} dBm "
              f"(strongest) = {hi['rx_med'] - lo['rx_med']:.0f} dB")

    if a.series:
        print("\nRX median per 4 s window within each run (dBm):")
        for i, g in enumerate(read_runs(a.csv)):
            rx = [(t, v) for t, n, v in g["pts"] if n == "RX"]
            wins, t = [], g["t0"]
            while t <= g["t1"]:
                w = [v for tt, v in rx if t <= tt < t + 4]
                if w:
                    wins.append(f"{int(np.median(w))}")
                t += 4
            print(f"{i:>2} {g['seg']:>14} {g['t0']:7.1f}-{g['t1']:6.1f} s | {' '.join(wins)}")


if __name__ == "__main__":
    main()
