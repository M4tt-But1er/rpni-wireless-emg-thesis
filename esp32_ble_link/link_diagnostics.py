#!/usr/bin/env python3
"""link_diagnostics.py -- a fixed set of checks on the measurement method itself, run with the boards held still.

It answers, one experiment at a time, whether the high link loss could be caused by how the test is set up:

  A  power readback    what transmit-power level does the radio REPORT for advertising / default / connection,
                       before and after the connection-power call that the ESP32-C3 documentation prescribes?
  B  power-setting API with and without the explicit post-connection power call (cpwr 0 / 1), at three levels:
                       does RSSI (and so the computed loss) change?
  C  PHY               1M versus 2M at the same placement: does the PHY change RSSI?
  D  direction         TX board at +9 dBm, RX board at -24 dBm: each direction must follow its own board's power
  E  repeatability     reconnecting 3 times at the same setting: how much does RSSI move with no change in placement?

Keep the boards still for the whole run (about 2.5 minutes) and note where they are with --note.

  python link_diagnostics.py --tx-port COM3 --rx-port COM4 --note "5 cm apart, flat on the desk, antennas facing"
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ble_link_test import Node, boot_check, kv  # noqa: E402


def run_config(tx, rx, name, tx_pwr, rx_pwr, phy=2, cpwr=1, secs=3.0, rate_ms=200):
    """Connect, wait for the link to settle, record RSSI at both boards for `secs`, disconnect."""
    tx.cmd("STOP"); rx.cmd("STOP")
    time.sleep(0.7)
    tx.cmd(f"SET txpwr={tx_pwr} phy={phy} cpwr={cpwr} verify=0 fwd=0")
    rx.cmd(f"SET txpwr={rx_pwr} phy={phy} cpwr={cpwr} ci=7.5 lat=0 tmo=2000 verify=0 fwd=0")
    tx.cmd("CLEAR"); rx.cmd("CLEAR")
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.cmd("ADV"); time.sleep(0.4); rx.cmd("SCAN")
    if not tx.wait("EV SUBSCRIBED 1", m_tx, 40) or not rx.wait("EV CONNECTED role=RX", m_rx, 5):
        raise RuntimeError(f"{name}: no connection")
    time.sleep(2.2)                                           # link setup + power calls happen about 0.4 s after connect
    s_tx, s_rx = tx.mark(), rx.mark()
    tx.cmd(f"MON {rate_ms}"); rx.cmd(f"MON {rate_ms}")
    time.sleep(secs)
    tx.send("MON 0"); rx.send("MON 0")
    time.sleep(0.3)
    rssi_rx = [int(x.split()[2]) for _, x in rx.lines_since(s_rx, "R ")]
    rssi_tx = [int(x.split()[2]) for _, x in tx.lines_since(s_tx, "R ")]
    pw = {"tx": [x for _, x in tx.lines_since(m_tx, "EV PWRDBG")], "rx": [x for _, x in rx.lines_since(m_rx, "EV PWRDBG")],
          "tx_set": [x for _, x in tx.lines_since(m_tx, "EV TXPWR")] + [x for _, x in tx.lines_since(m_tx, "EV CONNPWR")],
          "rx_set": [x for _, x in rx.lines_since(m_rx, "EV TXPWR")] + [x for _, x in rx.lines_since(m_rx, "EV CONNPWR")]}
    params = rx.lines_since(m_rx, "EV PARAMS")
    res = dict(name=name, tx_pwr=tx_pwr, rx_pwr=rx_pwr, phy=phy, cpwr=cpwr,
               rssi_rx_mean=float(np.mean(rssi_rx)) if rssi_rx else None, rssi_rx_sd=float(np.std(rssi_rx)) if rssi_rx else None,
               rssi_tx_mean=float(np.mean(rssi_tx)) if rssi_tx else None, n=len(rssi_rx), power_debug=pw,
               phy_reported=kv(params[-1][1]).get("phy_tx") if params else None)
    tx.cmd("STOP"); rx.cmd("STOP")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port", required=True)
    ap.add_argument("--rx-port", required=True)
    ap.add_argument("--note", default="", help="where the boards are (distance, orientation, surface) - recorded in the output")
    args = ap.parse_args()

    out_dir = HERE / "results" / f"link_diagnostics_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tx = Node("TX", args.tx_port, out_dir / "tx_serial.log")
    rx = Node("RX", args.rx_port, out_dir / "rx_serial.log")
    results = []
    try:
        for n in (tx, rx):
            print(f"{n.name} on {n.port}: {boot_check(n)}")
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        plan = []
        for p in (-12, 0, 9):                                  # B (and the readback for A)
            plan.append((f"B cpwr=0 @{p:+d} dBm", p, p, 2, 0))
            plan.append((f"B cpwr=1 @{p:+d} dBm", p, p, 2, 1))
        plan.append(("C phy 1M @0 dBm", 0, 0, 1, 1))           # C (the 2M case is "B cpwr=1 @0")
        plan.append(("D TX +9 / RX -24 dBm", 9, -24, 2, 1))    # D
        for i in (2, 3):
            plan.append((f"E repeat {i} @0 dBm", 0, 0, 2, 1))  # E (first is "B cpwr=1 @0")
        for name, tp, rp, phy, cp in plan:
            print(f"  running: {name} ...", flush=True)
            r = run_config(tx, rx, name, tp, rp, phy, cp)
            results.append(r)
            print(f"     RX hears {r['rssi_rx_mean']:.1f} dBm (sd {r['rssi_rx_sd']:.1f}), TX hears {r['rssi_tx_mean']:.1f} dBm, "
                  f"n={r['n']}, PHY reported {r['phy_reported']}M")
    finally:
        tx.close(); rx.close()

    json.dump({"note": args.note, "results": results}, open(out_dir / "diagnostics.json", "w"), indent=2)
    print(f"\nsaved {out_dir / 'diagnostics.json'}\n")
    print(f"placement note: {args.note or '(none given)'}\n")
    print(f"{'experiment':26} {'TX pwr':>6} {'RX pwr':>6} {'RX hears':>9} {'TX hears':>9} {'loss (TX->RX)':>14}")
    for r in results:
        print(f"{r['name']:26} {r['tx_pwr']:>6} {r['rx_pwr']:>6} {r['rssi_rx_mean']:>9.1f} {r['rssi_tx_mean']:>9.1f} "
              f"{r['tx_pwr'] - r['rssi_rx_mean']:>13.1f} dB")
    b0 = {r["tx_pwr"]: r for r in results if r["name"].startswith("B cpwr=0")}
    b1 = {r["tx_pwr"]: r for r in results if r["name"].startswith("B cpwr=1")}
    print("\nB: change in RSSI from adding the explicit connection-power call (cpwr 1 minus 0), dB:",
          {p: round(b1[p]["rssi_rx_mean"] - b0[p]["rssi_rx_mean"], 1) for p in b0})
    e = [r["rssi_rx_mean"] for r in results if r["name"].startswith("E ")] + [b1[0]["rssi_rx_mean"]]
    print(f"E: RSSI at 0 dBm over 3 reconnects: {[round(v, 1) for v in e]} (spread {max(e) - min(e):.1f} dB)")
    c = next(r for r in results if r["name"].startswith("C "))
    print(f"C: 1M minus 2M PHY RSSI: {c['rssi_rx_mean'] - b1[0]['rssi_rx_mean']:+.1f} dB")
    d = next(r for r in results if r["name"].startswith("D "))
    print(f"D: RX hears {d['rssi_rx_mean']:.1f}, TX hears {d['rssi_tx_mean']:.1f}: difference {d['rssi_rx_mean'] - d['rssi_tx_mean']:+.1f} dB "
          f"(power difference {d['tx_pwr'] - d['rx_pwr']} dB)")
    print("\nRaw power readback for the first cpwr=1 run (level index: 0=-24 dBm, 8=0 dBm, 11=+9 dBm, 255=invalid):")
    for line in b1[-12]["power_debug"]["tx"] + b1[-12]["power_debug"]["tx_set"]:
        print("  TX:", line)
    for line in b1[-12]["power_debug"]["rx"] + b1[-12]["power_debug"]["rx_set"]:
        print("  RX:", line)


if __name__ == "__main__":
    main()
