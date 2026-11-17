#!/usr/bin/env python3
"""power_calibration.py -- measure how the transmit-power LABELS of the ESP32-C3 boards relate to the signal that
actually arrives, using two independent receivers, and build a correction table.

What it can and cannot do
  * It measures the RELATIVE power staircase: how much stronger the signal gets at each label compared with label 0 dBm.
    The ESP32 documents 3 dB per step, but an earlier sweep showed a 9 dB jump between +9 and +12 dBm.
  * It cannot measure ABSOLUTE power (dBm). That needs a spectrum analyser or power meter. The correction table is therefore
    anchored at label 0 dBm (assumed correct) and says so.
  * It is not a spectrum analyser: the boards and the PC adapter report one RSSI number per packet.

How it works: for every label, in random order and repeated --reps times, the transmitter board advertises at that label while
  (1) the PC's own Bluetooth adapter (an independent receiver, through the Windows BLE API) records the RSSI of its advertisements,
  (2) the second ESP32 board records the RSSI of the same advertisements (its own scan result), then connects, and
  (3) records the RSSI of the connection, which is the mode the data is sent in.
(1) and (2) are two receivers of the same advertising-mode power: where they agree the receiver scale can be trusted. (3) is compared
with 1 dB per dB to give the correction for the connected link: that mode can radiate differently from advertising mode. The boards and the PC stay where they are for the whole run (about 10 minutes): note where with --note.

  python power_calibration.py --tx-port COM3 --rx-port COM4 --note "5 cm apart, flat, PC 1 m away"
  python power_calibration.py --tx-port COM4 --rx-port COM3          # the other board as transmitter
Output: results/power_cal_<time>/calibration.json (+ a plot). Use it with --cal in the other scripts.
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ble_link_test import Node, boot_check  # noqa: E402

LABELS = [-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 20]
ANCHOR = 0


async def _scan(mac, seconds):
    from bleak import BleakScanner
    got = []

    def cb(device, adv):
        if device.address.lower() == mac.lower():
            got.append(adv.rssi)
    async with BleakScanner(cb):
        await asyncio.sleep(seconds)
    return got


def pc_rssi(mac, seconds):
    return asyncio.run(_scan(mac, seconds))


def measure_label(tx, rx, mac, label, rx_ref, adv_s, mon_s, phy=2):
    tx.cmd("STOP"); rx.cmd("STOP", timeout=40)               # a receiver stuck in a failed connect() (up to 30 s) answers only afterwards
    time.sleep(0.7)
    tx.cmd(f"SET txpwr={label} phy={phy} cpwr=1 verify=0 fwd=0")
    rx.cmd(f"SET txpwr={rx_ref} phy={phy} cpwr=1 ci=7.5 lat=0 tmo=2000 verify=0 fwd=0")
    tx.cmd("CLEAR"); rx.cmd("CLEAR")
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.cmd("ADV")
    time.sleep(0.5)
    pc = pc_rssi(mac, adv_s)                                 # independent receiver: advertisement RSSI at the PC
    rx.cmd("SCAN")
    esp = []
    if tx.wait("EV SUBSCRIBED 1", m_tx, 18) and rx.wait("EV CONNECTED role=RX", m_rx, 5):   # weak labels never connect: do not wait the full scan
        time.sleep(2.0)
        s_rx = rx.mark()
        rx.cmd("MON 200"); time.sleep(mon_s); rx.send("MON 0"); time.sleep(0.3)
        esp = [int(x.split()[2]) for _, x in rx.lines_since(s_rx, "R ")]
    adv = [int(x.split("rssi=")[1]) for _, x in rx.lines_since(m_rx, "EV FOUND")][:1]   # the receiver's own advertisement reading (same radio as the connection)
    tx.cmd("STOP"); rx.cmd("STOP", timeout=40)               # a receiver stuck in a failed connect() (up to 30 s) answers only afterwards
    return pc, esp, adv


REP_AGREE_DB = 5.0     # repetitions of the same label that differ more than this mean something moved
CROSS_CHECK_DB = 3.0   # the two advertising-mode receivers must agree within this for a label to be corrected
CHIP_MAX_DBM = 20.0    # the highest label; an effective power well above it means the 0 dBm anchor assumption cannot hold
SOURCES = (("pc", 3), ("esp_adv", 1), ("esp", 3))   # (key, minimum samples per repetition)
NAMES = {"pc": "PC adapter (advertisements)", "esp_adv": "second ESP32 (advertisements)", "esp": "second ESP32 (connection)"}


def analyse(raw, use_reps=None):
    """raw: {label: {"pc": [[...rep1], ...], "esp_adv": [[...], ...], "esp": [[...], ...]}} -> per-label statistics and the table.

    Three views of the same transmitter: the PC adapter and the second ESP both hear its ADVERTISEMENTS (two independent receivers),
    and the second ESP also hears the CONNECTION, which is the mode the data is sent in. The correction for the connected link is the
    connection staircase's departure from 1 dB per dB, applied only where the two advertising-mode receivers agree (so the receiver's
    own RSSI scale is trustworthy) and the repetitions agree (so nothing moved).
    use_reps: analyse only the first N repetitions (e.g. when a later one was disturbed)."""
    stats = {}
    for lab in LABELS:
        d = raw[str(lab)]
        row = {}
        for src, min_n in SOURCES:
            reps = [r for r in d.get(src, [])[:use_reps] if len(r) >= min_n]
            allv = [v for r in reps for v in r]
            row[src] = dict(n=len(allv), mean=float(np.mean(allv)) if allv else None,
                            sd=float(np.std(allv)) if allv else None,
                            rep_means=[float(np.mean(r)) for r in reps])
        stats[str(lab)] = row
    dev = {}
    for src, _ in SOURCES:
        a = stats[str(ANCHOR)][src]["mean"]
        for lab in LABELS:
            m = stats[str(lab)][src]["mean"]
            dev.setdefault(str(lab), {})[src] = None if m is None or a is None else (m - a) - (lab - ANCHOR)
    table = {}
    for lab in LABELS:
        d = dev[str(lab)]
        spread = max([max(m) - min(m) for m in (stats[str(lab)][s]["rep_means"] for s, _ in SOURCES) if len(m) > 1] or [0.0])
        if d["esp"] is None:
            corr, verdict = 0.0, "second ESP never connected (too weak): connected power not measured"
        elif spread > REP_AGREE_DB:
            corr, verdict = 0.0, f"repetitions differ by {spread:.0f} dB (boards moved?): not corrected"
        elif d["pc"] is None or d["esp_adv"] is None:
            corr, verdict = 0.0, "only one advertising-mode receiver heard it"
        elif abs(d["pc"] - d["esp_adv"]) > CROSS_CHECK_DB:
            corr, verdict = 0.0, "advertising-mode receivers disagree: not corrected"
        else:
            corr, verdict = d["esp"], "both receivers agree in advertising mode: connection reading used"
        eff = lab + corr
        table[str(lab)] = dict(label_dbm=lab, correction_db=round(corr, 1), effective_dbm=round(eff, 1), verdict=verdict,
                               exceeds_chip_max=bool(eff > CHIP_MAX_DBM + 1.0), rep_spread_db=round(spread, 1),
                               rssi_pc=stats[str(lab)]["pc"]["mean"], rssi_esp_adv=stats[str(lab)]["esp_adv"]["mean"],
                               rssi_esp=stats[str(lab)]["esp"]["mean"],
                               dev_pc=d["pc"], dev_esp_adv=d["esp_adv"], dev_esp=d["esp"])
    return stats, table


def make_plot(table, path, note):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labs = LABELS
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    style = {"pc": ("o-", "C0"), "esp_adv": ("^-", "C2"), "esp": ("s--", "C1")}
    for src, _ in SOURCES:
        ref = table[str(ANCHOR)][f"rssi_{src}"]
        if ref is None:
            continue
        ys = [table[str(l)][f"rssi_{src}"] for l in labs]
        ax[0].plot(labs, [np.nan if y is None else y - ref for y in ys], style[src][0], color=style[src][1], label=NAMES[src])
    ax[0].plot(labs, [l - ANCHOR for l in labs], ":", color="grey", label="ideal: 1 dB per dB")
    ax[0].set_xlabel("transmit power label (dBm)"); ax[0].set_ylabel("received signal relative to the 0 dBm label (dB)")
    ax[0].set_title("Power staircase, three views of one transmitter"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    for src, _ in SOURCES:
        ax[1].plot(labs, [np.nan if table[str(l)][f"dev_{src}"] is None else table[str(l)][f"dev_{src}"] for l in labs],
                   style[src][0], color=style[src][1], label=NAMES[src])
    ax[1].axhline(0, color="grey", ls=":")
    ax[1].set_xlabel("transmit power label (dBm)"); ax[1].set_ylabel("departure from the ideal staircase (dB)")
    ax[1].set_title("Departure from 1 dB per dB (advertising agrees, connection departs?)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(note[:130] if note else "", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140)


def print_table(table, labels):
    print(f"{'label':>6} {'PC adv':>7} {'ESP adv':>8} {'ESP conn':>9} {'correction':>11} {'effective':>10}   (departures from 1 dB/dB, in dB; verdict)")
    for lab in labels:
        r = table[str(lab)]
        f = lambda v: "    n/a" if v is None else f"{v:7.1f}"
        flag = "  !! above the chip's maximum: the 0 dBm anchor cannot hold" if r.get("exceeds_chip_max") else ""
        print(f"{lab:>+6} {f(r['dev_pc']):>7} {f(r['dev_esp_adv']):>8} {f(r['dev_esp']):>9} {r['correction_db']:>+10.1f}  {r['effective_dbm']:>+9.1f}   {r['verdict']}{flag}")


def backfill_esp_adv(folder, raw):
    """Runs made before the advertising-mode reading was recorded: take it from the serial logs ('EV FOUND rssi=' on the receiver,
    one per measurement, in order). Only done when the counts match exactly, and only into the in-memory copy."""
    logs_tx, logs_rx = sorted(folder.glob("tx_serial*.log")), sorted(folder.glob("rx_serial*.log"))
    if len(logs_tx) != 1 or len(logs_rx) != 1:
        print("  (cannot back-fill the advertising-mode RSSI: the run was resumed, so the logs are split)")
        return raw
    tx = [int(m.group(1)) for m in re.finditer(r"OK SET txpwr=(-?\d+) phy=\d ci=\d+\.\d+ lat=\d+ tmo=\d+ pay=\d+ verify=0 fwd=0", logs_tx[0].read_text(errors="ignore"))]
    rx = [int(m.group(1)) for m in re.finditer(r"EV FOUND rssi=(-?\d+)", logs_rx[0].read_text(errors="ignore"))]
    n = sum(len(raw[str(l)]["pc"]) for l in LABELS)
    if not (len(tx) == len(rx) == n):
        print(f"  (cannot back-fill the advertising-mode RSSI: {len(tx)} power settings, {len(rx)} advertisements found, {n} measurements)")
        return raw
    for l in LABELS:
        raw[str(l)]["esp_adv"] = []
    for lab, v in zip(tx, rx):
        raw[str(lab)]["esp_adv"].append([v])
    print(f"  back-filled the advertising-mode RSSI of {n} measurements from the serial logs")
    return raw


def reanalyse(folder, use_reps):
    """Rebuild the table from a saved run. Never overwrites calibration.json: the new files carry the repetitions used in their name."""
    raw = json.load(open(folder / "raw_partial.json"))
    old = json.load(open(folder / "calibration.json")) if (folder / "calibration.json").exists() else {}
    if any(not v.get("esp_adv") for v in raw.values()):
        raw = backfill_esp_adv(folder, raw)
    stats, table = analyse(raw, use_reps)
    tag = f"_rep{use_reps}" if use_reps else "_reanalysed"
    note = (old.get("note", "") + f" [re-analysed from {folder.name}, first {use_reps} repetition(s)]" if use_reps else old.get("note", ""))
    cal = dict(created=datetime.now().isoformat(timespec="seconds"), note=note, tx_board_mac=old.get("tx_board_mac"), anchor_label_dbm=ANCHOR,
               anchor_assumption="label 0 dBm is assumed to radiate 0 dBm; ABSOLUTE power is not measured",
               reps=use_reps or old.get("reps"), table=table, raw=raw, stats=stats)
    json.dump(cal, open(folder / f"calibration{tag}.json", "w"), indent=2)
    make_plot(table, folder / f"power_staircase{tag}.png", note)
    print(f"saved {folder / f'calibration{tag}.json'}\n")
    print_table(table, LABELS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port", help="the board whose power labels are calibrated")
    ap.add_argument("--rx-port", help="the second board, used as one of the receivers")
    ap.add_argument("--reanalyse", help="results folder: rebuild the table and plot from its raw_partial.json without measuring (writes new files)")
    ap.add_argument("--use-reps", type=int, help="with --reanalyse: use only the first N repetitions")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--adv-s", type=float, default=8.0, help="seconds the PC listens to advertisements per label")
    ap.add_argument("--mon-s", type=float, default=3.0, help="seconds of ESP-to-ESP RSSI per label")
    ap.add_argument("--phy", type=int, choices=[1, 2], default=2,
                    help="PHY requested for the connection (default 2M, as in the data tests). 1M tests whether a connection-reading jump at the top labels comes from the 2M receive path")
    ap.add_argument("--rx-ref", type=int, default=0, help="power label of the receiving board (does not affect TX->RX RSSI)")
    ap.add_argument("--note", default="", help="where the boards and the PC are")
    ap.add_argument("--labels", help="comma list to test fewer labels (must include 0)")
    ap.add_argument("--resume", help="results folder of an interrupted run: continue it (raw_partial.json is saved after every label)")
    args = ap.parse_args()
    if args.reanalyse:
        reanalyse(Path(args.reanalyse), args.use_reps)
        return
    if not (args.tx_port and args.rx_port):
        ap.error("--tx-port and --rx-port are required (or use --reanalyse)")
    labels = [int(x) for x in args.labels.split(",")] if args.labels else LABELS
    assert ANCHOR in labels, "label 0 is the anchor and must be included"

    out_dir = Path(args.resume) if args.resume else HERE / "results" / f"power_cal_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{datetime.now():%H%M%S}" if args.resume else ""
    tx = Node("TX", args.tx_port, out_dir / f"tx_serial{suffix}.log")
    rx = Node("RX", args.rx_port, out_dir / f"rx_serial{suffix}.log")
    raw = {str(l): {"pc": [], "esp_adv": [], "esp": []} for l in LABELS}
    if args.resume and (out_dir / "raw_partial.json").exists():
        raw = json.load(open(out_dir / "raw_partial.json"))
        for v in raw.values():
            v.setdefault("esp_adv", [])
        print("resuming: measurements already done:", sum(len(v["pc"]) for v in raw.values()))
    try:
        info = {n.name: boot_check(n) for n in (tx, rx)}
        for k, v in info.items():
            print(f"{k}: {v}")
        mac = info["TX"]["mac"]
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        total = args.reps * len(labels)
        done = sum(min(len(raw[str(l)]["pc"]), args.reps) for l in labels)
        for rep in range(args.reps):
            todo = [l for l in labels if len(raw[str(l)]["pc"]) <= rep]      # skips what a resumed run already holds
            order = random.sample(todo, len(todo))
            for lab in order:
                pc, esp, adv = measure_label(tx, rx, mac, lab, args.rx_ref, args.adv_s, args.mon_s, args.phy)
                raw[str(lab)]["pc"].append(pc); raw[str(lab)]["esp"].append(esp); raw[str(lab)]["esp_adv"].append(adv)
                json.dump(raw, open(out_dir / "raw_partial.json", "w"))
                done += 1
                print(f"  [{done:2d}/{total}] label {lab:+3d} dBm: PC {np.mean(pc) if pc else float('nan'):6.1f} dBm (n={len(pc):2d}), "
                      f"ESP {np.mean(esp) if esp else float('nan'):6.1f} dBm (n={len(esp):2d}), "
                      f"ESP adv {adv[0] if adv else float('nan'):5.0f} dBm", flush=True)
    finally:
        tx.close(); rx.close()

    use = {k: v for k, v in raw.items() if int(k) in labels}
    missing = [l for l in LABELS if l not in labels]
    for l in missing:
        raw[str(l)] = {"pc": [[]], "esp_adv": [[]], "esp": [[]]}
    stats, table = analyse(raw)
    cal = dict(created=datetime.now().isoformat(timespec="seconds"), note=args.note, tx_board_mac=mac, anchor_label_dbm=ANCHOR,
               anchor_assumption="label 0 dBm is assumed to radiate 0 dBm; ABSOLUTE power is not measured",
               reps=args.reps, phy_requested=f"{args.phy}M", table=table, raw=raw, stats=stats)
    json.dump(cal, open(out_dir / "calibration.json", "w"), indent=2)
    make_plot(table, out_dir / "power_staircase.png", args.note)

    print(f"\nsaved {out_dir / 'calibration.json'}\n")
    print_table(table, labels)


if __name__ == "__main__":
    main()
