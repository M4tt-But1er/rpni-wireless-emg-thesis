#!/usr/bin/env python3
"""rssi_monitor.py -- live RSSI between the two boards, for finding out WHY the link loss is what it is.

Connects the boards (one advertises, one scans) and then prints the signal strength several times a second while
you move and rotate the boards. Nothing is transferred, the link just idles.

  python rssi_monitor.py --tx-port COM3 --rx-port COM4
  python rssi_monitor.py --tx-port COM3 --rx-port COM4 --txpwr 0 --rate-ms 200 --seconds 120

While it runs, type a label and press Enter each time you reach a new position or orientation (for example
"0deg", "90deg", "flat", "30cm"). The monitor closes the previous segment and, when you stop it (Ctrl+C or
--seconds), prints one table row per label with the mean / min / max RSSI and the implied link loss. The spread
between rows is the orientation or position effect. Everything is also written to a CSV.

How to read it:
  * RX RSSI = how strongly the receiver hears the transmitter; TX RSSI = the other direction.
  * link loss = transmit power setting minus RSSI (includes both antennas, so it is a property of the setup).
  * a swing of more than 10-15 dB when you only rotate a board points to antenna pattern / polarisation.
  * RSSI that changes when you touch or move the USB cables points to cable and ground coupling.
"""
import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ble_link_test import Node, boot_check, kv  # noqa: E402


def bar(rssi, lo=-100, hi=-20, width=40):
    n = int(round((min(max(rssi, lo), hi) - lo) / (hi - lo) * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port", required=True)
    ap.add_argument("--rx-port", required=True)
    ap.add_argument("--txpwr", type=int, default=0, help="transmit power for both boards in dBm (default 0)")
    ap.add_argument("--cal", help="power calibration from power_calibration.py (path, its folder, or 'latest'): the loss uses the EFFECTIVE power")
    ap.add_argument("--effective-dbm", action="store_true", help="with --cal: read --txpwr as the power you WANT and send the label that gives it")
    ap.add_argument("--phy", type=int, choices=[1, 2], default=2)
    ap.add_argument("--ci", type=float, default=7.5, help="connection interval in ms (default 7.5, as in the throughput tests)")
    ap.add_argument("--rate-ms", type=int, default=200, help="RSSI sampling period, 50 ms or more")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--note", default="", help="where the boards are (distance, orientation, surface); saved with the results")
    ap.add_argument("--plain", action="store_true", help="one line per sample instead of a live display")
    ap.add_argument("--out", help="CSV path (default results/rssi_monitor_<time>.csv)")
    args = ap.parse_args()

    power = float(args.txpwr)                                # the transmit power used in "loss"; args.txpwr stays the LABEL sent to the boards
    cal = None
    if args.cal:
        from power_cal import PowerCal, find_cal
        cal = PowerCal(find_cal(args.cal))
        if args.effective_dbm:
            args.txpwr = cal.label_for(args.txpwr)
        power = cal.effective(args.txpwr)
        print(cal.describe())
        print(f"\nlabel {args.txpwr:+d} dBm is sent to the boards; its effective power is {power:+.1f} dBm "
              f"({'verified' if cal.verified(args.txpwr) else 'NOT verified, nominal'}); loss below uses {power:+.1f} dBm.\n")
    elif args.effective_dbm:
        raise SystemExit("--effective-dbm needs --cal (a calibration from power_calibration.py)")
    pw_txt = f"{args.txpwr} dBm label" + (f", {power:+.1f} dBm effective" if cal else "")

    out = Path(args.out) if args.out else HERE / "results" / f"rssi_monitor_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out.parent / (out.stem + "_logs")
    log_dir.mkdir(exist_ok=True)

    tx = Node("TX", args.tx_port, log_dir / "tx_serial.log")
    rx = Node("RX", args.rx_port, log_dir / "rx_serial.log")
    marks = [(time.perf_counter(), "start")]                 # (time, label); segment i runs to marks[i+1]
    stop = threading.Event()

    try:
        import msvcrt                                        # Windows console: read keys without disturbing the live line
        use_keys = sys.stdin.isatty()
    except ImportError:
        msvcrt, use_keys = None, False

    def read_marks():                                        # piped stdin (or non-Windows): one label per line
        for line in sys.stdin:
            lab = line.strip()
            if lab:
                marks.append((time.perf_counter(), lab))
                print(f"\n  >> segment '{lab}' started" + " " * 30)
    if not use_keys:
        threading.Thread(target=read_marks, daemon=True).start()
    typed = None                                             # None = not typing a label; else the characters so far

    samples = []                                             # (t, node, rssi)
    lost_events = []                                         # (host time, last good RX RSSI) when the link dropped
    try:
        for n in (tx, rx):
            info = boot_check(n)
            print(f"{n.name} on {n.port}: {info.get('chip')} fw {info.get('fw')}")
            if cal and n is tx and cal.mac and cal.mac.lower() != info.get("mac", "").lower():
                print(f"  WARNING: the calibration was measured on board {cal.mac}, this transmitter is {info.get('mac')}: it may not apply.")
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        tx.cmd("STOP"); rx.cmd("STOP")
        time.sleep(0.6)
        tx.cmd(f"SET txpwr={args.txpwr} phy={args.phy} verify=0 fwd=0")
        rx.cmd(f"SET txpwr={args.txpwr} phy={args.phy} ci={args.ci} lat=0 tmo=2000 verify=0 fwd=0")
        tx.cmd("CLEAR"); rx.cmd("CLEAR")
        m_tx, m_rx = tx.mark(), rx.mark()
        tx.cmd("ADV"); time.sleep(0.4); rx.cmd("SCAN")
        if not tx.wait("EV SUBSCRIBED 1", m_tx, 40) or not rx.wait("EV CONNECTED role=RX", m_rx, 5):
            raise RuntimeError("The boards did not connect within 40 s")
        time.sleep(1.5)
        tx.cmd(f"MON {args.rate_ms}"); rx.cmd(f"MON {args.rate_ms}")
        print(f"\nConnected. BOTH boards TRANSMIT at {pw_txt}. The numbers shown are the signal each board RECEIVES "
              f"(RSSI; closer to 0 = stronger).\nloss = transmit power - RSSI. Start typing at any time to name a new position "
              f"(the display pauses while you type), press Enter to start that segment, Ctrl+C to finish.\n")
        t0 = time.perf_counter()
        i_tx = i_rx = 0
        recent = {"TX": deque(maxlen=15), "RX": deque(maxlen=15)}
        lo = {"TX": len(tx.lines), "RX": len(rx.lines)}      # only lines from now on: the STOP before connecting logs an old "EV DISCONNECTED"
        last_print = 0.0
        last_rx = time.perf_counter()                        # when the last RX reading arrived: a frozen display must not look alive
        usb_dropped = set()
        while not stop.is_set():
            for node, key in ((tx, "TX"), (rx, "RX")):
                with node.cv:
                    new = node.lines[lo[key]:]
                    lo[key] = len(node.lines)
                for t, text in new:
                    if text.startswith("R "):
                        try:
                            v = int(text.split()[2])
                        except (IndexError, ValueError):
                            continue
                        samples.append((t, key, v))
                        recent[key].append(v)
                        if key == "RX":
                            last_rx = time.perf_counter()
                    elif text.startswith("HOSTERR") and key not in usb_dropped:
                        usb_dropped.add(key)                 # the PC lost the board's USB serial port (cable, plug, power)
                        print(f"\n\n  !! USB SERIAL OF THE {key} BOARD DROPPED ({text[8:80]}). Check its cable and plug: the board loses "
                              f"power with it and the link goes too. Readings stop; restart when it is back.\n")
                    elif text.startswith("EV DISCONNECTED") and not lost_events:
                        # RSSI freezes at its last value once packets stop, so use readings from before the
                        # supervision timeout (2 s) plus margin
                        good = [v for tt, k, v in samples if k == "RX" and tt < t - 3.0][-5:]
                        last_good = int(np.median(good)) if good else None
                        lost_events.append((t, last_good))
                        msg = (f"signal just before it died about {last_good} dBm = loss {power - last_good:.0f} dB at "
                               f"{power:+.1f} dBm transmit power" if last_good is not None else "")
                        print(f"\n\n  !! LINK LOST: {msg}.\n  !! Either the signal was too weak (raise --txpwr or move the boards closer) "
                              f"or a USB cable or board was disturbed (look for a 'USB SERIAL ... DROPPED' message). Restart, "
                              f"or Ctrl+C to see the summary.\n")
            while use_keys and msvcrt.kbhit():               # label typing, without the status line overwriting it
                ch = msvcrt.getwch()
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\x00", "\xe0"):                   # arrow / function key: skip its second code
                    msvcrt.getwch()
                    continue
                if typed is None:
                    typed = []
                    print("\n  label> ", end="", flush=True)
                if ch in ("\r", "\n"):
                    lab = "".join(typed).strip()
                    typed = None
                    if lab:
                        marks.append((time.perf_counter(), lab))
                        print(f"\n  >> segment '{lab}' started", flush=True)
                    else:
                        print(flush=True)
                elif ch == "\x08":
                    if typed:
                        typed.pop()
                        print("\b \b", end="", flush=True)
                elif ch.isprintable():
                    typed.append(ch)
                    print(ch, end="", flush=True)
            if typed is None and time.perf_counter() - last_print >= max(args.rate_ms / 1000.0, 0.1) and recent["RX"]:
                last_print = time.perf_counter()
                # median of the last 5 readings: single packets can read 10+ dB off near the sensitivity limit
                r_rx = int(np.median(list(recent["RX"])[-5:]))
                r_tx = float(np.median(list(recent["TX"])[-5:])) if recent["TX"] else float("nan")
                loss = power - r_rx
                label = marks[-1][1]
                silent = time.perf_counter() - last_rx
                if silent > 2.0:                             # no new reading: do not repeat the frozen number as if it were live
                    msg = (f"[{label:>10}] !! NO READINGS FOR {silent:4.0f} s: the link is down or a board's USB cable was disturbed "
                           f"(last reading {r_rx} dBm). Data from now on is NOT being recorded. Restart after fixing it.        ")
                else:
                    msg = (f"[{label:>10}] RX board hears {r_rx:4d} dBm | TX board hears {r_tx:4.0f} dBm | "
                           f"loss {loss:5.1f} dB |{bar(r_rx)}| (median of 5; last 15 raw: min {min(recent['RX'])} max {max(recent['RX'])})")
                print(msg if args.plain else "\r" + msg, end="\n" if args.plain else "", flush=True)
            if args.seconds and time.perf_counter() - t0 >= args.seconds:
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tx.send("MON 0"); rx.send("MON 0"); time.sleep(0.2)
            tx.send("STOP"); rx.send("STOP"); time.sleep(0.3)
        except Exception:
            pass
        tx.close(); rx.close()

    # ---------------------------------------------------------------- summary
    marks.append((time.perf_counter(), "end"))
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "node", "rssi_dbm", "segment"])
        for t, key, v in samples:
            seg = next((marks[i][1] for i in range(len(marks) - 1) if marks[i][0] <= t < marks[i + 1][0]), "")
            w.writerow([f"{t - marks[0][0]:.3f}", key, v, seg])
    if args.note:
        out.with_suffix(".note.txt").write_text(args.note + "\n", encoding="utf-8")
    print(f"\n\nsaved {out}  ({len(samples)} samples)" + (f"  note: {args.note}" if args.note else "") + "\n")
    print(f"{'segment':>12} {'n':>5} {'RX hears':>9} {'min':>5} {'max':>5} {'TX hears':>9} {'link loss':>10}   (dBm, transmit power {pw_txt})")
    means = []
    for i in range(len(marks) - 1):
        a, b = marks[i][0], marks[i + 1][0]
        rxv = [v for t, k, v in samples if k == "RX" and a <= t < b]
        txv = [v for t, k, v in samples if k == "TX" and a <= t < b]
        if len(rxv) < 2:
            continue
        means.append(np.mean(rxv))
        print(f"{marks[i][1]:>12} {len(rxv):>5} {np.mean(rxv):>9.1f} {min(rxv):>5} {max(rxv):>5} "
              f"{(np.mean(txv) if txv else float('nan')):>9.1f} {power - np.mean(rxv):>9.1f} dB")
    for t_ev, rssi in lost_events:
        if rssi is not None:
            print(f"\nLINK LOST at {t_ev - marks[0][0]:.0f} s. Last good signal about {rssi} dBm, so the link tolerates a loss of "
                  f"about {power - rssi:.0f} dB at {power:+.1f} dBm transmit power (the working sensitivity of this link).")
    if len(means) > 1:
        print(f"\nspread between segment means: {max(means) - min(means):.1f} dB "
              f"(larger than about 10 dB = antenna pattern / polarisation / placement dominates)")


if __name__ == "__main__":
    main()
