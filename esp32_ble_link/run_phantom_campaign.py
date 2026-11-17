#!/usr/bin/env python3
"""run_phantom_campaign.py -- send several datasets at several bitrates through the link, then classify what arrived.

For every (dataset, rate) the reduced stream (down-sampled, fewer bits, fewer channels: reduced BEFORE the radio, see README
section 7) is sent in real time from the transmitter board to the receiver board, the receiver's copy is recorded, and
evaluate_received.py trains RF and SVM with the thesis pipeline at that configuration and classifies the received slice. Results are
collected into one table.

  python run_phantom_campaign.py --tx-port COM5 --rx-port COM6 --txpwr 0 --seconds 60 --note "gel phantom, 3.5 cm" --dry-run
  python run_phantom_campaign.py --tx-port COM5 --rx-port COM6 --txpwr 0 --seconds 60 --note "gel phantom, 3.5 cm" --eval
  python run_phantom_campaign.py --resume results/phantom_campaign_<time> --tx-port COM5 --rx-port COM6 --eval   # continue

The rate ladder (kb/s -> sample rate Hz, channels, bits):
  10 -> 250,5,8    32 -> 500,8,8    128 -> 1000,8,16    384 -> 3000,8,16    960 -> 7500,8,16
or give your own as fs:channels:bits (for example 2000:8:16). The sample rate must divide 30000. A recording with fewer channels than
asked simply sends fewer, and the table shows the rate actually offered.

Streams above --record-max-kbps are NOT recorded: forwarding every packet to the PC over the receiver's USB port would become the
limit, not the radio. They are still checked end to end by the CRC (transmitter, receiver and host must agree), and the link
numbers (throughput, stalls, loss, retries) are reported; the classifier result at such a rate equals the offline one because
delivery is bit-exact.

Progress is saved after every run (manifest.json), so an interrupted campaign continues with --resume.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RATES = {10: (250, 5, 8), 32: (500, 8, 8), 128: (1000, 8, 16), 384: (3000, 8, 16), 960: (7500, 8, 16)}


def parse_rate(tok):
    if ":" in tok:
        fs, ch, bits = (int(x) for x in tok.split(":"))
        return f"{fs}x{ch}x{bits}", (fs, ch, bits)
    k = int(tok)
    if k not in RATES:
        raise SystemExit(f"rate {k} is not in the ladder {sorted(RATES)}; give fs:channels:bits instead")
    return f"{k}kbps", RATES[k]


def run_key(source, rate_tag):
    return f"{source.replace(':', '-')}_{rate_tag}"


def load_manifest(path):
    return json.load(open(path)) if path.exists() else {"runs": {}}


def save_manifest(path, m):
    json.dump(m, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port"); ap.add_argument("--rx-port")
    ap.add_argument("--sources", default="vaskov:P1_1of12,vaskov:P2_1of10,wallace:S1,wallace:S2")
    ap.add_argument("--rates", default="10,32,128,384,960")
    ap.add_argument("--seconds", type=float, default=60, help="seconds of recording sent per run (more seconds = more test windows)")
    ap.add_argument("--txpwr", type=int, default=0, help="transmit power label in dBm")
    ap.add_argument("--note", default="", help="where the boards and antennas are; saved with every run")
    ap.add_argument("--eval", action="store_true", help="classify the received data after the transfers")
    ap.add_argument("--eval-max-kbps", type=float, default=400, help="only classify rates up to this (default 400)")
    ap.add_argument("--record-max-kbps", type=float, default=400, help="record the receiver's copy only up to this rate (default 400)")
    ap.add_argument("--no-realtime", action="store_true", help="send as fast as the link accepts instead of at the sampling rate")
    ap.add_argument("--resume", help="a campaign folder to continue")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the time estimate, touch nothing")
    a = ap.parse_args()

    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    rates = [parse_rate(t.strip()) for t in a.rates.split(",") if t.strip()]
    plan = [(s, tag, cfg) for s in sources for tag, cfg in rates]

    def offered_kbps(cfg):
        return cfg[0] * cfg[1] * cfg[2] / 1000.0

    print(f"{len(plan)} runs: {len(sources)} datasets x {len(rates)} rates, {a.seconds:g} s each at TX power {a.txpwr} dBm")
    per_run = a.seconds + 30
    n_eval = sum(1 for s, tag, cfg in plan if offered_kbps(cfg) <= a.eval_max_kbps) if a.eval else 0
    print(f"estimated time: transfers about {len(plan) * per_run / 60:.0f} min (+ one recording load per dataset, 20 to 80 s each), "
          f"classification about {n_eval * 1.5:.0f} min ({n_eval} runs, about 1 to 2 min each)")
    if a.dry_run:
        print(f"\n{'dataset':>18} {'rate':>12} {'fs':>6} {'ch':>3} {'bits':>4} {'offered kb/s':>12} {'recorded':>9} {'classified':>10}")
        for s, tag, cfg in plan:
            k = offered_kbps(cfg)
            print(f"{s:>18} {tag:>12} {cfg[0]:6d} {cfg[1]:3d} {cfg[2]:4d} {k:12.1f} {'yes' if k <= a.record_max_kbps else 'no':>9} "
                  f"{'yes' if (a.eval and k <= a.eval_max_kbps) else 'no':>10}")
        return
    if not (a.tx_port and a.rx_port):
        ap.error("--tx-port and --rx-port are required")

    out_dir = Path(a.resume) if a.resume else Path(a.out) / f"phantom_campaign_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = load_manifest(man_path)
    man.setdefault("settings", dict(seconds=a.seconds, txpwr=a.txpwr, note=a.note, sources=sources, rates=a.rates))
    save_manifest(man_path, man)
    print("campaign folder:", out_dir, flush=True)

    # ------------------------------------------------------------------ transfers
    import ble_link_test as B
    todo = [(s, tag, cfg) for s, tag, cfg in plan if man["runs"].get(run_key(s, tag), {}).get("status") != "sent"]
    if todo:
        tx, rx = B.Node("TX", a.tx_port, out_dir / "tx_serial.log"), B.Node("RX", a.rx_port, out_dir / "rx_serial.log")
        try:
            for n in (tx, rx):
                info = B.boot_check(n)
                print(f"{n.name} on {n.port}: {info.get('chip')} fw {info.get('fw')} mac {info.get('mac')}", flush=True)
            tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
            last_source = None
            for i, (s, tag, cfg) in enumerate(todo, 1):
                if s != last_source:
                    B._DF_CACHE.clear()                       # one recording in memory at a time
                    last_source = s
                key = run_key(s, tag)
                fs, ch, bits = cfg
                k = offered_kbps(cfg)
                run_dir = out_dir / key
                run_dir.mkdir(exist_ok=True)
                print(f"\n=== [{i}/{len(todo)}] {s}  {tag}  ({fs} Hz x {ch} ch x {bits} bit = {k:.0f} kb/s offered) ===", flush=True)
                args = SimpleNamespace(source=s, mode="stream", seconds=a.seconds, start="auto", no_record=(k > a.record_max_kbps),
                                       fs=fs, channels=ch, bits=bits, raw=False, realtime=not a.no_realtime, burst_seconds=10,
                                       burst_rate=0, txpwr=str(a.txpwr), cal=None, effective_dbm=False, phy=2, ci=7.5, payload=240,
                                       sensitivity=-94, repeat=1, note=a.note, out=str(out_dir), cal_obj=None)
                t0 = time.time()
                try:
                    stream, desc = B.build_stream(args)
                    res = B.run_trial(tx, rx, stream, desc, args, a.txpwr, run_dir)
                except RuntimeError as e:
                    if "REBOOTED" in str(e) or "HOSTERR" in str(e):
                        raise                                 # the hardware is in trouble: stop, do not carry on with wrong data
                    print(f"  FAILED: {e}", flush=True)
                    man["runs"][key] = dict(status="failed", reason=str(e), source=s, rate=tag)
                    save_manifest(man_path, man)
                    continue
                except SystemExit as e:
                    print(f"  FAILED: {e}", flush=True)
                    man["runs"][key] = dict(status="failed", reason=str(e), source=s, rate=tag)
                    save_manifest(man_path, man)
                    continue
                B.print_report(res)
                row = {k2: v for k2, v in res.items() if k2 != "_series"}
                json.dump([row], open(run_dir / "summary.json", "w"), indent=2)
                man["runs"][key] = dict(status="sent", source=s, rate=tag, fs=fs, channels_asked=ch, bits=bits,
                                        channels_sent=desc.get("channels", ch), offered_kbps=res.get("offered_kbps"),
                                        recorded=("recording" in res), run_dir=str(run_dir), seconds_wall=round(time.time() - t0, 1),
                                        result=row)
                save_manifest(man_path, man)
        finally:
            tx.close(); rx.close()
        B._DF_CACHE.clear()

    # ------------------------------------------------------------------ classification
    if a.eval:
        for key, r in man["runs"].items():
            if r.get("status") != "sent" or not r.get("recorded") or (r.get("offered_kbps") or 1e9) > a.eval_max_kbps:
                continue
            if r.get("evaluation"):
                continue
            run_dir = Path(r["run_dir"])
            print(f"\n--- classifying {key} ---", flush=True)
            with open(run_dir / "evaluation.log", "w") as lg:
                rc = subprocess.run([sys.executable, str(HERE / "evaluate_received.py"), str(run_dir)], cwd=str(HERE), stdout=lg,
                                    stderr=subprocess.STDOUT).returncode
            js = sorted(run_dir.glob("evaluation_*dBm.json"))
            if rc == 0 and js:
                r["evaluation"] = json.load(open(js[0]))
                print("  done:", {m: round(v["accuracy"]["received"] * 100, 1) for m, v in r["evaluation"]["models"].items()}, flush=True)
            else:
                r["evaluation_error"] = f"exit {rc}; see {run_dir / 'evaluation.log'}"
                print("  FAILED:", r["evaluation_error"], flush=True)
            save_manifest(man_path, man)

    # ------------------------------------------------------------------ summary
    cols = ["source", "rate", "fs_hz", "channels", "bits", "offered_kbps", "throughput_kbps", "packets_lost", "loss_pct", "retry_rate_pct",
            "host_stalled_s", "keeps_up_with_realtime", "max_gap_ms", "rssi_rx_mean_dbm", "link_loss_db", "integrity_ok", "recorded",
            "bit_exact", "test_windows", "damaged_windows", "RF_offline", "RF_received", "SVM_offline", "SVM_received",
            "RF_sent_vs_received", "SVM_sent_vs_received", "classes_missing_from_training"]
    rows = []
    for key, r in man["runs"].items():
        if r.get("status") != "sent":
            rows.append({"source": r.get("source"), "rate": r.get("rate"), "integrity_ok": f"FAILED: {r.get('reason', '')[:60]}"})
            continue
        x = r["result"]
        ev = r.get("evaluation") or {}
        mods = ev.get("models", {})
        good = ev.get("accuracy_meaningful", True)          # False when a test class has no training windows: accuracy is then an artefact
        acc = lambda m, k: (mods.get(m, {}).get("accuracy", {}).get(k) if good else None)
        rows.append(dict(source=r["source"], rate=r["rate"], fs_hz=r["fs"], channels=r["channels_sent"], bits=r["bits"],
                         offered_kbps=r["offered_kbps"], throughput_kbps=x.get("throughput_kbps"), packets_lost=x.get("packets_lost"),
                         loss_pct=x.get("loss_pct"), retry_rate_pct=x.get("retry_rate_pct"), host_stalled_s=x.get("host_stalled_s"),
                         keeps_up_with_realtime=x.get("keeps_up_with_realtime"), max_gap_ms=x.get("max_gap_ms"),
                         rssi_rx_mean_dbm=x.get("rssi_rx_mean_dbm"), link_loss_db=x.get("link_loss_db"), integrity_ok=x.get("integrity_ok"),
                         recorded=r["recorded"], bit_exact=(x.get("recording") or {}).get("bit_exact"),
                         test_windows=ev.get("test_windows"), damaged_windows=ev.get("damaged_windows"),
                         RF_offline=acc("RandomForest", "offline"), RF_received=acc("RandomForest", "received"),
                         SVM_offline=acc("SVM", "offline"), SVM_received=acc("SVM", "received"),
                         RF_sent_vs_received=mods.get("RandomForest", {}).get("agreement_sent_vs_received"),
                         SVM_sent_vs_received=mods.get("SVM", {}).get("agreement_sent_vs_received"),
                         classes_missing_from_training=(" ".join(str(c) for c in ev.get("classes_missing_from_training", [])) or None)))

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"
        return "" if v is None else str(v)
    with open(out_dir / "campaign_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    (out_dir / "campaign_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved {out_dir / 'campaign_summary.csv'} and campaign_summary.md ({len(rows)} rows)")
    print(f"{'dataset':>18} {'rate':>11} {'offered':>8} {'got kb/s':>9} {'lost':>5} {'stall s':>7} {'keeps up':>8} {'RF':>6} {'SVM':>6} "
          f"{'sent=recv':>9}   (accuracy shown only where it is meaningful)")
    for r in rows:
        if r.get("offered_kbps") is None:
            print(f"{r.get('source')!s:>18} {r.get('rate')!s:>11}  {r.get('integrity_ok')}")
            continue
        rf, sv = r.get("RF_received"), r.get("SVM_received")
        ag = r.get("RF_sent_vs_received")
        print(f"{r['source']:>18} {r['rate']:>11} {r['offered_kbps']:8.0f} {r['throughput_kbps']:9.1f} {r['packets_lost']:5d} "
              f"{r['host_stalled_s']:7.2f} {str(r['keeps_up_with_realtime']):>8} {('%.1f%%' % (rf * 100)) if rf is not None else '-':>6} "
              f"{('%.1f%%' % (sv * 100)) if sv is not None else '-':>6} {('%.0f%%' % (ag * 100)) if ag is not None else '-':>9}")


if __name__ == "__main__":
    main()
