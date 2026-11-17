#!/usr/bin/env python3
"""link_timing.py -- X2 of pipeline/docs/three_experiments_declared_protocol.md: delay and backlog of the BLE link, per packet.

The PC plays the sensor exactly as live_classify.py does (each frame is handed to the transmitter's USB port at its production time,
never earlier), but NO classifier runs, so the PC's compute cannot add to the timing. For every packet, on one PC clock:
  t_src  the last byte of the packet was produced by the sensor (ideal clock)
  t_fed  that byte was handed to the transmitter's USB port
  t_rx   the receiver delivered the packet to the PC
total delay = t_rx - t_src (packet fill + backlog + link); link transit = t_rx - t_fed (USB + BLE incl. retransmissions + transmitter queue);
sensor backlog = t_fed - t_src (data due but not accepted because the transmitter's buffer was full).

Examples (from this folder):
  python link_timing.py --tx-port COM5 --rx-port COM6 --rates 32,128,384,960 --txpwr 0 --seconds 60 --note "..."
  python link_timing.py --tx-port COM5 --rx-port COM6 --rates 32,128,384,960 --txpwr=-24 --seconds 60 --note "..."
"""
import argparse
import json
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ble_link_test as B                       # noqa: E402
from live_classify import connect, usb_rtt, pct  # noqa: E402
from run_phantom_campaign import parse_rate      # noqa: E402

STALL_MS = 22.5                                  # 3 connection intervals of 7.5 ms


def run_one(tx, rx, stream, desc, a, pay, run_dir):
    fs, C, bits = desc["fs"], desc["channels"], desc["bits"]
    frame_bytes = C * bits // 8
    Bps = fs * frame_bytes
    L = len(stream)
    n_frames = L // frame_bytes
    n_pk = -(-L // pay)
    link, idle = connect(tx, rx, a, a.txpwr, pay)
    print(f"  connected: interval {link['rx'].get('itvl_ms')} ms, PHY {link['rx'].get('phy_tx')}M, MTU {link['rx'].get('mtu')}, payload {pay} B "
          f"(fills in {1000 * pay / Bps:.1f} ms)", flush=True)
    rtt = {"TX": usb_rtt(tx), "RX": usb_rtt(rx)}
    t_rx = np.full(n_pk, np.nan)
    got = bytearray(L)

    def hook(seq, payload, t_read):              # serial reader thread: keep it cheap
        if seq < n_pk and np.isnan(t_rx[seq]):
            t_rx[seq] = t_read
            got[seq * pay: seq * pay + len(payload)] = payload

    rx.packets = {}
    rx.packet_hook = hook
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.consumed = 0
    tx.cmd("BEGIN stream")
    crc_host = zlib.crc32(stream) & 0xFFFFFFFF
    t0 = time.perf_counter() + 0.2
    wr_t, wr_cum = [], []
    sent, stalled = 0, 0.0
    limit_s = 60 + 3 * n_frames / fs
    tx.ser.write_timeout = 3
    print(f"  streaming {n_frames / fs:.0f} s in real time ({Bps * 8 / 1000:.0f} kb/s) ...", flush=True)
    while sent < L:
        now = time.perf_counter()
        if now - t0 > limit_s:
            raise RuntimeError("the stream is taking far too long, aborting")
        frames_due = min(n_frames, int((now - t0) * fs) + 1) if now >= t0 else 0
        n = min(frames_due * frame_bytes - sent, B.RING_WINDOW - (sent - tx.consumed), 4096)
        if n <= 0:
            time.sleep(0.0005)
            if frames_due * frame_bytes > sent:
                stalled += 0.0005
            continue
        t_call = time.perf_counter()
        tx.ser.write(f"D {n}\n".encode() + stream[sent:sent + n])
        sent += n
        wr_t.append(t_call); wr_cum.append(sent)
    tx.send("FIN")
    r_tx = tx.wait("TXSTAT", m_tx, 60 + n_frames / fs)
    r_rx = rx.wait("RXSTAT", m_rx, 30)
    time.sleep(0.3)
    rx.packet_hook = None
    if not r_tx or not r_rx:
        raise RuntimeError("Did not get TXSTAT/RXSTAT")
    txs, rxs = B.kv(r_tx[1]), B.kv(r_rx[1])
    rssi_run = [int(B.kv(x)["rssi"]) for _, x in rx.lines_since(m_rx, "T ") if int(B.kv(x)["rssi"]) != 0]

    # ---- per packet
    wr_t, wr_cum = np.array(wr_t), np.array(wr_cum)
    seq = np.arange(n_pk)
    last_byte = np.minimum((seq + 1) * pay, L) - 1
    t_src = t0 + (last_byte // frame_bytes) / fs
    t_fed = wr_t[np.minimum(np.searchsorted(wr_cum, last_byte, side="right"), len(wr_t) - 1)]
    df = pd.DataFrame(dict(seq=seq, t_src_s=t_src - t0, t_fed_s=t_fed - t0, t_rx_s=t_rx - t0))
    df["total_delay_ms"] = 1000 * (t_rx - t_src)
    df["link_transit_ms"] = 1000 * (t_rx - t_fed)
    df["sensor_backlog_ms"] = 1000 * (t_fed - t_src)
    df["fill_ms"] = 1000 * (t_src - (t0 + (seq * pay // frame_bytes) / fs))
    df.to_csv(run_dir / "packets.csv", index=False)
    ok = df.dropna(subset=["t_rx_s"])
    arr = np.sort(ok["t_rx_s"].to_numpy())
    gaps = np.diff(arr) * 1000 if len(arr) > 1 else np.array([])
    fill_ms = 1000 * pay / Bps
    stalls = gaps[gaps > fill_ms + STALL_MS] - fill_ms        # a stall = a gap longer than the packet spacing by more than 3 connection intervals
    # excess link delay over the run's fastest packets, in connection intervals: the time that retransmissions and queueing in the transmitter add
    ci_ms = float(link["rx"].get("itvl_ms", a.ci) or a.ci)
    base = float(np.percentile(ok["link_transit_ms"], 5))
    excess = ok["link_transit_ms"].to_numpy() - base
    n_ci = np.floor(np.maximum(excess, 0) / ci_ms)
    # backlog = data produced by the sensor but not yet at the PC, at every packet arrival (wherever it sits: PC, USB, transmitter buffer, air)
    # arr holds arrival times relative to t0 (s); the sensor has produced floor(t * fs) + 1 frames by time t
    produced = np.minimum(np.floor(arr * fs) + 1, n_frames) * frame_bytes
    received = np.minimum(np.arange(1, len(arr) + 1) * pay, L)
    backlog_B = np.maximum(produced - received, 0)
    over = {f"{kB}kB": (float(arr[np.argmax(backlog_B > kB * 1024)]) if (backlog_B > kB * 1024).any() else None) for kB in (4, 8, 16, 24, 64)}
    np.save(run_dir / "backlog_bytes.npy", np.stack([arr, backlog_B]))
    # backlog trend: straight-line fit of total delay against production time
    slope = float(np.polyfit(ok["t_src_s"], ok["total_delay_ms"], 1)[0]) if len(ok) > 10 else float("nan")
    end = ok[ok["t_src_s"] >= ok["t_src_s"].max() - 5]
    span = (arr[-1] - arr[0]) if len(arr) > 1 else float("nan")
    res = dict(source=desc["source"], fs=fs, channels=C, bits=bits, offered_kbps=Bps * 8 / 1000, payload_bytes=pay,
               packet_fill_ms=1000 * pay / Bps, tx_power_label_dbm=a.txpwr, link=link, rssi_idle_rx_dbm=idle,
               usb_rtt_ms={k: pct(v) for k, v in rtt.items()}, stream_seconds=n_frames / fs, host_stalled_s=stalled,
               packets_sent=int(txs["pkts"]), packets_received=int(rxs["pkts"]), packets_lost=int(rxs["lost"]),
               host_packets_missing=int(np.isnan(t_rx).sum()), tx_notify_retries=int(txs["retries"]),
               retry_rate_pct=100.0 * int(txs["retries"]) / max(int(txs["retries"]) + int(txs["pkts"]), 1),
               usb_forward_drops=int(rxs.get("fwd_drops", 0)),
               rssi_rx_during_run_dbm=(dict(mean=float(np.mean(rssi_run)), min=int(min(rssi_run)), max=int(max(rssi_run)), n=len(rssi_run)) if rssi_run else None),
               link_loss_db=(a.txpwr - float(np.mean(rssi_run))) if rssi_run else None,
               crc_host=f"{crc_host:08x}", crc_tx=txs["crc"], crc_rx=rxs["crc"], bit_exact=bool(bytes(got) == stream),
               total_delay_ms=pct(ok["total_delay_ms"]), link_transit_ms=pct(ok["link_transit_ms"]), sensor_backlog_ms=pct(ok["sensor_backlog_ms"]),
               fill_ms=pct(ok["fill_ms"]),
               backlog_slope_ms_per_s=slope, total_delay_last5s_ms=pct(end["total_delay_ms"]),
               stalls=dict(threshold_ms=STALL_MS, rule="gap - packet spacing > 3 connection intervals", count=int(len(stalls)), total_ms=float(stalls.sum()),
                           longest_ms=float(stalls.max()) if len(stalls) else 0.0, gap_p99_ms=float(np.percentile(gaps, 99)) if len(gaps) else float("nan")),
               connection_interval_ms=ci_ms, link_transit_p5_ms=base,
               excess_delay=dict(share_ge_1ci=float((n_ci >= 1).mean()), share_ge_2ci=float((n_ci >= 2).mean()), share_ge_4ci=float((n_ci >= 4).mean()),
                                 max_ci=int(n_ci.max()) if len(n_ci) else 0, mean_ms=float(np.maximum(excess, 0).mean())),
               backlog_bytes=dict(median=float(np.median(backlog_B)), p99=float(np.percentile(backlog_B, 99)), max=float(backlog_B.max()), final=float(backlog_B[-1])),
               backlog_first_exceeds_s=over,
               throughput_kbps=(8 * L / 1000 / span) if span and span > 0 else float("nan"),
               delivered_in_real_time=bool(slope < 1.0 and (end["total_delay_ms"].median() if len(end) else 1e9) < 100))
    res["integrity_ok"] = bool(res["bit_exact"] and res["crc_host"] == res["crc_rx"] and res["host_packets_missing"] == 0)
    json.dump(res, open(run_dir / "summary.json", "w"), indent=2)
    return res


def show(res):
    d, t = res["total_delay_ms"], res["link_transit_ms"]
    print(f"""  ---- {res['offered_kbps']:.0f} kb/s, label {res['tx_power_label_dbm']} dBm, payload {res['payload_bytes']} B ----
  link     : {res['packets_sent']} sent, {res['packets_received']} received, {res['packets_lost']} lost; refused sends {res['retry_rate_pct']:.1f} %; integrity {'OK' if res['integrity_ok'] else 'FAILED'}
  signal   : RSSI {res['rssi_rx_during_run_dbm']['mean'] if res['rssi_rx_during_run_dbm'] else float('nan'):.1f} dBm, link loss {res['link_loss_db'] if res['link_loss_db'] is not None else float('nan'):.1f} dB
  delay ms : total median {d['median']:.1f}, p99 {d['p99']:.1f}, max {d['max']:.1f}; link transit median {t['median']:.1f}, p99 {t['p99']:.1f}, max {t['max']:.1f}
  backlog  : slope {res['backlog_slope_ms_per_s']:.2f} ms per s; last 5 s median {res['total_delay_last5s_ms'].get('median', float('nan')):.1f} ms; sensor stalled {res['host_stalled_s']:.2f} s
  stalls   : {res['stalls']['count']} stalls (> 3 intervals beyond the packet spacing), total {res['stalls']['total_ms']:.0f} ms, longest {res['stalls']['longest_ms']:.1f} ms; throughput {res['throughput_kbps']:.0f} kb/s
  excess   : link delay above the run's fastest 5 %: >= 1 interval in {100 * res['excess_delay']['share_ge_1ci']:.1f} % of packets, >= 2 in {100 * res['excess_delay']['share_ge_2ci']:.1f} %, >= 4 in {100 * res['excess_delay']['share_ge_4ci']:.1f} %, most {res['excess_delay']['max_ci']} intervals
  queue    : backlog median {res['backlog_bytes']['median'] / 1024:.1f} kB, max {res['backlog_bytes']['max'] / 1024:.1f} kB, at the end {res['backlog_bytes']['final'] / 1024:.1f} kB; first above 24 kB at {res['backlog_first_exceeds_s']['24kB']} s""", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port", required=True); ap.add_argument("--rx-port", required=True)
    ap.add_argument("--source", default="vaskov:P1_1of12")
    ap.add_argument("--rates", default="32,128,384,960")
    ap.add_argument("--payload", type=int, default=240)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--txpwr", type=int, default=0)
    ap.add_argument("--phy", type=int, choices=[1, 2], default=2)
    ap.add_argument("--ci", type=float, default=7.5)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass
    sys.setswitchinterval(0.001)
    out_dir = Path(a.out) / f"link_timing_{a.txpwr}dBm_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {"settings": dict(source=a.source, seconds=a.seconds, txpwr=a.txpwr, phy=a.phy, ci=a.ci, payload=a.payload, note=a.note), "runs": {}}
    print("folder:", out_dir, flush=True)
    tx, rx = B.Node("TX", a.tx_port, out_dir / "tx_serial.log"), B.Node("RX", a.rx_port, out_dir / "rx_serial.log")
    try:
        for n in (tx, rx):
            info = B.boot_check(n)
            print(f"{n.name} on {n.port}: {info.get('chip')} fw {info.get('fw')} mac {info.get('mac')}", flush=True)
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        for tok in a.rates.split(","):
            tag, (fs, ch, bits) = parse_rate(tok.strip())
            run_dir = out_dir / tag
            run_dir.mkdir(exist_ok=True)
            print(f"\n=== {tag} ({fs} Hz x {ch} ch x {bits} bit), label {a.txpwr} dBm ===", flush=True)
            args = SimpleNamespace(source=a.source, mode="stream", seconds=a.seconds, start="auto", no_record=True, fs=fs, channels=ch, bits=bits,
                                   raw=False, realtime=True, burst_seconds=10, burst_rate=0, txpwr=str(a.txpwr), cal=None, effective_dbm=False,
                                   phy=a.phy, ci=a.ci, payload=a.payload, sensitivity=-94, repeat=1, note=a.note, out=str(out_dir), cal_obj=None)
            try:
                stream, desc = B.build_stream(args)
                res = run_one(tx, rx, stream, desc, a, a.payload, run_dir)
            except RuntimeError as e:
                if "REBOOTED" in str(e) or "HOSTERR" in str(e):
                    raise
                print(f"  FAILED: {e}", flush=True)
                man["runs"][tag] = dict(status="failed", reason=str(e))
                json.dump(man, open(out_dir / "manifest.json", "w"), indent=2)
                continue
            res["placement_note"] = a.note
            show(res)
            man["runs"][tag] = dict(status="done", result=res)
            json.dump(man, open(out_dir / "manifest.json", "w"), indent=2)
    finally:
        tx.close(); rx.close()
    print("\nsaved in", out_dir)


if __name__ == "__main__":
    main()
