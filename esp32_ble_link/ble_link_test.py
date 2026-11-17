#!/usr/bin/env python3
"""ble_link_test.py -- drive two ESP32 boards (flashed with firmware/emg_ble_link) over their USB serial
ports, make them connect over BLE, stream EMG data from one to the other and report link statistics.

Both boards stay plugged in to this one computer. Each shows up as its own COM port, and this script opens
BOTH at once (one reader thread per port), so a single command runs the whole test.

Examples (run from this folder):
  python ble_link_test.py --list-ports
  python ble_link_test.py --tx-port COM3 --rx-port COM4 --source vaskov:P1_1of12 --seconds 30
  python ble_link_test.py --tx-port COM3 --rx-port COM4 --source wallace:S1 --fs 1000 --channels 8 --bits 16 --realtime
  python ble_link_test.py --tx-port COM3 --rx-port COM4 --mode burst --burst-seconds 10         # maximum throughput
  python ble_link_test.py --tx-port COM3 --rx-port COM4 --mode burst --txpwr=-24,-12,0,9,20     # power sweep

Data: the sample stream is built exactly as in the thesis pipeline (band-pass 100-499 Hz at 30 kHz, decimate to
--fs, uniform quantisation to --bits over the 0.1-99.9 percentile range, samples of all channels interleaved,
bit-packed). It is sent to the transmitter board over USB serial; the transmitter forwards it over BLE in
240-byte notifications; the receiver board counts, checksums and times what arrives.
"""
import argparse
import csv
import json
import queue
import sys
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing: pip install pyserial")

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "pipeline"
RING_WINDOW = 16384          # bytes the host may have in flight to the transmitter (its ring is 24576)
CHUNK = 2048                 # bytes per USB write
ESP_VID = 0x303A             # Espressif native USB serial


def kv(line):
    """'RXSTAT why=end pkts=12' -> {'why': 'end', 'pkts': '12'}"""
    return dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)


# ============================================================================ serial node
class Node:
    def __init__(self, name, port, log_path):
        self.name, self.port = name, port
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate = port, 2_000_000
        self.ser.timeout, self.ser.write_timeout = 0.05, 10
        self.ser.dtr, self.ser.rts = True, False          # DTR on = "terminal connected"; RTS off = no reset pulse
        self.ser.open()
        self.lines = []                                    # (perf_counter, text)
        self.cv = threading.Condition()
        self.consumed = 0                                  # latest "K" credit from the transmitter
        self.packets = {}                                  # seq -> payload bytes forwarded by the receiver ("B" records)
        self.packet_hook = None                            # optional f(seq, payload, t_read): called for every "B" record as soon as it is complete
        self.t_last_data = time.perf_counter()             # when anything last arrived from this board
        self.armed = False                                # set once the board has answered PING; a later READY banner = it rebooted
        self.reboots = 0
        self.log = open(log_path, "w", encoding="utf-8")
        self.alive = True
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        """Splits the serial stream into text lines, except after a 'B <seq> <len>' header, where the next
        <len> bytes are a binary payload (which may itself contain newlines)."""
        buf = bytearray()
        need = None                                        # (seq, length) while waiting for a payload
        while self.alive:
            try:
                data = self.ser.read(max(1, self.ser.in_waiting))
            except Exception as e:                         # port vanished
                self._add(f"HOSTERR {e}")
                return
            if not data:
                continue
            t_read = time.perf_counter()                   # when this chunk came out of the USB driver (the arrival time of any packet it completes)
            self.t_last_data = t_read                      # lets a caller notice a port that has gone silent
            buf += data
            while True:
                if need is not None:
                    if len(buf) < need[1]:
                        break
                    payload = bytes(buf[:need[1]])
                    self.packets[need[0]] = payload
                    del buf[:need[1]]
                    hook = self.packet_hook
                    if hook is not None:
                        hook(need[0], payload, t_read)
                    need = None
                    continue
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = bytes(buf[:i]).decode("utf-8", "replace").strip()
                del buf[:i + 1]
                if line.startswith("B ") and len(line.split()) == 3 and line.split()[1].isdigit() and line.split()[2].isdigit():
                    need = (int(line.split()[1]), int(line.split()[2]))
                else:
                    self._add(line)

    def _add(self, text):
        if not text:
            return
        t = time.perf_counter()
        if text.startswith("READY") and self.armed:
            self.reboots += 1                                # the firmware crashed and restarted (it comes back with no role)
        if text.startswith("K "):
            try:
                self.consumed = int(text.split()[1])
            except ValueError:
                pass
            return                                          # credits are not logged
        with self.cv:
            self.lines.append((t, text))
            self.cv.notify_all()
        self.log.write(f"{t:.4f} {text}\n")
        self.log.flush()

    def send(self, text):
        self.ser.write((text + "\n").encode())

    def mark(self):
        with self.cv:
            return len(self.lines)

    def wait(self, prefix, since=0, timeout=10.0, contains=None):
        """First line at index >= since that starts with `prefix` (and contains `contains`)."""
        end = time.perf_counter() + timeout
        with self.cv:
            while True:
                for i in range(since, len(self.lines)):
                    t, text = self.lines[i]
                    if text.startswith(prefix) and (contains is None or contains in text):
                        return t, text
                left = end - time.perf_counter()
                if left <= 0:
                    return None
                self.cv.wait(left)

    def check_alive(self):
        if self.reboots:
            raise RuntimeError(f"[{self.name}] the board REBOOTED during the run (firmware crash). Its role is lost, so the "
                               f"results from here on would be wrong; see the serial log for the crash dump.")

    def cmd(self, text, ok_prefix=None, timeout=5.0):
        self.check_alive()
        since = self.mark()
        self.send(text)
        want = ok_prefix or "OK"
        res = self.wait(want, since, timeout)
        if res is None:
            raise RuntimeError(f"[{self.name}] no '{want}' reply to '{text}' within {timeout}s")
        return res[1]

    def lines_since(self, since, prefix):
        with self.cv:
            return [(t, x) for t, x in self.lines[since:] if x.startswith(prefix)]

    def close(self):
        self.alive = False
        try:
            self.ser.close()
        except Exception:
            pass
        self.log.close()


def boot_check(node):
    """The board may be mid-boot after the port opens; ping until it answers."""
    for _ in range(12):
        since = node.mark()
        node.send("PING")
        r = node.wait("PONG", since, 0.7)
        if r:
            node.armed = True
            return kv(r[1])
        time.sleep(0.2)
    raise RuntimeError(f"[{node.name}] {node.port} does not answer PING. Is the firmware flashed (USB CDC On Boot = Enabled)?")


# ============================================================================ data source
def pack_bits(codes, bits):
    flat = codes.reshape(-1)
    if bits == 16:
        return flat.astype("<u2").tobytes()
    if bits == 8:
        return flat.astype(np.uint8).tobytes()
    if bits == 12:
        if len(flat) % 2:
            flat = np.append(flat, 0)
        a, b = flat[0::2].astype(np.uint16), flat[1::2].astype(np.uint16)
        out = np.stack([a & 0xFF, (a >> 8) | ((b & 0x0F) << 4), b >> 4], axis=1)
        return out.astype(np.uint8).tobytes()
    raise ValueError("bits must be 8, 12 or 16")


_DF_CACHE = {}


def _cached(key, loader):
    """Loading a recording takes 20 to 80 s. A campaign that sends several rates of the same recording loads it once
    (the driver clears the cache between recordings to keep memory down)."""
    if key not in _DF_CACHE:
        _DF_CACHE[key] = loader()
    return _DF_CACHE[key]


def build_stream(args):
    """Returns (bytes, description dict)."""
    src = args.source
    if src == "synthetic":
        rng = np.random.default_rng(0)
        n = int(args.fs * args.seconds)
        codes = rng.integers(0, 2 ** args.bits, size=(n, args.channels))
        return pack_bits(codes, args.bits), dict(source="synthetic", fs=args.fs, channels=args.channels, bits=args.bits)

    sys.path.insert(0, str(PIPELINE))
    from config import RAW_FS, DATASETS
    from bitrate_sweep import variable_rate_filter_downsample, channel_importance_ranking

    kind, _, rest = src.partition(":")
    if kind == "vaskov":
        from data_io import load_raw
        name = rest or "P1_1of12"
        df = _cached(("vaskov", name), lambda: load_raw(name))
        chans = DATASETS[name]["channels"]
        try:
            chans = channel_importance_ranking(f"{DATASETS[name]['participant']}_combined_noRest", chans)
        except Exception as e:
            print(f"(channel ranking unavailable, using file order: {e})")
        label = f"Vaskov {name}"
        file_key, participant, ranges = name, DATASETS[name]["participant"], None
    elif kind == "wallace":
        from wallace_data import PARTICIPANTS, load_wallace_run
        from wallace_bitrate import channel_importance_ranking_wallace
        who, _, task = rest.partition(":")
        part = f"Wallace_{who or 'S1'}"
        runs = PARTICIPANTS[part]["run_for_task"]
        run = runs[int(task)] if task else next(iter(runs.values()))
        df = _cached(("wallace", part, run), lambda: load_wallace_run(part, run))
        chans = channel_importance_ranking_wallace(part)
        label = f"Wallace {part} run {run}"
        file_key, participant = run, part
        ranges = None
    else:
        raise SystemExit("--source must be synthetic, vaskov:<file> or wallace:<S1|S2>[:task]")

    chans = chans[:args.channels]
    factor = RAW_FS // args.fs
    n_dec = len(df) // factor
    if str(args.start) == "auto":                           # start 1 s before the first movement cue
        moving = np.flatnonzero(df["cue"].to_numpy() != 1)
        args.start = max(0.0, (moving[0] / RAW_FS - 1.0)) if len(moving) else 0.0
    i0d = int(float(args.start) * args.fs)
    i1d = min(n_dec, i0d + int(args.seconds * args.fs))
    if i1d - i0d < args.fs:
        raise SystemExit("Recording is too short for the requested --start/--seconds")
    if kind == "wallace" and not args.raw:                 # one fixed quantiser range per channel, pooled over all runs (as in the sweeps)
        from wallace_bitrate import global_quant_ranges
        ranges = global_quant_ranges(participant, args.fs, list(chans))
    cols, los, his = [], [], []
    for ch in chans:
        x = df[ch].to_numpy()                               # the WHOLE recording is filtered, exactly as in the sweeps
        if args.raw:
            y = x[::factor].astype(np.float32)
        else:
            y, _, _ = variable_rate_filter_downsample(x, args.fs)
        lo, hi = ranges[ch] if ranges else np.percentile(y, [0.1, 99.9])
        step = (hi - lo) / (2 ** args.bits - 1) if hi > lo else 1.0
        codes_full = np.clip(np.round((np.clip(y, lo, hi) - lo) / step), 0, 2 ** args.bits - 1).astype(np.uint16)
        cols.append(codes_full[i0d:i1d]); los.append(float(lo)); his.append(float(hi))
    codes = np.stack(cols, axis=1)
    return pack_bits(codes, args.bits), dict(
        source=label, kind=kind, participant=participant, file_key=file_key, fs=args.fs, channels=len(chans), bits=args.bits,
        channel_names=list(chans), lo=los, hi=his, i0d=i0d, n_frames=i1d - i0d, factor=factor, start_s=i0d / args.fs,
        seconds=(i1d - i0d) / args.fs, raw=bool(args.raw))


# ============================================================================ recording
def save_recording(stream, packets, desc, pay, txpwr, out_dir):
    """Rebuild the byte stream as the receiver saw it. Packets that never arrived stay zero and are flagged in
    the mask, so evaluate_received.py knows exactly which samples are missing."""
    got = bytearray(len(stream))
    missing = np.zeros(len(stream), dtype=bool)
    n_pk = -(-len(stream) // pay)
    n_missing = 0
    for s in range(n_pk):
        a, b = s * pay, min((s + 1) * pay, len(stream))
        p = packets.get(s)
        if p is not None and len(p) == b - a:
            got[a:b] = p
        else:
            missing[a:b] = True
            n_missing += 1
    (out_dir / "sent_stream.bin").write_bytes(stream)
    (out_dir / f"received_stream_{txpwr}dBm.bin").write_bytes(bytes(got))
    np.save(out_dir / f"received_missing_{txpwr}dBm.npy", missing)
    json.dump({**desc, "payload_bytes_per_packet": pay, "n_packets": n_pk, "stream_bytes": len(stream)},
              open(out_dir / "stream_meta.json", "w"), indent=1)
    return dict(packets_missing_in_recording=n_missing, bytes_missing=int(missing.sum()),
                bit_exact=bool(bytes(got) == stream))


# ============================================================================ one trial
def run_trial(tx, rx, stream, desc, args, txpwr, out_dir):
    burst = args.mode == "burst"
    tx.cmd("STOP"); rx.cmd("STOP", timeout=40)               # a receiver stuck in a failed connect() (up to 30 s) answers only afterwards
    time.sleep(0.6)
    m_pw = tx.mark()
    tx.cmd(f"SET txpwr={txpwr} phy={args.phy} pay={args.payload} verify=0")
    pw = tx.wait("EV TXPWR", m_pw, 1)
    record = (not burst) and not args.no_record
    rx.cmd(f"SET txpwr={txpwr} phy={args.phy} ci={args.ci} lat=0 tmo=2000 verify={1 if burst else 0} fwd={1 if record else 0}")
    rx.packets = {}
    tx.cmd("CLEAR"); rx.cmd("CLEAR")

    m_tx, m_rx = tx.mark(), rx.mark()
    tx.cmd("ADV"); time.sleep(0.4); rx.cmd("SCAN")
    if not tx.wait("EV SUBSCRIBED 1", m_tx, 40) or not rx.wait("EV CONNECTED role=RX", m_rx, 5):
        raise RuntimeError("The boards did not connect within 40 s (are they close together? is another central grabbing the TX?)")
    p_tx = tx.wait("EV PARAMS", m_tx, 8)
    p_rx = rx.wait("EV PARAMS", m_rx, 8)
    link = {"tx": kv(p_tx[1]) if p_tx else {}, "rx": kv(p_rx[1]) if p_rx else {}}
    print(f"  connected: interval {link['rx'].get('itvl_ms')} ms, PHY tx/rx {link['rx'].get('phy_tx')}/{link['rx'].get('phy_rx')}, "
          f"MTU {link['rx'].get('mtu')}")

    time.sleep(2.5)                                         # let the link settle and take a few idle RSSI samples
    idle_rx = [int(kv(x)["rssi"]) for _, x in rx.lines_since(m_rx, "T ") if int(kv(x)["rssi"]) != 0]
    # the first PARAMS line can predate the receiver's connection-interval update, so read the settled values
    i_tx, i_rx = tx.mark(), rx.mark()
    tx.send("INFO"); rx.send("INFO")
    p_tx, p_rx = tx.wait("EV PARAMS", i_tx, 3), rx.wait("EV PARAMS", i_rx, 3)
    link = {"tx": kv(p_tx[1]) if p_tx else link["tx"], "rx": kv(p_rx[1]) if p_rx else link["rx"]}
    print(f"  settled link: interval TX {link['tx'].get('itvl_ms')} ms / RX {link['rx'].get('itvl_ms')} ms, "
          f"PHY {link['rx'].get('phy_tx')}M/{link['rx'].get('phy_rx')}M, MTU {link['rx'].get('mtu')}")
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.consumed = 0
    if burst:
        rate = int(args.burst_rate)
        tx.cmd(f"BEGIN burst dur_ms={int(args.burst_seconds * 1000)} rate={rate}")
        expected_s = args.burst_seconds
        t_start = time.perf_counter()
        stalled, sent_total, crc_host = 0.0, 0, None
    else:
        tx.cmd("BEGIN stream")
        bps = (desc["fs"] * desc["channels"] * desc["bits"]) if args.realtime else 0
        t_start = time.perf_counter()
        sent_total, stalled = 0, 0.0
        data = stream
        crc_host = zlib.crc32(data) & 0xFFFFFFFF
        limit_s = 20 + 6 * len(data) * 8 / 100e3             # generous for a free-running transfer: assumes at least about 17 kb/s
        if bps:                                              # a real-time stream legitimately takes the stream's own duration
            limit_s = max(limit_s, 20 + 1.5 * len(data) * 8 / bps)
        while sent_total < len(data):
            room = RING_WINDOW - (sent_total - tx.consumed)
            n = min(CHUNK, room, len(data) - sent_total)
            if n < 256 and len(data) - sent_total >= 256:
                time.sleep(0.001); stalled += 0.001
                continue
            if bps and sent_total + n > (time.perf_counter() - t_start) * bps / 8 + CHUNK:
                time.sleep(0.002)
                continue
            tx.ser.write(f"D {n}\n".encode() + data[sent_total:sent_total + n])
            sent_total += n
            if time.perf_counter() - t_start > limit_s:
                raise RuntimeError("transfer is taking far too long, aborting")
        tx.send("FIN")
        expected_s = len(data) * 8 / max(bps, 1) if bps else 0
    t_fed = time.perf_counter()

    wait_s = 30 + expected_s + (len(stream) * 8 / 50e3 if not burst else 0)
    r_tx = tx.wait("TXSTAT", m_tx, wait_s)
    r_rx = rx.wait("RXSTAT", m_rx, 15)
    t_done = time.perf_counter()
    if not r_tx or not r_rx:
        raise RuntimeError("Did not get TXSTAT/RXSTAT. Check the *.log files in the output folder.")
    txs, rxs = kv(r_tx[1]), kv(r_rx[1])

    # ------------------------------------------------------------- time series
    series = []
    for t, line in rx.lines_since(m_rx, "T "):
        d = kv(line)
        series.append(dict(t_s=round(t - t_start, 3), rx_kbps=int(d["bytes"]) * 8 / 1000, rx_pkts=int(d["pkts"]),
                           lost=int(d["lost"]), rssi_rx=int(d["rssi"])))
    rssi_tx = [int(kv(x)["rssi"]) for _, x in tx.lines_since(m_tx, "S ") if int(kv(x)["rssi"]) != 0]
    rssi_rx = [s["rssi_rx"] for s in series if s["rssi_rx"] != 0]

    # ------------------------------------------------------------- derived numbers
    pk_tx, pk_rx = int(txs["pkts"]), int(rxs["pkts"])
    by_rx, lost = int(rxs["bytes"]), int(rxs["lost"])
    dur_s = int(rxs["dur_us"]) / 1e6
    eff_bytes = by_rx * (pk_rx - 1) / pk_rx if pk_rx > 1 else by_rx
    thr = eff_bytes * 8 / dur_s / 1000 if dur_s > 0 else float("nan")
    hist = [int(v) for v in rxs["hist"].split(",")]
    edges = ["<1 ms", "1-5 ms", "5-10 ms", "10-20 ms", "20-50 ms", "50-100 ms", ">=100 ms"]
    n_gaps = max(sum(hist), 1)
    txp = int(txs["txpwr"])
    applied = int(kv(pw[1]).get("applied", txp)) if pw else txp
    rssi_mean = float(np.mean(rssi_rx)) if rssi_rx else float("nan")
    res = dict(
        tx_power_setting_dbm=txp, tx_power_applied_dbm=applied, phy_requested={1: "1M", 2: "2M", 3: "Coded S2 (500 kb/s)", 4: "Coded S8 (125 kb/s)"}[args.phy], link=link,
        placement_note=args.note,
        mode=args.mode, source=desc, payload_bytes=int(txs["pay"]),
        packets_sent=pk_tx, packets_received=pk_rx, packets_lost=lost, loss_pct=100.0 * lost / max(pk_tx, 1),
        duplicates=int(rxs["dup"]), out_of_order=int(rxs["ooo"]), bad_payload=int(rxs["bad"]),
        tx_notify_retries=int(txs["retries"]), tx_errors=int(txs["errors"]), ring_drops=int(txs["ring_drops"]),
        retry_rate_pct=100.0 * int(txs["retries"]) / max(int(txs["retries"]) + pk_tx, 1),
        payload_bytes_sent=int(txs["bytes"]), payload_bytes_received=by_rx,
        rx_duration_s=dur_s, throughput_kbps=thr, throughput_kBps=thr / 8,
        host_total_s=t_done - t_start, host_feed_s=t_fed - t_start, host_stalled_s=stalled,
        max_gap_ms=int(rxs["max_gap_us"]) / 1000, gap_histogram=dict(zip(edges, hist)),
        gaps_over_20ms_pct=100.0 * sum(hist[4:]) / n_gaps,
        rssi_rx_mean_dbm=rssi_mean, rssi_rx_min_dbm=min(rssi_rx) if rssi_rx else None, rssi_rx_max_dbm=max(rssi_rx) if rssi_rx else None,
        rssi_tx_mean_dbm=float(np.mean(rssi_tx)) if rssi_tx else None, rssi_idle_rx_dbm=idle_rx,
        link_loss_db=applied - rssi_mean if rssi_rx else None,
        margin_over_sensitivity_db=rssi_mean - args.sensitivity if rssi_rx else None,
        stream_crc_tx=txs["crc"], stream_crc_rx=rxs["crc"],
    )
    res["fwd_drops"] = int(rxs.get("fwd_drops", 0))
    cal = getattr(args, "cal_obj", None)
    if cal:
        eff = cal.effective(applied) if applied in cal.rows else float(applied)
        res.update(tx_power_effective_dbm=eff, tx_power_label_verified=bool(applied in cal.rows and cal.verified(applied)),
                   link_loss_calibrated_db=(eff - rssi_mean) if rssi_rx else None, power_cal_file=str(cal.path),
                   power_cal_assumption=cal.assumption)
    if record:
        res["recording"] = save_recording(stream, rx.packets, desc, int(txs["pay"]), txpwr, out_dir)
    if not burst:
        res["offered_kbps"] = desc["fs"] * desc["channels"] * desc["bits"] / 1000
        res["stream_bytes"] = len(stream)
        res["crc_host"] = f"{crc_host:08x}"
        res["integrity_ok"] = (f"{crc_host:08x}" == rxs["crc"] and lost == 0 and by_rx == len(stream))
        res["keeps_up_with_realtime"] = bool(args.realtime and res["host_stalled_s"] < 0.5)
    else:
        res["integrity_ok"] = (res["bad_payload"] == 0 and lost == 0 and txs["crc"] == rxs["crc"])
    if series:
        with open(out_dir / f"timeseries_{txpwr}dBm.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(series[0].keys()))
            w.writeheader(); w.writerows(series)
    res["_series"] = series
    return res


def print_report(res):
    def f(v, nd=2):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    L = res["link"]["rx"]
    crc_host = (" / host " + res["crc_host"]) if "crc_host" in res else ""
    cal_line = ""
    if "tx_power_effective_dbm" in res:
        cal_line = (f"\n  calibrated      : effective TX power {res['tx_power_effective_dbm']:+.1f} dBm "
                    f"({'verified by two receivers' if res['tx_power_label_verified'] else 'label not verified, nominal used'}) "
                    f"-> link loss {f(res['link_loss_calibrated_db'], 1)} dB (relative to the 0 dBm label; absolute power not measured)")
    print(f"""
  ---- result: TX power {res['tx_power_setting_dbm']} dBm (applied {res['tx_power_applied_dbm']} dBm), {res['mode']} ----
  link            : interval {L.get('itvl_ms')} ms, PHY {L.get('phy_tx')}M/{L.get('phy_rx')}M, MTU {L.get('mtu')}, payload {res['payload_bytes']} B/packet
  throughput      : {f(res['throughput_kbps'])} kb/s  ({f(res['throughput_kBps'])} kB/s) application payload, measured at the receiver
  transfer time   : {f(res['rx_duration_s'])} s first-to-last packet (host wall time incl. USB feed: {f(res['host_total_s'])} s)
  packets         : sent {res['packets_sent']}, received {res['packets_received']}, lost {res['packets_lost']} ({f(res['loss_pct'])} %), dup {res['duplicates']}, out-of-order {res['out_of_order']}, bad payload {res['bad_payload']}
  retries         : {res['tx_notify_retries']} notify attempts refused by the BLE stack ({f(res['retry_rate_pct'])} % of attempts); other TX errors {res['tx_errors']}; USB ring drops {res['ring_drops']}
  inter-arrival   : max gap {f(res['max_gap_ms'], 1)} ms, {f(res['gaps_over_20ms_pct'], 1)} % of gaps >= 20 ms
  RSSI at receiver: mean {f(res['rssi_rx_mean_dbm'], 1)} dBm (min {res['rssi_rx_min_dbm']}, max {res['rssi_rx_max_dbm']}); at transmitter mean {f(res['rssi_tx_mean_dbm'], 1)} dBm
  link loss       : {f(res['link_loss_db'], 1)} dB  (= TX power - RSSI; includes both antennas, mismatch and the boards' own errors)
  margin          : {f(res['margin_over_sensitivity_db'], 1)} dB above the assumed receiver sensitivity{cal_line}
  integrity       : {'OK' if res['integrity_ok'] else 'FAILED'}  (stream CRC tx {res['stream_crc_tx']} / rx {res['stream_crc_rx']}{crc_host})""")
    if "offered_kbps" in res:
        print(f"  offered load    : {f(res['offered_kbps'])} kb/s; the host waited {f(res['host_stalled_s'])} s for the transmitter's buffer to drain")
    if "recording" in res:
        r = res["recording"]
        print(f"  recorded        : {'bit-exact copy of what was sent' if r['bit_exact'] else str(r['packets_missing_in_recording']) + ' packets missing'}"
              f" (USB forwarding drops {res['fwd_drops']}); saved for evaluate_received.py")


def make_plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for r in results:
        s = r["_series"]
        if not s:
            continue
        lab = f"{r['tx_power_setting_dbm']} dBm"
        ax[0].plot([x["t_s"] for x in s], [x["rx_kbps"] for x in s], marker="o", ms=3, label=lab)
        ax[1].plot([x["t_s"] for x in s], [x["rssi_rx"] for x in s], marker="o", ms=3, label=lab)
    ax[0].set_ylabel("received payload (kb/s)"); ax[1].set_ylabel("RSSI at receiver (dBm)"); ax[1].set_xlabel("time since start (s)")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "throughput_rssi.png", dpi=130)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--tx-port"); ap.add_argument("--rx-port")
    ap.add_argument("--source", default="vaskov:P1_1of12", help="synthetic | vaskov:<P1_1of12|P1_Grasps|P2_1of10|P2_Grasps> | wallace:<S1|S2>[:task]")
    ap.add_argument("--mode", choices=["stream", "burst"], default="stream")
    ap.add_argument("--seconds", type=float, default=30, help="seconds of recording to send (stream mode)")
    ap.add_argument("--start", default="auto", help="start offset in the recording (s), or 'auto' = 1 s before the first movement")
    ap.add_argument("--no-record", action="store_true", help="do not save the received samples (default: they are saved for evaluate_received.py)")
    ap.add_argument("--fs", type=int, default=1000); ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--bits", type=int, default=16, choices=[8, 12, 16])
    ap.add_argument("--raw", action="store_true", help="skip the band-pass filter (just decimate)")
    ap.add_argument("--realtime", action="store_true", help="pace the data at the sampling rate instead of as fast as possible")
    ap.add_argument("--burst-seconds", type=float, default=10); ap.add_argument("--burst-rate", type=int, default=0, help="bits/s, 0 = maximum")
    ap.add_argument("--txpwr", default="9", help="dBm, or a comma list for a sweep, e.g. -24,-12,0,9,20")
    ap.add_argument("--cal", help="power calibration from power_calibration.py (path, its folder, or 'latest'): results also report the EFFECTIVE power")
    ap.add_argument("--effective-dbm", action="store_true", help="with --cal: read --txpwr as the power you WANT and send the label that gives it")
    ap.add_argument("--phy", type=int, choices=[1, 2, 3, 4], default=2,
                    help="1 = 1M, 2 = 2M, 3 = LE Coded S=2 (500 kb/s), 4 = LE Coded S=8 (125 kb/s); 3 and 4 need firmware 1.2. The radio reports a coded link as PHY 3 for both S=2 and S=8")
    ap.add_argument("--ci", type=float, default=7.5, help="connection interval requested (ms)")
    ap.add_argument("--payload", type=int, default=240)
    ap.add_argument("--sensitivity", type=float, default=-94, help="receiver sensitivity assumed for the margin (dBm)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--note", default="", help="where the boards are (distance, orientation, surface); saved with the results")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if args.list_ports or not (args.tx_port and args.rx_port):
        print("Serial ports:")
        for p in ports:
            print(f"  {p.device:8} {p.description}  VID:PID={p.vid and hex(p.vid)}:{p.pid and hex(p.pid)}")
        if not args.list_ports:
            esp = [p.device for p in ports if p.vid == ESP_VID]
            print(f"\nGive both ports, e.g.  --tx-port {esp[0] if esp else 'COM3'} --rx-port {esp[1] if len(esp) > 1 else 'COM4'}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"{stamp}_{args.mode}_{args.source.replace(':', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stream, desc = (b"", dict(source="synthetic pattern", fs=args.fs, channels=args.channels, bits=args.bits)) if args.mode == "burst" else build_stream(args)
    if args.mode == "stream":
        print(f"Stream: {desc['source']}, {desc['channels']} ch x {desc['fs']} Hz x {desc['bits']} bit = "
              f"{desc['fs'] * desc['channels'] * desc['bits'] / 1000:.1f} kb/s offered, {len(stream) / 1024:.0f} kB total")

    tx = Node("TX", args.tx_port, out_dir / "tx_serial.log")
    rx = Node("RX", args.rx_port, out_dir / "rx_serial.log")
    results = []
    failed = []                                              # settings at which the boards never connected
    try:
        for n in (tx, rx):
            info = boot_check(n)
            print(f"{n.name} on {n.port}: {info.get('chip')} fw {info.get('fw')} mac {info.get('mac')}")
            if n is tx:
                tx_info = info
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        args.cal_obj = None
        if args.cal:
            from power_cal import PowerCal, find_cal
            args.cal_obj = PowerCal(find_cal(args.cal))
            print(args.cal_obj.describe())
            if args.cal_obj.mac and args.cal_obj.mac.lower() != tx_info.get("mac", "").lower():
                print(f"  WARNING: this table was measured on the board {args.cal_obj.mac}; the transmitter now is {tx_info.get('mac')}. "
                      f"The corrections may not apply.")
        elif args.effective_dbm:
            raise SystemExit("--effective-dbm needs --cal (a calibration from power_calibration.py)")
        for rep in range(args.repeat):
            for want in [float(x) for x in str(args.txpwr).split(",")]:
                pw = args.cal_obj.label_for(want) if (args.cal_obj and args.effective_dbm) else int(want)
                if args.cal_obj and args.effective_dbm:
                    print(f"\n== trial {rep + 1}/{args.repeat}: wanted {want:+.0f} dBm -> label {pw:+d} dBm "
                          f"(effective {args.cal_obj.effective(pw):+.1f} dBm) ==")
                else:
                    print(f"\n== trial {rep + 1}/{args.repeat}, TX power {pw} dBm ==")
                try:
                    res = run_trial(tx, rx, stream, desc, args, pw, out_dir)
                except RuntimeError as e:
                    if "did not connect" not in str(e):
                        raise                                # a reboot or a dead port must still stop the run
                    print(f"  NO CONNECTION at {pw} dBm (the link is too weak): carrying on with the next setting")
                    failed.append(dict(tx_power_setting_dbm=pw, repeat=rep + 1, reason=str(e)))
                    continue
                print_report(res)
                results.append(res)
        tx.cmd("STOP"); rx.cmd("STOP", timeout=40)
    finally:
        tx.close(); rx.close()

    summary = [{k: v for k, v in r.items() if k != "_series"} for r in results]
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    if failed:
        json.dump(failed, open(out_dir / "failed_to_connect.json", "w"), indent=2)
        print("\nNo connection at: " + ", ".join(f"{f['tx_power_setting_dbm']} dBm" for f in failed))
    if len(results) > 1:
        cols = ["tx_power_setting_dbm", "tx_power_applied_dbm", "throughput_kbps", "packets_lost", "loss_pct", "tx_notify_retries",
                "rssi_rx_mean_dbm", "link_loss_db", "margin_over_sensitivity_db", "integrity_ok"]
        if "tx_power_effective_dbm" in results[0]:
            cols[2:2] = ["tx_power_effective_dbm", "tx_power_label_verified"]
            cols.insert(cols.index("link_loss_db") + 1, "link_loss_calibrated_db")
        with open(out_dir / "sweep.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in results:
                w.writerow([r[c] for c in cols])
    make_plot(results, out_dir)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
