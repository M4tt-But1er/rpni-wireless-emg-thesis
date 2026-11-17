#!/usr/bin/env python3
"""live_classify.py -- stream EMG through the BLE link in real time and classify every window the moment it arrives.
Measures the latency of every decision and splits it into its parts.

What happens (all on one PC, the two ESP32 boards are plugged in over USB):
  sensor     this script plays the electrode amplifier: it produces one frame (all channels) every 1/fs seconds and hands
             it to the transmitter board over USB at that moment, never earlier (so the data cannot be "pre-delivered").
  link       the transmitter packs the bytes into BLE notifications; the receiver board forwards each packet to the PC.
  classify   a worker thread decodes each window the moment the packet that completes it has arrived, extracts the
             thesis features, and runs the Random Forest and the SVM (trained beforehand on everything OUTSIDE the
             transmitted slice, exactly as evaluate_received.py does). It sees only data that has arrived.

Which windows: the same ones as the offline evaluation (200 ms long, 50 ms hop, starting at the onset-trimmed segment start,
movement only). The window POSITIONS therefore come from the offline segmentation (which uses the cue labels), so this is a test
of the data path and the timing, not of a fully deployable controller that has to find movements by itself.

Latency of one decision, measured on the PC clock (every timestamp comes from the same perf_counter):
  t_src   the instant the LAST sample of the window was produced (ideal sensor clock, set by this script)
  parts   packet fill wait  (the transmitter sends only when a packet is full: payload / data rate)
        + feed lag          (this script handing the byte to USB later than the ideal instant; harness error)
        + link transit      (USB to the transmitter, BLE, USB from the receiver to the PC)
        + queue wait        (decoder busy with an earlier window)
        + feature extraction + model prediction
  Total = decision time - t_src.  The window itself is 200 ms long, so the newest data in a decision is 0 ms old at t_src and
  the oldest is 200 ms old; "controller delay" in the myoelectric literature is usually window length + processing.

Examples (from this folder):
  python live_classify.py --tx-port COM5 --rx-port COM6 --rates 128 --payloads 240 --seconds 60
  python live_classify.py --tx-port COM5 --rx-port COM6 --rates 10,128,960 --payloads 240,auto
  python live_classify.py --rates 10,128,960 --prepare-only          # train and cache the models, no hardware
"""
import argparse
import csv
import json
import pickle
import queue
import sys
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))

import ble_link_test as B                     # noqa: E402  (Node, build_stream, ...)
import evaluate_received as E                 # noqa: E402  (decode, window_features, clean)
from run_phantom_campaign import RATES, parse_rate   # noqa: E402

MODELS = ["RandomForest", "SVM"]
SHORT = {"RandomForest": "RF", "SVM": "SVM"}
CACHE_DIR = HERE / "results" / "live_model_cache"


class FastForest:
    """The fitted sklearn Random Forest, evaluated for ONE feature vector by walking all trees together with numpy.

    sklearn's predict() costs about 25 ms for a single window (95 ms with its default n_jobs=-1) and almost all of that is Python
    overhead, not the trees; that would hide the link latency this script is measuring. This class reads the SAME fitted trees
    (thresholds, features, leaf class fractions) and gives the same answer; run_live() checks that on every run before it
    starts (same predictions, probabilities equal to 1e-9), and the finished run is checked again against sklearn's batch predict."""

    def __init__(self, rf):
        trees = [e.tree_ for e in rf.estimators_]
        T, M, K = len(trees), max(t.node_count for t in trees), len(rf.classes_)
        self.left = np.full((T, M), -1, dtype=np.int32)
        self.right = np.full((T, M), -1, dtype=np.int32)
        self.feat = np.zeros((T, M), dtype=np.int32)
        self.thr = np.zeros((T, M), dtype=np.float64)
        self.prob = np.zeros((T, M, K), dtype=np.float64)
        for i, t in enumerate(trees):
            n = t.node_count
            self.left[i, :n], self.right[i, :n] = t.children_left, t.children_right
            self.feat[i, :n], self.thr[i, :n] = t.feature, t.threshold
            v = t.value[:, 0, :]
            self.prob[i, :n] = v / v.sum(axis=1, keepdims=True)
        self.depth = max(t.max_depth for t in trees)
        self.idx = np.arange(T)
        self.classes = rf.classes_

    def predict_proba(self, x):
        x32 = np.asarray(x, dtype=np.float32)                 # sklearn casts its input to float32 before comparing with the thresholds
        node = np.zeros(len(self.idx), dtype=np.int32)
        for _ in range(self.depth):
            l = self.left[self.idx, node]
            go_left = x32[self.feat[self.idx, node]] <= self.thr[self.idx, node]
            node = np.where(l < 0, node, np.where(go_left, l, self.right[self.idx, node]))   # a leaf has no children: stay
        return self.prob[self.idx, node].mean(axis=0)

    def predict(self, x):
        return int(self.classes[int(np.argmax(self.predict_proba(x)))])


# ====================================================================================== training (offline, before the clock starts)
def prep_key(desc):
    return (f"{desc['kind']}_{desc['participant']}_{desc['file_key']}_{desc['fs']}Hz_{desc['channels']}ch_{desc['bits']}bit"
            f"_start{desc['start_s']:.2f}_len{desc['seconds']:.1f}")


def prepare(desc, retrain=False):
    """Trains the classifiers for this configuration on everything outside the transmitted slice and returns them with the
    test windows. Same steps as evaluate_received.py (the same functions); cached, because building the features takes minutes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (prep_key(desc) + ".pkl")
    if path.exists() and not retrain:
        print(f"  models for this configuration: loaded from cache ({path.name})", flush=True)
        return pickle.load(open(path, "rb"))

    fs, bits, chans = desc["fs"], desc["bits"], desc["channel_names"]
    print(f"  building the offline pipeline features for {fs} Hz x {len(chans)} ch x {bits} bit (a few minutes) ...", flush=True)
    t0 = time.time()
    if desc["kind"] == "vaskov":
        from bitrate_sweep import build_one_config
        from run_bitrate_search import DATASET_SPECS
        raws = DATASET_SPECS[f"{desc['participant']}_combined"]["raw_names"]
        df = pd.concat([build_one_config(r, fs, chans, bits, "eval", snr_db=None) for r in raws], ignore_index=True)
    else:
        from wallace_bitrate import build_wallace_config
        rng = {c: (lo, hi) for c, lo, hi in zip(chans, desc["lo"], desc["hi"])}
        df = build_wallace_config(desc["participant"], fs, chans, bits, quant_ranges=rng)
    from bitrate_sweep import WINDOW_MS_SWEEP, HOP_MS_SWEEP
    win = int(round(WINDOW_MS_SWEEP * fs / 1000))
    hop = int(round(HOP_MS_SWEEP * fs / 1000))
    i0, n_frames = desc["i0d"], desc["n_frames"]
    widx = np.rint(df["window_time_s"].to_numpy() * fs).astype(int)
    in_file = df["segment_id"].str.startswith(desc["file_key"] + "_").to_numpy()
    test = in_file & (widx >= i0) & (widx + win <= i0 + n_frames)
    touches = in_file & (widx + win > i0) & (widx < i0 + n_frames)
    if not test.any():
        raise SystemExit("The transmitted slice contains no movement windows.")
    train = ~df["segment_id"].isin(set(df.loc[touches, "segment_id"]))
    test_classes = sorted(df.loc[test, "cue"].unique().tolist())
    missing = sorted(set(test_classes) - set(df.loc[train, "cue"].unique().tolist()))

    from run_packet_loss_test import reduce_features
    from models import make_model_pipeline
    from sklearn.preprocessing import StandardScaler
    reduced = reduce_features(df[train])
    df_test = df[test].reset_index(drop=True)
    y = df_test["cue"].to_numpy()
    starts = widx[test] - i0
    order = np.argsort(starts, kind="stable")                      # classify in time order
    Xoff = E.clean(df_test[reduced])
    Xtr, ytr = E.clean(df.loc[train, reduced]), df.loc[train, "cue"].to_numpy()
    models, pred_off = {}, {}
    for name in MODELS:
        scale, est = make_model_pipeline(name)
        sc = StandardScaler().fit(Xtr) if scale else None
        est.fit(sc.transform(Xtr) if sc else Xtr, ytr)
        models[name] = (sc, est)
        pred_off[name] = est.predict(sc.transform(Xoff) if sc else Xoff)
    prep = dict(reduced=reduced, models=models, starts=starts[order], y=y[order], pred_offline={k: v[order] for k, v in pred_off.items()},
                win=win, hop=hop, test_classes=[int(c) for c in test_classes], classes_missing_from_training=[int(c) for c in missing],
                accuracy_meaningful=not missing, n_train=int(train.sum()), n_test=int(test.sum()), n_features=len(reduced),
                build_seconds=round(time.time() - t0, 1))
    pickle.dump(prep, open(path, "wb"))
    print(f"  trained in {prep['build_seconds']:.0f} s: {prep['n_train']} training windows, {prep['n_test']} test windows, "
          f"{len(reduced)} features, classes in test {test_classes}" + (f", NO training data for {missing}" if missing else ""), flush=True)
    return prep


# ====================================================================================== one window, as the live system does it
def decode_window(buf, C, bits, lo, step):
    """bytes of one window -> (frames x channels) float32; identical arithmetic to evaluate_received.decode."""
    if bits == 16:
        codes = np.frombuffer(buf, dtype="<u2")
    elif bits == 8:
        codes = np.frombuffer(buf, dtype=np.uint8)
    else:
        raise SystemExit("live_classify supports 8 and 16 bit streams")
    return (codes.astype(np.float64).reshape(-1, C) * step + lo).astype(np.float32)


def window_vector(x, fs, chans, reduced):
    """Features of ONE window; the same feature code as window_features(), restricted to the selected columns."""
    from features import extract_channel_features
    cols = {}
    for j, ch in enumerate(chans):
        xw = np.ascontiguousarray(x[:, j])[None, :]
        for name, arr in extract_channel_features(xw, fs=float(fs)).items():
            cols[f"{ch}_{name}"] = arr[0]
    return np.nan_to_num(np.array([cols[c] for c in reduced], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class Classifier(threading.Thread):
    """Receives packets, and classifies each window the moment the packet that completes it has arrived."""

    def __init__(self, prep, desc, pay, stream_len, n_pk):
        super().__init__(daemon=True)
        self.q = queue.SimpleQueue()
        self.prep, self.desc, self.pay, self.L, self.n_pk = prep, desc, pay, stream_len, n_pk
        self.C, self.bits, self.fs = desc["channels"], desc["bits"], desc["fs"]
        self.frame_bytes = self.C * self.bits // 8
        self.lo = np.array(desc["lo"])
        hi = np.array(desc["hi"])
        self.step = np.where(hi > self.lo, (hi - self.lo) / (2 ** self.bits - 1), 1.0)
        self.win = prep["win"]
        self.end_byte = (prep["starts"] + self.win) * self.frame_bytes        # exclusive end of each window, in stream bytes
        self.got = bytearray(stream_len)
        self.have = np.zeros(n_pk, dtype=bool)
        self.t_pkt = np.full(n_pk, np.nan)
        self.next_seq = 0
        self.next_win = 0
        self.recs = []
        self.error = None
        self.last_print = 0.0

    def hook(self, seq, payload, t_read):                 # called from the receiver's serial reader thread: do nothing slow here
        self.q.put((seq, payload, t_read))

    def finish(self):
        self.q.put(None)

    def run(self):
        try:
            self._loop()
        except Exception as e:                            # surface it to the main thread instead of dying silently
            import traceback
            self.error = traceback.format_exc()

    def _loop(self):
        P, d = self.prep, self.desc
        n_win = len(P["starts"])
        while True:
            item = self.q.get()
            if item is None:
                return
            seq, payload, t_arr = item
            if seq >= self.n_pk or self.have[seq]:
                continue
            a = seq * self.pay
            self.got[a:a + len(payload)] = payload
            self.have[seq] = True
            self.t_pkt[seq] = t_arr
            while self.next_seq < self.n_pk and self.have[self.next_seq]:
                self.next_seq += 1
            contig = min(self.next_seq * self.pay, self.L)
            while self.next_win < n_win and self.end_byte[self.next_win] <= contig:
                self._classify(self.next_win, t_arr)
                self.next_win += 1

    def _classify(self, w, t_ready):
        P, d = self.prep, self.desc
        t_work0 = time.perf_counter()
        s = int(P["starts"][w])
        a, b = s * self.frame_bytes, (s + self.win) * self.frame_bytes
        x = decode_window(bytes(self.got[a:b]), self.C, self.bits, self.lo, self.step)
        vec = window_vector(x, self.fs, d["channel_names"], P["reduced"])
        t_f = time.perf_counter()
        rec = dict(w=w, t_ready=t_ready, t_work0=t_work0, d_feat=t_f - t_work0, x=vec)
        t_m0 = time.perf_counter()
        rec["pred_RF"] = P["fast_rf"].predict(vec)                            # the same fitted forest, see FastForest
        rec["d_RF"] = time.perf_counter() - t_m0
        sc, est = P["models"]["SVM"]
        t_m0 = time.perf_counter()
        rec["pred_SVM"] = int(est.predict(sc.transform(vec[None, :]))[0])
        rec["d_SVM"] = time.perf_counter() - t_m0
        self.recs.append(rec)
        if t_f - self.last_print > 5.0:
            self.last_print = t_f
            print(f"    ... {len(self.recs)} windows classified; last: true {int(P['y'][w])}, RF {rec['pred_RF']}, SVM {rec['pred_SVM']}; "
                  f"decided {1000 * (time.perf_counter() - t_ready):.1f} ms after its last packet arrived; backlog {self.q.qsize()} packets", flush=True)


# ====================================================================================== the live run
def usb_rtt(node, n=40):
    """PING round trip through the board's own USB port: roughly what the two USB hops add to every measurement."""
    out = []
    for _ in range(n):
        since = node.mark()
        t0 = time.perf_counter()
        node.send("PING")
        r = node.wait("PONG", since, 1.0)
        if r:
            out.append((r[0] - t0) * 1000)
        time.sleep(0.02)
    return out


def connect(tx, rx, a, txpwr, payload):
    tx.cmd("STOP"); rx.cmd("STOP", timeout=40)
    time.sleep(0.6)
    m_pw = tx.mark()
    tx.cmd(f"SET txpwr={txpwr} phy={a.phy} pay={payload} verify=0")
    tx.wait("EV TXPWR", m_pw, 1)
    rx.cmd(f"SET txpwr={txpwr} phy={a.phy} ci={a.ci} lat=0 tmo=2000 verify=0 fwd=1")
    rx.packets = {}
    tx.cmd("CLEAR"); rx.cmd("CLEAR")
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.cmd("ADV"); time.sleep(0.4); rx.cmd("SCAN")
    if not tx.wait("EV SUBSCRIBED 1", m_tx, 40) or not rx.wait("EV CONNECTED role=RX", m_rx, 5):
        raise RuntimeError("The boards did not connect within 40 s")
    time.sleep(2.5)
    i_tx, i_rx = tx.mark(), rx.mark()
    tx.send("INFO"); rx.send("INFO")
    p_tx, p_rx = tx.wait("EV PARAMS", i_tx, 3), rx.wait("EV PARAMS", i_rx, 3)
    link = {"tx": B.kv(p_tx[1]) if p_tx else {}, "rx": B.kv(p_rx[1]) if p_rx else {}}
    idle = [int(B.kv(x)["rssi"]) for _, x in rx.lines_since(m_rx, "T ") if int(B.kv(x)["rssi"]) != 0]
    return link, idle


def pc_state():
    """Read-only look at the PC's power state. A slow or throttled PC makes the compute part of the latency meaningless (seen on
    5 Nov 2026: feature extraction became 7 to 10 times slower in the middle of a run), so it is recorded with every result."""
    import subprocess
    out = {}
    try:
        o = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout
        out["power_scheme"] = o.split("(")[-1].split(")")[0].strip() if "(" in o else o.strip()
    except Exception as e:
        out["power_scheme"] = f"unknown ({e})"
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\").PercentProcessorPerformance"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        out["cpu_performance_pct_at_start"] = int(o)
    except Exception:
        out["cpu_performance_pct_at_start"] = None
    return out


def selfcheck(prep, desc, stream):
    """Before the clock starts: (1) the fast forest must agree with sklearn on every test window, built through the live code path;
    (2) time the pieces on their own, so the live latency can be read against them."""
    rf = prep["models"]["RandomForest"][1]
    rf.set_params(n_jobs=1)
    ff = prep["fast_rf"] = FastForest(rf)
    C, bits, fs, win = desc["channels"], desc["bits"], desc["fs"], prep["win"]
    fb = C * bits // 8
    lo = np.array(desc["lo"]); hi = np.array(desc["hi"])
    step = np.where(hi > lo, (hi - lo) / (2 ** bits - 1), 1.0)
    X, t_feat = [], []
    for s in prep["starts"]:
        t0 = time.perf_counter()
        X.append(window_vector(decode_window(stream[int(s) * fb:(int(s) + win) * fb], C, bits, lo, step), fs, desc["channel_names"], prep["reduced"]))
        t_feat.append((time.perf_counter() - t0) * 1000)
    X = np.stack(X)
    p_sk = rf.predict_proba(X)
    t_ff, p_ff = [], []
    for x in X:
        t0 = time.perf_counter()
        p_ff.append(ff.predict_proba(x))
        t_ff.append((time.perf_counter() - t0) * 1000)
    p_ff = np.stack(p_ff)
    same = bool((rf.classes_[p_sk.argmax(1)] == ff.classes[p_ff.argmax(1)]).all())
    diff = float(np.abs(p_sk - p_ff).max())
    if not same or diff > 1e-9:
        raise RuntimeError(f"FastForest does not reproduce sklearn (predictions equal: {same}, largest probability difference {diff:.2e}); not using it")
    res = {}
    t_sk = []

    def one_by_one():                                           # sklearn's own single-window call, from a worker thread as in the live run
        for x in X[:40]:
            t0 = time.perf_counter()
            rf.predict(x[None, :])
            t_sk.append((time.perf_counter() - t0) * 1000)
    th = threading.Thread(target=one_by_one); th.start(); th.join()
    res.update(fast_forest_matches_sklearn=same, fast_forest_max_prob_diff=diff, windows_checked=int(len(X)),
               features_standalone_ms=pct(t_feat), fast_rf_standalone_ms=pct(t_ff), rf_sklearn_single_call_ms=pct(t_sk))
    print(f"  self-check on {len(X)} windows: fast forest == sklearn forest (largest probability difference {diff:.1e}); standalone medians: "
          f"features {np.median(t_feat):.1f} ms, fast RF {np.median(t_ff):.2f} ms, sklearn RF single call {np.median(t_sk):.1f} ms", flush=True)
    return res


def pct(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(n=0)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()), min=float(a.min()))


def run_live(tx, rx, stream, desc, prep, a, payload, run_dir):
    fs, C, bits = desc["fs"], desc["channels"], desc["bits"]
    frame_bytes = C * bits // 8
    Bps = fs * frame_bytes
    L = len(stream)
    n_frames = L // frame_bytes
    pay = payload
    n_pk = -(-L // pay)

    link, idle = connect(tx, rx, a, a.txpwr, payload)
    print(f"  connected: interval {link['rx'].get('itvl_ms')} ms, PHY {link['rx'].get('phy_tx')}M/{link['rx'].get('phy_rx')}M, "
          f"MTU {link['rx'].get('mtu')}, payload {pay} B (fills in {1000 * pay / Bps:.1f} ms at this data rate)", flush=True)
    rtt = {"TX": usb_rtt(tx), "RX": usb_rtt(rx)}
    print(f"  USB round trip of the boards' own ports (PING): TX median {np.median(rtt['TX']):.1f} ms, RX median {np.median(rtt['RX']):.1f} ms "
          f"(the two USB hops of this test rig; a real implant has neither)", flush=True)

    check = selfcheck(prep, desc, stream)
    check["pc_state"] = pc_state()
    print(f"  PC: power plan '{check['pc_state']['power_scheme']}', CPU at {check['pc_state']['cpu_performance_pct_at_start']} % of rated performance at the start "
          f"(an idle CPU also reads low)", flush=True)
    clf = Classifier(prep, desc, pay, L, n_pk)
    rx.packets = {}
    rx.packet_hook = clf.hook
    clf.start()
    m_tx, m_rx = tx.mark(), rx.mark()
    tx.consumed = 0
    tx.cmd("BEGIN stream")

    # ---- the sensor: frame f is produced at t0 + f/fs and handed to USB then, never earlier
    crc_host = zlib.crc32(stream) & 0xFFFFFFFF
    t0 = time.perf_counter() + 0.2
    wr_t, wr_cum = [], []
    sent, stalled = 0, 0.0
    limit_s = 30 + 1.5 * n_frames / fs
    print(f"  streaming {n_frames / fs:.0f} s of recording in real time ({Bps * 8 / 1000:.0f} kb/s) and classifying as it arrives ...", flush=True)
    last_diag = time.perf_counter()
    tx.ser.write_timeout = 3
    while sent < L:
        now = time.perf_counter()
        if now - t0 > limit_s:
            raise RuntimeError("the live stream is taking far too long, aborting")
        if now - last_diag > 1.0:                         # a board or its USB port that goes quiet while streaming is a fault, say so at once
            last_diag = now
            for nd in (tx, rx):
                if now - nd.t_last_data > 3.0:
                    raise RuntimeError(f"HOSTERR no data from the {nd.name} board for {now - nd.t_last_data:.1f} s while streaming "
                                       f"({sent} bytes sent, {len(clf.recs)} windows classified, classifier backlog {clf.q.qsize()} packets)")
        frames_due = min(n_frames, int((now - t0) * fs) + 1) if now >= t0 else 0
        n = min(frames_due * frame_bytes - sent, B.RING_WINDOW - (sent - tx.consumed), 4096)
        if n <= 0:
            time.sleep(0.001)
            if frames_due * frame_bytes > sent:
                stalled += 0.001                          # data was due but the transmitter's buffer had no room
            continue
        t_call = time.perf_counter()
        try:
            tx.ser.write(f"D {n}\n".encode() + stream[sent:sent + n])
        except Exception as e:
            t_err = time.perf_counter()
            raise RuntimeError(f"HOSTERR the write to the transmitter's USB port failed after {sent} bytes ({e}); seconds since the TX port last "
                               f"sent anything {t_err - tx.t_last_data:.1f}, RX port {t_err - rx.t_last_data:.1f}; classifier backlog "
                               f"{clf.q.qsize()} packets, {len(clf.recs)} windows classified")
        sent += n
        wr_t.append(t_call); wr_cum.append(sent)
    tx.send("FIN")
    t_fed_all = time.perf_counter()

    r_tx = tx.wait("TXSTAT", m_tx, 30 + n_frames / fs)
    r_rx = rx.wait("RXSTAT", m_rx, 30)
    time.sleep(0.2)
    clf.finish(); clf.join(30)
    rx.packet_hook = None
    if clf.error:
        raise RuntimeError("the classifier thread failed:\n" + clf.error)
    if not r_tx or not r_rx:
        raise RuntimeError("Did not get TXSTAT/RXSTAT")
    txs, rxs = B.kv(r_tx[1]), B.kv(r_rx[1])
    rssi_run = [int(B.kv(x)["rssi"]) for _, x in rx.lines_since(m_rx, "T ") if int(B.kv(x)["rssi"]) != 0]   # receiver's reading, once a second, during the stream

    # ---- latency of every decision
    wr_t, wr_cum = np.array(wr_t), np.array(wr_cum)
    P = prep
    win = P["win"]
    rows = []
    for rec in clf.recs:
        w = rec["w"]
        s = int(P["starts"][w])
        e = s + win
        t_src = t0 + (e - 1) / fs                                              # last sample of the window produced
        last_byte = e * frame_bytes - 1
        p = last_byte // pay
        pk_last = min((p + 1) * pay, L) - 1                                    # last byte of the packet that completes the window
        t_src_pk = t0 + (pk_last // frame_bytes) / fs
        t_fed = wr_t[min(int(np.searchsorted(wr_cum, pk_last, side="right")), len(wr_t) - 1)]
        wait = t_src_pk - t_src
        feed = t_fed - t_src_pk
        transit = rec["t_ready"] - t_fed
        queue_ = rec["t_work0"] - rec["t_ready"]
        row = dict(window=w, start_s=s / fs, true_class=int(P["y"][w]), pred_RF=rec["pred_RF"], pred_SVM=rec["pred_SVM"],
                   pred_offline_RF=int(P["pred_offline"]["RandomForest"][w]), pred_offline_SVM=int(P["pred_offline"]["SVM"][w]),
                   last_packet=int(p), in_final_partial_packet=bool(p == n_pk - 1 and L % pay != 0),
                   packet_fill_wait_ms=1000 * wait, feed_lag_ms=1000 * feed, link_transit_ms=1000 * transit, queue_wait_ms=1000 * queue_,
                   features_ms=1000 * rec["d_feat"], predict_RF_ms=1000 * rec["d_RF"], predict_SVM_ms=1000 * rec["d_SVM"])
        pre = row["packet_fill_wait_ms"] + row["feed_lag_ms"] + row["link_transit_ms"] + row["queue_wait_ms"] + row["features_ms"]
        row["total_RF_ms"] = pre + row["predict_RF_ms"]
        row["total_SVM_ms"] = pre + row["predict_SVM_ms"]
        rows.append(row)
    df = pd.DataFrame(rows)

    # ---- integrity, and the same windows classified in one batch from the finished received stream
    bit_exact = bytes(clf.got) == stream
    lost = int((~clf.have).sum())
    miss_bytes = np.zeros(L, dtype=bool)
    for sq in np.flatnonzero(~clf.have):
        miss_bytes[sq * pay:(sq + 1) * pay] = True
    x_recv, _ = E.decode(bytes(clf.got), miss_bytes, desc, "zero")
    Xb = E.window_features(x_recv, P["starts"], fs, desc["channel_names"], win)[P["reduced"]]
    Xb = E.clean(Xb)
    pred_batch = {}
    for name in MODELS:
        sc, est = P["models"][name]
        pred_batch[SHORT[name]] = est.predict(sc.transform(Xb) if sc else Xb)
    done = df["window"].to_numpy()
    Xlive = np.stack([r["x"] for r in sorted(clf.recs, key=lambda r: r["w"])]) if len(clf.recs) else np.zeros((0, Xb.shape[1]))
    feat_max_diff = float(np.abs(Xlive - Xb[done]).max()) if len(done) else float("nan")
    agree_batch = {k: float((df[f"pred_{k}"].to_numpy() == v[done]).mean()) for k, v in pred_batch.items()}
    agree_offline = {k: float((df[f"pred_{k}"].to_numpy() == df[f"pred_offline_{k}"].to_numpy()).mean()) for k in ("RF", "SVM")}
    y = df["true_class"].to_numpy()
    acc = {k: float((df[f"pred_{k}"].to_numpy() == y).mean()) for k in ("RF", "SVM")}
    ok_rows = df[~df["in_final_partial_packet"]]

    parts = ("packet_fill_wait_ms", "feed_lag_ms", "link_transit_ms", "queue_wait_ms", "features_ms", "predict_RF_ms", "predict_SVM_ms",
             "total_RF_ms", "total_SVM_ms")
    lat = {k: pct(ok_rows[k]) for k in parts}
    hop_ms = 1000 * P["hop"] / fs
    # The decoder "kept up" if it was almost never waiting behind an earlier window. If it was not, the latency measures the PC's backlog, not the link.
    kept_up = bool(lat["queue_wait_ms"].get("p95", 0) < 0.5 * hop_ms and lat["features_ms"].get("p99", 0) < hop_ms)
    res = dict(compute_kept_up=kept_up,
        source=desc["source"], fs=fs, channels=C, bits=bits, offered_kbps=Bps * 8 / 1000, payload_bytes=pay, tx_power_label_dbm=a.txpwr,
        packet_fill_ms=1000 * pay / Bps, window_ms=1000 * win / fs, hop_ms=1000 * P["hop"] / fs, link=link, rssi_idle_rx_dbm=idle,
        usb_rtt_ms={k: pct(v) for k, v in rtt.items()},
        windows_expected=int(len(P["starts"])), windows_classified=int(len(df)), windows_in_final_partial_packet=int(df["in_final_partial_packet"].sum()),
        stream_seconds=n_frames / fs, host_stalled_s=stalled,
        packets_sent=int(txs["pkts"]), packets_received=int(rxs["pkts"]), packets_lost=int(rxs["lost"]), tx_notify_retries=int(txs["retries"]),
        usb_forward_drops=int(rxs.get("fwd_drops", 0)),
        rssi_rx_during_run_dbm=(dict(mean=float(np.mean(rssi_run)), min=int(min(rssi_run)), max=int(max(rssi_run)), n=len(rssi_run)) if rssi_run else None),
        link_loss_db=(a.txpwr - float(np.mean(rssi_run))) if rssi_run else None,
        retry_rate_pct=100.0 * int(txs["retries"]) / max(int(txs["retries"]) + int(txs["pkts"]), 1),
        crc_host=f"{crc_host:08x}", crc_tx=txs["crc"], crc_rx=rxs["crc"], bit_exact=bool(bit_exact), host_packets_missing=lost,
        integrity_ok=bool(bit_exact and f"{crc_host:08x}" == rxs["crc"] and lost == 0),
        latency_ms=lat, standalone_timings=check,
        accuracy_live=acc, accuracy_meaningful=bool(P["accuracy_meaningful"]), test_classes=P["test_classes"],
        live_equals_batch=agree_batch, live_equals_offline=agree_offline, feature_max_abs_diff_live_vs_batch=feat_max_diff,
        n_train=P["n_train"], n_features=P["n_features"],
    )
    df.to_csv(run_dir / "windows.csv", index=False)
    json.dump(res, open(run_dir / "summary.json", "w"), indent=2)
    return res


def print_run(res):
    L = res["latency_ms"]
    f = lambda k, f_="median": L[k].get(f_, float("nan"))
    print(f"""
  ---- live result: {res['source']}, {res['fs']} Hz x {res['channels']} ch x {res['bits']} bit = {res['offered_kbps']:.0f} kb/s, payload {res['payload_bytes']} B ----
  windows         : {res['windows_classified']} of {res['windows_expected']} classified live (movement windows only), window {res['window_ms']:.0f} ms, hop {res['hop_ms']:.0f} ms
  link            : {res['packets_sent']} packets sent, {res['packets_received']} received, {res['packets_lost']} lost; {res['tx_notify_retries']} refused sends ({res['retry_rate_pct']:.1f} %); integrity {'OK' if res['integrity_ok'] else 'FAILED'}
  signal          : TX label {res['tx_power_label_dbm']} dBm; receiver reads {('mean %.1f dBm (min %d, max %d)' % (res['rssi_rx_during_run_dbm']['mean'], res['rssi_rx_during_run_dbm']['min'], res['rssi_rx_during_run_dbm']['max'])) if res['rssi_rx_during_run_dbm'] else 'n/a'}; link loss {('%.1f dB' % res['link_loss_db']) if res['link_loss_db'] is not None else 'n/a'}
  latency after the LAST sample of a window (ms)          median      p95      p99      max
    waiting for the packet to fill (payload / rate)     {f('packet_fill_wait_ms'):8.1f} {f('packet_fill_wait_ms', 'p95'):8.1f} {f('packet_fill_wait_ms', 'p99'):8.1f} {f('packet_fill_wait_ms', 'max'):8.1f}
    rig: feed lag (script hands byte to USB late)       {f('feed_lag_ms'):8.1f} {f('feed_lag_ms', 'p95'):8.1f} {f('feed_lag_ms', 'p99'):8.1f} {f('feed_lag_ms', 'max'):8.1f}
    link transit (USB + BLE + USB)                      {f('link_transit_ms'):8.1f} {f('link_transit_ms', 'p95'):8.1f} {f('link_transit_ms', 'p99'):8.1f} {f('link_transit_ms', 'max'):8.1f}
    queue wait (decoder busy)                           {f('queue_wait_ms'):8.1f} {f('queue_wait_ms', 'p95'):8.1f} {f('queue_wait_ms', 'p99'):8.1f} {f('queue_wait_ms', 'max'):8.1f}
    feature extraction                                  {f('features_ms'):8.1f} {f('features_ms', 'p95'):8.1f} {f('features_ms', 'p99'):8.1f} {f('features_ms', 'max'):8.1f}
    prediction RF / SVM                                 {f('predict_RF_ms'):8.1f} {f('predict_RF_ms', 'p95'):8.1f} {f('predict_RF_ms', 'p99'):8.1f} {f('predict_RF_ms', 'max'):8.1f}   (SVM median {f('predict_SVM_ms'):.1f})
    TOTAL with RF                                       {f('total_RF_ms'):8.1f} {f('total_RF_ms', 'p95'):8.1f} {f('total_RF_ms', 'p99'):8.1f} {f('total_RF_ms', 'max'):8.1f}
    TOTAL with SVM                                      {f('total_SVM_ms'):8.1f} {f('total_SVM_ms', 'p95'):8.1f} {f('total_SVM_ms', 'p99'):8.1f} {f('total_SVM_ms', 'max'):8.1f}
  + the window itself ({res['window_ms']:.0f} ms) -> controller delay (window + processing), median RF {res['window_ms'] + f('total_RF_ms'):.0f} ms, p99 {res['window_ms'] + f('total_RF_ms', 'p99'):.0f} ms
  live == batch   : predictions {res['live_equals_batch']}, largest feature difference {res['feature_max_abs_diff_live_vs_batch']:.2e}
  live == offline : {res['live_equals_offline']} (offline = features that never went through the link)
  accuracy (live) : RF {100 * res['accuracy_live']['RF']:.1f} %, SVM {100 * res['accuracy_live']['SVM']:.1f} %""" +
          ("" if res["accuracy_meaningful"] else "   [NOT meaningful: a test class has no training windows]"))
    if not res.get("compute_kept_up", True):
        print("  *** WARNING: the PC could NOT keep up (windows queued behind each other: queue wait p95 "
              f"{L['queue_wait_ms']['p95']:.0f} ms, feature extraction p99 {L['features_ms']['p99']:.0f} ms against a {res['hop_ms']:.0f} ms hop). "
              "The totals above measure the PC's backlog, NOT the link. Only the link figures (packets, signal, refused sends) are usable. ***")


# ====================================================================================== driver
def auto_payload(Bps, max_ms):
    return int(min(240, max(20, round(Bps * max_ms / 1000))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-port"); ap.add_argument("--rx-port")
    ap.add_argument("--source", default="vaskov:P1_1of12")
    ap.add_argument("--rates", default="128", help="kb/s from the ladder 10,32,128,384,960, or fs:channels:bits")
    ap.add_argument("--payloads", default="240", help="bytes per BLE packet (max 240), or 'auto' = fills in --max-packet-ms; comma list")
    ap.add_argument("--max-packet-ms", type=float, default=10.0, help="for 'auto' payloads")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--txpwr", type=int, default=0)
    ap.add_argument("--phy", type=int, choices=[1, 2], default=2)
    ap.add_argument("--ci", type=float, default=7.5, help="connection interval requested (ms)")
    ap.add_argument("--prepare-only", action="store_true", help="train and cache the models, do not touch the hardware")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--resume", help="a live_classification_* folder to continue")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    rates = [parse_rate(t.strip()) for t in a.rates.split(",") if t.strip()]
    pays = [t.strip() for t in a.payloads.split(",") if t.strip()]
    plan = []
    for tag, (fs, ch, bits) in rates:
        Bps = fs * ch * bits / 8
        for p in pays:
            pay = auto_payload(Bps, a.max_packet_ms) if p == "auto" else min(240, int(p))
            if not any(x[0] == tag and x[2] == pay for x in plan):
                plan.append((tag, (fs, ch, bits), pay))
    print(f"{len(plan)} live runs of {a.seconds:g} s: " + ", ".join(f"{t} pay {p} B" for t, _, p in plan), flush=True)
    sys.setswitchinterval(0.001)                            # let the reader threads and the sensor loop run promptly while features are computed
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
    try:                                                    # the timing is only as good as this process gets scheduled
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)   # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass

    if a.prepare_only:
        for tag, (fs, ch, bits), pay in plan:
            args = SimpleNamespace(source=a.source, mode="stream", seconds=a.seconds, start="auto", no_record=True, fs=fs, channels=ch,
                                   bits=bits, raw=False, realtime=True, burst_seconds=10, burst_rate=0, txpwr="0", cal=None, effective_dbm=False,
                                   phy=2, ci=7.5, payload=240, sensitivity=-94, repeat=1, note="", out=a.out, cal_obj=None)
            _, desc = B.build_stream(args)
            prepare(desc, a.retrain)
            B._DF_CACHE.clear()
        return
    if not (a.tx_port and a.rx_port):
        ap.error("--tx-port and --rx-port are required")

    out_dir = Path(a.resume) if a.resume else Path(a.out) / f"live_classification_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = json.load(open(man_path)) if man_path.exists() else {"runs": {}}
    man.setdefault("settings", dict(source=a.source, seconds=a.seconds, txpwr=a.txpwr, phy=a.phy, ci=a.ci, note=a.note))
    json.dump(man, open(man_path, "w"), indent=2)
    print("folder:", out_dir, flush=True)

    tx, rx = B.Node("TX", a.tx_port, out_dir / "tx_serial.log"), B.Node("RX", a.rx_port, out_dir / "rx_serial.log")
    try:
        for n in (tx, rx):
            info = B.boot_check(n)
            print(f"{n.name} on {n.port}: {info.get('chip')} fw {info.get('fw')} mac {info.get('mac')}", flush=True)
        tx.cmd("ROLE TX"); rx.cmd("ROLE RX")
        for i, (tag, (fs, ch, bits), pay) in enumerate(plan, 1):
            key = f"{a.source.replace(':', '-')}_{tag}_pay{pay}"
            if man["runs"].get(key, {}).get("status") == "done":
                print(f"\n=== [{i}/{len(plan)}] {key}: already done, skipping ===")
                continue
            run_dir = out_dir / key
            run_dir.mkdir(exist_ok=True)
            print(f"\n=== [{i}/{len(plan)}] {a.source}  {tag}  ({fs} Hz x {ch} ch x {bits} bit), payload {pay} B ===", flush=True)
            args = SimpleNamespace(source=a.source, mode="stream", seconds=a.seconds, start="auto", no_record=True, fs=fs, channels=ch,
                                   bits=bits, raw=False, realtime=True, burst_seconds=10, burst_rate=0, txpwr=str(a.txpwr), cal=None,
                                   effective_dbm=False, phy=a.phy, ci=a.ci, payload=pay, sensitivity=-94, repeat=1, note=a.note,
                                   out=str(out_dir), cal_obj=None)
            try:
                stream, desc = B.build_stream(args)
                prep = prepare(desc, a.retrain)
                res = run_live(tx, rx, stream, desc, prep, a, pay, run_dir)
            except RuntimeError as e:
                if "REBOOTED" in str(e) or "HOSTERR" in str(e):
                    raise
                print(f"  FAILED: {e}", flush=True)
                man["runs"][key] = dict(status="failed", reason=str(e), rate=tag, payload=pay)
                json.dump(man, open(man_path, "w"), indent=2)
                continue
            print_run(res)
            man["runs"][key] = dict(status="done", rate=tag, payload=pay, result=res)
            json.dump(man, open(man_path, "w"), indent=2)
            if i < len(plan) and plan[i][0] != tag:
                B._DF_CACHE.clear()
    finally:
        tx.close(); rx.close()
        B._DF_CACHE.clear()

    # ---- summary table
    cols = ["rate", "payload_B", "packet_fill_ms", "rssi_dbm", "refused_sends_pct", "pc_kept_up", "windows", "link_transit_med", "link_transit_p99", "compute_RF_med", "total_RF_med", "total_RF_p95",
            "total_RF_p99", "total_RF_max", "total_SVM_med", "controller_delay_RF_med", "controller_delay_RF_p99", "live_eq_batch", "acc_RF", "acc_SVM", "integrity"]
    rows = []
    for key, r in man["runs"].items():
        if r.get("status") != "done":
            rows.append(dict(rate=r.get("rate"), payload_B=r.get("payload"), integrity=f"FAILED: {r.get('reason', '')[:50]}"))
            continue
        x = r["result"]
        L = x["latency_ms"]
        comp = L["features_ms"]["median"] + L["predict_RF_ms"]["median"] + L["queue_wait_ms"]["median"]
        rows.append(dict(rate=r["rate"], payload_B=r["payload"], packet_fill_ms=round(x["packet_fill_ms"], 1),
                         rssi_dbm=(round(x["rssi_rx_during_run_dbm"]["mean"], 1) if x.get("rssi_rx_during_run_dbm") else ""),
                         refused_sends_pct=round(x["retry_rate_pct"], 1), pc_kept_up=("yes" if x.get("compute_kept_up", True) else "NO"), windows=f"{x['windows_classified']}/{x['windows_expected']}",
                         link_transit_med=round(L["link_transit_ms"]["median"], 1), link_transit_p99=round(L["link_transit_ms"]["p99"], 1),
                         compute_RF_med=round(comp, 1), total_RF_med=round(L["total_RF_ms"]["median"], 1), total_RF_p95=round(L["total_RF_ms"]["p95"], 1),
                         total_RF_p99=round(L["total_RF_ms"]["p99"], 1), total_RF_max=round(L["total_RF_ms"]["max"], 1),
                         total_SVM_med=round(L["total_SVM_ms"]["median"], 1),
                         controller_delay_RF_med=round(x["window_ms"] + L["total_RF_ms"]["median"], 0), controller_delay_RF_p99=round(x["window_ms"] + L["total_RF_ms"]["p99"], 0),
                         live_eq_batch=f"{100 * min(x['live_equals_batch'].values()):.0f}%",
                         acc_RF=f"{100 * x['accuracy_live']['RF']:.1f}%" if x["accuracy_meaningful"] else "n/a", acc_SVM=f"{100 * x['accuracy_live']['SVM']:.1f}%" if x["accuracy_meaningful"] else "n/a",
                         integrity="OK" if x["integrity_ok"] else "FAILED"))
    with open(out_dir / "live_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out_dir / "live_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    try:
        plot_summary(man, out_dir)
    except Exception as e:                                  # the chart is a convenience; never lose the numbers over it
        print("(chart not made:", e, ")")
    print("\nsummary saved in", out_dir)


def plot_summary(man, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in man["runs"].values() if r.get("status") == "done"]
    if not runs:
        return
    parts = [("packet_fill_wait_ms", "waiting for the packet to fill", "#E69F00"), ("link_transit_ms", "link transit (USB + BLE + USB)", "#0072B2"),
             ("feed_lag_ms", "rig: feed lag", "#999999"), ("queue_wait_ms", "queue wait", "#CC79A7"),
             ("features_ms", "feature extraction", "#009E73"), ("predict_RF_ms", "Random Forest", "#D55E00")]
    labels = [f"{r['rate']}\n{r['payload']} B" for r in runs]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(runs) + 2), 4.6))
    bottom = np.zeros(len(runs))
    for key, name, col in parts:
        v = np.array([r["result"]["latency_ms"][key]["median"] for r in runs])
        ax.bar(labels, v, bottom=bottom, label=name, color=col, width=0.6)
        bottom += v
    p99 = [r["result"]["latency_ms"]["total_RF_ms"]["p99"] for r in runs]
    ax.plot(labels, p99, "k_", ms=18, mew=2, label="99th percentile of the total")
    ax.set_ylabel("ms after the last sample of the window")
    ax.set_title("Time to a decision after the last sample: median of each part, stacked\n"
                 "(sum of medians, so slightly different from the median of the total; the 200 ms window is not included)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "latency_breakdown.png", dpi=140)


if __name__ == "__main__":
    main()
