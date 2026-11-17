#!/usr/bin/env python3
"""Recompute the X2 metrics (pipeline/docs/three_experiments_declared_protocol.md, amendment 1) for link_timing runs from their packets.csv and
summary.json, so that every run is analysed by the same code, and write one table.

Per run: packets lost; refused sends; link transit (median, p99, max); excess link delay above the run's fastest 5 % in connection intervals
(the time that retransmissions and queueing in the transmitter add; the ESP32 does not report retransmission counts); total delay; backlog
(data produced by the sensor but not yet at the PC) over time, its maximum and final value, its slope, and when it would first exceed a
buffer of 4, 8, 16, 24 or 64 kB (an implant with that buffer would start to LOSE data then); stalls; throughput.

Usage: python link_timing_analyse.py results/link_timing_0dBm_* results/link_timing_-24dBm_* [--out results/link_timing_summary]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STALL_INTERVALS = 3
BUFFERS_KB = (4, 8, 16, 24, 64)


def run_metrics(run_dir):
    s = json.load(open(run_dir / "summary.json"))
    p = pd.read_csv(run_dir / "packets.csv")
    fs, C, bits, pay = s["fs"], s["channels"], s["bits"], s["payload_bytes"]
    frame_bytes = C * bits // 8
    n_frames = int(round(s["stream_seconds"] * fs))
    L = n_frames * frame_bytes
    ci = float(s["link"]["rx"].get("itvl_ms") or 7.5)
    ok = p.dropna(subset=["t_rx_s"])
    arr = np.sort(ok["t_rx_s"].to_numpy())
    spacing = 1000 * pay / (fs * frame_bytes)
    gaps = np.diff(arr) * 1000
    stalls = gaps[gaps > spacing + STALL_INTERVALS * ci] - spacing
    tr = ok["link_transit_ms"].to_numpy()
    base = np.percentile(tr, 5)
    n_ci = np.floor(np.maximum(tr - base, 0) / ci)
    produced = np.minimum(np.floor(arr * fs) + 1, n_frames) * frame_bytes
    received = np.minimum(np.arange(1, len(arr) + 1) * pay, L)
    backlog = np.maximum(produced - received, 0)
    slope = float(np.polyfit(ok["t_src_s"], ok["total_delay_ms"], 1)[0])
    first = {kB: (float(arr[np.argmax(backlog > kB * 1024)]) if (backlog > kB * 1024).any() else None) for kB in BUFFERS_KB}
    span = arr[-1] - arr[0]
    rssi = s.get("rssi_rx_during_run_dbm") or {}
    return dict(
        label_dbm=s["tx_power_label_dbm"], offered_kbps=s["offered_kbps"], rssi_dbm=rssi.get("mean"), link_loss_db=s.get("link_loss_db"),
        sent=s["packets_sent"], lost=s["packets_lost"], refused_pct=s["retry_rate_pct"], integrity=s["integrity_ok"],
        throughput_kbps=8 * L / 1000 / span if span > 0 else float("nan"),
        transit_median=float(np.median(tr)), transit_p99=float(np.percentile(tr, 99)), transit_max=float(tr.max()), transit_p5=float(base),
        excess_ge1=float((n_ci >= 1).mean()), excess_ge2=float((n_ci >= 2).mean()), excess_ge4=float((n_ci >= 4).mean()), excess_max_ci=int(n_ci.max()),
        delay_median=float(ok["total_delay_ms"].median()), delay_p99=float(np.percentile(ok["total_delay_ms"], 99)), delay_max=float(ok["total_delay_ms"].max()),
        delay_slope_ms_per_s=slope, backlog_max_kB=float(backlog.max() / 1024), backlog_final_kB=float(backlog[-1] / 1024),
        backlog_median_kB=float(np.median(backlog) / 1024), sensor_stalled_s=s["host_stalled_s"],
        stalls=int(len(stalls)), stall_total_ms=float(stalls.sum()), stall_longest_ms=float(stalls.max()) if len(stalls) else 0.0,
        **{f"exceeds_{kB}kB_at_s": first[kB] for kB in BUFFERS_KB},
        run=str(run_dir))


def main():
    args = sys.argv[1:]
    out = Path("results") / "link_timing_summary"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    folders = [Path(a) for a in args if not a.startswith("--")]
    rows = []
    for f in folders:
        for run in sorted(d for d in f.iterdir() if (d / "packets.csv").exists() and (d / "summary.json").exists()):
            rows.append(run_metrics(run))
    df = pd.DataFrame(rows).sort_values(["label_dbm", "offered_kbps"], ascending=[False, True])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "link_timing_summary.csv", index=False)
    cols = [("label_dbm", "Label (dBm)", "{:.0f}"), ("offered_kbps", "Offered (kb/s)", "{:.0f}"), ("rssi_dbm", "RSSI (dBm)", "{:.1f}"),
            ("throughput_kbps", "Delivered (kb/s)", "{:.0f}"), ("lost", "Lost", "{:.0f}"), ("refused_pct", "Refused sends (%)", "{:.1f}"),
            ("transit_median", "Transit median (ms)", "{:.1f}"), ("transit_p99", "Transit p99 (ms)", "{:.1f}"), ("transit_max", "Transit max (ms)", "{:.1f}"),
            ("excess_ge1", ">=1 extra interval (%)", "{:.1%}"), ("excess_max_ci", "Most extra intervals", "{:.0f}"),
            ("delay_median", "Delay median (ms)", "{:.1f}"), ("delay_max", "Delay max (ms)", "{:.0f}"), ("delay_slope_ms_per_s", "Delay slope (ms/s)", "{:.2f}"),
            ("backlog_max_kB", "Backlog max (kB)", "{:.1f}"), ("exceeds_24kB_at_s", "Backlog > 24 kB at (s)", "{:.1f}"),
            ("stalls", "Stalls", "{:.0f}"), ("stall_longest_ms", "Longest stall (ms)", "{:.0f}")]
    md = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for k, _, f in cols:
            v = r[k]
            cells.append("never" if (k.startswith("exceeds") and (v is None or (isinstance(v, float) and np.isnan(v)))) else
                         (f.format(v) if not (isinstance(v, float) and np.isnan(v)) else "n/a"))
        md.append("| " + " | ".join(cells) + " |")
    (out / "link_timing_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
