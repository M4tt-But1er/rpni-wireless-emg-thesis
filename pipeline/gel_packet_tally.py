"""Count every packet sent through the gel phantom (link-level only).

Reads the run records under esp32_ble_link/results/ that were made with the
transmitter inside the gel phantom (4-5 Nov 2026) and writes
cache/link/gel_packet_tally.json and .csv. Nothing is re-run and no result
file is changed.

Which folders count as "gel" (from esp32_ble_link/README.md sections 13-14):
  20261104_134402_burst_*        full-speed bursts at six power labels
  20261104_134557_stream_*       pilot stream, 128 kb/s
  phantom_campaign_20261104_141740   4 datasets x 5 rates, 0 dBm label
  phantom_campaign_20261104_160046   lowest power label (-24 dBm)
  live_classification_20261105_085900  live pilot
  live_classification_20261105_090141  live matrix (8 runs)
  live_classification_20261105_094315  live runs at -24 dBm (link figures valid,
                                       latency invalid; latency is not used here)
Left out: live_classification_20261105_083713 (failed attempt: the PC stopped
reading the USB ports; not a radio result) and every run made in air.

Only link quantities are collected (packets, loss, integrity, received signal).
No classification result is read.
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "esp32_ble_link", "results")
OUT = os.path.join(HERE, "cache", "link")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rx_rssi(res):
    """Mean received signal (dBm) as seen by the receiving board."""
    if "rssi_rx_mean_dbm" in res and res["rssi_rx_mean_dbm"] is not None:
        return float(res["rssi_rx_mean_dbm"])
    idle = res.get("rssi_idle_rx_dbm") or []
    vals = [float(v) for v in idle if v is not None]
    link_rx = res.get("link", {}).get("rx", {}).get("rssi")
    if link_rx not in (None, ""):
        vals.append(float(link_rx))
    return sum(vals) / len(vals) if vals else float("nan")


def row(group, name, res, label):
    sent = int(res["packets_sent"])
    recv = int(res.get("packets_received", sent - int(res["packets_lost"])))
    lost = int(res["packets_lost"])
    integ = res.get("integrity_ok")
    if integ is None:
        integ = res.get("stream_crc_tx") == res.get("stream_crc_rx")
    return {
        "group": group, "run": name, "tx_label_dbm": label,
        "mode": res.get("mode", "live"),
        "payload_bytes": res.get("payload_bytes"),
        "packets_sent": sent, "packets_received": recv, "packets_lost": lost,
        "integrity_ok": bool(integ),
        "rssi_rx_mean_dbm": round(rx_rssi(res), 2),
    }


def unpolarised_rows():
    """Runs whose note marks the UNPOLARISED (worst-case) orientation, added 9 Nov 2026: bursts and campaigns."""
    rows = []
    for d in sorted(Path(RES).glob("2026*_burst_*")):
        try:
            s = load(os.path.join(d, "summary.json"))
        except (OSError, ValueError):
            continue
        for r in (s if isinstance(s, list) else [s]):
            if "UNPOLARISED" in str(r.get("placement_note", "")):
                rows.append(row("burst_gel_unpolarised", "burst_%+d" % r["tx_power_setting_dbm"], r, r["tx_power_setting_dbm"]))
    for camp in sorted(Path(RES).glob("phantom_campaign_*")):
        try:
            m = load(os.path.join(camp, "manifest.json"))
        except (OSError, ValueError):
            continue
        for name, run in m["runs"].items():
            res = run.get("result")
            if res and "UNPOLARISED" in str(res.get("placement_note", "")):
                rows.append(row(camp.name, name, res, res["tx_power_setting_dbm"]))
    return rows


def summarise_rows(rows):
    n_sent = sum(r["packets_sent"] for r in rows)
    n_lost = sum(r["packets_lost"] for r in rows)
    rss = [r["rssi_rx_mean_dbm"] for r in rows if not math.isnan(r["rssi_rx_mean_dbm"])]
    by_label = {}
    for r in rows:
        b = by_label.setdefault(str(r["tx_label_dbm"]), {"runs": 0, "packets_sent": 0, "packets_lost": 0, "rssi": []})
        b["runs"] += 1
        b["packets_sent"] += r["packets_sent"]
        b["packets_lost"] += r["packets_lost"]
        b["rssi"].append(r["rssi_rx_mean_dbm"])
    for b in by_label.values():
        b["rssi_min"], b["rssi_max"] = min(b["rssi"]), max(b["rssi"])
        del b["rssi"]
    return {"runs": len(rows), "packets_sent": n_sent, "packets_lost": n_lost, "all_integrity_ok": all(r["integrity_ok"] for r in rows),
            "rssi_rx_mean_range_dbm": [min(rss), max(rss)],
            "loss_95pct_upper_bound": (1 - 0.05 ** (1 / n_sent)) if n_lost == 0 else None,
            "per_measured": n_lost / n_sent, "by_tx_label_dbm": by_label}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--unpolarised" in sys.argv:
        rows = unpolarised_rows()
        out = {"summary": summarise_rows(rows), "runs": rows,
               "note": "UNPOLARISED orientation (RX antenna turned 90 deg). Packet loss after the BLE link layer; the bound is indicative (packets not independent)."}
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "gel_packet_tally_unpolarised.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(OUT, "gel_packet_tally_unpolarised.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(out["summary"], indent=1))
        return
    rows = []
    # bursts: summary.json is a list, one entry per power label
    for r in load(os.path.join(RES, "20261104_134402_burst_vaskov-P1_1of12", "summary.json")):
        rows.append(row("burst_gel", "burst_%+d" % r["tx_power_setting_dbm"], r, r["tx_power_setting_dbm"]))
    # pilot stream
    s = load(os.path.join(RES, "20261104_134557_stream_vaskov-P1_1of12", "summary.json"))
    for r in (s if isinstance(s, list) else [s]):
        rows.append(row("pilot_stream_gel", "pilot_128kbps", r, r["tx_power_setting_dbm"]))
    # campaigns
    for folder in ("phantom_campaign_20261104_141740", "phantom_campaign_20261104_160046"):
        m = load(os.path.join(RES, folder, "manifest.json"))
        for name, run in m["runs"].items():
            res = run.get("result")
            if not res:
                continue
            rows.append(row(folder, name, res, res["tx_power_setting_dbm"]))
    # live classification (link figures only)
    for folder in ("live_classification_20261105_085900", "live_classification_20261105_090141",
                   "live_classification_20261105_094315"):
        m = load(os.path.join(RES, folder, "manifest.json"))
        for name, run in m["runs"].items():
            res = run.get("result")
            if not res or run.get("status") not in ("done", "sent", None):
                continue
            rows.append(row(folder, name, res, res.get("tx_power_label_dbm", m["settings"].get("txpwr"))))

    n_sent = sum(r["packets_sent"] for r in rows)
    n_lost = sum(r["packets_lost"] for r in rows)
    n_runs = len(rows)
    all_ok = all(r["integrity_ok"] for r in rows)
    rss = [r["rssi_rx_mean_dbm"] for r in rows if not math.isnan(r["rssi_rx_mean_dbm"])]
    by_label = {}
    for r in rows:
        b = by_label.setdefault(str(r["tx_label_dbm"]), {"runs": 0, "packets_sent": 0, "packets_lost": 0, "rssi": []})
        b["runs"] += 1
        b["packets_sent"] += r["packets_sent"]
        b["packets_lost"] += r["packets_lost"]
        b["rssi"].append(r["rssi_rx_mean_dbm"])
    for b in by_label.values():
        b["rssi_min"] = min(b["rssi"])
        b["rssi_max"] = max(b["rssi"])
        del b["rssi"]
    summary = {
        "runs": n_runs, "packets_sent": n_sent, "packets_lost": n_lost,
        "all_integrity_ok": all_ok,
        "rssi_rx_mean_range_dbm": [min(rss), max(rss)],
        # rule of three: with 0 events in N independent trials, the 95 % upper
        # bound on the event probability is about 3/N (exact: 1 - 0.05**(1/N))
        "loss_95pct_upper_bound": (1 - 0.05 ** (1 / n_sent)) if n_lost == 0 else None,
        "by_tx_label_dbm": by_label,
        "note": "Packet loss after the BLE link layer (which acknowledges and retransmits). "
                "Raw packet errors before retransmission are not visible to the application. "
                "Packets are not independent trials (same link, same minute), so the bound is indicative.",
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gel_packet_tally.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=1)
    with open(os.path.join(OUT, "gel_packet_tally.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
