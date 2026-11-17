"""V8 of the grid protocol: integrity of every grid cell (cache/grid/grid_results.json).
(a) the pooled confusion matrix reproduces the accuracy (trace / total), and its total equals the number of windows;
(b) all cells of one (patient, rate, bits) table share the same number of windows (channel subsets and feature sets differ only in columns);
(c) number of columns equals the expected structure: set W = 30 per-channel columns x channels + one xcorr per channel pair (amplitude 8 + shape 8 + sub-band 8 + window change 6 per channel);
    set A = 24 / 22 / 20 per channel at 1 kHz and above / 500 Hz / 250 Hz;
(d) bit rate = rate x channels x bits <= 1.4 Mb/s, 193 configurations x 2 sets x 3 models = 1,158 cells present, no NaN in any metric;
(e) extraction cost rises with the channel count (same rate, set).   Usage: python -W ignore check_grid_integrity.py"""
import collections
import json
import sys

import numpy as np

from config import CACHE_DIR

R = json.load(open(CACHE_DIR / "grid" / "grid_results.json"))
bad = collections.Counter()
windows = collections.defaultdict(set)
ext = {}
for key, r in R.items():
    part, rate, nch, bits, fset, model = key.split("|")
    rate, nch, bits = int(rate[:-2]), int(nch[:-2]), int(bits[:-3])
    cm = np.array(r["confusion_matrix"])
    if abs(np.trace(cm) / cm.sum() - r["accuracy"]) > 1e-12:
        bad["confusion matrix does not reproduce accuracy"] += 1
    if cm.sum() != r["n_windows"]:
        bad["confusion matrix total != windows"] += 1
    windows[(part, rate, bits)].add(r["n_windows"])
    per_ch = 30 if fset == "W" else (24 if rate >= 1000 else 22 if rate == 500 else 20)
    expect = per_ch * nch + (nch * (nch - 1) // 2 if fset == "W" else 0)
    if r["n_features"] != expect:
        bad[f"columns {r['n_features']} != expected {expect}"] += 1
    if r["bitrate_bps"] != rate * nch * bits or r["bitrate_bps"] > 1.4e6:
        bad["bit rate"] += 1
    if not all(np.isfinite([r["accuracy"], r["weighted_f1"], r["cohen_kappa"]])):
        bad["non-finite metric"] += 1
    ext.setdefault((part, rate, bits, fset, model), {})[nch] = r["extraction"]["flops"]
for k, v in windows.items():
    if len(v) != 1:
        bad[f"windows differ within table {k}: {sorted(v)}"] += 1
for k, d in ext.items():
    chans = sorted(d)
    if any(d[a] >= d[b] for a, b in zip(chans, chans[1:])):
        bad["extraction cost not increasing with channels"] += 1
print(f"{len(R)} cells (expected 1158); tables checked: {len(windows)}; problems: {dict(bad) if bad else 'none'}")
ok = not bad and len(R) == 1158 and len(windows) == 60
print("V8 PASSED" if ok else "V8 FAILED")
sys.exit(0 if ok else 1)
