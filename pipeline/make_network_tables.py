"""Markdown tables for the networks section of docs/grid_results.md, from cache/grid/{grid,networks}_results.json (no hand-typed numbers). Usage: python -W ignore make_network_tables.py > cache/grid/network_tables.md"""
import json

import numpy as np

from analyse_networks import MODELS, SHORT, WIN, cellrec, costs, fmt

REF = (1000, 8, 16)
PARTS = ["P1_combined", "P2_combined", "Wallace_S1", "Wallace_S2"]

print("### N1. Reference configuration (1000 Hz, 8 ch, 16 b = 128 kb/s): all six models, accuracy / weighted F1 / kappa, model MACs and total FLOPs per decision (extraction + 2 x MACs + comparisons)\n")
print("| Patient | Set | " + " | ".join(MODELS) + " |\n|---|---|" + "---|" * len(MODELS))
for p in PARTS:
    for s in ("A", "W"):
        cells = []
        for m in MODELS:
            r = cellrec(p, *REF[:1], REF[1], REF[2], s, m)
            macs, ops, ex, tot = costs(r, m)
            cells.append(f"{100 * r['accuracy']:.1f} / {100 * r['weighted_f1']:.1f} / {r['cohen_kappa']:.3f}<br>{macs / 1000:.1f} kMAC, {tot / 1000:.0f} kFLOP")
        print(f"| {SHORT[p]} | {s} | " + " | ".join(cells) + " |")

print("\n### N2. Mean accuracy over the winner cells of each patient and feature set (%), and the number of winner cells\n")
print("| Patient | Set | cells | " + " | ".join(MODELS) + " |\n|---|---|---|" + "---|" * len(MODELS))
for p in PARTS:
    for s in ("A", "W"):
        cfgs = WIN[f"{p}|{s}"]
        vals = [np.mean([100 * cellrec(p, r, n, b, s, m)["accuracy"] for r, n, b in cfgs]) for m in MODELS]
        print(f"| {SHORT[p]} | {s} | {len(cfgs)} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

print("\n### N3. The winner configurations per patient and feature set\n")
print("| Patient | Set | Winner configurations (rate, channels, bits) |\n|---|---|---|")
for p in PARTS:
    for s in ("A", "W"):
        print(f"| {SHORT[p]} | {s} | " + "; ".join(f"{fmt(c)} ({c[0] * c[1] * c[2] / 1000:.0f} kb/s)" for c in WIN[f"{p}|{s}"]) + " |")

print("\n### N4. LSTM tuning: configuration and epoch count chosen per fold (counts over all winner cells x folds)\n")
import collections
cfg, ep = collections.Counter(), []
N = json.load(open("cache/grid/networks_results.json"))
for k, r in N.items():
    if k.endswith("LSTM_tuned"):
        for i in r["fold_info"]:
            cfg[i["config"]] += 1
            ep.append(i["epochs"])
print("Configurations: " + ", ".join(f"{k}: {v}" for k, v in sorted(cfg.items())) + f". Epochs chosen: median {np.median(ep):.0f}, range {min(ep)} to {max(ep)} (the headline LSTM uses 40).")
