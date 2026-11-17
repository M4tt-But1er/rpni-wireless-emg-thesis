"""Markdown tables for docs/grid_results.md, generated from cache/grid/grid_cells.csv and grid_results.json (no hand-typed numbers).
Usage: python -W ignore make_grid_tables.py > cache/grid/grid_tables.md"""
import json

import numpy as np
import pandas as pd

from analyse_grid import cheapest_reaching, fmt, pick
from config import CACHE_DIR

df = pd.read_csv(CACHE_DIR / "grid" / "grid_cells.csv")
PARTS = ["P1_combined", "P2_combined", "Wallace_S1", "Wallace_S2"]
SH = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
MODELS = ["LDA", "SVM", "RandomForest"]
REF = (1000, 8, 16)


def sub(p, s, m):
    return df[(df.part == p) & (df.feature_set == s) & (df.model == m)]


def short(cfg):
    return f"{cfg[0]} Hz, {cfg[1]} ch, {cfg[2]} b"


print("## T1. Geometric accuracy elbow of the Pareto frontier against bit rate (configuration, bit rate, accuracy %)\n")
print("| Patient | Set | LDA | SVM | RF |\n|---|---|---|---|---|")
for p in PARTS:
    for s in ("A", "W"):
        row = []
        for m in MODELS:
            cfg, c, a = pick(sub(p, s, m), "accuracy", "bitrate_bps")
            row.append(f"{short(cfg)}: {c / 1000:.0f} kb/s, {100 * a:.1f}")
        print(f"| {SH[p]} | {s} | " + " | ".join(row) + " |")

print("\n## T2. The same elbow as coded in the earlier grid (elbow truncated to a whole number of decades; see docs/elbow_truncation_note.md)\n")
print("| Patient | Set | LDA | SVM | RF |\n|---|---|---|---|---|")
for p in PARTS:
    for s in ("A", "W"):
        row = []
        for m in MODELS:
            cfg, c, a = pick(sub(p, s, m), "accuracy", "bitrate_bps", geometric=False)
            row.append(f"{short(cfg)}: {c / 1000:.0f} kb/s, {100 * a:.1f}")
        print(f"| {SH[p]} | {s} | " + " | ".join(row) + " |")

for target in (0.90, 0.95):
    print(f"\n## T3. Cheapest configuration with accuracy >= {int(target * 100)}% (bit rate, configuration; 'not reached' if none)\n")
    print("| Patient | Set | LDA | SVM | RF |\n|---|---|---|---|---|")
    for p in PARTS:
        for s in ("A", "W"):
            row = []
            for m in MODELS:
                r = cheapest_reaching(sub(p, s, m), target)
                row.append("not reached" if r is None else f"{r[1] / 1000:.0f} kb/s ({short(r[0])}, {100 * r[2]:.1f}%)")
            print(f"| {SH[p]} | {s} | " + " | ".join(row) + " |")

print("\n## T4. Reference configuration (1000 Hz, 8 ch, 16 b) against the geometric accuracy elbow: accuracy / weighted F1 / kappa, bit rate, FLOPs and model operations per decision\n")
print("| Patient | Set | Model | Reference: acc / F1 / kappa | kb/s | kFLOP | model kOps | Elbow config | Elbow: acc / F1 / kappa | kb/s | kFLOP | model kOps |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for p in PARTS:
    for s in ("A", "W"):
        for m in MODELS:
            d = sub(p, s, m)
            r = d[(d.rate_hz == REF[0]) & (d.channels == REF[1]) & (d.bits == REF[2])].iloc[0]
            cfg, _, _ = pick(d, "accuracy", "bitrate_bps")
            e = d[(d.rate_hz == cfg[0]) & (d.channels == cfg[1]) & (d.bits == cfg[2])].iloc[0]
            print(f"| {SH[p]} | {s} | {m} | {100 * r.accuracy:.1f} / {100 * r.weighted_f1:.1f} / {r.kappa:.3f} | {r.bitrate_bps / 1000:.0f} | {r.total_flops / 1000:.0f} | {r.model_ops / 1000:.1f} | {short(cfg)} | "
                  f"{100 * e.accuracy:.1f} / {100 * e.weighted_f1:.1f} / {e.kappa:.3f} | {e.bitrate_bps / 1000:.0f} | {e.total_flops / 1000:.0f} | {e.model_ops / 1000:.1f} |")

print("\n## T5. Main effects: mean accuracy (%) over the other settings and over LDA, SVM, RF, per patient and feature set\n")
for axis, name in (("channels", "Channels"), ("rate_hz", "Sample rate (Hz)"), ("bits", "Bit depth")):
    g = df.groupby(["patient", "feature_set", axis]).accuracy.mean().unstack(axis) * 100
    print(f"\n**{name}**\n")
    print("| Patient | Set | " + " | ".join(str(c) for c in g.columns) + " |\n|---|---|" + "---|" * len(g.columns))
    for (pat, s), row in g.iterrows():
        print(f"| {pat} | {s} | " + " | ".join("" if np.isnan(v) else f"{v:.1f}" for v in row) + " |")

print("\n## T6. Extraction cost per decision (kFLOP, analytic) of the two feature sets at selected configurations (8 channels; S2 also 12)\n")
print("| Configuration | Set A | Set W |\n|---|---|---|")
for rate, ch, b in [(250, 8, 8), (500, 8, 8), (1000, 8, 16), (3000, 8, 16), (10000, 8, 16)]:
    vals = []
    for s in ("A", "W"):
        d = df[(df.part == "P1_combined") & (df.feature_set == s) & (df.model == "LDA") & (df.rate_hz == rate) & (df.channels == ch) & (df.bits == b)].iloc[0]
        vals.append(f"{d.extraction_flops / 1000:.0f} (F = {d.n_features})")
    print(f"| {rate} Hz, {ch} ch, {b} b (P1) | " + " | ".join(vals) + " |")

print("\n## T7. Model cost per decision at the reference configuration (kOps = MACs + comparisons): mean over the 4 patients\n")
print("| Set | LDA | SVM | RF |\n|---|---|---|---|")
for s in ("A", "W"):
    vals = []
    for m in MODELS:
        d = df[(df.feature_set == s) & (df.model == m) & (df.rate_hz == REF[0]) & (df.channels == REF[1]) & (df.bits == REF[2])]
        vals.append(f"{d.model_ops.mean() / 1000:.1f} (SVM support vectors drive this)" if m == "SVM" else f"{d.model_ops.mean() / 1000:.1f}")
    print(f"| {s} | " + " | ".join(vals) + " |")
