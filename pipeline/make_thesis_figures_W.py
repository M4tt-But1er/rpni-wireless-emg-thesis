"""Wide-set-only versions of two thesis figures (user, 11 Nov 2026: W for everything except the feature-set comparisons). The original figures are kept.
  fig_main_effects_W - mean accuracy against channel count, sample rate and bit depth, set W only (cache/grid/grid_results.json)
  fig_temporal_W     - (a) accuracy against the span of a causal vote on the 14 W winner cells (cache/grid/fair/fair_results.json) with the LSTM levels
                       (cache/grid/fair2/fair2_cells.csv, set W); (b) accuracy against time since the onset, W cells (cache/grid/fair2/fair_W_summary.md)
Usage: python -W ignore make_thesis_figures_W.py"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyse_grid as AG
import make_thesis_figures as M
from config import CACHE_DIR


def fig_main_effects_W(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    specs = [("channels", "Channels transmitted", None), ("rate_hz", "Sample rate (Hz)", "log"), ("bits", "Bit depth (bits per sample)", None)]
    for ax, (col, xlabel, scale) in zip(axes, specs):
        for p in ["P1", "P2", "S1", "S2"]:
            g = df[(df.patient == p) & (df.feature_set == "W")].groupby(col).accuracy.mean() * 100
            ax.plot(g.index, g.values, color=M.PATIENT_COLOUR[p], marker=M.PATIENT_MARKER[p], label=p, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xlabel(xlabel)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xticks([250, 500, 1000, 3000, 10000])
            ax.set_xticklabels(["250", "500", "1k", "3k", "10k"])
            ax.minorticks_off()
        elif col == "channels":
            ax.set_xticks([2, 5, 8, 12])
        else:
            ax.set_xticks([8, 12, 16])
    axes[0].set_ylabel("Mean accuracy (%)")
    axes[0].set_ylim(45, 100)
    for ax, t in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(t, loc="left", color=M.INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    M.save(fig, "fig_main_effects_W")


def fig_temporal_W():
    fr = json.load(open(CACHE_DIR / "grid" / "fair" / "fair_results.json"))["results"]
    ks = ["k1", "k3", "k5", "k9", "k15", "all"]
    span_ms = ["200", "300", "400", "600", "900", "all\n(≤2 s)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for m in ["LDA", "SVM", "RandomForest"]:
        y = [100 * np.mean([v["accuracy"] for k, v in fr.items() if "|W|" in k and k.endswith(f"|{m}/{kk}")]) for kk in ks]
        a1.plot(range(len(ks)), y, color=M.MODEL_COLOUR[m], marker=M.MODEL_MARKER[m], label=f"{M.MODEL_LABEL[m]} + vote", markeredgecolor="white", markeredgewidth=0.5)
        a1.annotate(f"{y[-1]:.1f}", (len(ks) - 1, y[-1]), xytext=(4, {"RandomForest": 8, "SVM": -1, "LDA": -7}[m]), textcoords="offset points", va="center", fontsize=8.5, color=M.INK2)
    cells = pd.read_csv(CACHE_DIR / "grid" / "fair2" / "fair2_cells.csv")
    cells = cells[cells["set"] == "W"]
    lstm = {name: 100 * cells[cells.method == name].accuracy.mean() for name in ["LSTM_tuned", "LSTM_best", "LSTM_best_ens5"]}
    for name, ls, lab in [("LSTM_best_ens5", "-", "LSTM, search, 5 seeds"), ("LSTM_tuned", "--", "LSTM, 3-setting tuning"), ("LSTM_best", ":", "LSTM, search, 1 seed")]:
        a1.axhline(lstm[name], color=M.INK2, ls=ls, lw=1.0, label=f"{lab} ({lstm[name]:.1f})")
    a1.set_xticks(range(len(ks)))
    a1.set_xticklabels(span_ms)
    a1.set_xlabel("Span of the causal vote (ms of signal)")
    a1.set_ylabel("Accuracy, 14 W cells (%)")
    a1.set_ylim(80, 96)
    a1.set_title("(a)", loc="left", color=M.INK2)
    a1.legend(loc="lower right", ncol=1, frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.2)

    rows = M.parse_md_table(open(CACHE_DIR / "grid" / "fair2" / "fair_W_summary.md", encoding="utf-8").read(), "## Time-resolved accuracy")
    head, data = rows[0], {r[0]: r[1:] for r in rows[1:]}
    xlab = ["1–2\n(≤250)", "3–5\n(≤400)", "6–10\n(≤650)", "11–20\n(≤1150)", "21+\n(>1150)"]
    series = [("RandomForest/P0", M.AQUA, "--", "^", "RF, no vote"), ("RandomForest/P3", M.AQUA, "-", "^", "RF + vote"), ("SVM/P3", M.ORANGE, "-", "s", "SVM + vote"),
              ("LDA/P3", M.BLUE, "-", "o", "LDA + vote"), ("LSTM_best_ens5", M.INK2, "-", "D", "LSTM, 5 seeds"), ("LSTM_best", M.INK2, ":", "D", "LSTM, 1 seed")]
    x = np.arange(len(head) - 1)
    for key, col, ls, mk, lab in series:
        a2.plot(x, [float(v) for v in data[key]], color=col, ls=ls, marker=mk, label=lab, markeredgecolor="white", markeredgewidth=0.5, lw=1.4)
    a2.set_xticks(x)
    a2.set_xticklabels(xlab)
    a2.set_xlabel("Window index since onset (decision time, ms)")
    a2.set_ylabel("Accuracy (%)")
    a2.set_ylim(78, 97.5)
    a2.set_title("(b)", loc="left", color=M.INK2)
    a2.legend(loc="lower right", frameon=False, fontsize=8, ncol=2, handlelength=1.8, columnspacing=0.8)
    fig.tight_layout()
    M.save(fig, "fig_temporal_W")


if __name__ == "__main__":
    fig_main_effects_W(AG.table())
    fig_temporal_W()
