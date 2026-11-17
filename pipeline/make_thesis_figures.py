"""Figures for the thesis chapters on the theoretical (offline) simulations: docs/thesis_sim/simulation_chapters.tex.
Every number is read from stored results (no model is run):
  fig_main_effects  - mean accuracy against channel count, sample rate and bit depth (bitrate grid, cache/grid/grid_results.json)
  fig_frontier      - Pareto frontier of accuracy against transmitted bit rate per patient, set W, with the geometric elbow (same file)
  fig_temporal      - (a) accuracy against the span of a causal vote (cache/grid/fair/fair_results.json) with the LSTM levels
                      (cache/grid/fair2/fair2_cells.csv); (b) accuracy against time since the onset (table F13 of cache/grid/fair2/fair2_tables.md)
Output: docs/thesis_sim/figures/<name>.pdf (vector, for LaTeX) and .png (preview).
Usage: python -W ignore make_thesis_figures.py"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyse_grid as AG
from bitrate_pareto import pareto_frontier
from config import CACHE_DIR

OUT = CACHE_DIR.parent / "docs" / "thesis_sim" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode); categorical slots in fixed order
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PATIENT_COLOUR = {"P1": BLUE, "P2": ORANGE, "S1": AQUA, "S2": VIOLET}
PATIENT_MARKER = {"P1": "o", "P2": "s", "S1": "^", "S2": "D"}
MODEL_COLOUR = {"LDA": BLUE, "SVM": ORANGE, "RandomForest": AQUA}
MODEL_MARKER = {"LDA": "o", "SVM": "s", "RandomForest": "^"}
MODEL_LABEL = {"LDA": "LDA", "SVM": "SVM", "RandomForest": "RF"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_main_effects(df):
    """Mean accuracy over the other two settings and over LDA, SVM and RF (= table T5 of docs/grid_results.md). Solid = set W, dashed = set A."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.1), sharey=True)
    specs = [("channels", "Channels transmitted", None), ("rate_hz", "Sample rate (Hz)", "log"), ("bits", "Bit depth (bits per sample)", None)]
    for ax, (col, xlabel, scale) in zip(axes, specs):
        for p in ["P1", "P2", "S1", "S2"]:
            for fset, ls, alpha in [("W", "-", 1.0), ("A", "--", 0.55)]:
                g = df[(df.patient == p) & (df.feature_set == fset)].groupby(col).accuracy.mean() * 100
                ax.plot(g.index, g.values, ls=ls, color=PATIENT_COLOUR[p], marker=PATIENT_MARKER[p], alpha=alpha, markersize=3.8 if fset == "W" else 3,
                        label=f"{p}, set {fset}", markeredgecolor="white", markeredgewidth=0.5)
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
        ax.set_title(t, loc="left", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "fig_main_effects")


def fig_frontier(df):
    """Pareto frontier (best accuracy at or below each bit rate) of each model with set W, per patient; filled marker = geometric elbow (analyse_grid.pick), ring = reference 1 kHz x 8 ch x 16 bit."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.0), sharey=True)
    for ax, p in zip(axes, ["P1", "P2", "S1", "S2"]):
        for m in ["LDA", "SVM", "RandomForest"]:
            d = df[(df.patient == p) & (df.feature_set == "W") & (df.model == m)]
            ax.scatter(d.bitrate_bps / 1000, d.accuracy * 100, s=4, color=MODEL_COLOUR[m], alpha=0.18, linewidths=0)
            fr = pareto_frontier(list(zip(d.bitrate_bps.astype(float), d.accuracy.astype(float))))
            fx, fy = zip(*sorted(fr))
            ax.step(np.array(fx) / 1000, np.array(fy) * 100, where="post", color=MODEL_COLOUR[m], lw=1.3, label=MODEL_LABEL[m])
            cfg, c, a = AG.pick(d, "accuracy", "bitrate_bps", geometric=True)
            ax.plot(c / 1000, a * 100, marker=MODEL_MARKER[m], color=MODEL_COLOUR[m], markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            ref = d[(d.rate_hz == 1000) & (d.channels == 8) & (d.bits == 16)]
            ax.plot(ref.bitrate_bps / 1000, ref.accuracy * 100, marker="o", mfc="none", color=MODEL_COLOUR[m], markersize=6, mew=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.set_title(p, loc="left", color=INK2)
        ax.set_xlabel("Bit rate (kb/s)")
        ax.set_ylim(40, 100)
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="o", color=INK2, ls="none", markersize=6, markeredgecolor="white"), Line2D([], [], marker="o", mfc="none", color=INK2, ls="none", markersize=6)]
    l += ["elbow (filled)", "reference 128 kb/s (ring)"]
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig_frontier")


def parse_md_table(text, heading):
    block = text.split(heading, 1)[1]
    lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
        if len(rows) > 40:
            break
    return rows


def fig_temporal():
    fr = json.load(open(CACHE_DIR / "grid" / "fair" / "fair_results.json"))["results"]
    ks = ["k1", "k3", "k5", "k9", "k15", "all"]
    span_ms = ["200", "300", "400", "600", "900", "all\n(≤2 s)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for m in ["LDA", "SVM", "RandomForest"]:
        y = [100 * np.mean([v["accuracy"] for k, v in fr.items() if k.endswith(f"|{m}/{kk}")]) for kk in ks]
        a1.plot(range(len(ks)), y, color=MODEL_COLOUR[m], marker=MODEL_MARKER[m], label=f"{MODEL_LABEL[m]} + vote", markeredgecolor="white", markeredgewidth=0.5)
        a1.annotate(f"{y[-1]:.1f}", (len(ks) - 1, y[-1]), xytext=(4, {"RandomForest": 3, "SVM": -2, "LDA": 0}[m]), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    cells = pd.read_csv(CACHE_DIR / "grid" / "fair2" / "fair2_cells.csv")
    lstm = {name: 100 * cells[cells.method == name].accuracy.mean() for name in ["LSTM_tuned", "LSTM_best", "LSTM_best_ens5"]}
    for name, ls, lab in [("LSTM_best_ens5", "-", "LSTM, search, 5 seeds"), ("LSTM_tuned", "--", "LSTM, 3-setting tuning"), ("LSTM_best", ":", "LSTM, search, 1 seed")]:
        a1.axhline(lstm[name], color=INK2, ls=ls, lw=1.0, label=f"{lab} ({lstm[name]:.1f})")
    a1.set_xticks(range(len(ks)))
    a1.set_xticklabels(span_ms)
    a1.set_xlabel("Span of the causal vote (ms of signal)")
    a1.set_ylabel("Accuracy, 28 cells (%)")
    a1.set_ylim(78, 95.5)
    a1.set_title("(a)", loc="left", color=INK2)
    a1.legend(loc="lower right", ncol=1, frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0.2)

    rows = parse_md_table(open(CACHE_DIR / "grid" / "fair2" / "fair2_tables.md", encoding="utf-8").read(), "## F13.")
    head, data = rows[0], {r[0]: r[1:] for r in rows[1:]}
    xlab = ["1–2\n(≤250)", "3–5\n(≤400)", "6–10\n(≤650)", "11–20\n(≤1150)", "21+\n(>1150)"]
    series = [("RF/P0", AQUA, "--", "^", "RF, no vote"), ("RF/P3", AQUA, "-", "^", "RF + vote"), ("SVM/P3", ORANGE, "-", "s", "SVM + vote"), ("LDA/P3", BLUE, "-", "o", "LDA + vote"),
              ("LSTM_best_ens5", INK2, "-", "D", "LSTM, 5 seeds"), ("LSTM_best", INK2, ":", "D", "LSTM, 1 seed")]
    x = np.arange(len(head) - 1)
    for key, col, ls, mk, lab in series:
        y = [float(v) for v in data[key]]
        a2.plot(x, y, color=col, ls=ls, marker=mk, label=lab, markeredgecolor="white", markeredgewidth=0.5, lw=1.4)
    a2.set_xticks(x)
    a2.set_xticklabels(xlab)
    a2.set_xlabel("Window index since onset (decision time, ms)")
    a2.set_ylabel("Accuracy (%)")
    a2.set_ylim(72, 97.5)
    a2.set_title("(b)", loc="left", color=INK2)
    a2.legend(loc="lower right", frameon=False, fontsize=8, ncol=2, handlelength=1.8, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig_temporal")


def main():
    df = AG.table()
    fig_main_effects(df)
    fig_frontier(df)
    fig_temporal()


if __name__ == "__main__":
    main()
