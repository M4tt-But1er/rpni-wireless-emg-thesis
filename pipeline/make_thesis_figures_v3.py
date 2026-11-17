"""Figures for the V3 thesis revision (14 Nov 2026), in the style of make_thesis_figures.py. New files only; nothing existing is overwritten.
  fig_temporal_OS     - fig_temporal_W with the feature set renamed OS (W -> OS in the thesis); same data and code path
  fig_channel_order   - accuracy against channels kept on the raw stream: grid order vs backward-elimination keep order
                        (cache/channel_importance/plan.json, evals.json; mean of LDA, SVM, RF)
  fig_rpni_vs_muscle  - (a) RPNI channels only vs every equal-size set of residual-muscle channels only (raw stream);
                        (b) mean single-channel accuracy against sample rate (16 bit) for RPNI and muscle channels
                        (cache/rpni_vs_muscle/evals.json)
Usage: python -W ignore make_thesis_figures_v3.py"""
import json

import matplotlib.pyplot as plt
import numpy as np

import make_thesis_figures as M
import make_thesis_figures_W as MW
from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from run_channel_importance import MODELS, RPNI, skey

SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
RP, MU = M.BLUE, M.MUTED


def fig_temporal_OS():
    """Same as make_thesis_figures_W.fig_temporal_W; only the axis label and file name change."""
    orig_save, orig_ylabel = M.save, plt.Axes.set_ylabel

    def ylabel(self, text, *a, **k):
        return orig_ylabel(self, text.replace("14 W cells", "14 candidate configurations"), *a, **k)

    plt.Axes.set_ylabel = ylabel
    M.save = lambda fig, name: orig_save(fig, "fig_temporal_OS")
    try:
        MW.fig_temporal_W()
    finally:
        M.save, plt.Axes.set_ylabel = orig_save, orig_ylabel


def fig_channel_order():
    plan = json.load(open(CACHE_DIR / "channel_importance" / "plan.json"))
    ev = json.load(open(CACHE_DIR / "channel_importance" / "evals.json"))

    def mean_acc(part, subset):
        return 100 * np.mean([ev[f"{part}|{m}|{skey(part, subset)}"]["accuracy"] for m in MODELS])

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), sharey=True)
    for ax, part in zip(axes, PARTS):
        p = plan[part]
        n = len(p["channels"])
        x = np.arange(1, n + 1)
        grid = [mean_acc(part, p["grid_order"][:k]) for k in x]
        keep = [mean_acc(part, p["be_keep_order"][:k]) for k in x]
        ax.plot(x, grid, color=M.ORANGE, marker="s", label="Grid order", markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(x, keep, color=M.BLUE, marker="o", label="Keep order", markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title(SHORT[part], color=M.INK2)
        ax.set_xticks(x if n <= 8 else x[::2])
        ax.set_xlabel("Channels kept")
    axes[0].set_ylabel("Mean accuracy (%)")
    axes[0].set_ylim(40, 100)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    M.save(fig, "fig_channel_order")


def fig_rpni_vs_muscle():
    E = json.load(open(CACHE_DIR / "rpni_vs_muscle" / "evals.json"))
    from run_rpni_vs_muscle import T30, muscle_subsets

    def acc(part, table, subset):
        return 100 * np.mean([E[f"{part}|{table}|{m}|{skey(part, subset)}"]["accuracy"] for m in MODELS])

    rng = np.random.default_rng(0)
    rates = [250, 500, 1000, 3000, 10000]
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    for i, part in enumerate(PARTS):
        mu = [acc(part, T30, s) for s in muscle_subsets(part)]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(mu)), mu, s=12, color=MU, alpha=0.75, zorder=2,
                   label="Muscle channels only\n(every equal-size set)" if i == 0 else None)
        ax.scatter([i], [acc(part, T30, RPNI[part])], s=46, marker="D", color=RP, zorder=3, edgecolor="white", linewidth=0.6,
                   label="RPNI channels only" if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{SHORT[p]}\n({len(RPNI[p])} ch)" for p in PARTS])
    ax.set_ylabel("Accuracy, raw stream (%)")
    ax.set_ylim(55, 95)
    ax.set_title("(a)", loc="left", color=M.INK2)
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    x = np.arange(len(rates))
    for j, part in enumerate(PARTS):
        a = fig.add_subplot(gs[j // 2, 1 + j % 2])
        ch = channels_of(part)
        for typ, col in (("muscle", MU), ("RPNI", RP)):
            cs = [c for c in ch if (c in RPNI[part]) == (typ == "RPNI")]
            Y = np.array([[acc(part, f"{r}Hz_16bit", [c]) for r in rates] for c in cs])
            for y in Y:
                a.plot(x, y, color=col, lw=0.6, alpha=0.35, marker=None)
            a.plot(x, Y.mean(axis=0), color=col, lw=1.8, marker="o", ms=3.2, label=f"{typ} (mean)", markeredgecolor="white", markeredgewidth=0.4)
        a.set_xticks(x)
        a.set_xticklabels(["250", "500", "1k", "3k", "10k"], fontsize=8)
        a.set_ylim(10, 80)
        a.tick_params(axis="y", labelsize=8)
        a.set_title(("(b) " if j == 0 else "") + SHORT[part], loc="left", color=M.INK2)
        if j % 2 == 0:
            a.set_ylabel("Single channel (%)", fontsize=8.5)
        if j >= 2:
            a.set_xlabel("Sample rate (Hz), 16 bit", fontsize=8.5)
        if j == 1:
            a.legend(loc="lower right", frameon=False, fontsize=7.5)
    M.save(fig, "fig_rpni_vs_muscle")


if __name__ == "__main__":
    fig_temporal_OS()
    fig_channel_order()
    fig_rpni_vs_muscle()
