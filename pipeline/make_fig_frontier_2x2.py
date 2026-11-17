"""Figure 4.2 of the thesis (Final V4) as a 2x2 panel: the same Pareto frontiers as make_thesis_figures.fig_frontier
(same data cache/grid/grid_results.json via analyse_grid.table(), same style and colours), arranged 2x2 instead of 1x4 for legibility.
No model is run. Output: docs/thesis_sim/figures/fig_frontier_2x2.pdf (+ .png).
Usage: python -W ignore make_fig_frontier_2x2.py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import make_thesis_figures as MF
import analyse_grid as AG
from bitrate_pareto import pareto_frontier


def fig_frontier_2x2(df):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), sharey=True, sharex=True)
    for ax, p in zip(axes.ravel(), ["P1", "P2", "S1", "S2"]):
        for m in ["LDA", "SVM", "RandomForest"]:
            d = df[(df.patient == p) & (df.feature_set == "W") & (df.model == m)]
            ax.scatter(d.bitrate_bps / 1000, d.accuracy * 100, s=5, color=MF.MODEL_COLOUR[m], alpha=0.18, linewidths=0)
            fr = pareto_frontier(list(zip(d.bitrate_bps.astype(float), d.accuracy.astype(float))))
            fx, fy = zip(*sorted(fr))
            ax.step(np.array(fx) / 1000, np.array(fy) * 100, where="post", color=MF.MODEL_COLOUR[m], lw=1.3, label=MF.MODEL_LABEL[m])
            cfg, c, a = AG.pick(d, "accuracy", "bitrate_bps", geometric=True)
            ax.plot(c / 1000, a * 100, marker=MF.MODEL_MARKER[m], color=MF.MODEL_COLOUR[m], markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            ref = d[(d.rate_hz == 1000) & (d.channels == 8) & (d.bits == 16)]
            ax.plot(ref.bitrate_bps / 1000, ref.accuracy * 100, marker="o", mfc="none", color=MF.MODEL_COLOUR[m], markersize=6, mew=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.set_title(p, loc="left", color=MF.INK2)
        ax.set_ylim(40, 100)
    for ax in axes[1]:
        ax.set_xlabel("Bit rate (kb/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color=MF.INK2, ls="none", markersize=6, markeredgecolor="white"),
          Line2D([], [], marker="o", mfc="none", color=MF.INK2, ls="none", markersize=6)]
    l += ["elbow (filled)", "reference 128 kb/s (ring)"]
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    MF.save(fig, "fig_frontier_2x2")


if __name__ == "__main__":
    fig_frontier_2x2(AG.table())
