"""Pareto-frontier + elbow (knee) selection for the bitrate-vs-accuracy/F1
tradeoff -- picks the "best bitrate to F1 or accuracy ratio" config per the
user's request, reusing feature_selection.py's find_elbow (a generic
max-perpendicular-distance-from-chord knee finder) on log10(bitrate) vs the
metric, since bitrate spans orders of magnitude (log axis is the standard
framing for a rate-distortion-style tradeoff, and linear bitrate would make
the frontier's early, cheap points visually/geometrically negligible).
"""
import numpy as np

from feature_selection import find_elbow


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points: (bitrate_bps, metric) pairs. Returns the non-dominated subset
    (lower bitrate AND higher metric is better), sorted by bitrate ascending
    -- for increasing bitrate, keep only points that set a new best metric."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))  # equal bitrate: best metric first, so a worse tie is dropped
    frontier = []
    best_metric = -np.inf
    for bitrate, metric in pts:
        if metric > best_metric:
            frontier.append((bitrate, metric))
            best_metric = metric
    return frontier


def best_ratio_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Returns the (bitrate, metric) point at the elbow of the Pareto
    frontier -- the best tradeoff, not just the cheapest or the most
    accurate point.

    WARNING (added 9 Nov 2026, comment only, behaviour unchanged): this is
    BUGGY. find_elbow truncates its result to an int, so on the log10(bitrate)
    axis the elbow is cut to a whole number of decades and then snapped to the
    nearest frontier point, which gave 10 kb/s (250 Hz x 5 ch x 8 bit) for every
    participant whatever the data. Use check_elbow_truncation.
    best_ratio_point_geometric instead. See docs/elbow_truncation_note.md."""
    frontier = pareto_frontier(points)
    if len(frontier) == 1:
        return frontier[0]
    bitrates = np.array([p[0] for p in frontier], dtype=float)
    metrics = np.array([p[1] for p in frontier], dtype=float)
    log_bitrates = np.log10(bitrates)
    elbow_log_bitrate = find_elbow(log_bitrates, metrics)
    idx = int(np.argmin(np.abs(log_bitrates - elbow_log_bitrate)))
    return frontier[idx]
