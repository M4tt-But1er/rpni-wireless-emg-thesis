"""Does the earlier grid's elbow ('every participant lands on 250 Hz x 5 ch x 8 bit = 10 kbps') depend on an integer truncation in feature_selection.find_elbow?
find_elbow ends with `return int(k_grid[elbow_idx])` (written for integer K grids); bitrate_pareto.best_ratio_point passes it log10(bit rate), so the elbow's log value is cut to a whole number
(for example 4.2 -> 4) and then snapped to the nearest frontier point (log10 = 4.0 is exactly 10 kb/s). This script recomputes the old grid's winners (same data, same selection of the best config at each bit rate)
(a) with the code as it was and (b) with the geometric rule the docstring describes (max perpendicular distance on normalised axes, no truncation).  Read-only: writes only cache/elbow_truncation_check.txt.
Usage: python -W ignore check_elbow_truncation.py"""
import json

import numpy as np

from bitrate_pareto import best_ratio_point, pareto_frontier
from config import CACHE_DIR

MAX_BPS = 1_280_000
MODELS = ["RandomForest", "SVM"]


def load():
    t1 = json.load(open(CACHE_DIR / "bitrate_grid_full_results.json"))
    fixed = json.load(open(CACHE_DIR / "bitrate_grid_wallace_fixedrange.json"))
    for ds in ("Wallace_S1", "Wallace_S2"):
        t1[ds] = fixed[ds]
    for d in t1.values():
        d["configs"] = {k: e for k, e in d["configs"].items() if e["bitrate_bps"] <= MAX_BPS}
    return t1


def elbow_index(x, y):
    """Geometric elbow: index of the point with max perpendicular distance from the chord first-last, on axes normalised to [0, 1] (no truncation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    d = [np.linalg.norm((np.array([a, b]) - p1) - np.dot(np.array([a, b]) - p1, u) * u) for a, b in zip(xn, yn)]
    return int(np.argmax(d))


def best_ratio_point_geometric(points):
    fr = pareto_frontier(points)
    if len(fr) == 1:
        return fr[0]
    b = np.array([p[0] for p in fr], float)
    m = np.array([p[1] for p in fr], float)
    return fr[elbow_index(np.log10(b), m)]


def main():
    t1 = load()
    n_same = n = 0
    for ds, d in t1.items():
        for model in MODELS:
            best_at = {}
            for key, e in d["configs"].items():
                b, m = e["bitrate_bps"], e[model]["accuracy"]
                if b not in best_at or m > best_at[b][0]:
                    best_at[b] = (m, key)
            pts = [(b, m) for b, (m, _) in best_at.items()]
            old = best_ratio_point(pts)
            new = best_ratio_point_geometric(pts)
            fr = pareto_frontier(pts)
            n += 1
            n_same += old[0] == new[0]
            print(f"{ds:11s} {model:12s}: as coded: {old[0] / 1000:.0f} kb/s ({100 * old[1]:.1f}%, {best_at[old[0]][1]}); geometric: {new[0] / 1000:.0f} kb/s ({100 * new[1]:.1f}%, {best_at[new[0]][1]}); "
                  f"frontier log10(bps) of the points: {[round(float(np.log10(p[0])), 2) for p in fr]}")
    print(f"\nSame bit rate with and without the truncation in {n_same} of {n} cases.")


if __name__ == "__main__":
    main()
