"""V10a, V10b and V10e of docs/fair_lstm_declared_protocol.md.
V10a: the classical raw predictions (variant k1) reproduce cache/grid/grid_results.json for every cell and model (accuracy, confusion matrix, fold accuracies) - must be exact.
V10b: the LSTM rerun against the stored first run (cache/grid/networks_results.json): per-cell accuracy difference; reported, not required to be zero.
V10e: the stored predictions are aligned with the table rows (fold ids equal compute_cv_folds; every row predicted; classes in range).
Usage: python -W ignore check_fair_v10.py > cache/grid/fair/check_v10abe.log"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR
from grid_tables import table_path
from models import compute_cv_folds
from run_models_norest import compute_metrics

FAIR = CACHE_DIR / "grid" / "fair"
G = json.load(open(CACHE_DIR / "grid" / "grid_results.json"))
N = json.load(open(CACHE_DIR / "grid" / "networks_results.json"))
WIN = json.load(open(CACHE_DIR / "grid" / "winners.json"))


def npz(kind, key):
    return np.load(FAIR / kind / (key.replace("|", "__") + ".npz"))


def main():
    n_cm = n_acc = n_fold = n_total = n_align = 0
    worst = 0.0
    lstm_diffs = []
    for pf, cfgs in WIN.items():
        part, fset = pf.split("|")
        for rate, nch, bits in cfgs:
            t = pd.read_parquet(table_path(part, rate, bits), columns=["cue", "segment_id", "window_time_s"])
            y = t["cue"].to_numpy()
            folds = compute_cv_folds(t)
            labels = sorted(np.unique(y).tolist())
            for model in ("LDA", "SVM", "RandomForest"):
                key = f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{model}"
                z = npz("classical", key)
                pooled = z["pooled"]
                assert (z["folds"] == folds).all() and len(pooled) == len(y)
                n_align += 1
                m = compute_metrics(y, pooled, labels)
                g = G[key]
                fa = [float((pooled[folds == f] == y[folds == f]).mean()) for f in sorted(np.unique(folds))]
                n_total += 1
                n_acc += m["accuracy"] == g["accuracy"]
                n_cm += m["confusion_matrix"] == g["confusion_matrix"]
                n_fold += np.allclose(fa, g["fold_accs"], rtol=0, atol=1e-15)
                worst = max(worst, abs(m["accuracy"] - g["accuracy"]))
            for method in ("LSTM_default", "LSTM_tuned"):
                key = f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{method}"
                z = npz("lstm", key)
                pred = z["pred"]
                assert (z["folds"] == folds).all() and len(pred) == len(y) and set(np.unique(pred)) <= set(labels)
                n_align += 1
                acc = float((pred == y).mean())
                lstm_diffs.append((method, key, acc - N[key]["accuracy"], np.array(z["fold_accs"]) - np.array(N[key]["fold_accs"])))
    print("V10a: classical raw predictions (variant k1) against the grid")
    print(f"  cells x models checked: {n_total}; accuracy identical: {n_acc}; confusion matrix identical: {n_cm}; fold accuracies identical: {n_fold}; largest accuracy difference {worst:.2e}")
    print("  PASSED" if n_acc == n_cm == n_fold == n_total else "  FAILED")
    print("\nV10b: LSTM rerun against the stored first run (difference in points of pooled accuracy)")
    for method in ("LSTM_default", "LSTM_tuned"):
        d = np.array([100 * x[2] for x in lstm_diffs if x[0] == method])
        same = int((np.abs(d) < 1e-9).sum())
        print(f"  {method}: {len(d)} cells; identical in {same}; mean difference {d.mean():+.3f}; mean absolute {np.abs(d).mean():.3f}; largest absolute {np.abs(d).max():.3f}")
        for x in sorted([x for x in lstm_diffs if x[0] == method], key=lambda x: -abs(x[2]))[:3]:
            print(f"      largest: {x[1]}: {100 * x[2]:+.3f} points")
    print("\nV10e: alignment of stored predictions with table rows: fold ids equal compute_cv_folds, one prediction per row, classes in range: " + f"{n_align} files checked, PASSED")


if __name__ == "__main__":
    main()
