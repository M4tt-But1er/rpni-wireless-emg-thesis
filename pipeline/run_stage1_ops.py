"""Stage 1 runs of docs/stage1_declared_protocol.md: LDA, SVM, RF and the untuned RBFNN on feature sets A, B, C for every (patient, operating point) table
built by build_wide_tables.py. No feature selection, so nothing is nested; same grouped, class-stratified 5-fold CV as everywhere (models.compute_cv_folds on each table).

Output (new file): cache/stage1_ops_results.json, keyed "<patient>|<op>" -> "<set>|<model>" -> {fold_accs, fold_sizes, n_features, n_windows, seconds, accuracy, weighted_f1, ...}.
Resumable at the level of one (patient, operating point) table.
Usage: python -W ignore -u run_stage1_ops.py [--workers N] [patient op]
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import OPS, PARTS, TAG, table_path
from config import CACHE_DIR
from models import compute_cv_folds
from run_models_norest import compute_metrics
from run_own_feature_selection import fit_predict
from wide_features import set_columns

OUT = CACHE_DIR / f"stage1_ops_results{'_' + TAG if TAG else ''}.json"   # STAGE1_TAG=fixedrange -> stage1_ops_results_fixedrange.json (see build_wide_tables.py)
MODELS = ["LDA", "SVM", "RandomForest", "RBFNN"]
SETS = ["A", "B", "C"]


def run_task(part, op):
    warnings.filterwarnings("ignore")
    t = pd.read_parquet(table_path(part, op))
    cols = list(t.columns[5:])
    y = t["cue"].to_numpy()
    folds = compute_cv_folds(t)
    labels = sorted(t["cue"].unique().tolist())
    res = {}
    for s in SETS:
        sc = set_columns(cols, s)
        X = t[sc].to_numpy(dtype=np.float64)
        for m in MODELS:
            t0 = time.time()
            pooled = np.empty_like(y)
            accs, sizes = [], []
            for f in sorted(np.unique(folds)):
                tr, te = folds != f, folds == f
                pred = fit_predict(m, X[tr], y[tr], X[te])
                pooled[te] = pred
                accs.append(float((pred == y[te]).mean()))
                sizes.append(int(te.sum()))
            r = {"fold_accs": accs, "fold_sizes": sizes, "n_features": len(sc), "n_windows": len(t), "seconds": time.time() - t0}
            r.update(compute_metrics(y, pooled, labels))
            res[f"{s}|{m}"] = r
            print(f"  {part} {op} set {s} {m:12s} F={len(sc):4d}  acc {100 * r['accuracy']:.2f}  ({r['seconds']:.0f}s)", flush=True)
    return f"{part}|{op}", res


def main():
    args = sys.argv[1:]
    workers = 1
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    tasks = [(args[0], args[1])] if len(args) == 2 else [(p, o) for p in PARTS for o in OPS]
    results = json.load(open(OUT)) if OUT.exists() else {}
    todo = [(p, o) for p, o in tasks if f"{p}|{o}" not in results]
    print(f"{len(todo)} tables to run, {workers} worker(s)", flush=True)
    for key, res in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(p, o) for p, o in todo):
        results[key] = res
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"== saved {key}", flush=True)
    print("All done")


if __name__ == "__main__":
    main()
