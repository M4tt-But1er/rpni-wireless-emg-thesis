"""Tuned RBFNN for the Stage 1 comparison (docs/stage1_declared_protocol.md, Amendment 2).

One task = one (patient, operating point, feature set, outer fold): the hyper-parameters (centres x width multiplier x C, grid of run_rbfnn_tuning.py) are chosen by the grouped,
stratified inner 3-fold CV on the outer-training part only (run_rbfnn_tuning.tune_rbfnn), the model is refitted on the whole outer-training part and scored on the untouched outer fold.
Same outer folds as everything else (models.compute_cv_folds on each table).

Output (new file): cache/stage1_rbfnn_tuned_results.json, keyed "<patient>|<op>|<set>|<fold>" -> {pred, test_index, acc, n_centers, sigma_scale, C, inner_cv_acc, n_features, seconds}.
Resumable at the level of one task. compare_stage1_ops.py pools the folds into accuracy / weighted F1 / kappa.
Usage: python -W ignore -u run_stage1_rbfnn_tuned.py [--workers N] [--sets A B C] [--ops ref low]
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from build_wide_tables import OPS, PARTS, TAG, table_path
from config import CACHE_DIR
from models import RBFNetClassifier, compute_cv_folds
from run_rbfnn_tuning import tune_rbfnn
from wide_features import set_columns

OUT = CACHE_DIR / f"stage1_rbfnn_tuned_results{'_' + TAG if TAG else ''}.json"


def run_task(part, op, s, fold, threads):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t = pd.read_parquet(table_path(part, op))
        cols = set_columns(list(t.columns[5:]), s)
        X = t[cols].to_numpy(dtype=np.float64)
        y = t["cue"].to_numpy()
        groups = t["segment_id"].to_numpy()
        folds = compute_cv_folds(t)
        tr, te = folds != fold, folds == fold
        (n_c, sig, C), inner, _ = tune_rbfnn(X[tr], y[tr], groups[tr])
        sc = StandardScaler().fit(X[tr])
        clf = RBFNetClassifier(n_centers=n_c, C=C, sigma_scale=sig).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        acc = float((pred == y[te]).mean())
        rec = {"pred": pred.tolist(), "test_index": np.flatnonzero(te).tolist(), "acc": acc, "n_centers": int(n_c), "sigma_scale": float(sig),
               "C": float(C), "inner_cv_acc": float(inner), "n_features": len(cols), "seconds": time.time() - t0}
    print(f"  {part} {op} set {s} fold {fold}: F={len(cols)} centres={n_c} width x{sig} C={C} inner {inner:.4f} outer {acc:.4f} ({rec['seconds']:.0f}s)", flush=True)
    return f"{part}|{op}|{s}|{fold}", rec


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=str, many=False):
        if name not in args:
            return default
        i = args.index(name)
        vals = []
        for a in args[i + 1:]:
            if a.startswith("--"):
                break
            vals.append(cast(a))
        return vals if many else vals[0]

    workers = opt("--workers", 1, int)
    sets = opt("--sets", ["C", "B", "A"], many=True)
    ops = opt("--ops", list(OPS), many=True)
    results = json.load(open(OUT)) if OUT.exists() else {}
    tasks = [(p, o, s, f) for s in sets for o in ops for p in PARTS for f in range(5) if f"{p}|{o}|{s}|{f}" not in results]
    threads = max(1, 16 // workers)
    print(f"{len(tasks)} tasks, {workers} worker(s), {threads} thread(s) each", flush=True)
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(p, o, s, f, threads) for p, o, s, f in tasks):
        results[key] = rec
        with open(OUT, "w") as fh:
            json.dump(results, fh)
    print("All done")


if __name__ == "__main__":
    main()
