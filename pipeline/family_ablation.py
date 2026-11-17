"""Family ablation of docs/family_ablation_declared_protocol.md: 28 feature subsets x LDA/SVM x 8 tables (4 patients x ref/low), grouped 5-fold CV (same folds as Stage 1).
Needs STAGE1_TAG=fixedrange (Wallace tables with one quantiser range per channel). Output (new file): cache/family_ablation_results.json, keyed "<part>|<op>|<subset>|<model>".
Resumable per cell. Usage: STAGE1_TAG=fixedrange python -W ignore -u family_ablation.py [--workers N] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import OPS, PARTS, TAG, table_path
from config import CACHE_DIR
from families import subsets_of
from models import compute_cv_folds
from run_models_norest import compute_metrics
from run_own_feature_selection import fit_predict
from wide_features import set_columns

OUT = CACHE_DIR / "family_ablation_results.json"
MODELS = ["LDA", "SVM"]


def run_cell(part, op, subset, model):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        t0 = time.time()
        t = pd.read_parquet(table_path(part, op))
        cols = list(t.columns[5:])
        sub = subsets_of(cols, set_columns(cols, "A"))[subset]
        X = t[sub].to_numpy(dtype=np.float64)
        y = t["cue"].to_numpy()
        folds = compute_cv_folds(t)
        pooled = np.empty_like(y)
        accs = []
        for f in sorted(np.unique(folds)):
            tr, te = folds != f, folds == f
            pred = fit_predict(model, X[tr], y[tr], X[te])
            pooled[te] = pred
            accs.append(float((pred == y[te]).mean()))
        m = compute_metrics(y, pooled, sorted(t["cue"].unique().tolist()))
        rec = {"fold_accs": accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"],
               "n_features": len(sub), "seconds": time.time() - t0}
    print(f"  {part} {op} {subset:32s} {model:4s} F={len(sub):4d} acc {100 * rec['accuracy']:.2f} ({rec['seconds']:.0f}s)", flush=True)
    return f"{part}|{op}|{subset}|{model}", rec


def main():
    assert TAG == "fixedrange", "run with STAGE1_TAG=fixedrange (the Wallace tables must use a fixed quantiser range)"
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    results = json.load(open(OUT)) if OUT.exists() else {}
    names = list(subsets_of(list(pd.read_parquet(table_path("P1_combined", "ref")).columns[5:]), set_columns(list(pd.read_parquet(table_path("P1_combined", "ref")).columns[5:]), "A")))
    tasks = [(p, o, s, m) for m in MODELS for s in names for o in OPS for p in PARTS if f"{p}|{o}|{s}|{m}" not in results]
    if "--smoke" in args:
        tasks = [(p, o, s, "LDA") for p, o, s, m in tasks if m == "LDA" and s in ("A", "C", "drop:wavelet") and p == "P1_combined" and o == "ref"]
    print(f"{len(names)} subsets; {len(tasks)} cells to run, {workers} worker(s)", flush=True)
    n = 0
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_cell)(*t) for t in tasks):
        results[key] = rec
        n += 1
        if n % 8 == 0 or n == len(tasks):
            with open(OUT, "w") as fh:
                json.dump(results, fh)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print("All done")


if __name__ == "__main__":
    main()
