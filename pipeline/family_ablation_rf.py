"""EXPLORATORY (user, 13 Nov 2026; replaces nothing): Random Forest for the E4 family ablation of family_ablation.py
(28 feature subsets x 8 tables = 4 participants x ref/low streams, grouped 5-fold CV, same tables and folds). The published
LDA/SVM results (cache/family_ablation_results.json) and the wide set W (chosen by an LDA-only rule) are only read.
RF is fitted with run_grid.fit_predict_info (n_jobs=2; same trees as n_jobs=-1, random_state fixed) so that many cells can run at once.
Checks: (1) the A and C subsets must reproduce the Stage-1 RF values (cache/stage1_ops_results_fixedrange.json, same tables and folds);
(2) --check also reruns two published LDA cells with this script's code path.
Output: NEW file cache/family_ablation_rf/results.json, keyed as family_ablation_results.json.
Usage: STAGE1_TAG=fixedrange python -W ignore -u family_ablation_rf.py [--workers N] [--check]"""
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
from wide_features import set_columns

OUT_DIR = CACHE_DIR / "family_ablation_rf"
OUT = OUT_DIR / "results.json"
PUBLISHED = CACHE_DIR / "family_ablation_results.json"
STAGE1 = CACHE_DIR / "stage1_ops_results_fixedrange.json"


def run_cell(part, op, subset, model):
    """As family_ablation.run_cell, with the grid's fitting function (RF on 2 threads)."""
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    from run_grid import fit_predict_info
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
            pred, _ = fit_predict_info(model, X[tr], y[tr], X[te])
            pooled[te] = pred
            accs.append(float((pred == y[te]).mean()))
        m = compute_metrics(y, pooled, sorted(t["cue"].unique().tolist()))
        rec = {"fold_accs": accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"],
               "n_features": len(sub), "seconds": time.time() - t0}
    return f"{part}|{op}|{subset}|{model}", rec


def main():
    assert TAG == "fixedrange", "run with STAGE1_TAG=fixedrange (the Wallace tables must use a fixed quantiser range)"
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 8
    cols = list(pd.read_parquet(table_path("P1_combined", "ref")).columns[5:])
    names = list(subsets_of(cols, set_columns(cols, "A")))
    if "--check" in args:
        pub = json.load(open(PUBLISHED))
        for key, rec in Parallel(n_jobs=2)(delayed(run_cell)(*k) for k in [("P1_combined", "ref", "drop:wavelet", "LDA"), ("Wallace_S2", "low", "add:subband_power", "LDA")]):
            same = rec["fold_accs"] == pub[key]["fold_accs"]
            print(f"  {key}: {100 * rec['accuracy']:.3f} vs published {100 * pub[key]['accuracy']:.3f} {'identical' if same else 'DIFFERENT'}", flush=True)
            assert same, key
        print("check passed")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = json.load(open(OUT)) if OUT.exists() else {}
    tasks = [(p, o, s, "RandomForest") for s in names for o in OPS for p in PARTS if f"{p}|{o}|{s}|RandomForest" not in res]
    print(f"{len(names)} subsets; {len(tasks)} RF cells to run, {workers} workers x 2 threads", flush=True)
    t0, n = time.time(), 0
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_cell)(*t) for t in tasks):
        res[key] = rec
        n += 1
        if n % 8 == 0 or n == len(tasks):
            json.dump(res, open(OUT, "w"))
            print(f"  {n}/{len(tasks)} cells, {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
    # check against Stage 1 (sets A and C are the same columns on the same tables and folds)
    s1 = json.load(open(STAGE1))
    for p in PARTS:
        for o in OPS:
            for sset in ("A", "C"):
                a, b = res[f"{p}|{o}|{sset}|RandomForest"]["accuracy"], s1[f"{p}|{o}"][f"{sset}|RandomForest"]["accuracy"]
                print(f"  Stage-1 check {p} {o} {sset}: {100 * a:.3f} vs {100 * b:.3f} {'identical' if a == b else 'DIFFERENT'}", flush=True)
    print("All done ->", OUT)


if __name__ == "__main__":
    main()
