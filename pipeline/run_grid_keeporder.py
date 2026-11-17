"""EXPLORATORY (user, 12 Nov 2026; replaces nothing): the bit-rate grid's channel cells again, with channels kept in the backward-elimination
keep order of the channel-importance study (cache/channel_importance/plan.json, chosen on the 30 kHz stream with set W, mean of LDA/SVM/RF)
instead of the grid's order (best mutual-information rank of any feature of the channel). Everything else is the grid's own: tables, sets
A and W, models, folds, metrics (run_grid.fit_predict_info, grid_slices.slice_matrix, run_models_norest.compute_metrics).
Only cells whose channel SET changes are run; the others are identical to the grid by construction.
Caveat: the keep order was chosen on these participants' data (as the grid order was), so differences are a sensitivity check, not an unbiased estimate.
Output: NEW file cache/grid_keeporder/results.json, same keys as cache/grid/grid_results.json. cache/grid/ is only read.
Usage: python -W ignore -u run_grid_keeporder.py [--workers N] [--check]
  --check: rerun a sample of cells with the GRID order and assert they equal cache/grid/grid_results.json (verifies this script)."""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from feature_cost import extraction_cost
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import BITS, RATES, table_path
from models import compute_cv_folds
from run_grid import MODELS, SETS, W_PATH, cell_key, configs, fit_predict_info
from run_models_norest import compute_metrics
from wide_features import set_columns

GRID = CACHE_DIR / "grid" / "grid_results.json"
OUT_DIR = CACHE_DIR / "grid_keeporder"
OUT = OUT_DIR / "results.json"
KEEP = {p: v["be_keep_order"] for p, v in json.load(open(CACHE_DIR / "channel_importance" / "plan.json")).items()}


def order_of(part, which):
    return KEEP[part] if which == "keep" else configs(part)[1]


def run_cells(part, rate, bits, model, wanted, which):
    """As run_grid.run_task, with the channel order chosen by `which` ('keep' or 'grid')."""
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    W = set(json.load(open(W_PATH))["kept_leaves"])
    out = []
    with threadpool_limits(limits=1):
        t = pd.read_parquet(table_path(part, rate, bits))
        cols = list(t.columns[5:])
        channels = channels_of(part)
        order = order_of(part, which)
        y, folds = t["cue"].to_numpy(), compute_cv_folds(t)
        labels = sorted(t["cue"].unique().tolist())
        a_cols = set(set_columns(cols, "A"))
        set_cols = {"A": [c for c in cols if c in a_cols], "W": [c for c in cols if leaf_of(c) in W]}
        for nch, fset in wanted:
            t0 = time.time()
            keep = order[:nch]
            X, names = slice_matrix(t, set_cols[fset], channels, keep)
            pooled = np.empty_like(y)
            accs, infos = [], []
            for f in sorted(np.unique(folds)):
                tr, te = folds != f, folds == f
                pred, info = fit_predict_info(model, X[tr], y[tr], X[te])
                pooled[te] = pred
                accs.append(float((pred == y[te]).mean()))
                infos.append(info)
            m = compute_metrics(y, pooled, labels)
            out.append((cell_key(part, rate, nch, bits, fset, model),
                        {"fold_accs": accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"],
                         "confusion_matrix": m["confusion_matrix"], "class_labels": m["class_labels"], "n_features": len(names),
                         "n_windows": int(len(t)), "fold_info": infos, "extraction": extraction_cost(names, channels, rate),
                         "bitrate_bps": rate * nch * bits, "channels": list(keep), "order": which, "seconds": time.time() - t0}))
    return out


def changed(part, nch):
    return set(KEEP[part][:nch]) != set(configs(part)[1][:nch])


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 12
    grid = json.load(open(GRID))
    if "--check" in args:
        sample = [("P1_combined", 1000, 16, "LDA"), ("Wallace_S2", 250, 8, "SVM"), ("P2_combined", 500, 12, "RandomForest")]
        jobs = [(p, r, b, m, [(n, s) for (rr, n, bb) in configs(p)[0] if rr == r and bb == b for s in SETS]) for p, r, b, m in sample]
        for batch in Parallel(n_jobs=len(jobs))(delayed(run_cells)(*j, "grid") for j in jobs):
            for k, rec in batch:
                g = grid[k]
                same = rec["accuracy"] == g["accuracy"] and rec["fold_accs"] == g["fold_accs"] and rec["channels"] == g["channels"]
                print(f"  {k}: new {100 * rec['accuracy']:.3f} grid {100 * g['accuracy']:.3f} {'identical' if same else 'DIFFERENT'}", flush=True)
                assert same, k
        print("check passed: the script reproduces the grid exactly with the grid order")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = json.load(open(OUT)) if OUT.exists() else {}
    tasks = []
    for part in PARTS:
        print(f"{part}: keep order {KEEP[part]}; grid order {configs(part)[1][:len(KEEP[part])]}; "
              f"changed channel counts {sorted({n for (_, n, _) in configs(part)[0] if changed(part, n)})}", flush=True)
        for model in MODELS:
            for rate in RATES:
                for bits in BITS:
                    wanted = [(n, s) for (r, n, b) in configs(part)[0] if r == rate and b == bits and changed(part, n)
                              for s in SETS if cell_key(part, rate, n, bits, s, model) not in res]
                    if wanted:
                        tasks.append((part, rate, bits, model, wanted))
    print(f"{sum(len(t[4]) for t in tasks)} cells in {len(tasks)} tasks, {workers} workers", flush=True)
    t0, n = time.time(), 0
    for batch in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_cells)(*t, "keep") for t in tasks):
        for k, rec in batch:
            res[k] = rec
        n += 1
        if n % 10 == 0 or n == len(tasks):
            json.dump(res, open(OUT, "w"))
            print(f"  {n}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"))
    print("All done ->", OUT)


if __name__ == "__main__":
    main()
