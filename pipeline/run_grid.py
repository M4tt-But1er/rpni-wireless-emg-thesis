"""Bitrate grid models (docs/grid_declared_protocol.md): 193 configurations x feature sets A and W x LDA/SVM/RF, grouped 5-fold CV on the all-channel tables of grid_tables.py,
channel subsets sliced out by grid_slices.py. One task = one (patient, rate, bits, model) table; resumable per task. Output: cache/grid/grid_results.json keyed
"<patient>|<rate>Hz|<n>ch|<bits>bit|<set>|<model>" -> {fold_accs, accuracy, weighted_f1, cohen_kappa, confusion_matrix, class_labels, n_features, n_windows, fold_info, extraction, bitrate_bps, channels, seconds}.
Usage: python -W ignore -u run_grid.py [--workers N] [--models LDA SVM RandomForest] [--rates ...] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from feature_cost import extraction_cost
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import BITS, RATES, table_path
from models import compute_cv_folds, make_model_pipeline
from run_bitrate_search import get_ranking_and_grid
from run_models_norest import compute_metrics
from wide_features import set_columns

OUT = CACHE_DIR / "grid" / "grid_results.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
MODELS = ["LDA", "SVM", "RandomForest"]
SETS = ["A", "W"]
MAX_BPS = 1.4e6


def configs(part):
    """[(rate, n_channels, bits)] of the grid, bit rate <= 1.4 Mb/s, plus the channel ranking."""
    ranking, grid = get_ranking_and_grid(part)
    return [(r, n, b) for r in RATES for n in grid for b in BITS if r * n * b <= MAX_BPS], ranking


def cell_key(part, rate, nch, bits, fset, model):
    return f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{model}"


def fit_predict_info(model, Xtr, ytr, Xte):
    need, est = make_model_pipeline(model)
    if model == "RandomForest":
        est.set_params(n_jobs=2)           # same trees whatever n_jobs is (random_state fixed)
    if need:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    info = {}
    if model == "SVM":
        info["n_sv"] = int(est.support_vectors_.shape[0])
    elif model == "RandomForest":
        depths = [t.tree_.max_depth for t in est.estimators_]
        info.update(trees=len(depths), max_depth=int(max(depths)), mean_depth=float(np.mean(depths)), nodes=int(sum(t.tree_.node_count for t in est.estimators_)))
    return pred, info


def run_task(part, rate, bits, model, wanted):
    """`wanted` = [(n_channels, set)] cells of this table and model that are still to run."""
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    W = set(json.load(open(W_PATH))["kept_leaves"])
    out = []
    with threadpool_limits(limits=1):
        t = pd.read_parquet(table_path(part, rate, bits))
        cols = list(t.columns[5:])
        channels = channels_of(part)
        ranking = configs(part)[1]
        y, folds = t["cue"].to_numpy(), compute_cv_folds(t)
        labels = sorted(t["cue"].unique().tolist())
        a_cols = set(set_columns(cols, "A"))
        set_cols = {"A": [c for c in cols if c in a_cols], "W": [c for c in cols if leaf_of(c) in W]}
        for nch, fset in wanted:
            t0 = time.time()
            keep = ranking[:nch]
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
            rec = {"fold_accs": accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"],
                   "confusion_matrix": m["confusion_matrix"], "class_labels": m["class_labels"], "n_features": len(names), "n_windows": int(len(t)),
                   "fold_info": infos, "extraction": extraction_cost(names, channels, rate), "bitrate_bps": rate * nch * bits, "channels": list(keep),
                   "seconds": time.time() - t0}
            print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} {model:12s} F={len(names):4d} acc {100 * rec['accuracy']:.2f} ({rec['seconds']:.0f}s)", flush=True)
            out.append((cell_key(part, rate, nch, bits, fset, model), rec))
    return out


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=str, many=False):
        if name not in args:
            return default
        vals = []
        for a in args[args.index(name) + 1:]:
            if a.startswith("--"):
                break
            vals.append(cast(a))
        return vals if many else vals[0]

    workers = opt("--workers", 1, int)
    models = opt("--models", MODELS, str, many=True)
    rates = opt("--rates", RATES, int, many=True)
    assert W_PATH.exists(), "run family_backward_elimination.py first (cache/wide_set_W.json)"
    results = json.load(open(OUT)) if OUT.exists() else {}
    all_cfg = {p: configs(p)[0] for p in PARTS}
    assert [len(all_cfg[p]) for p in PARTS] == [45, 45, 45, 58], [len(all_cfg[p]) for p in PARTS]
    tasks = []
    # up to 1 kHz before 3 and 10 kHz; within each group LDA, then SVM, then RF
    for group in ([r for r in rates if r <= 1000], [r for r in rates if r > 1000]):
        for model in [m for m in MODELS if m in models]:
            for rate in group:
                for part in PARTS:
                    for bits in BITS:
                        wanted = [(n, s) for (r, n, b) in all_cfg[part] if r == rate and b == bits for s in SETS if cell_key(part, rate, n, bits, s, model) not in results]
                        if wanted:
                            tasks.append((part, rate, bits, model, wanted))
    if "--smoke" in args:
        tasks = tasks[:2]
    print(f"{sum(len(t[4]) for t in tasks)} cells in {len(tasks)} tasks, {workers} worker(s)", flush=True)
    n = 0
    for batch in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks):
        for key, rec in batch:
            results[key] = rec
        n += 1
        if n % 4 == 0 or n == len(tasks):
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "w") as fh:
                json.dump(results, fh)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print("All done")


if __name__ == "__main__":
    main()
