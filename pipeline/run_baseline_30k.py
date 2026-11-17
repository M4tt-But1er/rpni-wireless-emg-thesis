"""X1 of docs/three_experiments_declared_protocol.md: baseline on the full-rate stream (30 kHz, all channels, 16 bit) with the wide set W.

Step 1 builds one all-channel table per participant with the grid's own code at rate 30000 (cache/baseline30k/tables/<part>__30000Hz_16bit.parquet).
Step 2 runs LDA, SVM, RF (grid settings), RBFNN default and tuned, LSTM default and tuned on set W with the grid's grouped 5-fold CV, and the
causal hard vote (all decisions since the onset) on LDA, SVM and RF as a secondary result.
Output: cache/baseline30k/results.json keyed "<part>|<method>" (resumable), pooled predictions in cache/baseline30k/pred_<part>_<method>.npy.
Usage: python -W ignore -u run_baseline_30k.py [--workers N] [--tables-only] [--parts ...] [--methods ...]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from fair_vote import causal_vote
from feature_cost import extraction_cost
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import build_table, decimated_sources, wallace_ranges
from models import RBFNetClassifier, compute_cv_folds
from run_bitrate_search import get_ranking_and_grid
from run_models_norest import compute_metrics

RATE, BITS = 30000, 16
OUT_DIR = CACHE_DIR / "baseline30k"
TAB_DIR = OUT_DIR / "tables"
RES = OUT_DIR / "results.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
METHODS = ["LDA", "SVM", "RandomForest", "RBFNN_default", "RBFNN_tuned", "LSTM_default", "LSTM_tuned"]


def tab_path(part):
    return TAB_DIR / f"{part}__{RATE}Hz_{BITS}bit.parquet"


def build(part):
    warnings.filterwarnings("ignore")
    if tab_path(part).exists():
        return f"{part}: exists"
    t0 = time.time()
    srcs = decimated_sources(part, RATE)
    ranges = wallace_ranges(srcs, channels_of(part)) if part.startswith("Wallace") else None
    out, bad = build_table(srcs, part, RATE, BITS, ranges)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(tab_path(part), index=False)
    return f"{part}: {len(out)} windows, {out.shape[1] - 5} columns, {time.time() - t0:.0f}s" + (f", non-finite set to 0 in {len(bad)} columns" if bad else "")


def prepare(part):
    W = set(json.load(open(W_PATH))["kept_leaves"])
    t = pd.read_parquet(tab_path(part))
    cols = list(t.columns[5:])
    chosen = [c for c in cols if leaf_of(c) in W]
    keep = get_ranking_and_grid(part)[0]                      # all channels, in ranking order (as the grid's all-channel cells)
    keep = keep[:len(channels_of(part))]
    X, names = slice_matrix(t, chosen, channels_of(part), keep)
    return t, X, names, list(keep), compute_cv_folds(t)


def run(part, method, threads):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    import run_grid as G
    import run_grid_networks as N
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t, X, names, keep, folds = prepare(part)
        y = t["cue"].to_numpy()
        labels = sorted(np.unique(y).tolist())
        extra, order_ok = {}, True
        if method in ("LDA", "SVM", "RandomForest", "RBFNN_default"):
            pooled = np.empty_like(y)
            accs = []
            for f in sorted(np.unique(folds)):
                tr, te = folds != f, folds == f
                if method == "RBFNN_default":
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(X[tr])
                    pred = RBFNetClassifier(n_centers=64, C=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
                else:
                    pred, _ = G.fit_predict_info(method, X[tr], y[tr], X[te])
                pooled[te] = pred
                accs.append(float((pred == y[te]).mean()))
            if method != "RBFNN_default":                     # secondary: causal hard vote over all decisions since the onset
                c2i = {c: i for i, c in enumerate(labels)}
                pi = np.array([c2i[c] for c in pooled])
                yi = np.array([c2i[c] for c in y])
                vi = causal_vote(pi, t["segment_id"].to_numpy(), None, len(labels))
                extra["vote_all"] = dict(accuracy=float((vi == yi).mean()),
                                         fold_accs=[float((vi[folds == f] == yi[folds == f]).mean()) for f in sorted(np.unique(folds))])
            y_true, y_pred = y, pooled
        elif method == "RBFNN_tuned":
            y_true, y_pred, accs, ex = N.rbfnn_tuned(t, X, folds)
            extra.update(ex)
        else:
            loaded = N.lstm_loaded(t, X, names, folds)
            y_true, y_pred, accs, ex, labels = (N.lstm_default if method == "LSTM_default" else N.lstm_tuned)(loaded)
            extra.update(ex)
            order_ok = False
        m = compute_metrics(y_true, y_pred, labels)
        rec = {"fold_accs": accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"],
               "confusion_matrix": m["confusion_matrix"], "class_labels": m["class_labels"], "n_features": len(names), "n_windows": int(len(t)),
               "extraction": extraction_cost(names, channels_of(part), RATE), "bitrate_bps": RATE * len(keep) * BITS, "channels": keep,
               "seconds": time.time() - t0, "pred_in_table_order": order_ok}
        rec.update(extra)
        np.save(OUT_DIR / f"pred_{part}_{method}.npy", np.asarray(y_pred))
    v = f", vote {100 * extra['vote_all']['accuracy']:.2f}" if "vote_all" in extra else ""
    print(f"  {part} {method:14s} F={len(names):4d} acc {100 * rec['accuracy']:.2f}{v} ({rec['seconds']:.0f}s)", flush=True)
    return f"{part}|{method}", rec


def main():
    args = sys.argv[1:]

    def opt(name, default, many=False, cast=str):
        if name not in args:
            return default
        vals = []
        for a in args[args.index(name) + 1:]:
            if a.startswith("--"):
                break
            vals.append(cast(a))
        return vals if many else vals[0]

    workers = opt("--workers", 2, cast=int)
    parts = opt("--parts", PARTS, many=True)
    methods = opt("--methods", METHODS, many=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for msg in Parallel(n_jobs=min(workers, 2))(delayed(build)(p) for p in parts):   # 30 kHz sources are large: at most 2 at once
        print(msg, flush=True)
    if "--tables-only" in args:
        return
    results = json.load(open(RES)) if RES.exists() else {}
    tasks = [(p, m) for m in methods for p in parts if f"{p}|{m}" not in results]
    threads = max(1, 15 // workers)
    print(f"{len(tasks)} tasks, {workers} worker(s), {threads} thread(s) each", flush=True)
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run)(p, m, threads) for p, m in tasks):
        results[key] = rec
        with open(RES, "w") as fh:
            json.dump(results, fh)
    print("All done")


if __name__ == "__main__":
    main()
