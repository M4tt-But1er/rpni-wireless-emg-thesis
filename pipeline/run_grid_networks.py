"""Step 3 of docs/grid_declared_protocol.md: tuned RBFNN, LSTM (headline setting) and tuned LSTM at the Pareto winners of the grid (cache/grid/winners.json, written by analyse_grid.py).
Same all-channel tables, channel slices, feature sets and folds as run_grid.py. One task = one (winner, method). Output: cache/grid/networks_results.json keyed
"<patient>|<rate>Hz|<n>ch|<bits>bit|<set>|<method>" with method in RBFNN_tuned, LSTM_default, LSTM_tuned; record layout as in the grid plus method details. Resumable per task.
Usage: python -W ignore -u run_grid_networks.py [--workers N] [--methods RBFNN_tuned LSTM_default LSTM_tuned] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import run_lstm_training_check as L
from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from feature_cost import extraction_cost
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import table_path
from models import RBFNetClassifier, compute_cv_folds
from run_bitrate_search import get_ranking_and_grid
from run_models_norest import compute_metrics
from run_rbfnn_tuning import tune_rbfnn
from wide_features import set_columns

OUT = CACHE_DIR / "grid" / "networks_results.json"
WIN = CACHE_DIR / "grid" / "winners.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
METHODS = ["RBFNN_tuned", "LSTM_default", "LSTM_tuned"]


def key_of(part, rate, nch, bits, fset, method):
    return f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{method}"


def prepare_cell(part, rate, nch, bits, fset):
    W = set(json.load(open(W_PATH))["kept_leaves"])
    t = pd.read_parquet(table_path(part, rate, bits))
    cols = list(t.columns[5:])
    a_cols = set(set_columns(cols, "A"))
    chosen = [c for c in cols if (c in a_cols if fset == "A" else leaf_of(c) in W)]
    keep = get_ranking_and_grid(part)[0][:nch]
    X, names = slice_matrix(t, chosen, channels_of(part), keep)
    return t, X, names, list(keep), compute_cv_folds(t)


def base_record(t, names, keep, rate, nch, bits, part, y_true, y_pred, labels, fold_accs, seconds, extra):
    m = compute_metrics(y_true, y_pred, labels)
    rec = {"fold_accs": fold_accs, "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"], "cohen_kappa": m["cohen_kappa"], "confusion_matrix": m["confusion_matrix"],
           "class_labels": m["class_labels"], "n_features": len(names), "n_windows": int(len(t)), "extraction": extraction_cost(names, channels_of(part), rate),
           "bitrate_bps": rate * nch * bits, "channels": keep, "seconds": seconds}
    rec.update(extra)
    return rec


def rbfnn_tuned(t, X, folds):
    y, groups = t["cue"].to_numpy(), t["segment_id"].to_numpy()
    pooled = np.empty_like(y)
    accs, chosen = [], []
    for f in sorted(np.unique(folds)):
        tr, te = folds != f, folds == f
        (n_c, s, C), inner, _ = tune_rbfnn(X[tr], y[tr], groups[tr])
        sc = StandardScaler().fit(X[tr])
        clf = RBFNetClassifier(n_centers=n_c, C=C, sigma_scale=s).fit(sc.transform(X[tr]), y[tr])
        pooled[te] = clf.predict(sc.transform(X[te]))
        accs.append(float((pooled[te] == y[te]).mean()))
        chosen.append(dict(n_centers=int(n_c), sigma_scale=float(s), C=float(C), inner_cv_acc=float(inner)))
    return y, pooled, accs, {"fold_info": chosen}


def lstm_loaded(t, X, names, folds):
    df = pd.concat([t.iloc[:, :5].reset_index(drop=True), pd.DataFrame(X, columns=names)], axis=1)
    return L.prepare(df, names, folds)


def lstm_default(loaded):
    df, feats, classes, c2i, by_id, seg_fold = loaded
    i2c = {i: c for c, i in c2i.items()}
    true_all, pred_all, accs = [], [], []
    for fold in sorted(seg_fold.unique()):
        tr, te, _, _ = L.fold_data(by_id, seg_fold, fold)
        model, _ = L.train(tr, {}, c2i, len(classes), len(feats), 40, **L.CONFIGS["base"])
        acc, _, pred, true = L.evaluate(model, te, c2i, len(classes))
        accs.append(float(acc))
        true_all.append(true)
        pred_all.append(pred)
    return (np.array([i2c[i] for i in np.concatenate(true_all)]), np.array([i2c[i] for i in np.concatenate(pred_all)]), accs,
            {"fold_info": [dict(config="base", epochs=40)] * len(accs)}, classes)


def lstm_tuned(loaded):
    """Same procedure as run_lstm_training_check.part2 (config and epoch chosen on a 20% validation split of the outer-training sequences), but returns the pooled predictions."""
    df, feats, classes, c2i, by_id, seg_fold = loaded
    n_classes = len(classes)
    i2c = {i: c for c, i in c2i.items()}
    true_all, pred_all, accs, info = [], [], [], []
    for fold in sorted(seg_fold.unique()):
        tr, te, _, _ = L.fold_data(by_id, seg_fold, fold)
        cues = np.array([c for _, _, c in tr])
        val_size = max(n_classes, int(np.ceil(0.2 * len(tr))))
        itr_idx, iva_idx = next(StratifiedShuffleSplit(1, test_size=val_size, random_state=L.SEED).split(np.zeros(len(tr)), cues))
        itr, iva = [tr[i] for i in itr_idx], [tr[i] for i in iva_idx]
        best = None
        for cname, cfg in L.CONFIGS.items():
            _, hist = L.train(itr, {"val": iva}, c2i, n_classes, len(feats), L.MAX_EPOCHS, eval_every=5, **cfg)
            r_best = min([r for r in hist if "val_loss" in r], key=lambda r: r["val_loss"])
            if best is None or r_best["val_loss"] < best[2]:
                best = (cname, r_best["epoch"], r_best["val_loss"])
        cname, epoch, _ = best
        model, _ = L.train(tr, {}, c2i, n_classes, len(feats), epoch, **L.CONFIGS[cname])
        acc, _, pred, true = L.evaluate(model, te, c2i, n_classes)
        accs.append(float(acc))
        true_all.append(true)
        pred_all.append(pred)
        info.append(dict(config=cname, epochs=int(epoch)))
    return (np.array([i2c[i] for i in np.concatenate(true_all)]), np.array([i2c[i] for i in np.concatenate(pred_all)]), accs, {"fold_info": info}, classes)


def run_task(part, rate, nch, bits, fset, method, threads):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t, X, names, keep, folds = prepare_cell(part, rate, nch, bits, fset)
        labels = sorted(t["cue"].unique().tolist())
        if method == "RBFNN_tuned":
            y, pred, accs, extra = rbfnn_tuned(t, X, folds)
        else:
            loaded = lstm_loaded(t, X, names, folds)
            y, pred, accs, extra, labels = (lstm_default if method == "LSTM_default" else lstm_tuned)(loaded)
        rec = base_record(t, names, keep, rate, nch, bits, part, y, pred, labels, accs, time.time() - t0, extra)
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} {method:12s} F={len(names):4d} acc {100 * rec['accuracy']:.2f} ({rec['seconds']:.0f}s)", flush=True)
    return key_of(part, rate, nch, bits, fset, method), rec


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    methods = []
    if "--methods" in args:
        for a in args[args.index("--methods") + 1:]:
            if a.startswith("--"):
                break
            methods.append(a)
    methods = methods or METHODS
    winners = json.load(open(WIN))
    results = json.load(open(OUT)) if OUT.exists() else {}
    tasks = []
    for method in methods:
        for pf, cfgs in winners.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                if key_of(part, rate, nch, bits, fset, method) not in results:
                    tasks.append((part, rate, nch, bits, fset, method))
    if "--smoke" in args:
        tasks = [t for t in tasks if t[5] == "LSTM_default"][:1] + [t for t in tasks if t[5] == "LSTM_tuned"][:1]
    threads = max(1, 15 // workers)
    print(f"{len(tasks)} tasks, {workers} worker(s), {threads} thread(s) each", flush=True)
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t, threads) for t in tasks):
        results[key] = rec
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(results, fh)
    print("All done")


if __name__ == "__main__":
    main()
