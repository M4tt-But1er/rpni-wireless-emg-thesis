"""Fair LSTM comparison, classical side (docs/fair_lstm_declared_protocol.md): LDA, SVM and RF at the 28 winner cells of the grid, keeping the raw per-window out-of-fold predictions (same features, same
channel slice, same outer folds, same estimators as run_grid.py via run_grid.fit_predict_info), plus the nested choice of the vote span k in every outer fold (3-fold grouped, class-stratified inner CV on the
outer-training windows). The votes themselves are applied in analyse_fair.py from the stored raw predictions (fair_vote.causal_vote), so every span can be evaluated without refitting.
One task = one (cell, model); output one .npz per task in cache/grid/fair/classical/ (arrays: pooled raw predictions per table row, outer fold per row, chosen k per outer fold (0 = all), inner accuracy per
fold and candidate); resumable.
Usage: python -W ignore -u run_fair_classical.py [--workers N] [--models LDA SVM RandomForest] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from build_wide_tables import channels_of
from config import CACHE_DIR
from fair_vote import choose_k, segment_starts
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import table_path
from models import compute_cv_folds
from run_bitrate_search import get_ranking_and_grid
from run_grid import fit_predict_info
from wide_features import set_columns

OUT = CACHE_DIR / "grid" / "fair" / "classical"
WIN = CACHE_DIR / "grid" / "winners.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
MODELS = ["LDA", "SVM", "RandomForest"]


def key_of(part, rate, nch, bits, fset, model):
    return f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{model}"


def prepare_cell(part, rate, nch, bits, fset):
    """Same construction as run_grid_networks.prepare_cell (copied so that no torch import is needed; V10a proves the features are the grid's)."""
    W = set(json.load(open(W_PATH))["kept_leaves"])
    t = pd.read_parquet(table_path(part, rate, bits))
    cols = list(t.columns[5:])
    a_cols = set(set_columns(cols, "A"))
    chosen = [c for c in cols if (c in a_cols if fset == "A" else leaf_of(c) in W)]
    keep = get_ranking_and_grid(part)[0][:nch]
    X, names = slice_matrix(t, chosen, channels_of(part), keep)
    return t, X, names, list(keep), compute_cv_folds(t)


def assert_table_order(t):
    """V10e (part): windows of a segment are contiguous and strictly increasing in time."""
    seg, tm = t["segment_id"].to_numpy(), t["window_time_s"].to_numpy()
    b = segment_starts(seg)
    assert len(b) - 1 == len(np.unique(seg)), "segments are not contiguous in the table"
    for i, j in zip(b[:-1], b[1:]):
        assert (np.diff(tm[i:j]) > 0).all(), "windows of a segment are not in time order"


def task_path(part, rate, nch, bits, fset, model):
    return OUT / (key_of(part, rate, nch, bits, fset, model).replace("|", "__") + ".npz")


def run_task(part, rate, nch, bits, fset, model):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    with threadpool_limits(limits=1):
        t, X, names, keep, folds = prepare_cell(part, rate, nch, bits, fset)
        assert_table_order(t)
        y, seg = t["cue"].to_numpy(), t["segment_id"].to_numpy()
        labels = sorted(np.unique(y).tolist())
        c2i = {c: i for i, c in enumerate(labels)}
        yi = np.array([c2i[c] for c in y])
        pooled = np.empty_like(y)
        chosen_k, inner_acc = [], []
        for f in sorted(np.unique(folds)):
            tr, te = folds != f, folds == f
            pooled[te], _ = fit_predict_info(model, X[tr], y[tr], X[te])
            Xtr, ytr, gtr = X[tr], y[tr], seg[tr]
            inner = np.empty_like(ytr)
            for a, b in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(len(ytr)), ytr, gtr):
                inner[b], _ = fit_predict_info(model, Xtr[a], ytr[a], Xtr[b])
            k, accs = choose_k(np.array([c2i[c] for c in inner]), yi[tr], gtr, len(labels))
            chosen_k.append(0 if k is None else k)
            inner_acc.append(accs)
    np.savez_compressed(task_path(part, rate, nch, bits, fset, model), pooled=pooled, folds=folds, chosen_k=np.array(chosen_k), inner_acc=np.array(inner_acc), labels=np.array(labels),
                        n_features=len(names), seconds=time.time() - t0)
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} {model:12s} F={len(names):4d} raw acc {100 * (pooled == y).mean():.2f}  chosen k per fold {chosen_k} ({time.time() - t0:.0f}s)", flush=True)
    return key_of(part, rate, nch, bits, fset, model)


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    models = []
    if "--models" in args:
        for a in args[args.index("--models") + 1:]:
            if a.startswith("--"):
                break
            models.append(a)
    models = models or MODELS
    OUT.mkdir(parents=True, exist_ok=True)
    winners = json.load(open(WIN))
    tasks = []
    for model in [m for m in MODELS if m in models]:                 # LDA first, then SVM, then RF
        for pf, cfgs in winners.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                if not task_path(part, rate, nch, bits, fset, model).exists():
                    tasks.append((part, rate, nch, bits, fset, model))
    if "--smoke" in args:
        tasks = tasks[:1]
    print(f"{len(tasks)} tasks, {workers} worker(s)", flush=True)
    for _ in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks):
        pass
    print("All done")


if __name__ == "__main__":
    main()
