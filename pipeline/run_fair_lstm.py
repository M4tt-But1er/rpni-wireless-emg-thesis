"""Fair LSTM comparison, LSTM side (docs/fair_lstm_declared_protocol.md): rerun LSTM_default and LSTM_tuned at the 28 winner cells with exactly the code of run_grid_networks.py (same features, folds,
training, nested tuning), but keep the per-window out-of-fold predictions, mapped back to the rows of the cell's table (V10e: labels and order are asserted). The first run (cache/grid/networks_results.json)
kept only aggregates. One task = one (cell, method); output one .npz per task in cache/grid/fair/lstm/ (pred per table row = predicted cue, folds per row, fold accuracies, fold info as JSON); resumable.
Usage: python -W ignore -u run_fair_lstm.py [--workers N] [--methods LSTM_default LSTM_tuned] [--smoke]   (the first run used 3 workers x 5 threads)"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

import run_grid_networks as R
from config import CACHE_DIR
from fair_vote import segment_starts
from run_fair_classical import assert_table_order

OUT = CACHE_DIR / "grid" / "fair" / "lstm"
METHODS = ["LSTM_default", "LSTM_tuned"]


def task_path(part, rate, nch, bits, fset, method):
    return OUT / (R.key_of(part, rate, nch, bits, fset, method).replace("|", "__") + ".npz")


def run_task(part, rate, nch, bits, fset, method, threads):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t, X, names, keep, folds = R.prepare_cell(part, rate, nch, bits, fset)
        assert_table_order(t)
        loaded = R.lstm_loaded(t, X, names, folds)
        y_true, y_pred, accs, extra, classes = (R.lstm_default if method == "LSTM_default" else R.lstm_tuned)(loaded)
        _, _, _, _, by_id, seg_fold = loaded
        seg = t["segment_id"].to_numpy()
        b = segment_starts(seg)
        rows_of = {seg[i]: np.arange(i, j) for i, j in zip(b[:-1], b[1:])}
        order = np.concatenate([rows_of[s] for fold in sorted(seg_fold.unique()) for s in seg_fold.index if seg_fold[s] == fold])
        cue = t["cue"].to_numpy()
        assert len(order) == len(t) == len(y_true) and len(np.unique(order)) == len(t), "row mapping is not a permutation of the table"
        assert (cue[order] == y_true).all(), "labels do not match the table order (V10e)"
        pred = np.empty_like(y_true)
        pred[order] = y_pred
    np.savez_compressed(task_path(part, rate, nch, bits, fset, method), pred=pred, folds=folds, fold_accs=np.array(accs), fold_info=json.dumps(extra), seconds=time.time() - t0)
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} {method:12s} F={len(names):4d} acc {100 * (pred == cue).mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
    return R.key_of(part, rate, nch, bits, fset, method)


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
    OUT.mkdir(parents=True, exist_ok=True)
    winners = json.load(open(R.WIN))
    tasks = []
    for method in [m for m in METHODS if m in methods]:             # default (seconds) before tuned (minutes)
        for pf, cfgs in winners.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                if not task_path(part, rate, nch, bits, fset, method).exists():
                    tasks.append((part, rate, nch, bits, fset, method))
    if "--smoke" in args:
        tasks = tasks[:1]
    threads = max(1, 15 // workers)
    print(f"{len(tasks)} tasks, {workers} worker(s), {threads} thread(s) each", flush=True)
    for _ in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t, threads) for t in tasks):
        pass
    print("All done")


if __name__ == "__main__":
    main()
