"""LSTM pilot (docs/fair_followup_declared_protocol.md, B2 and amendment A2): configurations on the 8 reference cells (1000 Hz, 8 channels, 16 bit, one per patient/set), per outer fold, on the old inner split
(a stratified hold-out of the outer-training sequences). Only inner-validation numbers are stored (best epoch, validation loss and accuracy at the loss-optimal epoch, epochs run, seconds); NO outer-test
prediction is made. Round 1: the 35 configurations of lstm_tuning.all_configs (3 old + 32 random) -> cache/grid/fair2/lstm_pilot/; round 2 (--round 2): the 32 new random configurations of
lstm_tuning.random_configs_r2 -> cache/grid/fair2/lstm_pilot_r2/. One task = (cell, outer fold) -> <cell>__f<k>.json; resumable.
Usage: python -W ignore -u run_lstm_pilot.py [--round 1|2] [--workers N] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

import lstm_tuning as T
import run_grid_networks as R
import run_lstm_training_check as L
from config import CACHE_DIR
from run_fair_classical import assert_table_order

OUT = CACHE_DIR / "grid" / "fair2" / "lstm_pilot"                    # round 1
OUT_R2 = CACHE_DIR / "grid" / "fair2" / "lstm_pilot_r2"              # round 2
REF = (1000, 8, 16)


def out_dir(rnd):
    return OUT if rnd == 1 else OUT_R2


def round_configs(rnd):
    return T.all_configs() if rnd == 1 else T.random_configs_r2()


def pilot_cells():
    w = json.load(open(R.WIN))
    return [(pf.split("|")[0], *REF, pf.split("|")[1]) for pf, cfgs in w.items() if list(REF) in [list(c) for c in cfgs]]


def task_path(part, rate, nch, bits, fset, fold, rnd=1):
    return out_dir(rnd) / (R.key_of(part, rate, nch, bits, fset, "pilot").replace("|", "__") + f"__f{fold}.json")


def inner_split(tr, n_classes):
    cues = np.array([c for _, _, c in tr])
    val_size = max(n_classes, int(np.ceil(0.2 * len(tr))))
    a, b = next(StratifiedShuffleSplit(1, test_size=val_size, random_state=L.SEED).split(np.zeros(len(tr)), cues))
    return [tr[i] for i in a], [tr[i] for i in b]


def run_task(part, rate, nch, bits, fset, fold, threads, rnd=1):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t, X, names, keep, folds = R.prepare_cell(part, rate, nch, bits, fset)
        assert_table_order(t)
        df, feats, classes, c2i, by_id, seg_fold = R.lstm_loaded(t, X, names, folds)
        tr, _, _, _ = L.fold_data(by_id, seg_fold, fold)
        itr, iva = inner_split(tr, len(classes))
        res = {}
        for name, cfg in round_configs(rnd).items():
            _, info = T.train_cfg(itr, c2i, len(classes), len(feats), cfg, val_seqs=iva)
            res[name] = info
        out = dict(cell=[part, rate, nch, bits, fset], fold=int(fold), round=rnd, n_train_seqs=len(itr), n_val_seqs=len(iva), n_features=len(feats), results=res, seconds=time.time() - t0)
    task_path(part, rate, nch, bits, fset, fold, rnd).write_text(json.dumps(out))
    best = max(res.items(), key=lambda kv: kv[1]["val_acc"])
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} fold {fold}: {len(res)} configs, best val acc {best[0]} {best[1]['val_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return fold


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    rnd = int(args[args.index("--round") + 1]) if "--round" in args else 1
    out_dir(rnd).mkdir(parents=True, exist_ok=True)
    tasks = [(*c, f) for c in pilot_cells() for f in range(5) if not task_path(*c, f, rnd).exists()]
    if "--smoke" in args:
        tasks = tasks[:1]
    print(f"round {rnd}: {len(tasks)} tasks ({len(pilot_cells())} pilot cells x 5 folds), {workers} worker(s), {len(round_configs(rnd))} configs", flush=True)
    for _ in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t, 1, rnd) for t in tasks):
        pass
    print("All done")


if __name__ == "__main__":
    main()
