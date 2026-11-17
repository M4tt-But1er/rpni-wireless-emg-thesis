"""EXPLORATORY (docs/rpni_vs_muscle_declared_protocol.md), tests 1, 2 and 4: RPNI-only vs every equal-size muscle-only subset (30 kHz),
per-movement recall of every channel alone (30 kHz), and single-channel accuracy on every grid table (5 rates x 3 bit depths) plus 30 kHz.
Same data, subsets, models and folds as run_channel_importance.py. Nothing existing is overwritten: output cache/rpni_vs_muscle/evals.json
keyed "<part>|<table>|<model>|<subset>" -> {accuracy, fold_accs, labels, recall, n_features} (resumable, saved after every task).
Usage: python -W ignore -u run_rpni_vs_muscle.py [--workers N]"""
import itertools
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import BITS, RATES, table_path
from models import compute_cv_folds
from run_channel_importance import RPNI, skey

OUT_DIR = CACHE_DIR / "rpni_vs_muscle"
EVALS = OUT_DIR / "evals.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
MODELS = ["LDA", "SVM", "RandomForest"]
T30 = "30000Hz_16bit"
TABLES = [T30] + [f"{r}Hz_{b}bit" for r in RATES for b in BITS]


def tpath(part, table):
    if table == T30:
        return CACHE_DIR / "baseline30k" / "tables" / f"{part}__{T30}.parquet"
    r, b = table.replace("Hz", "").replace("bit", "").split("_")
    return table_path(part, int(r), int(b))


def muscle_subsets(part):
    rp = RPNI[part]
    mus = [c for c in channels_of(part) if c not in rp]
    return [list(s) for s in itertools.combinations(mus, len(rp))]


def run_task(part, table, model, keys):
    """All channel subsets `keys` of one (participant, table, model)."""
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    import run_grid as G
    out = {}
    with threadpool_limits(limits=1):
        W = set(json.load(open(W_PATH))["kept_leaves"])
        t = pd.read_parquet(tpath(part, table))
        chosen = [c for c in t.columns[5:] if leaf_of(c) in W]
        y, folds = t["cue"].to_numpy(), compute_cv_folds(t)
        labels = sorted(np.unique(y).tolist())
        for key in keys:
            X, names = slice_matrix(t, chosen, channels_of(part), key.split("+"))
            pooled = np.empty_like(y)
            accs = []
            for f in sorted(np.unique(folds)):
                tr, te = folds != f, folds == f
                pred, _ = G.fit_predict_info(model, X[tr], y[tr], X[te])
                pooled[te] = pred
                accs.append(float((pred == y[te]).mean()))
            recall = [float((pooled[y == c] == c).mean()) for c in labels]
            out[f"{part}|{table}|{model}|{key}"] = dict(accuracy=float((pooled == y).mean()), fold_accs=accs, labels=[int(c) for c in labels],
                                                        recall=recall, n_features=len(names))
    return out


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.load(open(EVALS)) if EVALS.exists() else {}
    tasks = []
    for part in PARTS:
        singles = [skey(part, [c]) for c in channels_of(part)]
        for table in TABLES:
            keys = list(singles)
            if table == T30:     # test 1 subsets
                keys += [skey(part, RPNI[part])] + [skey(part, s) for s in muscle_subsets(part)]
            for m in MODELS:
                todo = [k for k in keys if f"{part}|{table}|{m}|{k}" not in d]
                if todo:
                    tasks.append((part, table, m, todo))
    # 30 kHz first (tests 1 and 2), RF first within a table group (longest)
    tasks.sort(key=lambda a: (a[1] != T30, a[2] != "RandomForest", -len(a[3])))
    print(f"{len(tasks)} tasks, {sum(len(a[3]) for a in tasks)} evaluations, {workers} workers", flush=True)
    t0 = time.time()
    for i, r in enumerate(Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*a) for a in tasks), 1):
        d.update(r)
        json.dump(d, open(EVALS, "w"))
        k = next(iter(r))
        print(f"  [{i}/{len(tasks)}] {k.rsplit('|', 1)[0]} ({len(r)} subsets)  {time.time() - t0:.0f}s", flush=True)
    print("All done ->", EVALS, flush=True)


if __name__ == "__main__":
    main()
