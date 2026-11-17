"""Fair comparison follow-up, Part A (docs/fair_followup_declared_protocol.md): LDA, SVM and RF at the 28 winner cells with history in the features (stack variants none / ST3 / ST5 / CUM) and per-window class
probabilities, so that every vote candidate (hard / mean / log / ema) can be evaluated from stored arrays. Same cells, features, outer folds and estimators as run_grid.py (SVM gets probability=True,
which does not change predict()). For every variant and outer fold: the outer test decisions + probabilities, and the accuracy of every candidate on the inner out-of-fold predictions of the outer-training
windows (3-fold grouped, class-stratified inner CV, as in the earlier nested k) for the nested choice. One task = one (cell, model); one .npz per task in cache/grid/fair2/classical/; resumable.
Arrays per variant v: hard__v (n,) class index per table row, proba__v (n, C) float32, inner__v (folds, 22) candidate accuracies; plus folds, labels, info__v (JSON), seconds__v.
Usage: python -W ignore -u run_fair_classical2.py [--workers N] [--models LDA SVM RandomForest] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import CACHE_DIR
from fair_history import CANDIDATES, STACK_VARIANTS, candidate_accuracies, stack_features
from models import make_model_pipeline
from run_fair_classical import MODELS, WIN, assert_table_order, key_of, prepare_cell

OUT = CACHE_DIR / "grid" / "fair2" / "classical"


def task_path(part, rate, nch, bits, fset, model):
    return OUT / (key_of(part, rate, nch, bits, fset, model).replace("|", "__") + ".npz")


def fit_predict_proba(model, Xtr, ytr, Xte, labels):
    """Hard decisions as class indices (predict(), exactly as in the grid) and probabilities (n_te, n_classes) with columns in the order of `labels`."""
    need, est = make_model_pipeline(model)
    if model == "RandomForest":
        est.set_params(n_jobs=2)
    if model == "SVM":
        est.set_params(probability=True)
    if need:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    est.fit(Xtr, ytr)
    c2i = {c: i for i, c in enumerate(labels)}
    hard = np.array([c2i[c] for c in est.predict(Xte)])
    P = np.zeros((len(Xte), len(labels)))
    P[:, [c2i[c] for c in est.classes_]] = est.predict_proba(Xte)
    info = {}
    if model == "SVM":
        info["n_sv"] = int(est.support_vectors_.shape[0])
    elif model == "RandomForest":
        info["nodes"] = int(sum(t.tree_.node_count for t in est.estimators_))
    return hard, P, info


def run_task(part, rate, nch, bits, fset, model):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    arrays = {}
    with threadpool_limits(limits=1):
        t, X, names, keep, folds = prepare_cell(part, rate, nch, bits, fset)
        assert_table_order(t)
        y, seg = t["cue"].to_numpy(), t["segment_id"].to_numpy()
        labels = sorted(np.unique(y).tolist())
        c2i = {c: i for i, c in enumerate(labels)}
        yi = np.array([c2i[c] for c in y])
        C = len(labels)
        msg = []
        for v in STACK_VARIANTS:
            tv = time.time()
            Xv = stack_features(X, seg, v)
            hard = np.empty(len(y), dtype=int)
            proba = np.zeros((len(y), C), dtype=np.float32)
            inner_acc, infos = [], []
            for f in sorted(np.unique(folds)):
                tr, te = folds != f, folds == f
                hard[te], P, info = fit_predict_proba(model, Xv[tr], y[tr], Xv[te], labels)
                proba[te] = P
                infos.append(info)
                Xtr, ytr, gtr = Xv[tr], y[tr], seg[tr]
                ih, iP = np.empty(len(ytr), dtype=int), np.zeros((len(ytr), C))
                for a, b in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(len(ytr)), ytr, gtr):
                    ih[b], iP[b], _ = fit_predict_proba(model, Xtr[a], ytr[a], Xtr[b], labels)
                inner_acc.append(candidate_accuracies(ih, iP, yi[tr], gtr, C))
            arrays[f"hard__{v}"], arrays[f"proba__{v}"] = hard, proba
            arrays[f"inner__{v}"] = np.array(inner_acc)
            arrays[f"info__{v}"] = json.dumps(infos)
            arrays[f"seconds__{v}"] = time.time() - tv
            msg.append(f"{v} {100 * (hard == yi).mean():.2f} ({time.time() - tv:.0f}s)")
    np.savez_compressed(task_path(part, rate, nch, bits, fset, model), folds=folds, labels=np.array(labels), n_features=len(names), seconds=time.time() - t0, **arrays)
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} {model:12s} F={len(names):4d} raw acc by variant: {'; '.join(msg)} [{time.time() - t0:.0f}s]", flush=True)
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
    for model in [m for m in MODELS if m in models]:
        for pf, cfgs in winners.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                if not task_path(part, rate, nch, bits, fset, model).exists():
                    tasks.append((part, rate, nch, bits, fset, model))
    if "--smoke" in args:
        tasks = [x for x in tasks if (x[0], x[1], x[2], x[3], x[4]) == ("Wallace_S2", 1000, 8, 16, "W")][:3] if "--wide" in args else tasks[:1]
    print(f"{len(tasks)} tasks, {workers} worker(s), {len(CANDIDATES)} candidates, variants {STACK_VARIANTS}", flush=True)
    for _ in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t) for t in tasks):
        pass
    print("All done")


if __name__ == "__main__":
    main()
