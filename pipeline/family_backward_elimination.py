"""Backward elimination over the 10 leaf families (rule fixed in docs/family_ablation_declared_protocol.md): LDA, objective = mean pooled accuracy over the 8 tables,
remove the family whose removal gives the highest objective (ties within 0.05 points: the leaf with more columns first), stop before a removal that would put the objective more than 0.3 points below C.
Output: cache/wide_set_W.json (the kept leaves = wide set W, the removal order, the objective at every step) and a printed table. Needs STAGE1_TAG=fixedrange.
Usage: STAGE1_TAG=fixedrange python -W ignore -u family_backward_elimination.py [--workers N]"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_wide_tables import OPS, PARTS, TAG, table_path
from config import CACHE_DIR
from families import LEAVES, leaf_of
from models import compute_cv_folds
from run_own_feature_selection import fit_predict

CACHE = CACHE_DIR / "family_elimination_cache.json"
OUT = CACHE_DIR / "wide_set_W.json"
TABLES = [(p, o) for p in PARTS for o in OPS]
STOP_DROP = 0.3     # points below C
TIE = 0.05          # points


def lda_acc(part, op, kept):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        t = pd.read_parquet(table_path(part, op))
        cols = [c for c in t.columns[5:] if leaf_of(c) in kept]
        X, y, folds = t[cols].to_numpy(dtype=np.float64), t["cue"].to_numpy(), compute_cv_folds(t)
        pooled = np.empty_like(y)
        for f in sorted(np.unique(folds)):
            tr, te = folds != f, folds == f
            pooled[te] = fit_predict("LDA", X[tr], y[tr], X[te])
    return f"{'+'.join(sorted(kept))}|{part}|{op}", float((pooled == y).mean())


def objective(cache, kept):
    return 100 * float(np.mean([cache[f"{'+'.join(sorted(kept))}|{p}|{o}"] for p, o in TABLES]))


def ensure(cache, candidates, workers):
    todo = [(p, o, k) for k in candidates for p, o in TABLES if f"{'+'.join(sorted(k))}|{p}|{o}" not in cache]
    for key, acc in Parallel(n_jobs=workers)(delayed(lda_acc)(p, o, k) for p, o, k in todo):
        cache[key] = acc
    json.dump(cache, open(CACHE, "w"))


def main():
    assert TAG == "fixedrange"
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    ncol = {l: sum(leaf_of(c) == l for c in pd.read_parquet(table_path("P1_combined", "ref")).columns[5:]) for l in LEAVES}
    kept = set(LEAVES)
    ensure(cache, [kept], workers)
    obj_c = objective(cache, kept)
    print(f"C (all 10 families): objective {obj_c:.3f}")
    order, steps = [], [dict(removed=None, objective=obj_c, kept=sorted(kept))]
    while len(kept) > 1:
        cands = [kept - {l} for l in kept]
        ensure(cache, cands, workers)
        scored = sorted(((objective(cache, kept - {l}), l) for l in kept), reverse=True)
        top = scored[0][0]
        tied = [l for o, l in scored if o >= top - TIE]
        pick = max(tied, key=lambda l: (ncol[l], l))
        obj_after = objective(cache, kept - {pick})
        print(f"step {len(order) + 1}: candidates " + ", ".join(f"{l} {o:.2f}" for o, l in scored[:4]) + f" ... -> remove {pick} (objective {obj_after:.3f}, {obj_after - obj_c:+.3f} against C)")
        if obj_after < obj_c - STOP_DROP:
            print(f"STOP: removing {pick} would put the objective {obj_c - obj_after:.2f} points below C (limit {STOP_DROP})")
            break
        kept = kept - {pick}
        order.append(pick)
        steps.append(dict(removed=pick, objective=obj_after, kept=sorted(kept)))
    W = [l for l in LEAVES if l in kept]
    cols_p1 = sum(ncol[l] for l in W)
    json.dump(dict(kept_leaves=W, removal_order=order, steps=steps, objective_C=obj_c, objective_W=objective(cache, kept),
                   columns_P1_ref=cols_p1, columns_C_P1_ref=sum(ncol.values()), rule=dict(stop_drop=STOP_DROP, tie=TIE)), open(OUT, "w"), indent=1)
    print(f"\nW = {W}\nremoved in order: {order}\nobjective W {objective(cache, kept):.3f} vs C {obj_c:.3f}; columns at P1 ref: W {cols_p1} of {sum(ncol.values())}")


if __name__ == "__main__":
    main()
