"""Channel importance (docs/channel_importance_declared_protocol.md): LOCO, single channel, backward elimination, grid-order prefixes and the
RPNI group test, on the E1 stream (30 kHz, all channels, 16 bit, set W) with LDA, SVM and RF (grid settings) and the grid's grouped 5-fold CV.
Every (participant, channel subset, model) evaluation is stored once in cache/channel_importance/evals.json (resumable) and reused by all measures.
Usage: python -W ignore -u run_channel_importance.py [--workers N]"""
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
from models import compute_cv_folds
from run_bitrate_search import get_ranking_and_grid

OUT_DIR = CACHE_DIR / "channel_importance"
EVALS = OUT_DIR / "evals.json"
TAB = CACHE_DIR / "baseline30k" / "tables"
W_PATH = CACHE_DIR / "wide_set_W.json"
MODELS = ["LDA", "SVM", "RandomForest"]
RPNI = {"P1_combined": ["RPNI_ulnar", "RPNI_median"], "P2_combined": ["RPNI_median", "RPNI_ulnar1", "RPNI_ulnar2"],
        "Wallace_S1": ["Median1", "Ulnar1", "Ulnar2"], "Wallace_S2": ["Radial", "Median1", "Median2", "Median3", "Median4"]}
_cache = {}


def grid_order(part):
    return get_ranking_and_grid(part)[0][:len(channels_of(part))]


def skey(part, subset):
    """Canonical subset key: channels in the participant's own channel order."""
    s = set(subset)
    return "+".join(c for c in channels_of(part) if c in s)


def load(part):
    if part not in _cache:
        W = set(json.load(open(W_PATH))["kept_leaves"])
        t = pd.read_parquet(TAB / f"{part}__30000Hz_16bit.parquet")
        chosen = [c for c in t.columns[5:] if leaf_of(c) in W]
        _cache[part] = (t, chosen, t["cue"].to_numpy(), compute_cv_folds(t))
    return _cache[part]


def evaluate(part, key, model):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    import run_grid as G
    t0 = time.time()
    with threadpool_limits(limits=1):
        t, chosen, y, folds = load(part)
        X, names = slice_matrix(t, chosen, channels_of(part), key.split("+"))
        pooled = np.empty_like(y)
        accs = []
        for f in sorted(np.unique(folds)):
            tr, te = folds != f, folds == f
            pred, _ = G.fit_predict_info(model, X[tr], y[tr], X[te])
            pooled[te] = pred
            accs.append(float((pred == y[te]).mean()))
    return f"{part}|{model}|{key}", dict(accuracy=float((pooled == y).mean()), fold_accs=accs, n_features=len(names), seconds=round(time.time() - t0, 1))


class Store:
    def __init__(self, workers):
        self.workers = workers
        self.d = json.load(open(EVALS)) if EVALS.exists() else {}

    def ensure(self, part, subsets):
        keys = sorted({skey(part, s) for s in subsets})
        todo = [(part, k, m) for k in keys for m in MODELS if f"{part}|{m}|{k}" not in self.d]
        if todo:
            t0 = time.time()
            for k, r in Parallel(n_jobs=self.workers, return_as="generator_unordered")(delayed(evaluate)(*a) for a in todo):
                self.d[k] = r
            json.dump(self.d, open(EVALS, "w"))
            print(f"    {part}: {len(todo)} evaluations in {time.time() - t0:.0f}s", flush=True)

    def mean(self, part, subset):
        return float(np.mean([self.d[f"{part}|{m}|{skey(part, subset)}"]["accuracy"] for m in MODELS]))


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S = Store(workers)
    plan = {}
    for part in PARTS:
        ch = channels_of(part)
        order = grid_order(part)
        rp = RPNI[part]
        mus = [c for c in ch if c not in rp]
        assert set(rp) <= set(ch) and set(order) == set(ch), part
        print(f"=== {part}: {len(ch)} channels ({len(rp)} RPNI)", flush=True)
        full = list(ch)
        loco = [[c for c in ch if c != x] for x in ch]
        single = [[x] for x in ch]
        prefixes = [order[:n] for n in range(1, len(ch) + 1)]
        controls = [[c for c in ch if c not in drop] for drop in itertools.combinations(mus, len(rp))]
        S.ensure(part, [full] + loco + single + prefixes + controls + [mus, rp])
        # backward elimination on the mean accuracy over the three models
        kept, removed, curve = list(ch), [], [(len(ch), S.mean(part, ch))]
        while len(kept) > 1:
            cands = [[c for c in kept if c != x] for x in kept]
            S.ensure(part, cands)
            best = max(kept, key=lambda x: (S.mean(part, [c for c in kept if c != x]), order.index(x)))
            kept.remove(best)
            removed.append(best)
            curve.append((len(kept), S.mean(part, kept)))
        keep_order = kept + removed[::-1]
        plan[part] = dict(channels=ch, rpni=rp, grid_order=order, be_removed=removed, be_keep_order=keep_order, be_curve=curve,
                          controls=[skey(part, s) for s in controls])
        print(f"  keep order (backward elimination): {keep_order}", flush=True)
        print(f"  grid order:                        {order}", flush=True)
        json.dump(plan, open(OUT_DIR / "plan.json", "w"), indent=1)
    print("All done ->", OUT_DIR)


if __name__ == "__main__":
    main()
