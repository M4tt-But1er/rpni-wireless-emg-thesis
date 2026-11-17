"""An independent feature-selection method, evaluated fully nested on the four participants,
compared fold-by-fold with the project's 3-stage cascade.

DESIGN (fixed on 7 Nov 2026 BEFORE any result of this method was seen; not tuned afterwards):

  Candidate pool   all 24 features x every channel (192 columns; 288 for Wallace S2), exactly the cached
                   movement-only tables Vaskov P1/P2 and Wallace S1/S2 (cache/*_noRest.parquet).
  Ranking          mRMR, "F-test / correlation quotient" form (Ding and Peng 2005): the first feature is the one with the
                   largest ANOVA F against the posture label; every next feature maximises
                       F_j / (mean |r| between j and the features already chosen + 0.001).
                   Relevance (F) and redundancy (r) are both scale-free, so no scaling is needed for the ranking.
  Number of features K
                   chosen PER MODEL from the grid {5, 8, 12, 16, 20, 30, 40, 60} by an inner 3-fold segment-grouped,
                   class-stratified CV on the OUTER TRAINING FOLD, scoring the real model (not an LDA proxy).
                   The ranking is recomputed inside each inner training split. Best mean inner accuracy wins; ties go to the smaller K.
  Evaluation       the same 5 outer folds as every other result in this project (models.compute_cv_folds), the same model
                   settings (models.make_model_pipeline), accuracy / weighted F1 / kappa as in run_models_norest.py.
                   Everything (ranking, K, scaler, model) is fitted on the outer training fold only.
  Models           LDA, RandomForest, SVM, RBFNN.

What it differs from the cascade on: no hard 0.95 correlation cut-off (a soft penalty instead), F instead of kNN mutual
information, K chosen per model with the model itself instead of one K from an LDA curve, and Stages 1-2 are nested too.

Output: cache/own_selection_results.json (new file; resumable). Usage:  python run_own_feature_selection.py [--smoke]

ADDED AFTER A SMOKE TEST (P1, outer fold 0 only; all four models chose K = 60, the top of the grid, so the declared grid may
truncate the search). The declared grid above stays the PRIMARY result. Two clearly labelled extras, chosen after seeing that:
  --grid extended      same method, grid {5, 8, 12, 16, 20, 30, 40, 60, 80, 120}   -> cache/own_selection_results_extended_grid.json
  --grid none          control: every candidate feature, no selection at all        -> cache/own_selection_results_all_features.json
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import CACHE_DIR
from feature_selection import get_feature_columns
from models import make_model_pipeline, compute_cv_folds, RANDOM_STATE
from run_models_norest import compute_metrics

warnings.filterwarnings("ignore", message=".*least populated class.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

PARTICIPANTS = ["P1_combined_noRest", "P2_combined_noRest", "Wallace_S1_noRest", "Wallace_S2_noRest"]
MODELS = ["LDA", "RandomForest", "SVM", "RBFNN"]
K_GRID = [5, 8, 12, 16, 20, 30, 40, 60]        # declared grid (primary)
INNER_SPLITS = 3
EPS = 1e-3
NO_SELECTION = False                            # --grid none: use every candidate feature
RESULTS_PATH = CACHE_DIR / "own_selection_results.json"


def apply_grid_option():
    """--grid extended | none   (default: the declared grid, unchanged)."""
    global K_GRID, NO_SELECTION, RESULTS_PATH
    if "--grid" in sys.argv:
        mode = sys.argv[sys.argv.index("--grid") + 1]
        if mode == "extended":
            K_GRID = [5, 8, 12, 16, 20, 30, 40, 60, 80, 120]
            RESULTS_PATH = CACHE_DIR / "own_selection_results_extended_grid.json"
        elif mode == "none":
            NO_SELECTION = True
            RESULTS_PATH = CACHE_DIR / "own_selection_results_all_features.json"
        else:
            raise SystemExit(f"unknown --grid {mode}")


def mrmr_rank(X, y, k_max):
    """Column indices of X in mRMR (F-test / correlation quotient) order."""
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)
    n = X.shape[1]
    first = int(np.argmax(F))
    selected = [first]
    avail = np.ones(n, dtype=bool)
    avail[first] = False
    corr_sum = C[:, first].copy()
    while len(selected) < min(k_max, n):
        score = np.where(avail, F / (corr_sum / len(selected) + EPS), -np.inf)
        j = int(np.argmax(score))
        selected.append(j)
        avail[j] = False
        corr_sum += C[:, j]
    return selected


def fit_predict(model_name, Xtr, ytr, Xte):
    need, est = make_model_pipeline(model_name)
    if need:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def choose_k(model_name, X, y, groups):
    """Inner grouped CV on the (outer-training) data. Returns best K and the mean inner accuracy per K."""
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = {k: [] for k in K_GRID}
    for itr, iva in inner.split(X, y, groups):
        order = mrmr_rank(X[itr], y[itr], max(K_GRID))
        for k in K_GRID:
            cols = order[:k]
            pred = fit_predict(model_name, X[itr][:, cols], y[itr], X[iva][:, cols])
            scores[k].append(float((pred == y[iva]).mean()))
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(mean.values())
    k_best = min(k for k, v in mean.items() if v == best)
    return k_best, mean


def evaluate(df, cols, model_name, smoke=False):
    X = df[cols].to_numpy(dtype=np.float64)
    assert np.isfinite(X).all(), "non-finite feature values"
    y = df["cue"].to_numpy()
    groups = df["segment_id"].to_numpy()
    folds = compute_cv_folds(df)
    pooled = np.empty_like(y)
    out = {"fold_accs": [], "fold_sizes": [], "k_per_fold": [], "features_per_fold": [], "inner_curve_per_fold": []}
    for f in sorted(np.unique(folds)):
        if smoke and f > 0:
            break
        tr, te = folds != f, folds == f
        t0 = time.time()
        if NO_SELECTION:
            k, curve, sel = X.shape[1], {}, list(range(X.shape[1]))
        else:
            k, curve = choose_k(model_name, X[tr], y[tr], groups[tr])
            sel = mrmr_rank(X[tr], y[tr], k)[:k]
        pred = fit_predict(model_name, X[tr][:, sel], y[tr], X[te][:, sel])
        pooled[te] = pred
        acc = float((pred == y[te]).mean())
        out["fold_accs"].append(acc)
        out["fold_sizes"].append(int(te.sum()))
        out["k_per_fold"].append(int(k))
        out["features_per_fold"].append([cols[i] for i in sel])
        out["inner_curve_per_fold"].append({str(kk): v for kk, v in curve.items()})
        print(f"    outer fold {f}: K={k:2d}  acc={acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    if not smoke:
        labels = sorted(df["cue"].unique().tolist())
        out.update(compute_metrics(y, pooled, labels))
    return out


def main():
    apply_grid_option()
    smoke = "--smoke" in sys.argv
    print(f"grid={K_GRID if not NO_SELECTION else 'none (all features)'}  output={RESULTS_PATH.name}", flush=True)
    results = json.load(open(RESULTS_PATH)) if (RESULTS_PATH.exists() and not smoke) else {}
    for name in PARTICIPANTS:
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        cols = get_feature_columns(df)
        print(f"=== {name}: {len(df)} windows, {len(cols)} candidate features, {df['cue'].nunique()} classes ===", flush=True)
        results.setdefault(name, {})
        for model_name in MODELS:
            if model_name in results[name] and not smoke:
                continue
            t0 = time.time()
            print(f"  -- {model_name}", flush=True)
            r = evaluate(df, cols, model_name, smoke=smoke)
            r["seconds"] = time.time() - t0
            if not smoke:
                print(f"  {model_name}: accuracy {r['accuracy'] * 100:.2f}  K per fold {r['k_per_fold']}  ({r['seconds']:.0f}s)", flush=True)
                results[name][model_name] = r
                json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        if smoke:
            break
    print("All done" if not smoke else "Smoke test done (nothing saved)")


if __name__ == "__main__":
    main()
