"""Does tuning the RBFNN close its gap to the SVM?

The RBFNN (64 k-means centres, width = mean nearest-centre distance, logistic readout with C=1)
was never tuned and scored 93.2 / 83.5 against the SVM's 97.5 / 92.3 on P1 / P2, with the same data
and a similar Gaussian kernel. This runs the same nested protocol as run_hyperparameter_check_clean.py:
for each outer fold, take the Stage-3 feature set that fold already used (from model_results_norest.json,
so the features are identical to the untuned run), pick hyperparameters by a grouped, stratified inner
3-fold CV on the training part only, refit, and score the untouched outer fold.

Grid: centres {16, 32, 64, 128, 256} x width multiplier {0.5, 1, 2, 4} x C {0.1, 1, 10, 100}.
The untuned setting (64, 1, 1) is in the grid. K-means is fitted once per (inner fold, centre count)
and reused across widths and C.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import CACHE_DIR
from models import RBFNetClassifier, compute_cv_folds, RANDOM_STATE
from run_models_norest import compute_metrics

RESULTS_PATH = CACHE_DIR / "rbfnn_tuning.json"
PARTICIPANTS = ["P1_combined_noRest", "P2_combined_noRest"]
N_CENTERS = [16, 32, 64, 128, 256]
SIGMA_SCALES = [0.5, 1.0, 2.0, 4.0]
C_VALUES = [0.1, 1.0, 10.0, 100.0]
N_INNER = 3


def tune_rbfnn(X, y, groups, n_centers=None, sigma_scales=None, c_values=None):
    """Returns ((n_centers, sigma_scale, C), inner_cv_accuracy, {combo: mean inner accuracy}).
    The three grid arguments default to the module-level grids."""
    n_centers = n_centers or N_CENTERS
    sigma_scales = sigma_scales or SIGMA_SCALES
    c_values = c_values or C_VALUES
    cv = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for tr, va in cv.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        for n_c in n_centers:
            km = KMeans(n_clusters=min(n_c, len(Xtr)), random_state=RANDOM_STATE, n_init=10).fit(Xtr)
            centers = km.cluster_centers_
            dc = cdist(centers, centers)
            np.fill_diagonal(dc, np.inf)
            base = dc.min(axis=1).mean()
            if base <= 0:
                base = 1.0
            d2_tr, d2_va = cdist(Xtr, centers, "sqeuclidean"), cdist(Xva, centers, "sqeuclidean")
            for s in sigma_scales:
                sigma = base * s
                Ptr, Pva = np.exp(-d2_tr / (2 * sigma ** 2)), np.exp(-d2_va / (2 * sigma ** 2))
                for C in c_values:
                    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(Ptr, y[tr])
                    scores.setdefault((n_c, s, C), []).append(float((clf.predict(Pva) == y[va]).mean()))
    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(mean_scores, key=lambda k: (mean_scores[k], -k[0]))   # ties -> fewer centres
    return best, mean_scores[best], mean_scores


def check_equivalence(X, y, groups):
    """The hand-rolled inner loop must agree with RBFNetClassifier on the untuned setting."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    for s in (1.0, 2.0):
        ref = RBFNetClassifier(n_centers=64, C=1.0, sigma_scale=s).fit(Xs, y)
        km = KMeans(n_clusters=64, random_state=RANDOM_STATE, n_init=10).fit(Xs)
        dc = cdist(km.cluster_centers_, km.cluster_centers_)
        np.fill_diagonal(dc, np.inf)
        expected = dc.min(axis=1).mean() * s
        # k-means is only reproducible to ~1e-8 (threaded float sums), so compare relatively
        assert abs(ref.sigma_ - expected) < 1e-5 * expected, "width mismatch"


def main():
    results = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}
    untuned = json.load(open(CACHE_DIR / "model_results_norest.json"))
    for name in PARTICIPANTS:
        if name in results:
            print(f"{name}: already done")
            continue
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        y, groups = df["cue"].to_numpy(), df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)
        class_labels = sorted(df["cue"].unique().tolist())
        per_fold_features = untuned[name]["RBFNN"]["nested_per_fold_features"]
        check_equivalence(df[per_fold_features[0]].to_numpy(), y, groups)

        y_pred = np.empty_like(y)
        per_fold = []
        for fold in sorted(np.unique(fold_ids)):
            tr, te = fold_ids != fold, fold_ids == fold
            X = df[per_fold_features[fold]].to_numpy()
            (n_c, s, C), inner, _ = tune_rbfnn(X[tr], y[tr], groups[tr])
            scaler = StandardScaler().fit(X[tr])
            clf = RBFNetClassifier(n_centers=n_c, C=C, sigma_scale=s).fit(scaler.transform(X[tr]), y[tr])
            pred = clf.predict(scaler.transform(X[te]))
            y_pred[te] = pred
            acc = float((pred == y[te]).mean())
            per_fold.append(dict(n_centers=n_c, sigma_scale=s, C=C, inner_cv_acc=inner, outer_test_acc=acc,
                                 n_features=len(per_fold_features[fold])))
            print(f"  {name} outer fold {fold}: {len(per_fold_features[fold])} features, centres={n_c}, "
                  f"width x{s}, C={C}, inner={inner:.4f}, outer={acc:.4f}", flush=True)
        m = compute_metrics(y, y_pred, class_labels)
        default_acc = untuned[name]["RBFNN"]["accuracy"]
        results[name] = dict(tuned_accuracy=m["accuracy"], tuned_weighted_f1=m["weighted_f1"],
                             tuned_kappa=m["cohen_kappa"], default_accuracy=default_acc, per_fold=per_fold,
                             svm_accuracy=untuned[name]["SVM"]["accuracy"],
                             lda_accuracy=untuned[name]["LDA"]["accuracy"])
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        print(f"== {name}: tuned {m['accuracy']*100:.2f}% vs default {default_acc*100:.2f}% "
              f"(delta {(m['accuracy']-default_acc)*100:+.2f}); SVM {untuned[name]['SVM']['accuracy']*100:.2f}%",
              flush=True)
    print("All done ->", RESULTS_PATH)


if __name__ == "__main__":
    main()
