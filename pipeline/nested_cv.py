"""Proper nested CV: outer 5-fold loop for the reported accuracy, with stage-3
feature selection re-run from scratch inside each outer training fold only
(the held-out outer fold never influences which features get selected).

This replaces the earlier run_models.py numbers for LDA/RandomForest/SVM/RBFNN,
which reused the exact same 5-fold partition for both stage-3 selection *and*
final evaluation -- optimistic leakage. LSTM already did not have this problem
(no CV-driven feature subset selection), so its numbers are unchanged.

The final "feature set per model" reported to the user is still the one
selected once on the *full* dataset (from run_models.py / cache/model_results.json)
-- that's the actual recommended feature set to deploy. The nested procedure
here exists only to get an honest accuracy *estimate*, since a different,
slightly different feature subset is (correctly) selected in each outer fold.
"""
import json
import time

import numpy as np
import pandas as pd

from config import CACHE_DIR
from models import compute_cv_folds, make_model_pipeline, rfecv_select, permutation_forward_select
from sklearn.preprocessing import StandardScaler

RESULTS_PATH = CACHE_DIR / "model_results_nested.json"
PARTICIPANT_DATASETS = ["P1_combined", "P2_combined"]


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def nested_evaluate(model_name, X, y, groups, feature_names, outer_fold_ids, selector):
    per_fold_acc = []
    per_fold_features = []
    for fold in sorted(np.unique(outer_fold_ids)):
        train_mask = outer_fold_ids != fold
        test_mask = outer_fold_ids == fold
        Xtr, ytr, gtr = X[train_mask], y[train_mask], groups[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        selected = selector(model_name, Xtr, ytr, gtr, feature_names)
        idx = [feature_names.index(f) for f in selected]
        Xtr_sel, Xte_sel = Xtr[:, idx], Xte[:, idx]

        needs_scaling, est = make_model_pipeline(model_name)
        if needs_scaling:
            scaler = StandardScaler().fit(Xtr_sel)
            Xtr_sel, Xte_sel = scaler.transform(Xtr_sel), scaler.transform(Xte_sel)

        est.fit(Xtr_sel, ytr)
        preds = est.predict(Xte_sel)
        acc = (preds == yte).mean()
        per_fold_acc.append(acc)
        per_fold_features.append(selected)
        print(f"    outer fold {fold}: {len(selected)} features selected, acc={acc:.4f}")

    return np.array(per_fold_acc), per_fold_features


def rfecv_selector(model_name, Xtr, ytr, gtr, feature_names):
    return rfecv_select(model_name, Xtr, ytr, gtr, feature_names)


def perm_forward_selector(model_name, Xtr, ytr, gtr, feature_names):
    selected, _, _ = permutation_forward_select(model_name, Xtr, ytr, gtr, feature_names)
    return selected


SELECTORS = {
    "LDA": rfecv_selector,
    "RandomForest": rfecv_selector,
    "SVM": perm_forward_selector,
    "RBFNN": perm_forward_selector,
}


def main():
    results = load_results()
    fs = json.load(open(CACHE_DIR / "feature_selection_results.json"))

    for name in PARTICIPANT_DATASETS:
        results.setdefault(name, {})
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        reduced = fs[name]["reduced_features"]
        X_reduced = df[reduced].to_numpy()
        y = df["cue"].to_numpy()
        groups = df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)

        for model_name, selector in SELECTORS.items():
            if model_name in results[name]:
                continue
            print(f"=== {name} / {model_name} (nested) ===")
            t0 = time.time()
            accs, per_fold_features = nested_evaluate(
                model_name, X_reduced, y, groups, reduced, fold_ids, selector)
            print(f"  nested acc={accs.mean():.4f} +/- {accs.std():.4f}  ({time.time()-t0:.1f}s)")
            results[name][model_name] = {
                "fold_accs": accs.tolist(),
                "mean_acc": accs.mean(),
                "std_acc": accs.std(),
                "per_fold_features": per_fold_features,
            }
            save_results(results)

    print("\nAll done. Results saved to", RESULTS_PATH)


if __name__ == "__main__":
    main()
