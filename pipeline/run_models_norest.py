"""Movement-only (rest excluded) model training + evaluation for LDA,
RandomForest, SVM, RBFNN, LSTM on P1_combined_noRest / P2_combined_noRest.

For each model:
  - Nested CV (correct procedure, per nested_cv.py's fix) gives an honest
    accuracy estimate AND, new here, pools every sample's out-of-fold
    prediction into one array -- since the 5 held-out folds together cover
    every row exactly once, this pooled (y_true, y_pred) pair is used to
    compute accuracy, weighted-F1, Cohen's kappa, and one confusion matrix
    that reflects genuine held-out performance, not train-set performance.
  - A separate FINAL fit on the full dataset (stage-3 selection done once,
    same as run_models.py) produces the model that actually gets saved to
    disk -- a nested fold's model was only ever meant for scoring, not
    deployment (same reasoning as run_models.py/nested_cv.py already
    documented for the original, rest-included models).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from config import CACHE_DIR, CUE_LEGEND
from models import compute_cv_folds, evaluate_with_folds, make_model_pipeline
from nested_cv import rfecv_selector, perm_forward_selector, SELECTORS
from lstm_model import evaluate_lstm_with_folds_with_predictions
from save_models import save_model, save_lstm_model

RESULTS_PATH = CACHE_DIR / "model_results_norest.json"
PARTICIPANT_DATASETS = ["P1_combined_noRest", "P2_combined_noRest"]


def load_results():
    return json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}


def save_results(results):
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)


def nested_evaluate_with_predictions(model_name, X, y, groups, feature_names, outer_fold_ids, selector):
    y_pred_pooled = np.empty_like(y)
    per_fold_acc, per_fold_features = [], []

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
        y_pred_pooled[test_mask] = preds

        acc = (preds == yte).mean()
        per_fold_acc.append(acc)
        per_fold_features.append(selected)
        print(f"    outer fold {fold}: {len(selected)} features, acc={acc:.4f}")

    return np.array(per_fold_acc), per_fold_features, y_pred_pooled


def compute_metrics(y_true, y_pred, class_labels):
    acc = float((y_true == y_pred).mean())
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=class_labels))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=class_labels))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "class_labels": [int(c) for c in class_labels],
        "class_names": [CUE_LEGEND[int(c)] for c in class_labels],
    }


def final_fit_for_deployment(model_name, df, reduced_features, selector):
    """Stage-3 selection on the FULL dataset (no held-out fold) -- this is
    the feature set + fitted model actually saved for reuse."""
    X_reduced = df[reduced_features].to_numpy()
    y = df["cue"].to_numpy()
    groups = df["segment_id"].to_numpy()

    selected = selector(model_name, X_reduced, y, groups, reduced_features)
    idx = [reduced_features.index(f) for f in selected]
    X_sel = X_reduced[:, idx]

    needs_scaling, est = make_model_pipeline(model_name)
    scaler = None
    if needs_scaling:
        scaler = StandardScaler().fit(X_sel)
        X_sel = scaler.transform(X_sel)
    est.fit(X_sel, y)
    return est, scaler, selected


def main():
    results = load_results()

    for name in PARTICIPANT_DATASETS:
        results.setdefault(name, {})
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        fs = json.load(open(CACHE_DIR / "feature_selection_results_norest.json"))
        reduced = fs[name]["reduced_features"]

        X_reduced = df[reduced].to_numpy()
        y = df["cue"].to_numpy()
        groups = df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)
        class_labels = sorted(df["cue"].unique().tolist())

        for model_name, selector in SELECTORS.items():
            if model_name in results[name]:
                continue
            print(f"=== {name} / {model_name} (nested, movement-only) ===")
            accs, per_fold_features, y_pred_pooled = nested_evaluate_with_predictions(
                model_name, X_reduced, y, groups, reduced, fold_ids, selector)
            metrics = compute_metrics(y, y_pred_pooled, class_labels)
            print(f"  acc={metrics['accuracy']:.4f}  weighted_F1={metrics['weighted_f1']:.4f}  "
                  f"kappa={metrics['cohen_kappa']:.4f}")

            final_est, final_scaler, final_features = final_fit_for_deployment(
                model_name, df, reduced, selector)

            results[name][model_name] = {
                "nested_fold_accs": accs.tolist(),
                "nested_per_fold_features": per_fold_features,
                **metrics,
                "deployed_feature_set": final_features,
            }
            save_results(results)
            save_model(name, model_name, final_est, final_scaler, final_features, metrics)

        if "LSTM" not in results[name]:
            print(f"=== {name} / LSTM (movement-only) ===")
            accs, y_true_pooled, y_pred_pooled = evaluate_lstm_with_folds_with_predictions(
                df, reduced, fold_ids, epochs=40, hidden_size=64, batch_size=16)
            metrics = compute_metrics(y_true_pooled, y_pred_pooled, class_labels)
            print(f"  acc={metrics['accuracy']:.4f}  weighted_F1={metrics['weighted_f1']:.4f}  "
                  f"kappa={metrics['cohen_kappa']:.4f}")
            results[name]["LSTM"] = {
                "nested_fold_accs": accs.tolist(),
                **metrics,
                "deployed_feature_set": reduced,
            }
            save_results(results)
            save_lstm_model(name, reduced, df, fold_ids, metrics)

    print("\nAll done ->", RESULTS_PATH)


if __name__ == "__main__":
    main()
