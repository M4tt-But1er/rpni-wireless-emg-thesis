"""Pilot: does properly nested hyperparameter search change RF/SVM accuracy
on the noise-round-2 sweep (4 severities x 2 participants = 8 conditions),
compared to the fixed-default hyperparameters used everywhere else in this
project? Directly comparable to model_results_noisy_round2.json's numbers
(same outer folds, same Stage-3 feature selection, same metrics) -- the
ONLY thing that changes is whether hyperparameters are tuned per outer fold
or held at the project's defaults.
"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR
from models import compute_cv_folds, make_model_pipeline
from nested_cv import rfecv_selector, perm_forward_selector
from run_models_norest import compute_metrics
from hyperparameter_tuning import tune_hyperparameters, fit_tuned_model
from save_models import save_model
from noise_injection import ATTENUATION_LEVELS_DB
from sklearn.preprocessing import StandardScaler

SELECTORS = {"RandomForest": rfecv_selector, "SVM": perm_forward_selector}
PARTICIPANTS = ["P1_combined", "P2_combined"]

RESULTS_PATH = CACHE_DIR / "model_results_noisy_round2_tuned_pilot.json"
UNTUNED_PATH = CACHE_DIR / "model_results_noisy_round2.json"
FS_PATH = CACHE_DIR / "feature_selection_results_noisy_round2.json"


def load_json(path):
    return json.load(open(path)) if path.exists() else {}


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=2)


def nested_evaluate_tuned(model_name, X, y, groups, feature_names, outer_fold_ids):
    selector = SELECTORS[model_name]
    y_pred_pooled = np.empty_like(y)
    per_fold_features, per_fold_params = [], []

    for fold in sorted(np.unique(outer_fold_ids)):
        train_mask = outer_fold_ids != fold
        test_mask = outer_fold_ids == fold
        Xtr, ytr, gtr = X[train_mask], y[train_mask], groups[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        selected = selector(model_name, Xtr, ytr, gtr, feature_names)
        idx = [feature_names.index(f) for f in selected]
        Xtr_sel, Xte_sel = Xtr[:, idx], Xte[:, idx]

        best_params, inner_score = tune_hyperparameters(model_name, Xtr_sel, ytr, gtr)
        fitted, scaler = fit_tuned_model(model_name, Xtr_sel, ytr, best_params)
        Xte_final = scaler.transform(Xte_sel) if scaler is not None else Xte_sel
        preds = fitted.predict(Xte_final)
        y_pred_pooled[test_mask] = preds

        per_fold_features.append(selected)
        per_fold_params.append(best_params)
        print(f"    outer fold {fold}: {len(selected)} features, best_params={best_params}, "
              f"inner_cv_acc={inner_score:.4f}, outer_test_acc={(preds==yte).mean():.4f}")

    return y_pred_pooled, per_fold_features, per_fold_params


def final_tuned_fit(model_name, df, reduced_features):
    selector = SELECTORS[model_name]
    X = df[reduced_features].to_numpy()
    y = df["cue"].to_numpy()
    groups = df["segment_id"].to_numpy()

    selected = selector(model_name, X, y, groups, reduced_features)
    idx = [reduced_features.index(f) for f in selected]
    X_sel = X[:, idx]
    best_params, _ = tune_hyperparameters(model_name, X_sel, y, groups)
    fitted, scaler = fit_tuned_model(model_name, X_sel, y, best_params)
    return fitted, scaler, selected, best_params


def main():
    results = load_json(RESULTS_PATH)
    untuned = load_json(UNTUNED_PATH)
    fs = load_json(FS_PATH)

    for severity in ATTENUATION_LEVELS_DB:
        results.setdefault(severity, {})
        for participant in PARTICIPANTS:
            name = f"{participant}_noRest_{severity}"
            df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
            reduced = fs[severity][participant]["reduced_features"]

            X = df[reduced].to_numpy()
            y = df["cue"].to_numpy()
            groups = df["segment_id"].to_numpy()
            fold_ids = compute_cv_folds(df)
            class_labels = sorted(df["cue"].unique().tolist())

            results[severity].setdefault(participant, {})
            for model_name in ["RandomForest", "SVM"]:
                if model_name in results[severity][participant]:
                    continue
                print(f"=== {name} / {model_name} (TUNED) ===")
                y_pred_pooled, per_fold_features, per_fold_params = nested_evaluate_tuned(
                    model_name, X, y, groups, reduced, fold_ids)
                metrics = compute_metrics(y, y_pred_pooled, class_labels)

                untuned_acc = untuned[severity][participant][model_name]["accuracy"]
                print(f"  TUNED acc={metrics['accuracy']:.4f}  |  UNTUNED (defaults) acc={untuned_acc:.4f}  "
                      f"|  delta={metrics['accuracy']-untuned_acc:+.4f}")

                final_est, final_scaler, final_features, final_params = final_tuned_fit(
                    model_name, df, reduced)

                results[severity][participant][model_name] = {
                    **metrics,
                    "per_fold_features": per_fold_features,
                    "per_fold_best_params": per_fold_params,
                    "deployed_feature_set": final_features,
                    "deployed_best_params": final_params,
                    "untuned_accuracy_for_comparison": untuned_acc,
                }
                save_json(results, RESULTS_PATH)
                save_model(f"{name}_tuned", model_name, final_est, final_scaler, final_features, metrics)

    print("\nAll done ->", RESULTS_PATH)
    print("\n=== SUMMARY: tuned vs untuned ===")
    for severity in results:
        for participant in results[severity]:
            for model_name, r in results[severity][participant].items():
                u = r["untuned_accuracy_for_comparison"]
                t = r["accuracy"]
                print(f"{severity:10s} {participant:14s} {model_name:14s} "
                      f"untuned={u*100:5.1f}%  tuned={t*100:5.1f}%  delta={t-u:+.1%}")


if __name__ == "__main__":
    main()
