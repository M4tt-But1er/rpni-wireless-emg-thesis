"""Does properly nested hyperparameter search change RandomForest / SVM accuracy
on the clean, movement-only P1/P2 data?

Every model in this project uses fixed, a-priori hyperparameters (see
models.make_model_pipeline). The earlier pilot (run_noise_round2_tuned_pilot.py)
was aimed at the noisy sweep and never completed. This runs the same nested
procedure on the CLEAN headline results so the defence can say, with evidence,
whether tuning would have changed them.

Per outer fold: Stage-3 feature selection (default hyperparameters) on the
training rows only, then a grouped, stratified 3-fold grid search over the
hyperparameters on those selected features (still training rows only), then
fit the best setting and score the held-out fold. The grids contain the
project's defaults (RF 300 trees / unlimited depth / leaf 1; SVM C=10 /
gamma='scale'), so tuning can only match or beat them on the inner CV.
"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR
from models import compute_cv_folds
from run_models_norest import compute_metrics
from run_noise_round2_tuned_pilot import nested_evaluate_tuned

DATASETS = ["P1_combined_noRest", "P2_combined_noRest"]
MODELS = ["RandomForest", "SVM"]
RESULTS_PATH = CACHE_DIR / "hyperparameter_check_clean.json"


def main():
    fs = json.load(open(CACHE_DIR / "feature_selection_results_norest.json"))
    untuned = json.load(open(CACHE_DIR / "model_results_norest.json"))
    results = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}

    for name in DATASETS:
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        reduced = fs[name]["reduced_features"]
        X = df[reduced].to_numpy()
        y = df["cue"].to_numpy()
        groups = df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)
        class_labels = sorted(df["cue"].unique().tolist())
        results.setdefault(name, {})
        for model_name in MODELS:
            if model_name in results[name]:
                print(f"{name}/{model_name}: already done")
                continue
            print(f"=== {name} / {model_name} (nested hyperparameter search) ===", flush=True)
            y_pred, per_fold_features, per_fold_params = nested_evaluate_tuned(
                model_name, X, y, groups, reduced, fold_ids)
            m = compute_metrics(y, y_pred, class_labels)
            results[name][model_name] = {
                "tuned_accuracy": m["accuracy"], "tuned_weighted_f1": m["weighted_f1"],
                "tuned_kappa": m["cohen_kappa"],
                "default_accuracy": untuned[name][model_name]["accuracy"],
                "per_fold_best_params": per_fold_params,
                "per_fold_n_features": [len(f) for f in per_fold_features],
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  tuned {m['accuracy']*100:.2f}% vs default {untuned[name][model_name]['accuracy']*100:.2f}% "
                  f"(delta {(m['accuracy']-untuned[name][model_name]['accuracy'])*100:+.2f})", flush=True)
    print("All done ->", RESULTS_PATH)


if __name__ == "__main__":
    main()
