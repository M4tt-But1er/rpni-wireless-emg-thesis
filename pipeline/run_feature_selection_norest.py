"""Stage 1+2 feature selection for the rest-excluded ("movement-only")
datasets. Separate from feature_selection_results.json -- removing the
majority class entirely changes the label distribution and mutual-information
rankings, so this cannot reuse the original selection results.
"""
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CACHE_DIR, FIGS_DIR
from feature_selection import get_feature_columns, stage1_filter_and_rank, stage2_elbow_curve, find_elbow

PARTICIPANT_DATASETS = ["P1_combined_noRest", "P2_combined_noRest"]


def main():
    results = {}
    for name in PARTICIPANT_DATASETS:
        print(f"=== {name} ===")
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        cols = get_feature_columns(df)
        ranked = stage1_filter_and_rank(df, cols)
        k_grid, accs = stage2_elbow_curve(df, ranked)
        elbow_k = find_elbow(k_grid, accs)

        plt.figure(figsize=(8, 5))
        plt.plot(k_grid, accs, marker="o", ms=3)
        plt.axvline(elbow_k, color="r", ls="--", label=f"elbow K={elbow_k}")
        plt.xlabel("K (top-K MI-ranked features)")
        plt.ylabel("LDA grouped-stratified 5-fold CV accuracy")
        plt.title(f"{name}: Stage 2 accuracy-vs-K elbow curve (movement-only, no rest)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / f"{name}_elbow_curve.png", dpi=100)
        plt.close()

        results[name] = {
            "ranked_features": ranked, "k_grid": k_grid.tolist(), "accs": accs.tolist(),
            "elbow_k": elbow_k, "reduced_features": ranked[:elbow_k],
        }
        print(f"{name}: elbow K={elbow_k}, reduced set = {ranked[:elbow_k]}")

    with open(CACHE_DIR / "feature_selection_results_norest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved cache/feature_selection_results_norest.json")


if __name__ == "__main__":
    main()
