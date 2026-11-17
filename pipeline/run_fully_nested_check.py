"""Fully nested check of the movement-only results: Stage 1 (MI ranking +
correlation pruning), Stage 2 (LDA elbow for K) AND Stage 3 (per-model wrapper
selection) are all re-run from scratch on each outer fold's TRAINING rows only.

Why: nested_cv.py / run_models_norest.py nest only Stage 3. Stages 1-2 were
computed once on the full dataset (feature_selection_results_norest.json), so
the held-out fold's labels influenced which ~18-25 candidate features each
model was allowed to choose from. This script measures how much that matters.
Same outer folds (models.compute_cv_folds) as every other result, so numbers
are directly comparable.

Also saves every out-of-fold prediction with its segment_id so a segment-level
(cluster) bootstrap can give honest confidence intervals -- windows within a
segment are near-duplicates (75% overlap), so the effective sample size is the
number of segments (96 for P1, 65 for P2), not the number of windows.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from config import CACHE_DIR
from feature_selection import get_feature_columns, stage1_filter_and_rank, stage2_elbow_curve, find_elbow
from models import compute_cv_folds, make_model_pipeline
from nested_cv import SELECTORS

DATASETS = ["P1_combined_noRest", "P2_combined_noRest"]
MODELS = ["LDA", "RandomForest", "SVM"]
RESULTS_PATH = CACHE_DIR / "fully_nested_check.json"
N_BOOT = 2000


def cluster_bootstrap_ci(y_true, y_pred, seg_ids, n_boot=N_BOOT, seed=0):
    """95% CI for accuracy, resampling whole segments with replacement."""
    rng = np.random.default_rng(seed)
    segs = np.unique(seg_ids)
    idx_by_seg = {s: np.flatnonzero(seg_ids == s) for s in segs}
    correct = (y_true == y_pred)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(segs, size=len(segs), replace=True)
        idx = np.concatenate([idx_by_seg[s] for s in pick])
        accs[b] = correct[idx].mean()
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)), accs


def main():
    results = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}
    for name in DATASETS:
        if name in results and all(m in results[name]["models"] for m in MODELS):
            print(f"{name}: already done")
            continue
        df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        y_all = df["cue"].to_numpy()
        groups_all = df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)
        preds = {m: np.empty_like(y_all) for m in MODELS}
        per_fold = {m: [] for m in MODELS}
        per_fold_k = []
        per_fold_sel = {m: [] for m in MODELS}

        for fold in sorted(np.unique(fold_ids)):
            tr, te = fold_ids != fold, fold_ids == fold
            train_df = df[tr].reset_index(drop=True)
            cols = get_feature_columns(train_df)
            ranked = stage1_filter_and_rank(train_df, cols)
            k_grid, accs = stage2_elbow_curve(train_df, ranked)
            elbow_k = find_elbow(k_grid, accs)
            reduced = ranked[:elbow_k]
            per_fold_k.append(int(elbow_k))
            print(f"[{name}] fold {fold}: stage1 survivors {len(ranked)}, stage2 elbow K={elbow_k}", flush=True)

            Xtr_all = df.loc[tr, reduced].to_numpy()
            Xte_all = df.loc[te, reduced].to_numpy()
            ytr, yte, gtr = y_all[tr], y_all[te], groups_all[tr]
            for m in MODELS:
                selected = SELECTORS[m](m, Xtr_all, ytr, gtr, reduced)
                idx = [reduced.index(f) for f in selected]
                Xtr, Xte = Xtr_all[:, idx], Xte_all[:, idx]
                needs_scaling, est = make_model_pipeline(m)
                if needs_scaling:
                    sc = StandardScaler().fit(Xtr)
                    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                est.fit(Xtr, ytr)
                p = est.predict(Xte)
                preds[m][te] = p
                per_fold[m].append(float((p == yte).mean()))
                per_fold_sel[m].append(len(selected))
                print(f"    {m:12s} {len(selected):2d} features, fold acc {per_fold[m][-1]:.4f}", flush=True)

        entry = {"models": {}, "stage2_elbow_k_per_fold": per_fold_k}
        for m in MODELS:
            acc = float((preds[m] == y_all).mean())
            lo, hi, _ = cluster_bootstrap_ci(y_all, preds[m], groups_all)
            entry["models"][m] = {
                "accuracy": acc, "ci95_segment_bootstrap": [lo, hi],
                "weighted_f1": float(f1_score(y_all, preds[m], average="weighted")),
                "cohen_kappa": float(cohen_kappa_score(y_all, preds[m])),
                "per_fold_acc": per_fold[m], "n_selected_per_fold": per_fold_sel[m],
                "y_true": y_all.tolist(), "y_pred": preds[m].tolist(), "segment_id": groups_all.tolist(),
            }
        # paired segment bootstrap of RF - SVM, RF - LDA
        rng = np.random.default_rng(1)
        segs = np.unique(groups_all)
        idx_by_seg = {s: np.flatnonzero(groups_all == s) for s in segs}
        c = {m: (preds[m] == y_all) for m in MODELS}
        diffs = {"RandomForest-SVM": [], "RandomForest-LDA": [], "SVM-LDA": []}
        for _ in range(N_BOOT):
            idx = np.concatenate([idx_by_seg[s] for s in rng.choice(segs, size=len(segs), replace=True)])
            diffs["RandomForest-SVM"].append(c["RandomForest"][idx].mean() - c["SVM"][idx].mean())
            diffs["RandomForest-LDA"].append(c["RandomForest"][idx].mean() - c["LDA"][idx].mean())
            diffs["SVM-LDA"].append(c["SVM"][idx].mean() - c["LDA"][idx].mean())
        entry["paired_diff_ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v))]
                                      for k, v in diffs.items()}
        results[name] = entry
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f)
        print(f"== {name} done ==", flush=True)
        for m in MODELS:
            e = entry["models"][m]
            print(f"  {m:12s} acc {e['accuracy']*100:.2f}%  95% CI [{e['ci95_segment_bootstrap'][0]*100:.1f}, "
                  f"{e['ci95_segment_bootstrap'][1]*100:.1f}]  kappa {e['cohen_kappa']:.3f}", flush=True)
        for k, v in entry["paired_diff_ci95"].items():
            print(f"  paired {k}: mean {v[2]*100:+.2f} pts, 95% CI [{v[0]*100:+.2f}, {v[1]*100:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
