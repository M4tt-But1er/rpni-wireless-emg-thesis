"""Local-maxima (coordinate ascent) search over (sample_rate, n_channels,
bit_depth) for RandomForest/SVM on P1_combined_noRest, P2_combined_noRest,
Wallace_S1_noRest, Wallace_S2_noRest, with NO noise (clean signal) -- per the
user's "lots of combinations here, so do a local maxima search strategy"
request, instead of a full grid (up to 5x8x3 = 120 combos x 4 datasets x 2
models).

Algorithm: coordinate ascent (hill-climbing) on the discrete 3D grid. From a
starting point, evaluate every neighbor reachable by changing exactly ONE
dimension by one step; move to whichever neighbor most improves accuracy for
the model being optimized; repeat until no neighbor improves (a local
optimum). Run from 3 spread-out starting points per (dataset, model) and
keep the best local optimum found (the user's chosen tradeoff -- lower risk
of a mediocre local max than a single start, without paying for a full grid).

Channel grid is capped per dataset at how many channels actually exist:
P1/P2 (Vaskov) and Wallace_S1 top out at 8 (their real channel count);
Wallace_S2 tops out at 12 (implanted channels only -- surface channels were
explicitly excluded from this test, so the 20-channel case isn't run here).

Every (dataset, sample_rate, n_channels, bit_depth) actually evaluated is
cached -- both RF and SVM scored together, since building the feature table
(not fitting the model) is the expensive step -- so repeated visits across
different starts/models are free. Stage 3 (per-model wrapper selection) is
deliberately skipped, matching bitrate_sweep.py's own precedent for a sweep
this size: Stage 1+2 only, plain grouped-stratified 5-fold CV.
"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR, P1_CHANNELS, P2_CHANNELS
from bitrate_sweep import build_one_config, channel_importance_ranking, evaluate_config
from wallace_bitrate import build_wallace_config, channel_importance_ranking_wallace
from feature_selection import get_feature_columns, stage1_filter_and_rank, stage2_elbow_curve, find_elbow
from models import compute_cv_folds

RESULTS_PATH = CACHE_DIR / "bitrate_search_results.json"
SAMPLE_RATES = [250, 500, 1000, 3000, 10000]
BIT_DEPTHS = [8, 12, 16]
CHANNEL_GRID_FULL = [2, 5, 8, 12, 16, 20]
MODELS = ["RandomForest", "SVM"]

DATASET_SPECS = {
    "P1_combined": dict(raw_names=["P1_1of12", "P1_Grasps"], all_channels=P1_CHANNELS, kind="vaskov"),
    "P2_combined": dict(raw_names=["P2_1of10", "P2_Grasps"], all_channels=P2_CHANNELS, kind="vaskov"),
    "Wallace_S1": dict(raw_names=None, all_channels=None, kind="wallace"),
    "Wallace_S2": dict(raw_names=None, all_channels=None, kind="wallace"),
}


def load_results():
    return json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def get_ranking_and_grid(dataset: str):
    spec = DATASET_SPECS[dataset]
    if spec["kind"] == "vaskov":
        ranking = channel_importance_ranking(f"{dataset}_noRest", spec["all_channels"])
        max_ch = len(spec["all_channels"])
    else:
        ranking = channel_importance_ranking_wallace(dataset)
        max_ch = len(ranking)
    grid = [c for c in CHANNEL_GRID_FULL if c <= max_ch]
    return ranking, grid


def build_and_score(dataset: str, ranking: list[str], sample_rate: int, n_channels: int, bit_depth: int) -> dict:
    spec = DATASET_SPECS[dataset]
    keep = ranking[:n_channels]
    if spec["kind"] == "vaskov":
        frames = [build_one_config(raw, sample_rate, keep, bit_depth, f"{dataset}_search", snr_db=None)
                  for raw in spec["raw_names"]]
        df = pd.concat(frames, axis=0, ignore_index=True)
    else:
        df = build_wallace_config(dataset, sample_rate, keep, bit_depth)

    cols = get_feature_columns(df)
    ranked = stage1_filter_and_rank(df, cols)
    max_k = len(ranked)
    k_grid = sorted(set([2, 3, 5] + list(range(5, max_k + 1, 5)) + [max_k])) if max_k >= 2 else [max_k]
    k_grid, accs = stage2_elbow_curve(df, ranked, k_grid=k_grid)
    elbow_k = find_elbow(k_grid, accs)
    reduced = ranked[:elbow_k]

    X = df[reduced].to_numpy()
    y = df["cue"].to_numpy()
    fold_ids = compute_cv_folds(df)

    result = {"n_reduced_features": int(elbow_k), "n_windows": int(len(df)), "channels_used": keep}
    for model_name in MODELS:
        result[model_name] = float(evaluate_config(model_name, X, y, fold_ids))
    return result


def build_and_score_full(dataset: str, ranking: list[str], sample_rate: int, n_channels: int,
                          bit_depth: int) -> dict:
    """Same build + Stage1+2 as build_and_score, but also pools out-of-fold
    predictions to report weighted F1/kappa alongside accuracy (one fit per
    fold, not a duplicate fit) -- needed for the full exhaustive bitrate
    sweep's F1-vs-bitrate curve, not just accuracy-vs-bitrate."""
    from models import make_model_pipeline
    from run_models_norest import compute_metrics
    from sklearn.preprocessing import StandardScaler

    spec = DATASET_SPECS[dataset]
    keep = ranking[:n_channels]
    if spec["kind"] == "vaskov":
        frames = [build_one_config(raw, sample_rate, keep, bit_depth, f"{dataset}_search", snr_db=None)
                  for raw in spec["raw_names"]]
        df = pd.concat(frames, axis=0, ignore_index=True)
    else:
        df = build_wallace_config(dataset, sample_rate, keep, bit_depth)

    cols = get_feature_columns(df)
    ranked = stage1_filter_and_rank(df, cols)
    max_k = len(ranked)
    k_grid = sorted(set([2, 3, 5] + list(range(5, max_k + 1, 5)) + [max_k])) if max_k >= 2 else [max_k]
    k_grid, accs = stage2_elbow_curve(df, ranked, k_grid=k_grid)
    elbow_k = find_elbow(k_grid, accs)
    reduced = ranked[:elbow_k]

    X = df[reduced].to_numpy()
    y = df["cue"].to_numpy()
    fold_ids = compute_cv_folds(df)
    class_labels = sorted(df["cue"].unique().tolist())
    bitrate_bps = sample_rate * n_channels * bit_depth

    result = {"n_reduced_features": int(elbow_k), "n_windows": int(len(df)), "channels_used": keep,
              "bitrate_bps": int(bitrate_bps)}
    for model_name in MODELS:
        y_pred_pooled = np.empty_like(y)
        for fold in range(fold_ids.max() + 1):
            train_mask, test_mask = fold_ids != fold, fold_ids == fold
            needs_scaling, est = make_model_pipeline(model_name)
            Xtr, Xte = X[train_mask], X[test_mask]
            if needs_scaling:
                scaler = StandardScaler().fit(Xtr)
                Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
            est.fit(Xtr, y[train_mask])
            y_pred_pooled[test_mask] = est.predict(Xte)
        m = compute_metrics(y, y_pred_pooled, class_labels)
        result[model_name] = {"accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"],
                               "cohen_kappa": m["cohen_kappa"]}
    return result


def config_key(sample_rate, n_channels, bit_depth) -> str:
    return f"{sample_rate}Hz_{n_channels}ch_{bit_depth}bit"


def get_or_eval(dataset, ranking, channel_grid, idx, cache) -> dict:
    sr_idx, ch_idx, bd_idx = idx
    sample_rate, n_channels, bit_depth = SAMPLE_RATES[sr_idx], channel_grid[ch_idx], BIT_DEPTHS[bd_idx]
    key = config_key(sample_rate, n_channels, bit_depth)
    if key not in cache:
        print(f"    evaluating {dataset} / {key} ...")
        cache[key] = build_and_score(dataset, ranking, sample_rate, n_channels, bit_depth)
        cache["_dirty"] = True
    return cache[key]


def neighbors(idx, dims_sizes):
    out = []
    for dim in range(3):
        for delta in (-1, 1):
            new_idx = list(idx)
            new_idx[dim] += delta
            if 0 <= new_idx[dim] < dims_sizes[dim]:
                out.append(tuple(new_idx))
    return out


def coordinate_ascent(dataset, ranking, channel_grid, model_name, start_idx, cache):
    dims_sizes = (len(SAMPLE_RATES), len(channel_grid), len(BIT_DEPTHS))
    current = start_idx
    current_acc = get_or_eval(dataset, ranking, channel_grid, current, cache)[model_name]
    path = [{"idx": current, "accuracy": current_acc}]
    while True:
        best_neighbor, best_acc = None, current_acc
        for nb in neighbors(current, dims_sizes):
            acc = get_or_eval(dataset, ranking, channel_grid, nb, cache)[model_name]
            if acc > best_acc:
                best_acc, best_neighbor = acc, nb
        if best_neighbor is None:
            break
        current, current_acc = best_neighbor, best_acc
        path.append({"idx": current, "accuracy": current_acc})
    return current, current_acc, path


def start_points(channel_grid):
    n_sr, n_ch, n_bd = len(SAMPLE_RATES), len(channel_grid), len(BIT_DEPTHS)
    return [
        (0, 0, 0),
        (n_sr // 2, n_ch // 2, n_bd // 2),
        (n_sr - 1, n_ch - 1, n_bd - 1),
    ]


def idx_to_config(channel_grid, idx):
    sr_idx, ch_idx, bd_idx = idx
    return {"sample_rate": SAMPLE_RATES[sr_idx], "n_channels": channel_grid[ch_idx], "bit_depth": BIT_DEPTHS[bd_idx]}


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    for dataset in DATASET_SPECS:
        print(f"=== {dataset} ===")
        ranking, channel_grid = get_ranking_and_grid(dataset)
        results.setdefault(dataset, {"channel_ranking": ranking, "channel_grid": channel_grid,
                                      "evaluated_configs": {}})
        cache = results[dataset]["evaluated_configs"]

        for model_name in MODELS:
            if model_name in results[dataset] and "best_config" in results[dataset][model_name]:
                print(f"  {model_name}: already done, best={results[dataset][model_name]['best_accuracy']:.4f}")
                continue
            print(f"  --- {model_name}: coordinate ascent, 3 restarts ---")
            restarts = []
            for start in start_points(channel_grid):
                final_idx, final_acc, path = coordinate_ascent(dataset, ranking, channel_grid, model_name, start, cache)
                restarts.append({
                    "start_config": idx_to_config(channel_grid, start),
                    "final_config": idx_to_config(channel_grid, final_idx),
                    "final_accuracy": final_acc,
                    "n_steps": len(path) - 1,
                    "path": [{"config": idx_to_config(channel_grid, p["idx"]), "accuracy": p["accuracy"]} for p in path],
                })
                print(f"    restart from {idx_to_config(channel_grid, start)}: "
                      f"-> {idx_to_config(channel_grid, final_idx)} acc={final_acc:.4f} ({len(path)-1} steps)")
                cache.pop("_dirty", None)
                save_results(results)

            best = max(restarts, key=lambda r: r["final_accuracy"])
            results[dataset][model_name] = {
                "best_config": best["final_config"], "best_accuracy": best["final_accuracy"],
                "restarts": restarts,
            }
            save_results(results)
            print(f"  {model_name} best: {best['final_config']} acc={best['final_accuracy']:.4f}")

    print("\nAll done ->", RESULTS_PATH)
    print("\n=== SUMMARY ===")
    for dataset, r in results.items():
        for model_name in MODELS:
            if model_name in r:
                print(f"{dataset:14s} {model_name:14s} {r[model_name]['best_config']} "
                      f"acc={r[model_name]['best_accuracy']*100:5.1f}%")
        print(f"  ({len(r['evaluated_configs'])} unique configs evaluated for this dataset)")


if __name__ == "__main__":
    main()
