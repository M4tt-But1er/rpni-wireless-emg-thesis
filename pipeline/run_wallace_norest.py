"""Movement-only (rest excluded) Wallace S1/S2 evaluation, built to be
directly comparable to run_models_norest.py's Vaskov P1_combined_noRest /
P2_combined_noRest numbers: same nested Stage-3 wrapper selection (imported
unchanged from nested_cv.py), same metrics (compute_metrics/final_fit_for_
deployment imported unchanged from run_models_norest.py), same CUE_LEGEND
class codes (Wallace and Vaskov use the same Chestek-lab movement-code
convention, confirmed against docs/channel_mapping.md).

Models: RandomForest + SVM only, matching this project's established
convention for extensions beyond the original 5-model Vaskov build (noise
round 2, bitrate round 2, hyperparameter pilot all used just these two,
identified earlier as the two best-performing models). Not LDA/RBFNN/LSTM --
say so if you want the full 5-model set too.

See wallace_data.py's docstring for the class-set rationale (S1: 13 classes,
the full Vaskov CUE_LEGEND; S2: 12 classes, missing only Point).
"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR, REST_CUE
from preprocessing import preprocess_dataframe
from segmentation import build_segments
from features import extract_channel_features, _sliding_windows, window_starts, FEATURE_NAMES
from feature_selection import get_feature_columns, stage1_filter_and_rank, stage2_elbow_curve, find_elbow
from models import compute_cv_folds
from nested_cv import SELECTORS
from run_models_norest import nested_evaluate_with_predictions, compute_metrics, final_fit_for_deployment
from save_models import save_model
from wallace_data import PARTICIPANTS, load_wallace_run

MODELS = ["RandomForest", "SVM"]
FS_PATH = CACHE_DIR / "feature_selection_results_wallace.json"
RESULTS_PATH = CACHE_DIR / "model_results_wallace_norest.json"


def build_feature_matrix_for_run(participant: str, run_name: str, noise_snr_db: float = None,
                                  noise_color: str = "white", noise_seed: int = 0) -> pd.DataFrame:
    """noise_snr_db=None (default) preserves exact existing behavior -- the
    already-cached Wallace_S1/S2_noRest.parquet stay valid/reproducible.
    Added for the SNR-vs-accuracy shoulder search, which needs this same
    build with noise injected at the raw 30kHz stage, same as
    build_feature_matrix does for Vaskov."""
    channels = PARTICIPANTS[participant]["channels"]
    df = load_wallace_run(participant, run_name)
    if noise_snr_db is not None:
        from noise_injection import calibrate_and_inject
        rng = np.random.default_rng(noise_seed)
        for ch in channels:
            df[ch] = calibrate_and_inject(df[ch].to_numpy(), 30000, noise_snr_db, rng, noise_color=noise_color)
    dfp = preprocess_dataframe(df)
    del df
    segs = build_segments(dfp, run_name)
    segs = [s for s in segs if s.cue != REST_CUE]  # movement-only, per user request
    print(f"  [{run_name}] {len(segs)} movement segments after preprocessing")

    rows_meta = []
    channel_feat_cols = {(ch, fname): [] for ch in channels for fname in FEATURE_NAMES}

    for seg_id, seg in enumerate(segs):
        seg_start, seg_end = seg.trimmed_start, seg.end
        n_windows = None
        for ch in channels:
            x = dfp[ch].to_numpy()[seg_start:seg_end]
            xw = _sliding_windows(x)
            if n_windows is None:
                n_windows = xw.shape[0]
            feats = extract_channel_features(xw)
            for fname in FEATURE_NAMES:
                channel_feat_cols[(ch, fname)].append(feats[fname])
        if not n_windows:
            continue
        w_starts = window_starts(seg_end - seg_start)
        w_times = dfp["time_s"].to_numpy()[seg_start + w_starts]
        for w_i in range(n_windows):
            rows_meta.append({
                "dataset": f"{participant}_noRest", "participant": participant, "cue": seg.cue,
                "segment_id": f"{run_name}_{seg_id}",
                "onset_flagged": seg.onset_flagged, "window_time_s": w_times[w_i],
            })

    meta_df = pd.DataFrame(rows_meta)
    feat_data = {}
    for (ch, fname), arrays in channel_feat_cols.items():
        non_empty = [a for a in arrays if len(a) > 0]
        feat_data[f"{ch}_{fname}"] = np.concatenate(non_empty) if non_empty else np.array([])
    return pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(feat_data).reset_index(drop=True)], axis=1)


def build_and_cache(participant: str) -> pd.DataFrame:
    out_path = CACHE_DIR / f"{participant}_noRest.parquet"
    if out_path.exists():
        print(f"[{participant}] using cached {out_path}")
        return pd.read_parquet(out_path)

    run_for_task = PARTICIPANTS[participant]["run_for_task"]
    print(f"[{participant}] building feature matrix from {len(run_for_task)} SNR runs...")
    frames = [build_feature_matrix_for_run(participant, run_name) for run_name in run_for_task.values()]
    df = pd.concat(frames, axis=0, ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(f"[{participant}] cached -> {out_path} {df.shape}")
    return df


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fs_results = json.load(open(FS_PATH)) if FS_PATH.exists() else {}
    results = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}

    for participant in PARTICIPANTS:
        df = build_and_cache(participant)
        print(f"[{participant}] {df.shape}, cues={sorted(df['cue'].unique().tolist())}, "
              f"windows/cue={df['cue'].value_counts().to_dict()}")

        if participant not in fs_results:
            cols = get_feature_columns(df)
            ranked = stage1_filter_and_rank(df, cols)
            max_k = len(ranked)
            k_grid = sorted(set([2, 3, 5] + list(range(5, max_k + 1, 5)) + [max_k]))
            k_grid, accs = stage2_elbow_curve(df, ranked, k_grid=k_grid)
            elbow_k = find_elbow(k_grid, accs)
            reduced = ranked[:elbow_k]
            fs_results[participant] = {"ranked_features": ranked, "elbow_k": int(elbow_k),
                                        "reduced_features": reduced}
            json.dump(fs_results, open(FS_PATH, "w"), indent=2)
            print(f"[{participant}] Stage1+2: {len(ranked)} ranked -> elbow K={elbow_k}")
        reduced = fs_results[participant]["reduced_features"]

        X_reduced = df[reduced].to_numpy()
        y = df["cue"].to_numpy()
        groups = df["segment_id"].to_numpy()
        fold_ids = compute_cv_folds(df)
        class_labels = sorted(df["cue"].unique().tolist())

        results.setdefault(participant, {})
        for model_name in MODELS:
            if model_name in results[participant]:
                continue
            selector = SELECTORS[model_name]
            print(f"=== {participant} / {model_name} (nested Stage-3, movement-only) ===")
            accs, per_fold_features, y_pred_pooled = nested_evaluate_with_predictions(
                model_name, X_reduced, y, groups, reduced, fold_ids, selector)
            metrics = compute_metrics(y, y_pred_pooled, class_labels)
            print(f"  acc={metrics['accuracy']:.4f}  weighted_F1={metrics['weighted_f1']:.4f}  "
                  f"kappa={metrics['cohen_kappa']:.4f}")

            final_est, final_scaler, final_features = final_fit_for_deployment(
                model_name, df, reduced, selector)

            results[participant][model_name] = {
                "nested_fold_accs": accs.tolist(),
                "nested_per_fold_features": per_fold_features,
                **metrics,
                "deployed_feature_set": final_features,
            }
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)
            save_model(f"{participant}_noRest", model_name, final_est, final_scaler, final_features, metrics)

    print("\nAll done ->", RESULTS_PATH)
    print("\n=== SUMMARY ===")
    for participant, models in results.items():
        for model_name, r in models.items():
            print(f"{participant:12s} {model_name:14s} acc={r['accuracy']*100:5.1f}%  "
                  f"F1={r['weighted_f1']*100:5.1f}%  kappa={r['cohen_kappa']:.3f}")


if __name__ == "__main__":
    main()
