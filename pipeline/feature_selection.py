"""3-stage feature selection cascade.

Stage 1 (once, model-agnostic): drop near-zero-variance features, greedily
drop one of each highly-correlated pair (|corr|>0.95, keeping the
higher-mutual-information member of the pair), rank survivors by mutual
information with the label.

Stage 2 (once, cheap proxy = LDA): accuracy-vs-K curve over top-K
mutual-info-ranked features (grouped stratified 5-fold CV), elbow pick ->
reduced set.

Stage 3: per-model wrapper selection, implemented in models.py (needs the
actual estimator objects).
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

VAR_EPS = 1e-8
CORR_THRESHOLD = 0.95
N_SPLITS = 5
RANDOM_STATE = 0


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    meta_cols = {"dataset", "participant", "cue", "segment_id", "onset_flagged", "window_time_s"}
    return [c for c in df.columns if c not in meta_cols]


def stage1_filter_and_rank(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    X = df[feature_cols].to_numpy()
    y = df["cue"].to_numpy()

    stds = X.std(axis=0)
    keep_mask = stds > VAR_EPS
    dropped_var = [c for c, k in zip(feature_cols, keep_mask) if not k]
    survivors = [c for c, k in zip(feature_cols, keep_mask) if k]
    X = X[:, keep_mask]

    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    order = np.argsort(mi)[::-1]
    survivors_ranked = [survivors[i] for i in order]
    mi_ranked = mi[order]

    corr = np.corrcoef(X.T)
    name_to_idx = {name: i for i, name in enumerate(survivors)}

    kept: list[str] = []
    dropped_corr = []
    for name in survivors_ranked:
        idx = name_to_idx[name]
        redundant = False
        for kept_name in kept:
            kidx = name_to_idx[kept_name]
            if abs(corr[idx, kidx]) > CORR_THRESHOLD:
                redundant = True
                break
        if redundant:
            dropped_corr.append(name)
        else:
            kept.append(name)

    print(f"stage1: {len(feature_cols)} input features -> "
          f"{len(dropped_var)} dropped (near-zero variance), "
          f"{len(dropped_corr)} dropped (|corr|>{CORR_THRESHOLD}), "
          f"{len(kept)} survivors")

    return kept


def make_group_cv(df: pd.DataFrame, n_splits: int = N_SPLITS):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def stage2_elbow_curve(df: pd.DataFrame, ranked_features: list[str], k_grid=None):
    y = df["cue"].to_numpy()
    groups = df["segment_id"].to_numpy()
    cv = make_group_cv(df)

    if k_grid is None:
        max_k = len(ranked_features)
        k_grid = sorted(set(list(range(2, min(20, max_k) + 1)) +
                            list(range(20, max_k + 1, 5))))
        if max_k not in k_grid:
            k_grid.append(max_k)

    accs = []
    for k in k_grid:
        X = df[ranked_features[:k]].to_numpy()
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy", n_jobs=-1)
        accs.append(scores.mean())
        print(f"  K={k:3d}  acc={scores.mean():.4f} +/- {scores.std():.4f}")

    return np.array(k_grid), np.array(accs)


def find_elbow(k_grid: np.ndarray, accs: np.ndarray) -> int:
    """Kneedle-style: point of max perpendicular distance from the chord
    connecting the first and last (k, acc) points, on normalized axes.

    WARNING (added 9 Nov 2026, comment only, behaviour unchanged): the return
    value is int(k_grid[elbow_idx]). That is right for the integer K grid of
    the feature-selection curve. On any non-integer axis (log10 bit rate, SNR
    in dB) it TRUNCATES: for log10 bit rate it snapped every elbow to 10 kb/s
    (docs/elbow_truncation_note.md). For non-integer axes use
    check_elbow_truncation.elbow_index / best_ratio_point_geometric."""
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    a_norm = (accs - accs.min()) / (accs.max() - accs.min() + 1e-12)

    p1 = np.array([k_norm[0], a_norm[0]])
    p2 = np.array([k_norm[-1], a_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    distances = []
    for kx, ay in zip(k_norm, a_norm):
        p = np.array([kx, ay]) - p1
        proj_len = np.dot(p, line_unit)
        proj = proj_len * line_unit
        perp = p - proj
        distances.append(np.linalg.norm(perp))

    elbow_idx = int(np.argmax(distances))
    return int(k_grid[elbow_idx])
