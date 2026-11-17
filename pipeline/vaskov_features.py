"""Feature extraction for the Vaskov et al. 2020 benchmark replication.

Two feature sets, per the paper text (Sections II-B and VI-B):
  1. MAV only (per channel) -- the paper's real-time/NB baseline feature.
  2. 5 time-domain features (MAV, Waveform Length, Variance, Slope Sign
     Changes, Zero Crossings -- the paper's actual list, NOT the classic
     Hudgins MAV-Slope set) + 6th-order autoregressive coefficients (Burg's
     method) per channel = 11 features/channel -- the paper's LDA/SVM set.

Windows are variable-length (paper's 7 window lengths: 10-250ms), fixed
50ms hop (see config.py VASKOV_HOP_MS docstring for why), computed within a
single trimmed segment only -- reuses segmentation.py's Segment objects
directly (same onset-trimmed boundaries as the rest of this project).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import TARGET_FS


def window_starts(seg_len: int, window_samples: int, hop_samples: int) -> np.ndarray:
    if seg_len < window_samples:
        return np.array([], dtype=int)
    last_start = seg_len - window_samples
    return np.arange(0, last_start + 1, hop_samples)


def sliding_windows(x: np.ndarray, window_samples: int, hop_samples: int) -> np.ndarray:
    starts = window_starts(len(x), window_samples, hop_samples)
    if len(starts) == 0:
        return np.empty((0, window_samples), dtype=x.dtype)
    all_windows = sliding_window_view(x, window_samples)
    return all_windows[starts]


def burg_ar_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """Burg's method for AR coefficient estimation (MATLAB's `arburg`, the
    conventional choice for short-segment AR spectral estimation -- not
    explicitly named in the paper, which just says "Matlab 2018 built-ins").

    Returns `order` AR coefficients a[1..order] such that
    x[n] ~= -sum_{i=1}^{order} a[i] * x[n-i]. Sign convention is irrelevant
    here since these are only used as classifier input features.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    m = min(order, N - 1)
    if m < 1:
        return np.zeros(order)

    a = np.zeros(m + 1)
    a[0] = 1.0
    f = x.copy()
    b = x.copy()

    for k in range(1, m + 1):
        f_cur = f[1:]
        b_prev = b[:-1]
        denom = np.dot(f_cur, f_cur) + np.dot(b_prev, b_prev)
        k_m = -2.0 * np.dot(f_cur, b_prev) / denom if denom > 0 else 0.0

        a_prev = a.copy()
        for i in range(k + 1):
            a[i] = a_prev[i] + k_m * a_prev[k - i]

        f, b = f_cur + k_m * b_prev, b_prev + k_m * f_cur

    coeffs = a[1:]
    if m < order:
        coeffs = np.concatenate([coeffs, np.zeros(order - m)])
    return coeffs


def mav_only_features(x_windows: np.ndarray) -> dict:
    return {"MAV": np.abs(x_windows).mean(axis=1)}


def td5_ar6_features(x_windows: np.ndarray, ar_order: int = 6) -> dict:
    n = x_windows.shape[0]
    feats = {}

    absx = np.abs(x_windows)
    feats["MAV"] = absx.mean(axis=1)
    feats["WL"] = np.abs(np.diff(x_windows, axis=1)).sum(axis=1)
    feats["Var"] = x_windows.var(axis=1, ddof=1) if x_windows.shape[1] > 1 else x_windows.var(axis=1)

    win_std = x_windows.std(axis=1, keepdims=True)
    deadband = 0.02 * win_std
    s1, s2 = x_windows[:, :-1], x_windows[:, 1:]
    zc_mask = (s1 * s2 < 0) & (np.abs(s1) > deadband) & (np.abs(s2) > deadband)
    feats["ZC"] = zc_mask.sum(axis=1)

    diffs = np.diff(x_windows, axis=1)
    d1, d2 = diffs[:, :-1], diffs[:, 1:]
    ssc_mask = (d1 * d2 < 0) & (np.abs(d1 - d2) > deadband)
    feats["SSC"] = ssc_mask.sum(axis=1)

    ar_names = [f"AR{i+1}" for i in range(ar_order)]
    if n > 0:
        ar_rows = np.array([burg_ar_coeffs(w, ar_order) for w in x_windows])
        for j, name in enumerate(ar_names):
            feats[name] = ar_rows[:, j]
    else:
        for name in ar_names:
            feats[name] = np.array([])

    return feats


MAV_ONLY_FEATURE_NAMES = ["MAV"]
TD5_AR6_FEATURE_NAMES = ["MAV", "WL", "Var", "ZC", "SSC"] + [f"AR{i+1}" for i in range(6)]
