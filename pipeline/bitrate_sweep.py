"""Bitrate-minimization sweep: on top of the "moderate" (10dB) noisy signal,
jointly reduce sample rate, channel count, and bit depth, and measure how
much RandomForest/SVM accuracy actually costs at each combination.

3 levels per dimension (27 combos), evaluated per participant:
  - Sample rate (Hz): 1000 (baseline) / 500 / 250 -- bandpass upper cutoff
    co-adjusts down with the new Nyquist (100Hz lower bound is fixed; a
    250Hz sample rate leaves only a 100-124Hz passband, which is expected
    to hurt badly -- that's a real finding, not a bug).
  - Channels: 8 (all) / 5 / 2 -- ranked by each participant's OWN existing
    Stage-1 mutual-information ranking from the clean, unnoised, all-8-
    channel model (cache/feature_selection_results_norest.json) --
    "keep whichever channel has a feature that ranked best", not arbitrary.
  - Bit depth: 16 (~native resolution, effectively lossless) / 10 / 6 --
    applied to the signal AFTER downsampling (i.e. to what's actually
    transmitted at the reduced rate), by requantizing to 2^bits levels
    spanning a robust (0.1-99.9 percentile) per-channel range.

Bitrate = sample_rate x n_channels x bits (bits/second).

Stage 3 (per-model wrapper feature selection) is deliberately skipped here
-- this sweep's job is comparing 27 configurations honestly on equal
footing, not hand-optimizing each one. Just Stage 1+2, then a single
(non-nested) grouped-stratified 5-fold CV fit/score for RandomForest and
SVM on the Stage-2 reduced set. This is a legitimate scope reduction for a
sweep this size, not the final honest-accuracy claim the other results in
this project are -- flagged in the report.
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

from config import CACHE_DIR, DATASETS, COMBINED_GROUPS, REST_CUE, RAW_FS
from data_io import load_raw
from noise_injection import calibrate_and_inject, SNR_LEVELS_DB
from segmentation import rolling_mav, find_sustained_crossing, identify_segments
from features import extract_channel_features
from vaskov_features import window_starts, sliding_windows  # variable window/hop primitives
from feature_selection import get_feature_columns, stage1_filter_and_rank, stage2_elbow_curve, find_elbow
from models import make_model_pipeline, compute_cv_folds
from sklearn.preprocessing import StandardScaler

SAMPLE_RATES = [1000, 500, 250]
CHANNEL_COUNTS = [8, 5, 2]
BIT_DEPTHS = [16, 10, 6]
MODERATE_SNR_DB = SNR_LEVELS_DB["moderate"]

WINDOW_MS_SWEEP = 200
HOP_MS_SWEEP = 50
ONSET_MAV_WINDOW_MS = 100
ONSET_THRESHOLD_MULT = 4.0
ONSET_SUSTAIN_MS = 200
REST_GUARD_MS = 250


def channel_importance_ranking(participant_combined: str, channels: list[str]) -> list[str]:
    """Ranks channels by the best (lowest-index) rank of any of their
    features in the existing, already-computed clean-data Stage-1 MI
    ranking -- reusing real prior results rather than a fresh, redundant
    computation."""
    fs_results = json.load(open(CACHE_DIR / "feature_selection_results_norest.json"))
    ranked = fs_results[participant_combined]["ranked_features"]
    best_rank = {}
    for ch in channels:
        positions = [i for i, f in enumerate(ranked) if f.startswith(f"{ch}_")]
        best_rank[ch] = min(positions) if positions else len(ranked)
    return sorted(channels, key=lambda ch: best_rank[ch])


def mains_notch_sos(centres_hz, bandwidth_hz: float = 2.0) -> np.ndarray:
    """Cascade of second-order notches at the given frequencies (constant -3 dB width of
    `bandwidth_hz`, so Q = f / bandwidth), stacked into one sos array at RAW_FS."""
    return np.vstack([tf2sos(*iirnotch(h, h / bandwidth_hz, fs=RAW_FS)) for h in centres_hz])


def variable_rate_filter_downsample(x_raw: np.ndarray, target_fs: int, band: tuple = None,
                                    notch_hz=None) -> tuple[np.ndarray, float, float]:
    """Filters at RAW_FS with a Nyquist-safe upper cutoff for target_fs,
    then decimates. Returns (downsampled_signal, low_cutoff, high_cutoff).

    band=(low_hz, high_hz) overrides the default 100 Hz to min(499, fs/2 - 1)
    pass-band; used only by the pass-band sweep (run_band_sweep.py).
    notch_hz=[f1, f2, ...] first removes those frequencies with zero-phase notches at RAW_FS
    (mains harmonics, run_band_sweep_notch.py); None (default) applies no notch."""
    factor = RAW_FS // target_fs
    assert RAW_FS % target_fs == 0, f"{RAW_FS} not divisible by {target_fs}"
    if band is None:
        low = 100.0
        high = min(499.0, target_fs / 2 - 1)
    else:
        low, high = band
        assert high < target_fs / 2, f"upper edge {high} Hz must be below Nyquist {target_fs / 2} Hz"
    x = x_raw.astype(np.float64)
    if notch_hz:
        x = sosfiltfilt(mains_notch_sos(notch_hz), x)
    sos = butter(4, [low, high], btype="bandpass", fs=RAW_FS, output="sos")
    filtered = sosfiltfilt(sos, x)
    return filtered[::factor].astype(np.float32), low, high


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Requantizes to 2**bits levels spanning a robust (0.1-99.9 pct) range."""
    lo, hi = np.percentile(x, [0.1, 99.9])
    if hi <= lo:
        return x.copy()
    n_levels = 2 ** bits
    step = (hi - lo) / (n_levels - 1)
    q = np.round((np.clip(x, lo, hi) - lo) / step) * step + lo
    return q.astype(np.float32)


def build_segments_variable_rate(cue: np.ndarray, channels_data: dict, fs: int):
    """Minimal re-implementation of segmentation.build_segments for a
    variable sample rate -- the original hardcodes TARGET_FS-derived sample
    counts from config.py at call time; this recomputes them from the
    actual `fs` in use. Reuses the same underlying primitives
    (rolling_mav/find_sustained_crossing) and the identical onset rule."""
    mav_window = max(1, int(round(ONSET_MAV_WINDOW_MS * fs / 1000)))
    sustain_samples = max(1, int(round(ONSET_SUSTAIN_MS * fs / 1000)))
    guard_samples = max(1, int(round(REST_GUARD_MS * fs / 1000)))

    raw_segs = identify_segments(cue)
    segments = []
    for i, (cue_val, start, end) in enumerate(raw_segs):
        if cue_val == REST_CUE:
            if (end - start) > 2 * guard_samples:
                segments.append((cue_val, start + guard_samples, end - guard_samples, start + guard_samples))
            else:
                segments.append((cue_val, start, end, start))
            continue

        rest_ref = None
        if i > 0 and raw_segs[i - 1][0] == REST_CUE:
            rest_ref = raw_segs[i - 1]
        elif i < len(raw_segs) - 1 and raw_segs[i + 1][0] == REST_CUE:
            rest_ref = raw_segs[i + 1]

        onset_candidates = []
        for ch, x in channels_data.items():
            seg_x = x[start:end]
            rest_x = x[rest_ref[1]:rest_ref[2]] if rest_ref else x[cue == REST_CUE]
            rest_mav = rolling_mav(rest_x, mav_window)
            threshold = rest_mav.mean() + ONSET_THRESHOLD_MULT * rest_mav.std()
            mav = rolling_mav(seg_x, mav_window)
            idx = find_sustained_crossing(mav > threshold, sustain_samples)
            if idx is not None:
                onset_candidates.append(idx)

        trimmed_start = start + min(onset_candidates) if onset_candidates else start
        segments.append((cue_val, start, end, trimmed_start))

    # fallback for segments with no detected onset -> median lag of detected ones
    detected_lags = [ts - s for (c, s, e, ts) in segments if c != REST_CUE and ts != s]
    if detected_lags:
        median_lag = int(np.median(detected_lags))
        segments = [
            (c, s, e, s if c == REST_CUE else (ts if ts != s else min(s + median_lag, e - 1)))
            for (c, s, e, ts) in segments
        ]
    return segments


def build_one_config(raw_name: str, sample_rate: int, keep_channels: list[str], bit_depth: int,
                      dataset_label: str, snr_db: float = MODERATE_SNR_DB, band: tuple = None,
                      notch_hz=None) -> pd.DataFrame:
    """Builds the movement-only feature table for ONE raw file at ONE
    (sample_rate, channel_subset, bit_depth) configuration, on top of the
    given noise condition (defaults to round 1's "moderate"/10dB SNR, for
    exact backward compatibility -- round 2 passes -15dB explicitly).

    snr_db=None skips noise injection entirely (clean signal) -- added for
    the channel/bitrate local-maxima search, which is run with no noise
    attenuation at all, separate from the noise-severity sweeps."""
    spec = DATASETS[raw_name]
    df = load_raw(raw_name)
    rng = np.random.default_rng(0)

    channels_data = {}
    for ch in keep_channels:
        raw_or_noisy = (df[ch].to_numpy() if snr_db is None
                        else calibrate_and_inject(df[ch].to_numpy(), RAW_FS, snr_db, rng))
        down, low, high = variable_rate_filter_downsample(raw_or_noisy, sample_rate, band=band, notch_hz=notch_hz)
        channels_data[ch] = quantize(down, bit_depth)

    factor = RAW_FS // sample_rate
    cue = df["cue"].to_numpy()[::factor]
    time_s = df["time_s"].to_numpy()[::factor]
    del df

    segments = build_segments_variable_rate(cue, channels_data, sample_rate)
    segments = [s for s in segments if s[0] != REST_CUE]

    window_samples = int(round(WINDOW_MS_SWEEP * sample_rate / 1000))
    hop_samples = max(1, int(round(HOP_MS_SWEEP * sample_rate / 1000)))

    rows_meta = []
    channel_feat_cols = {}
    for seg_id, (cue_val, start, end, trimmed_start) in enumerate(segments):
        n_windows = None
        seg_feats = {}
        for ch in keep_channels:
            x = channels_data[ch][trimmed_start:end]
            xw = sliding_windows(x, window_samples, hop_samples)
            if n_windows is None:
                n_windows = xw.shape[0]
            feats = extract_channel_features(xw, fs=float(sample_rate), band=band)
            for fname, arr in feats.items():
                key = (ch, fname)
                channel_feat_cols.setdefault(key, [])
                seg_feats[key] = arr
        if n_windows == 0:
            continue
        for key, arr in seg_feats.items():
            channel_feat_cols[key].append(arr)

        w_starts = window_starts(end - trimmed_start, window_samples, hop_samples)
        w_times = time_s[trimmed_start + w_starts]
        for w_i in range(n_windows):
            rows_meta.append({
                "dataset": dataset_label, "participant": spec["participant"], "cue": cue_val,
                "segment_id": f"{raw_name}_{seg_id}", "window_time_s": w_times[w_i],
            })

    meta_df = pd.DataFrame(rows_meta)
    feat_data = {}
    for (ch, fname), arrays in channel_feat_cols.items():
        non_empty = [a for a in arrays if len(a) > 0]
        feat_data[f"{ch}_{fname}"] = np.concatenate(non_empty) if non_empty else np.array([])
    return pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(feat_data).reset_index(drop=True)], axis=1)


def evaluate_config(model_name: str, X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> float:
    """Plain (non-nested) grouped-stratified 5-fold CV -- Stage 3 is
    skipped for this sweep, see module docstring."""
    accs = []
    for fold in range(fold_ids.max() + 1):
        train_mask, test_mask = fold_ids != fold, fold_ids == fold
        needs_scaling, est = make_model_pipeline(model_name)
        Xtr, Xte = X[train_mask], X[test_mask]
        if needs_scaling:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        est.fit(Xtr, y[train_mask])
        accs.append((est.predict(Xte) == y[test_mask]).mean())
    return float(np.mean(accs))
