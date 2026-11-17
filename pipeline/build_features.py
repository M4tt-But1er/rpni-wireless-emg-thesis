"""Full per-dataset feature matrix assembly + Parquet caching.

One row per (segment, window). Columns: metadata (dataset, participant, cue,
segment_id, onset_flagged, window start time) + {channel}_{feature} for every
channel x feature-family combination.
"""
import time

import numpy as np
import pandas as pd

from config import DATASETS, COMBINED_GROUPS, CACHE_DIR, TARGET_FS, REST_CUE
from data_io import load_raw
from preprocessing import preprocess_dataframe
from segmentation import build_segments
from features import extract_channel_features, _sliding_windows, FEATURE_NAMES, WINDOW_SAMPLES


def build_feature_matrix(name: str, verbose: bool = True, exclude_cues=None,
                          dataset_label: str = None, noise_snr_db: float = None,
                          noise_seed: int = 0, noise_color: str = "shaped") -> pd.DataFrame:
    """exclude_cues/dataset_label are additive, opt-in params (default
    behavior unchanged) used to build cue-filtered variants of an existing
    raw dataset -- e.g. P1_1of12 with the paper-benchmark's excluded grasp
    cues removed -- without touching the original cached feature matrices.

    noise_snr_db: if given, injects calibrated frequency-shaped Gaussian
    noise (see noise_injection.py) into every channel of the RAW signal
    before the standard filter/downsample/segment/feature pipeline runs --
    each channel is calibrated independently (channels have very different
    native amplitude scales) to hit this exact in-band SNR."""
    spec = DATASETS[name]
    channels = spec["channels"]
    exclude_cues = set(exclude_cues or [])
    label = dataset_label or name

    t0 = time.time()
    df = load_raw(name)
    if noise_snr_db is not None:
        from noise_injection import calibrate_and_inject
        from config import RAW_FS
        rng = np.random.default_rng(noise_seed)
        for ch in channels:
            df[ch] = calibrate_and_inject(df[ch].to_numpy(), RAW_FS, noise_snr_db, rng, noise_color=noise_color)
    dfp = preprocess_dataframe(df)
    del df
    segs = build_segments(dfp, name)
    if exclude_cues:
        segs = [s for s in segs if s.cue not in exclude_cues]
    if verbose:
        print(f"[{label}] loaded+preprocessed in {time.time()-t0:.1f}s, {len(segs)} segments"
              + (f" ({len(exclude_cues)} cue(s) excluded)" if exclude_cues else ""))

    rows_meta = []
    channel_feat_cols = {}  # (channel, feature_name) -> list of arrays (one per segment)

    for ch in channels:
        for fname in FEATURE_NAMES:
            channel_feat_cols[(ch, fname)] = []

    t0 = time.time()
    for seg_id, seg in enumerate(segs):
        seg_start, seg_end = seg.trimmed_start, seg.end
        seg_len = seg_end - seg_start
        starts = None
        n_windows = None

        for ch in channels:
            x = dfp[ch].to_numpy()[seg_start:seg_end]
            xw = _sliding_windows(x)
            if n_windows is None:
                n_windows = xw.shape[0]
            feats = extract_channel_features(xw)
            for fname in FEATURE_NAMES:
                channel_feat_cols[(ch, fname)].append(feats[fname])

        if n_windows == 0:
            continue

        from features import window_starts
        w_starts = window_starts(seg_len)
        window_times = dfp["time_s"].to_numpy()[seg_start + w_starts]

        for w_i in range(n_windows):
            rows_meta.append({
                "dataset": label,
                "participant": spec["participant"],
                "cue": seg.cue,
                "segment_id": f"{name}_{seg_id}",
                "onset_flagged": seg.onset_flagged if seg.cue != REST_CUE else False,
                "window_time_s": window_times[w_i],
            })

    meta_df = pd.DataFrame(rows_meta)

    feat_data = {}
    for (ch, fname), arrays in channel_feat_cols.items():
        non_empty = [a for a in arrays if len(a) > 0]
        if non_empty:
            feat_data[f"{ch}_{fname}"] = np.concatenate(non_empty)
        else:
            feat_data[f"{ch}_{fname}"] = np.array([])

    feat_df = pd.DataFrame(feat_data)
    result = pd.concat([meta_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    if verbose:
        print(f"[{label}] feature extraction in {time.time()-t0:.1f}s -> shape {result.shape}")

    return result


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset = {}
    for name in DATASETS:
        fm = build_feature_matrix(name)
        out_path = CACHE_DIR / f"{name}.parquet"
        fm.to_parquet(out_path, index=False)
        print(f"cached {name} -> {out_path} ({fm.shape})")
        per_dataset[name] = fm

    for combined_name, members in COMBINED_GROUPS.items():
        combined = pd.concat([per_dataset[m] for m in members], axis=0, ignore_index=True)
        out_path = CACHE_DIR / f"{combined_name}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"cached {combined_name} -> {out_path} ({combined.shape})")


if __name__ == "__main__":
    main()
