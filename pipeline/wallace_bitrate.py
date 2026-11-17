"""Wallace equivalent of bitrate_sweep.py's build_one_config -- variable
sample-rate/channel-subset/bit-depth feature tables for Wallace_S1/S2,
reusing the exact same primitives (variable_rate_filter_downsample, quantize,
build_segments_variable_rate, extract_channel_features) so results are
methodologically identical to the Vaskov bitrate sweep. The only thing that
differs is sourcing: Wallace's data comes from multiple per-grip SNR .mat
runs (wallace_data.load_wallace_run) instead of one continuous raw CSV, so
each run is downsampled/quantized/segmented separately, then the resulting
per-run feature tables are concatenated -- mirroring run_wallace_norest.py's
multi-run assembly, just at a variable (sample_rate, channels, bit_depth)
instead of the native 30kHz/full-channel/full-resolution config.

Rest is excluded after segmentation (movement-only, matching every other
Wallace_*_noRest table in this project). No noise injection here -- this is
the channel/bitrate local-maxima search, run clean per the user's request.
"""
import json

import numpy as np
import pandas as pd

from config import CACHE_DIR, REST_CUE
from bitrate_sweep import variable_rate_filter_downsample, quantize, build_segments_variable_rate
from features import extract_channel_features
from vaskov_features import window_starts, sliding_windows
from wallace_data import PARTICIPANTS, load_wallace_run

WINDOW_MS = 200
HOP_MS = 50


def channel_importance_ranking_wallace(participant: str) -> list[str]:
    """Same idea as bitrate_sweep.channel_importance_ranking, sourced from
    this project's Wallace Stage-1 MI ranking instead of the Vaskov one."""
    fs_results = json.load(open(CACHE_DIR / "feature_selection_results_wallace.json"))
    ranked = fs_results[participant]["ranked_features"]
    channels = PARTICIPANTS[participant]["channels"]
    best_rank = {}
    for ch in channels:
        positions = [i for i, f in enumerate(ranked) if f.startswith(f"{ch}_")]
        best_rank[ch] = min(positions) if positions else len(ranked)
    return sorted(channels, key=lambda ch: best_rank[ch])


def global_quant_ranges(participant: str, sample_rate: int, keep_channels: list[str], band: tuple = None,
                        notch_hz=None) -> dict:
    """Fixed per-channel (lo, hi) quantiser range from ALL of a participant's runs
    pooled, same 0.1-99.9 percentile rule as bitrate_sweep.quantize. A real ADC has one
    fixed input range, not a range re-fitted to each recording."""
    pooled = {ch: [] for ch in keep_channels}
    for run_name in PARTICIPANTS[participant]["run_for_task"].values():
        raw_df = load_wallace_run(participant, run_name)
        for ch in keep_channels:
            down, _, _ = variable_rate_filter_downsample(raw_df[ch].to_numpy(), sample_rate, band=band,
                                                         notch_hz=notch_hz)
            pooled[ch].append(down)
    return {ch: tuple(np.percentile(np.concatenate(v), [0.1, 99.9])) for ch, v in pooled.items()}


def quantize_fixed(x: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    step = (hi - lo) / (2 ** bits - 1)
    return (np.round((np.clip(x, lo, hi) - lo) / step) * step + lo).astype(np.float32)


def build_wallace_config(participant: str, sample_rate: int, keep_channels: list[str],
                          bit_depth: int, quant_ranges: dict = None, band: tuple = None,
                          notch_hz=None) -> pd.DataFrame:
    """quant_ranges=None keeps the original behaviour (each run quantised to its own
    percentile range). Pass global_quant_ranges(...) for one fixed range per channel."""
    """Builds the movement-only feature table for ONE Wallace participant at
    ONE (sample_rate, channel_subset, bit_depth) configuration, no noise."""
    run_for_task = PARTICIPANTS[participant]["run_for_task"]
    window_samples = int(round(WINDOW_MS * sample_rate / 1000))
    hop_samples = max(1, int(round(HOP_MS * sample_rate / 1000)))

    frames = []
    for task, run_name in run_for_task.items():
        raw_df = load_wallace_run(participant, run_name)
        factor = 30000 // sample_rate
        assert 30000 % sample_rate == 0, f"30000 not divisible by {sample_rate}"

        channels_data = {}
        for ch in keep_channels:
            down, _, _ = variable_rate_filter_downsample(raw_df[ch].to_numpy(), sample_rate, band=band,
                                                         notch_hz=notch_hz)
            if quant_ranges is None:
                channels_data[ch] = quantize(down, bit_depth)
            else:
                channels_data[ch] = quantize_fixed(down, bit_depth, *quant_ranges[ch])
        cue = raw_df["cue"].to_numpy()[::factor]
        time_s = raw_df["time_s"].to_numpy()[::factor]

        segments = build_segments_variable_rate(cue, channels_data, sample_rate)
        segments = [s for s in segments if s[0] != REST_CUE]

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
            if not n_windows:
                continue
            for key, arr in seg_feats.items():
                channel_feat_cols[key].append(arr)

            w_starts = window_starts(end - trimmed_start, window_samples, hop_samples)
            w_times = time_s[trimmed_start + w_starts]
            for w_i in range(n_windows):
                rows_meta.append({
                    "dataset": f"{participant}_bitrate", "participant": participant, "cue": cue_val,
                    "segment_id": f"{run_name}_{seg_id}", "window_time_s": w_times[w_i],
                })

        meta_df = pd.DataFrame(rows_meta)
        feat_data = {}
        for (ch, fname), arrays in channel_feat_cols.items():
            non_empty = [a for a in arrays if len(a) > 0]
            feat_data[f"{ch}_{fname}"] = np.concatenate(non_empty) if non_empty else np.array([])
        frames.append(pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(feat_data).reset_index(drop=True)], axis=1))

    return pd.concat(frames, axis=0, ignore_index=True)
