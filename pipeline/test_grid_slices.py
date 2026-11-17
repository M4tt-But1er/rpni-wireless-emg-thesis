"""V5 of the grid protocol: a channel-subset slice of the all-channel columns equals direct computation of the wide features on the subset's windows
(per-channel, xcorr, bandcorr: exact; rmsshare: recomputed). Also: the in-memory fixed quantiser ranges equal wallace_bitrate.global_quant_ranges.
Usage: python -W ignore test_grid_slices.py"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from bitrate_sweep import build_segments_variable_rate, quantize
from build_wide_tables import channels_of
from config import REST_CUE
from grid_slices import column_channels, subset_columns
from grid_tables import decimated_sources, wallace_ranges
from vaskov_features import sliding_windows
from wallace_bitrate import HOP_MS, WINDOW_MS, global_quant_ranges, quantize_fixed
from wide_features import TINY, segment_columns


def canon(name, channels):
    """Pair columns are named a__b in the order of the channel list; direct computation on a subset may order the pair differently (values are symmetric), so compare by prefix + sorted pair."""
    kind, chs = column_channels(name, channels)
    return name.split("_")[0] + "|" + "&".join(sorted(chs)) if kind == "pair" else name


def check(part, rate, bits, keep_idx):
    channels = channels_of(part)
    keep = [channels[i] for i in keep_idx]
    srcs = decimated_sources(part, rate)
    ranges = wallace_ranges(srcs, channels) if part.startswith("Wallace") else None
    data_f, cue, time_s, *_ = srcs[0]
    data = {ch: (quantize(data_f[ch], bits) if part.startswith("P") else quantize_fixed(data_f[ch], bits, *ranges[ch])) for ch in channels}
    ws, hs = int(round(WINDOW_MS * rate / 1000)), max(1, int(round(HOP_MS * rate / 1000)))
    segs = [s for s in build_segments_variable_rate(cue, data, rate) if s[0] != REST_CUE][:4]
    worst, n_cols, by_kind = 0.0, 0, {}
    for _, start, end, ts in segs:
        xws = {ch: sliding_windows(data[ch][ts:end], ws, hs) for ch in channels}
        if xws[channels[0]].shape[0] == 0:
            continue
        full = segment_columns([xws[ch] for ch in channels], channels, rate)
        direct = segment_columns([xws[ch] for ch in keep], keep, rate)
        rms = np.stack([full[f"{ch}_RMS"] for ch in keep]); tot = rms.sum(axis=0) + TINY
        sliced = {}
        for c in subset_columns(list(full), channels, keep):
            kind, chs = column_channels(c, channels)
            sliced[c] = rms[keep.index(chs[0])] / tot if kind == "share" else full[c]
        d = {canon(k, keep): v for k, v in direct.items()}
        s = {canon(k, channels): v for k, v in sliced.items()}
        assert set(d) == set(s), (sorted(set(d) ^ set(s))[:6])
        for k in d:
            diff = float(np.abs(d[k] - s[k]).max())
            kind = "rmsshare" if k.startswith("rmsshare") else k.split("|")[0] if "|" in k else "per-channel"
            by_kind[kind] = max(by_kind.get(kind, 0.0), diff)
            worst = max(worst, diff)
        n_cols = len(s)
    print(f"V5 {part} {rate} Hz {bits} bit, channels {keep_idx}: {n_cols} columns compared on {len(segs)} segments, max abs difference {worst:g}; by kind "
          + ", ".join(f"{k} {v:.2g}" for k, v in sorted(by_kind.items())))
    # per-channel and pair columns must be bit-identical or equal to float32 rounding; rmsshare is recomputed in float64 from float32 inputs (float32 eps = 1.2e-7)
    return all(v == 0.0 for k, v in by_kind.items() if k == "per-channel") and worst < 1e-6


if __name__ == "__main__":
    ok = True
    ok &= check("Wallace_S1", 1000, 8, [0, 2, 5])
    ok &= check("P1_combined", 500, 8, [1, 3])
    ok &= check("Wallace_S2", 250, 12, [0, 4, 7, 9, 11])
    srcs = decimated_sources("Wallace_S1", 500)
    mine = wallace_ranges(srcs, channels_of("Wallace_S1"))
    ref = global_quant_ranges("Wallace_S1", 500, channels_of("Wallace_S1"))
    same = all(np.allclose(mine[c], ref[c], rtol=0, atol=0) for c in mine)
    print(f"fixed quantiser ranges equal wallace_bitrate.global_quant_ranges: {same}")
    ok &= same
    print("V5 PASSED" if ok else "V5 FAILED")
    sys.exit(0 if ok else 1)
