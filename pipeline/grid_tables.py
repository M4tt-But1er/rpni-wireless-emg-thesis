"""All-channel wide-feature tables for the bitrate grid (docs/grid_declared_protocol.md): one table per (patient, sample rate, bit depth), every column of set C (A, B, C families, pairs).
Channel subsets are cut out of these tables later (grid_slices.py); every channel subset therefore shares the same windows, labels and folds (a labelled deviation from the old sweep, which segmented per channel subset).

Signal path = the sweep's own: filter at 30 kHz (100 Hz to min(499, fs/2 - 1)) -> decimate -> quantise -> variable-rate segmentation -> 200 ms windows with 50 ms hop.
Vaskov: quantiser range per channel per raw file (those files mix many grips, checked 9 Nov 2026). Wallace: ONE range per channel pooled over the participant's runs (a real ADC), never per recording.
Output: cache/grid/tables/<patient>__<rate>Hz_<bits>bit.parquet (5 meta columns + feature columns), log cache/grid/tables.log (stdout).
Usage: python -W ignore -u grid_tables.py [--workers N] [--rates 250 500 ...] [--parts P1_combined ...]"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bitrate_sweep import build_segments_variable_rate, quantize, variable_rate_filter_downsample
from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR, DATASETS, RAW_FS, REST_CUE
from data_io import load_raw
from run_bitrate_search import DATASET_SPECS
from vaskov_features import sliding_windows, window_starts
from wallace_bitrate import HOP_MS, WINDOW_MS, quantize_fixed
from wallace_data import load_wallace_run, PARTICIPANTS as WALLACE
from wide_features import segment_columns

RATES = [250, 500, 1000, 3000, 10000]
BITS = [8, 12, 16]
GRID_DIR = CACHE_DIR / "grid"
TABLE_DIR = GRID_DIR / "tables"


def table_path(part, rate, bits):
    return TABLE_DIR / f"{part}__{rate}Hz_{bits}bit.parquet"


def decimated_sources(part, rate):
    """[(data: ch -> filtered, decimated float32, cue, time_s, segment-id prefix, participant, dataset label)] per raw file / run, not yet quantised."""
    channels = channels_of(part)
    out = []
    if part.startswith("P"):
        for raw_name in DATASET_SPECS[part]["raw_names"]:
            df = load_raw(raw_name)
            data = {ch: variable_rate_filter_downsample(df[ch].to_numpy(), rate)[0] for ch in channels}
            factor = RAW_FS // rate
            out.append((data, df["cue"].to_numpy()[::factor], df["time_s"].to_numpy()[::factor], raw_name, DATASETS[raw_name]["participant"], f"{part}_grid"))
            del df
    else:
        for run_name in WALLACE[part]["run_for_task"].values():
            raw_df = load_wallace_run(part, run_name)
            factor = 30000 // rate
            data = {ch: variable_rate_filter_downsample(raw_df[ch].to_numpy(), rate)[0] for ch in channels}
            out.append((data, raw_df["cue"].to_numpy()[::factor], raw_df["time_s"].to_numpy()[::factor], run_name, part, f"{part}_grid"))
    return out


def wallace_ranges(srcs, channels):
    """One (lo, hi) per channel, 0.1-99.9 percentile of ALL runs pooled (same rule as wallace_bitrate.global_quant_ranges)."""
    return {ch: tuple(np.percentile(np.concatenate([s[0][ch] for s in srcs]), [0.1, 99.9])) for ch in channels}


def build_table(srcs, part, rate, bits, ranges):
    channels = channels_of(part)
    window_samples = int(round(WINDOW_MS * rate / 1000))
    hop_samples = max(1, int(round(HOP_MS * rate / 1000)))
    metas, colchunks = [], {}
    for data_f, cue, time_s, prefix, participant, label in srcs:
        if part.startswith("P"):
            data = {ch: quantize(data_f[ch], bits) for ch in channels}
        else:
            data = {ch: quantize_fixed(data_f[ch], bits, *ranges[ch]) for ch in channels}
        segments = [s for s in build_segments_variable_rate(cue, data, rate) if s[0] != REST_CUE]
        for seg_id, (cue_val, start, end, trimmed_start) in enumerate(segments):
            xws = [sliding_windows(data[ch][trimmed_start:end], window_samples, hop_samples) for ch in channels]
            n = xws[0].shape[0]
            if n == 0:
                continue
            for name, arr in segment_columns(xws, channels, rate).items():
                colchunks.setdefault(name, []).append(arr)
            w_times = time_s[trimmed_start + window_starts(end - trimmed_start, window_samples, hop_samples)]
            metas.append(pd.DataFrame({"dataset": label, "participant": participant, "cue": cue_val,
                                       "segment_id": f"{prefix}_{seg_id}", "window_time_s": w_times}))
    meta = pd.concat(metas, ignore_index=True)
    feats = pd.DataFrame({k: np.concatenate(v) for k, v in colchunks.items()})
    assert len(meta) == len(feats)
    # Amendment 1 of docs/grid_declared_protocol.md: a window whose quantised samples are all equal (low bit depth, weak channel) has zero variance, so the ORIGINAL Skew and Kurtosis are NaN there
    # (scipy). Non-finite values are set to 0 (symmetric / normal-like); the count per column is logged.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    bad = feats.isna().sum()
    bad = {k: int(v) for k, v in bad.items() if v}
    feats = feats.fillna(0.0)
    assert np.isfinite(feats.to_numpy(dtype=float)).all()
    return pd.concat([meta, feats], axis=1), bad


def run_part_rate(part, rate):
    import warnings
    warnings.filterwarnings("ignore")
    todo = [b for b in BITS if not table_path(part, rate, b).exists()]
    if not todo:
        return f"{part} {rate} Hz: all exist"
    t0 = time.time()
    srcs = decimated_sources(part, rate)
    ranges = wallace_ranges(srcs, channels_of(part)) if part.startswith("Wallace") else None
    t_dec = time.time() - t0
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    msg = []
    for bits in todo:
        t1 = time.time()
        out, bad = build_table(srcs, part, rate, bits, ranges)
        out.to_parquet(table_path(part, rate, bits), index=False)
        msg.append(f"{bits} bit: {len(out)} windows, {out.shape[1] - 5} columns, {time.time() - t1:.0f}s"
                   + (f", NON-FINITE set to 0: {sum(bad.values())} values in {len(bad)} columns {list(bad.items())[:4]}" if bad else ""))
    return f"{part} {rate} Hz (load+filter {t_dec:.0f}s): " + "; ".join(msg)


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=str, many=False):
        if name not in args:
            return default
        vals = []
        for a in args[args.index(name) + 1:]:
            if a.startswith("--"):
                break
            vals.append(cast(a))
        return vals if many else vals[0]

    workers = opt("--workers", 1, int)
    rates = opt("--rates", RATES, int, many=True)
    parts = opt("--parts", PARTS, str, many=True)
    tasks = [(p, r) for r in rates for p in parts]
    print(f"{len(tasks)} (patient, rate) tasks, {workers} worker(s)", flush=True)
    for msg in Parallel(n_jobs=workers, return_as="generator")(delayed(run_part_rate)(p, r) for p, r in tasks):
        print(msg, flush=True)
    print("All done")


if __name__ == "__main__":
    main()
