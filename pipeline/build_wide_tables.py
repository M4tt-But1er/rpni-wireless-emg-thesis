"""Build the Set A/B/C feature tables of docs/stage1_declared_protocol.md for the four patients at the two declared operating points,
through exactly the sweep's own signal path (bitrate_sweep.build_one_config for Vaskov P1/P2, wallace_bitrate.build_wallace_config for Wallace S1/S2):
filter at 30 kHz -> decimate -> requantise -> variable-rate segmentation -> sliding windows.

One table per (patient, operating point): cache/stage1/<patient>__<op>.parquet, with the metadata columns of the sweep tables and every column of set C
(sets A and B are column subsets: wide_features.set_columns). Nothing existing is overwritten.

Usage:
  python build_wide_tables.py                      build all 8 tables (skips existing ones)
  python build_wide_tables.py P1_combined low      build one
  python build_wide_tables.py --validate           checks V1 (set A equals the sweep builders exactly), V2 (flatness equals build_extra_features at 1 kHz), V3, V4
"""
import sys
import time

import os

import numpy as np
import pandas as pd

from bitrate_sweep import (build_one_config, build_segments_variable_rate, quantize, variable_rate_filter_downsample)
from config import CACHE_DIR, DATASETS, RAW_FS, REST_CUE
from data_io import load_raw
from run_bitrate_search import DATASET_SPECS
from vaskov_features import sliding_windows, window_starts
from wallace_bitrate import HOP_MS, WINDOW_MS, build_wallace_config, global_quant_ranges, quantize_fixed
from wallace_data import PARTICIPANTS as WALLACE, load_wallace_run
from wide_features import feature_group, segment_columns, set_columns

OPS = {"ref": (1000, 16), "low": (500, 8)}
PARTS = ["P1_combined", "P2_combined", "Wallace_S1", "Wallace_S2"]
OUT_DIR = CACHE_DIR / "stage1"
# STAGE1_TAG=fixedrange: Wallace tables use ONE quantiser range per channel (pooled over the participant's runs, as a real ADC) and live in cache/stage1_fixedrange/.
# Added 9 Nov 2026 after finding that the per-recording range leaks the grip class for Wallace (docs/stage1_results.md, correction banner). P1/P2 are unaffected and keep their tables.
TAG = os.environ.get("STAGE1_TAG", "")


def table_path(part, op):
    if TAG and part.startswith("Wallace"):
        return CACHE_DIR / f"stage1_{TAG}" / f"{part}__{op}.parquet"
    return OUT_DIR / f"{part}__{op}.parquet"


def channels_of(part):
    return DATASET_SPECS[part]["all_channels"] if part.startswith("P") else WALLACE[part]["channels"]


def sources(part, sr, bits):
    """Yield (channels_data, cue, time_s, segment-id prefix, participant, dataset label) per raw source, as the sweep builders do."""
    channels = channels_of(part)
    if part.startswith("P"):
        for raw_name in DATASET_SPECS[part]["raw_names"]:
            df = load_raw(raw_name)
            data = {ch: quantize(variable_rate_filter_downsample(df[ch].to_numpy(), sr)[0], bits) for ch in channels}
            factor = RAW_FS // sr
            cue, time_s = df["cue"].to_numpy()[::factor], df["time_s"].to_numpy()[::factor]
            del df
            yield data, cue, time_s, raw_name, DATASETS[raw_name]["participant"], f"{part}_search"
    else:
        ranges = global_quant_ranges(part, sr, channels) if TAG else None
        for run_name in WALLACE[part]["run_for_task"].values():
            raw_df = load_wallace_run(part, run_name)
            factor = 30000 // sr
            if TAG:
                data = {ch: quantize_fixed(variable_rate_filter_downsample(raw_df[ch].to_numpy(), sr)[0], bits, *ranges[ch]) for ch in channels}
            else:
                data = {ch: quantize(variable_rate_filter_downsample(raw_df[ch].to_numpy(), sr)[0], bits) for ch in channels}
            cue, time_s = raw_df["cue"].to_numpy()[::factor], raw_df["time_s"].to_numpy()[::factor]
            yield data, cue, time_s, run_name, part, f"{part}_bitrate"


def build(part, op):
    sr, bits = OPS[op]
    t0 = time.time()
    channels = channels_of(part)
    window_samples = int(round(WINDOW_MS * sr / 1000))
    hop_samples = max(1, int(round(HOP_MS * sr / 1000)))
    metas, colchunks = [], {}
    for data, cue, time_s, prefix, participant, label in sources(part, sr, bits):
        segments = [s for s in build_segments_variable_rate(cue, data, sr) if s[0] != REST_CUE]
        for seg_id, (cue_val, start, end, trimmed_start) in enumerate(segments):
            xws = [sliding_windows(data[ch][trimmed_start:end], window_samples, hop_samples) for ch in channels]
            n = xws[0].shape[0]
            if n == 0:
                continue
            for name, arr in segment_columns(xws, channels, sr).items():
                colchunks.setdefault(name, []).append(arr)
            w_times = time_s[trimmed_start + window_starts(end - trimmed_start, window_samples, hop_samples)]
            metas.append(pd.DataFrame({"dataset": label, "participant": participant, "cue": cue_val,
                                       "segment_id": f"{prefix}_{seg_id}", "window_time_s": w_times}))
    meta = pd.concat(metas, ignore_index=True)
    feats = pd.DataFrame({k: np.concatenate(v) for k, v in colchunks.items()})
    assert len(meta) == len(feats)
    assert np.isfinite(feats.to_numpy(dtype=float)).all(), "non-finite feature value"
    out = pd.concat([meta, feats], axis=1)
    table_path(part, op).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(table_path(part, op), index=False)
    cols = list(feats.columns)
    print(f"{part} {op}: {len(out)} windows; columns A {len(set_columns(cols, 'A'))}, B {len(set_columns(cols, 'B'))}, C {len(cols)}; {time.time() - t0:.0f}s", flush=True)


def validate():
    """V1-V4 of the declared protocol. Raises on failure."""
    from build_extra_features import extra_channel_features
    from wide_features import passband, spec_flat
    ok = True
    for part in ["P1_combined", "Wallace_S1"]:
        for op in OPS:
            sr, bits = OPS[op]
            new = pd.read_parquet(table_path(part, op))
            channels = channels_of(part)
            old = (pd.concat([build_one_config(r, sr, channels, bits, f"{part}_search", snr_db=None) for r in DATASET_SPECS[part]["raw_names"]], ignore_index=True)
                   if part.startswith("P") else build_wallace_config(part, sr, channels, bits, quant_ranges=(global_quant_ranges(part, sr, channels) if TAG else None)))
            same_meta = all((old[c].to_numpy() == new[c].to_numpy()).all() for c in ["cue", "segment_id"]) and np.allclose(old["window_time_s"], new["window_time_s"])
            old_feats = [c for c in old.columns if c not in ("dataset", "participant", "cue", "segment_id", "window_time_s")]
            a_cols = set_columns(list(new.columns[5:]), "A")
            exact = len(old) == len(new) and set(old_feats) == set(a_cols) and all(np.array_equal(old[c].to_numpy(), new[c].to_numpy()) for c in old_feats)
            maxdiff = max(float(np.abs(old[c].to_numpy() - new[c].to_numpy()).max()) for c in old_feats) if len(old) == len(new) else float("nan")
            print(f"V1 {part} {op}: rows {len(old)} vs {len(new)}; meta equal {same_meta}; {len(old_feats)} A columns exactly equal: {exact} (max abs diff {maxdiff:g})")
            ok &= bool(same_meta and exact)
    # V2: the fs-aware flatness equals the build_extra_features expression at 1 kHz on real windows
    df = load_raw("P1_1of12")
    x = quantize(variable_rate_filter_downsample(df[channels_of("P1_combined")[0]].to_numpy()[:3_000_000], 1000)[0], 16)
    xw = sliding_windows(x[:20000], 200, 50).astype(np.float64)
    P = np.abs(np.fft.rfft(xw, axis=1)) ** 2
    freqs = np.fft.rfftfreq(200, d=1e-3)
    lo, hi = passband(1000)
    eq = np.array_equal(spec_flat(P, freqs, lo, hi), extra_channel_features(xw, 1000)["SpecFlat"])
    print(f"V2 flatness at 1 kHz equals build_extra_features exactly on {len(xw)} real windows: {eq}")
    ok &= bool(eq)
    # V3 / V4 on all existing tables
    for part in PARTS:
        for op in OPS:
            if not table_path(part, op).exists():
                continue
            t = pd.read_parquet(table_path(part, op))
            cols = list(t.columns[5:])
            finite = np.isfinite(t[cols].to_numpy(dtype=float)).all()
            const = [c for c in cols if t[c].std() == 0]
            nA, nB, nC = (len(set_columns(cols, s)) for s in "ABC")
            print(f"V3/V4 {part} {op}: finite {finite}; columns A {nA}, B {nB}, C {nC}; constant columns {len(const)} {const[:6]}")
            ok &= bool(finite)
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--validate"]:
        sys.exit(0 if validate() else 1)
    todo = [(args[0], args[1])] if len(args) == 2 else [(p, o) for p in PARTS for o in OPS]
    for p, o in todo:
        if table_path(p, o).exists():
            print(f"{p} {o}: exists, skipped", flush=True)
            continue
        build(p, o)
    print("All done")
