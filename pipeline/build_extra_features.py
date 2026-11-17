"""Build the extra (new-type) feature columns for the four movement-only participants, aligned row-for-row with the cached tables.

Uses the SAME raw -> preprocessing -> segmentation -> sliding-window path as build_features_norest.py / run_wallace_norest.py, so the windows
are identical; nothing existing is touched. Output (new files): cache/{P1_combined,P2_combined,Wallace_S1,Wallace_S2}_noRest_extra.parquet.

Checks run before saving (a failed check stops the script): same row count, cue, segment_id and window time as the cached table, AND the RMS and
WL recomputed here on every channel equal the cached RMS and WL columns (proves the windows are the same samples).

Choices made once, not tuned: AR order 4 (Yule-Walker on the demeaned window), cepstral coefficients from the AR coefficients with the standard
recursion (as remembered, not checked against a reference), WAMP threshold = 25 % of the window SD, spectral flatness over the 100-499 Hz band.

Usage: python build_extra_features.py [P1_combined_noRest ...]   (default: all four)
"""
import sys
import time

import numpy as np
import pandas as pd

from config import DATASETS, COMBINED_GROUPS, CACHE_DIR, REST_CUE, TARGET_FS
from data_io import load_raw
from preprocessing import preprocess_dataframe
from segmentation import build_segments
from features import _sliding_windows, window_starts
from extended_pool import EXTRA_NAMES, extra_path
from wallace_data import PARTICIPANTS as WALLACE, load_wallace_run

AR_ORDER = 4
WAMP_FRAC = 0.25
TINY = 1e-300


def extra_channel_features(xw, fs=TARGET_FS):
    """xw: (n_windows, N). Returns {name: (n_windows,)} for EXTRA_NAMES."""
    n, N = xw.shape
    out = {}
    xc = xw - xw.mean(axis=1, keepdims=True)
    dx = np.diff(xw, axis=1)
    ddx = np.diff(dx, axis=1)
    sd = xw.std(axis=1)
    rms = np.sqrt((xw ** 2).mean(axis=1))

    out["DASDV"] = np.sqrt((dx ** 2).mean(axis=1))
    out["LogDet"] = np.exp(np.log(np.abs(xw) + 1e-12 + 1e-9 * rms[:, None]).mean(axis=1))
    out["WAMP"] = (np.abs(dx) > WAMP_FRAC * sd[:, None]).sum(axis=1).astype(float)
    out["MaxAbs"] = np.abs(xw).max(axis=1)

    var_x, var_dx, var_ddx = xw.var(axis=1), dx.var(axis=1), ddx.var(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mob = np.sqrt(np.where(var_x > 0, var_dx / var_x, 0.0))
        mob_dx = np.sqrt(np.where(var_dx > 0, var_ddx / var_dx, 0.0))
        out["HjMob"] = mob
        out["HjCx"] = np.where(mob > 0, mob_dx / mob, 0.0)

    # AR coefficients: Yule-Walker, x[t] ~ sum_k phi_k x[t-k]
    r = np.stack([(xc[:, : N - k] * xc[:, k:]).sum(axis=1) / N for k in range(AR_ORDER + 1)], axis=1)  # (n, 5)
    idx = np.abs(np.arange(AR_ORDER)[:, None] - np.arange(AR_ORDER)[None, :])
    R = r[:, idx]                                    # (n, 4, 4) Toeplitz
    rhs = r[:, 1: AR_ORDER + 1].copy()
    R = R + (1e-9 * r[:, 0])[:, None, None] * np.eye(AR_ORDER)[None]
    bad = r[:, 0] <= 0
    R[bad] = np.eye(AR_ORDER)
    rhs[bad] = 0.0
    phi = np.linalg.solve(R, rhs[..., None])[..., 0]  # (n, 4)
    for k in range(AR_ORDER):
        out[f"AR{k + 1}"] = phi[:, k]
    # cepstral coefficients from the AR coefficients (a = -phi): c1 = phi1; c_k = phi_k + sum_{i<k} (1 - i/k) phi_i c_{k-i}
    c = np.zeros((n, AR_ORDER))
    for k in range(1, AR_ORDER + 1):
        ck = phi[:, k - 1].copy()
        for i in range(1, k):
            ck += (1 - i / k) * phi[:, i - 1] * c[:, k - i - 1]
        c[:, k - 1] = ck
        out[f"CC{k}"] = ck

    # spectral spread and flatness
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    P = np.abs(np.fft.rfft(xw, axis=1)) ** 2
    tot = np.where(P.sum(axis=1) > 0, P.sum(axis=1), 1.0)
    mnf = (P * freqs[None, :]).sum(axis=1) / tot
    out["SpecSpread"] = np.sqrt((P * (freqs[None, :] - mnf[:, None]) ** 2).sum(axis=1) / tot)
    band = (freqs >= 100.0) & (freqs <= 499.0)
    Pb = P[:, band]
    mean_pb = Pb.mean(axis=1)
    eps = 1e-12 * mean_pb[:, None] + TINY
    out["SpecFlat"] = np.where(mean_pb > 0, np.exp(np.log(Pb + eps).mean(axis=1)) / np.where(mean_pb > 0, mean_pb, 1.0), 0.0)
    assert set(out) == set(EXTRA_NAMES), set(out) ^ set(EXTRA_NAMES)
    return out


def process(dfp, channels, segs, prefix):
    """Mirror of the existing builders' segment/window loop. Returns (meta, feature columns, RMS/WL check columns)."""
    meta = {"segment_id": [], "cue": [], "window_time_s": []}
    cols = {}
    check = {}
    for seg_id, seg in enumerate(segs):
        s0, s1 = seg.trimmed_start, seg.end
        xws = [_sliding_windows(dfp[ch].to_numpy()[s0:s1]) for ch in channels]
        nw = xws[0].shape[0]
        if not nw:
            continue
        rmss = []
        for ch, xw in zip(channels, xws):
            for k, v in extra_channel_features(xw).items():
                cols.setdefault(f"{ch}_{k}", []).append(v)
            rms = np.sqrt((xw ** 2).mean(axis=1))
            rmss.append(rms)
            check.setdefault(f"{ch}_RMS", []).append(rms)
            check.setdefault(f"{ch}_WL", []).append(np.abs(np.diff(xw, axis=1)).sum(axis=1))
        # cross-channel: Pearson correlation of every channel pair inside the window; each channel's share of total RMS
        Z = []
        for xw in xws:
            xc = xw - xw.mean(axis=1, keepdims=True)
            Z.append(xc / (np.sqrt((xc ** 2).sum(axis=1, keepdims=True)) + TINY))
        for a in range(len(channels)):
            for b in range(a + 1, len(channels)):
                cols.setdefault(f"xcorr_{channels[a]}__{channels[b]}", []).append((Z[a] * Z[b]).sum(axis=1))
        tot = np.sum(rmss, axis=0) + TINY
        for ch, rms in zip(channels, rmss):
            cols.setdefault(f"rmsshare_{ch}", []).append(rms / tot)
        w_times = dfp["time_s"].to_numpy()[s0 + window_starts(s1 - s0)]
        meta["segment_id"] += [f"{prefix}_{seg_id}"] * nw
        meta["cue"] += [seg.cue] * nw
        meta["window_time_s"] += list(w_times)
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return pd.DataFrame(meta), pd.DataFrame(cat(cols)), pd.DataFrame(cat(check))


def build_vaskov_member(name):
    spec = DATASETS[name]
    dfp = preprocess_dataframe(load_raw(name))
    segs = [s for s in build_segments(dfp, name) if s.cue != REST_CUE]
    return process(dfp, spec["channels"], segs, name)


def build_wallace_run(participant, run_name):
    channels = WALLACE[participant]["channels"]
    dfp = preprocess_dataframe(load_wallace_run(participant, run_name))
    segs = [s for s in build_segments(dfp, run_name) if s.cue != REST_CUE]
    return process(dfp, channels, segs, run_name)


def build(part):
    t0 = time.time()
    base_name = part.replace("_noRest", "")
    if base_name in COMBINED_GROUPS:
        pieces = [build_vaskov_member(m) for m in COMBINED_GROUPS[base_name]]
    else:
        pieces = [build_wallace_run(base_name, rn) for rn in WALLACE[base_name]["run_for_task"].values()]
    meta = pd.concat([p[0] for p in pieces], ignore_index=True)
    feats = pd.concat([p[1] for p in pieces], ignore_index=True)
    chk = pd.concat([p[2] for p in pieces], ignore_index=True)

    base = pd.read_parquet(CACHE_DIR / f"{part}.parquet")
    assert len(base) == len(meta), f"row count {len(base)} vs {len(meta)}"
    assert (base["segment_id"].to_numpy() == meta["segment_id"].to_numpy()).all(), "segment_id differs"
    assert (base["cue"].to_numpy() == meta["cue"].to_numpy()).all(), "cue differs"
    assert np.allclose(base["window_time_s"].to_numpy(), meta["window_time_s"].to_numpy()), "window times differ"
    for c in chk.columns:
        assert np.allclose(base[c].to_numpy(), chk[c].to_numpy(), rtol=1e-9, atol=0), f"recomputed {c} differs from the cached column"
    assert np.isfinite(feats.to_numpy(dtype=float)).all(), "non-finite new feature value"
    out = pd.concat([meta, feats], axis=1)
    out.to_parquet(extra_path(part), index=False)
    print(f"{part}: {len(out)} windows, {feats.shape[1]} new columns; alignment and RMS/WL recomputation checks PASSED ({time.time() - t0:.0f}s) -> {extra_path(part).name}", flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["P1_combined_noRest", "P2_combined_noRest", "Wallace_S1_noRest", "Wallace_S2_noRest"]
    for p in parts:
        build(p)
    print("All done")
