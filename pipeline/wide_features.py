"""The wide feature library (sets A, B, C) exactly as declared in docs/stage1_declared_protocol.md (written before any result).

Set A  the project's 24 per-channel types (features.extract_channel_features, unchanged)
Set B  A + the 16 per-channel types of build_extra_features.py + xcorr + rmsshare   (only change: spectral-flatness band follows the pass-band)
Set C  B + BP1-8 (sub-band powers), WP1-8 (wavelet-packet energies), SpecSkew, SpecKurt,
       dMAV1/2, dRMS1/2, dWL1/2 (change against the window 1 and 2 hops earlier in the same segment), bandcorr1-3 (band-limited correlation per channel pair)

`segment_columns(xws, channels, fs)` takes the windowed signal of ONE segment (one (n_windows, N) array per channel) and returns {column name: (n_windows,) array}.
`feature_group(col)` gives the smallest set ('A', 'B' or 'C') that contains a column; `set_columns(cols, 'B')` selects the columns of a set.
"""
import re

import numpy as np
import pywt

from build_extra_features import extra_channel_features
from extended_pool import EXTRA_NAMES
from features import FEATURE_NAMES, extract_channel_features

TINY = 1e-300
BP_NAMES = [f"BP{i}" for i in range(1, 9)]
WP_NAMES = [f"WP{i}" for i in range(1, 9)]
SHAPE_NAMES = ["SpecSkew", "SpecKurt"]   # time-domain skewness / kurtosis already exist in Set A ('Skew', 'Kurtosis'); see Amendment 1 of the protocol
DYN_NAMES = [f"d{b}{k}" for b in ("MAV", "RMS", "WL") for k in (1, 2)]
C_ONLY_NAMES = BP_NAMES + WP_NAMES + SHAPE_NAMES + DYN_NAMES
N_BANDCORR = 3


def passband(fs):
    """The same pass-band the sweep's filter uses: 100 Hz to min(499, fs/2 - 1)."""
    return 100.0, min(499.0, fs / 2 - 1)


def spec_flat(P, freqs, lo, hi):
    """Spectral flatness over [lo, hi] -- the expression of build_extra_features.py, with the band as arguments."""
    band = (freqs >= lo) & (freqs <= hi)
    Pb = P[:, band]
    mean_pb = Pb.mean(axis=1)
    eps = 1e-12 * mean_pb[:, None] + TINY
    return np.where(mean_pb > 0, np.exp(np.log(Pb + eps).mean(axis=1)) / np.where(mean_pb > 0, mean_pb, 1.0), 0.0)


def _band_masks(freqs, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    return [(freqs >= edges[k]) & ((freqs < edges[k + 1]) if k < n - 1 else (freqs <= edges[k + 1])) for k in range(n)]


def wavelet_packet_log_energies(xw, level=3, wavelet="db4"):
    nodes = [xw]
    for _ in range(level):
        nxt = []
        for a in nodes:
            ca, cd = pywt.dwt(a, wavelet, mode="symmetric", axis=1)
            nxt += [ca, cd]
        nodes = nxt
    return [np.log10((c ** 2).mean(axis=1) + TINY) for c in nodes]


def _zero_nan(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def channel_b_c(xw64, fs):
    """Per-channel B extras and C extras for float64 windows. Returns (B dict, C dict (without dynamics), complex rfft)."""
    n, N = xw64.shape
    lo, hi = passband(fs)
    B = extra_channel_features(xw64, fs)
    X = np.fft.rfft(xw64, axis=1)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    B["SpecFlat"] = spec_flat(P, freqs, lo, hi)

    C = {}
    total = P.sum(axis=1)
    for k, m in enumerate(_band_masks(freqs, lo, hi, 8)):
        C[f"BP{k + 1}"] = np.log10(P[:, m].sum(axis=1) + 1e-12 * total + TINY)
    for k, e in enumerate(wavelet_packet_log_energies(xw64)):
        C[f"WP{k + 1}"] = e
    mb = (freqs >= lo) & (freqs <= hi)
    Pb, fb = P[:, mb], freqs[mb]
    sb = Pb.sum(axis=1)
    w = Pb / np.where(sb > 0, sb, 1.0)[:, None]
    mean_f = (w * fb[None, :]).sum(axis=1)
    var_f = (w * (fb[None, :] - mean_f[:, None]) ** 2).sum(axis=1)
    ok = var_f > 0
    safe = np.where(ok, var_f, 1.0)
    C["SpecSkew"] = np.where(ok, (w * (fb[None, :] - mean_f[:, None]) ** 3).sum(axis=1) / safe ** 1.5, 0.0)
    C["SpecKurt"] = np.where(ok, (w * (fb[None, :] - mean_f[:, None]) ** 4).sum(axis=1) / safe ** 2 - 3.0, 0.0)
    return B, C, X


def segment_columns(xws, channels, fs):
    """xws: list (one per channel) of (n_windows, N) windows of one segment, as the sweep builds them. Returns an ordered dict of columns."""
    cols = {}
    Xs, rmss, Zs = [], [], []
    lo, hi = passband(fs)
    n = xws[0].shape[0]
    idx1 = np.maximum(np.arange(n) - 1, 0)
    idx2 = np.maximum(np.arange(n) - 2, 0)
    for ch, xw in zip(channels, xws):
        A = extract_channel_features(xw, fs=float(fs), band=None)              # exactly what the sweep computes
        for name, arr in A.items():
            cols[f"{ch}_{name}"] = arr
        xw64 = xw.astype(np.float64)
        B, C, X = channel_b_c(xw64, fs)
        for name in EXTRA_NAMES:
            cols[f"{ch}_{name}"] = B[name]
        for name in C_ONLY_NAMES:
            if name in C:
                cols[f"{ch}_{name}"] = C[name]
        for base in ("MAV", "RMS", "WL"):
            f = np.asarray(A[base], dtype=np.float64)
            cols[f"{ch}_d{base}1"] = f - f[idx1]
            cols[f"{ch}_d{base}2"] = f - f[idx2]
        Xs.append(X)
        rmss.append(np.sqrt((xw64 ** 2).mean(axis=1)))
        xc = xw64 - xw64.mean(axis=1, keepdims=True)
        Zs.append(xc / (np.sqrt((xc ** 2).sum(axis=1, keepdims=True)) + TINY))

    N = xws[0].shape[1]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    masks3 = _band_masks(freqs, lo, hi, N_BANDCORR)
    pw = [np.abs(X) ** 2 for X in Xs]
    for a in range(len(channels)):
        for b in range(a + 1, len(channels)):
            cols[f"xcorr_{channels[a]}__{channels[b]}"] = (Zs[a] * Zs[b]).sum(axis=1)
            for k, m in enumerate(masks3):
                num = (Xs[a][:, m] * np.conj(Xs[b][:, m])).real.sum(axis=1)
                den = np.sqrt(pw[a][:, m].sum(axis=1) * pw[b][:, m].sum(axis=1)) + TINY
                cols[f"bandcorr{k + 1}_{channels[a]}__{channels[b]}"] = num / den
    tot = np.sum(rmss, axis=0) + TINY
    for ch, r in zip(channels, rmss):
        cols[f"rmsshare_{ch}"] = r / tot
    return cols


_B_ONLY = set(EXTRA_NAMES)
_C_ONLY = set(C_ONLY_NAMES)


def feature_group(col):
    """'A' for the original 24 types, 'B' for the extras and cross-channel xcorr/rmsshare, 'C' for the new families."""
    if col.startswith("xcorr_") or col.startswith("rmsshare_"):
        return "B"
    if col.startswith("bandcorr"):
        return "C"
    if re.search(r"_wav_(energy|entropy)_[AD]\d$", col):   # the wavelet level of Set A depends on the window length (A4 at 200 samples, A3 at 100)
        return "A"
    for name in sorted(FEATURE_NAMES + EXTRA_NAMES + C_ONLY_NAMES, key=len, reverse=True):
        if col.endswith("_" + name):
            return "A" if name in FEATURE_NAMES else ("B" if name in _B_ONLY else "C")
    raise ValueError(col)


def set_columns(cols, which):
    allowed = {"A": {"A"}, "B": {"A", "B"}, "C": {"A", "B", "C"}}[which]
    return [c for c in cols if feature_group(c) in allowed]
