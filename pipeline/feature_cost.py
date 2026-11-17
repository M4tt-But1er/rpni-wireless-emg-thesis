"""Analytic operation counts for the wide features and for the classifiers (docs/grid_declared_protocol.md, section 'FLOPs and MACs'). Platform-independent COUNTS, not measurements.
Conventions (declared before any grid result): 1 MAC = 2 FLOPs; add, subtract, multiply, compare, abs, max count 1 FLOP each; divisions, square roots, logs, exps and the like are counted
separately as 'special' operations and NOT folded into FLOPs; every intermediate (|x|, mean, variance, first difference, FFT and power spectrum, wavelet decomposition) is computed once per window and channel and
shared by the features that use it (charged only if some selected feature needs it); each window is processed from scratch (no sliding-sum tricks, no overlap reuse).
They describe a straightforward implementation, probably within a factor of about 2 of an efficient one; they are meant for comparing configurations, not for predicting cycles."""
import math
import re

import numpy as np

from wide_features import BP_NAMES, DYN_NAMES, EXTRA_NAMES, FEATURE_NAMES, WP_NAMES, passband

DECISIONS_PER_SECOND = 20          # 200 ms window, 50 ms hop
FILTER_LEN = 8                     # db4
_WAV_RE = re.compile(r"^wav_(energy|entropy)_([AD])(\d)$")
_ALL_TYPES = sorted(set(FEATURE_NAMES) | set(EXTRA_NAMES) | set(BP_NAMES) | set(WP_NAMES) | set(DYN_NAMES) | {"SpecSkew", "SpecKurt"}, key=len, reverse=True)


def _dwt_out(n):
    return (n + FILTER_LEN - 1) // 2


def _dwt_flops(n):
    """One DWT of n samples: two filters x _dwt_out(n) outputs x 8 MACs, 2 FLOPs per MAC."""
    return 2 * (2 * _dwt_out(n) * FILTER_LEN)


def wavedec_levels(N, level):
    """(decomposition FLOPs, [band lengths: A_level, D_level, ..., D1]) of the A-set wavelet decomposition (pywt.wavedec)."""
    flops, n, d_lens = 0, N, []
    for _ in range(level):
        flops += _dwt_flops(n)
        n = _dwt_out(n)
        d_lens.append(n)
    return flops, [d_lens[-1]] + d_lens[::-1]


def wp_nodes(N):
    """(decomposition FLOPs, node length) of the level-3 wavelet packet (both branches split at each level)."""
    flops, n = 0, N
    for lev in range(3):
        flops += (2 ** lev) * _dwt_flops(n)
        n = _dwt_out(n)
    return flops, n


def window_geometry(rate):
    N = int(round(200 * rate / 1000))
    Nf = N // 2 + 1
    lo, hi = passband(rate)
    freqs = np.fft.rfftfreq(N, d=1.0 / rate)
    nb = int(((freqs >= lo) & (freqs <= hi)).sum())
    import pywt
    level = max(1, min(4, pywt.dwt_max_level(N, FILTER_LEN)))
    return N, Nf, nb, level


def type_of(col, channels):
    """Feature type of a column: ('chan', type), ('pair', 'xcorr' | 'bandcorr'), ('share', 'rmsshare')."""
    if col.startswith("xcorr_"):
        return "pair", "xcorr"
    if col.startswith("bandcorr"):
        return "pair", "bandcorr"
    if col.startswith("rmsshare_"):
        return "share", "rmsshare"
    if re.search(r"_wav_(energy|entropy)_[AD]\d$", col):
        return "chan", re.search(r"(wav_(?:energy|entropy)_[AD]\d)$", col).group(1)
    for t in _ALL_TYPES:
        if col.endswith("_" + t):
            return "chan", t
    raise ValueError(col)


def _spec_needed(types):
    return bool(types & ({"MNF", "MDF", "PeakFreq", "BandPowerRatio", "SpecSpread", "SpecFlat", "SpecSkew", "SpecKurt"} | set(BP_NAMES)))


def per_channel_cost(types, rate, pair_fft=False):
    """(flops, special_ops) per window for ONE channel computing the per-channel feature `types` (a set of type names). pair_fft=True: the channel's FFT is also needed for band-limited pair correlation."""
    N, Nf, nb, L = window_geometry(rate)
    wav = {t for t in types if _WAV_RE.match(t)}
    f = s = 0.0
    # shared intermediates, charged if needed
    needs_abs = types & {"MAV", "IEMG", "MaxAbs", "LogDet", "ZC", "SSC"}
    needs_var = types & {"Var", "Skew", "Kurtosis", "ZC", "SSC", "WAMP", "HjMob", "HjCx", "AR1", "CC1"}
    needs_diff = types & {"WL", "SSC", "WAMP", "DASDV", "HjMob", "HjCx"}
    if needs_abs:
        f += N
    if needs_var:
        f += N + 3 * N          # mean, then centred squares and sum
        s += 1                  # sqrt for the standard deviation
    if needs_diff:
        f += N
    if _spec_needed(types) or pair_fft:
        f += 2.5 * N * math.log2(N) + 3 * Nf   # real FFT and power spectrum
    if wav:
        dec, lens = wavedec_levels(N, L)
        f += dec
    # per-type costs
    own = {"MAV": (N, 1), "RMS": (2 * N, 2), "WL": (2 * N, 0), "IEMG": (0, 0), "ZC": (6 * N, 0), "SSC": (8 * N, 0),
           "MNF": (2 * Nf, 1), "MDF": (2 * Nf, 0), "PeakFreq": (Nf, 0), "BandPowerRatio": (Nf, 1),
           "Var": (0, 0), "Skew": (4 * N, 2), "Kurtosis": (4 * N, 2), "ShannonEntropy": (3 * N + 64, 16),
           "DASDV": (2 * N, 1), "LogDet": (3 * N, N + 1), "WAMP": (3 * N, 0), "MaxAbs": (N, 0),
           "HjMob": (3 * N, 2), "HjCx": (4 * N, 3),
           "AR1": (10 * N + 80, 0), "AR2": (0, 0), "AR3": (0, 0), "AR4": (0, 0),
           "CC1": (40, 0), "CC2": (0, 0), "CC3": (0, 0), "CC4": (0, 0),
           "SpecSpread": (6 * Nf, 2), "SpecFlat": (3 * nb, nb + 1),
           "SpecSkew": (10 * nb, 4), "SpecKurt": (4 * nb, 2)}
    for k in range(8):
        own[f"BP{k + 1}"] = (nb / 8 + 2, 1)
    for t in DYN_NAMES:
        own[t] = (1, 0)
    wpt_dec, n4 = wp_nodes(N)
    for k in range(8):
        own[f"WP{k + 1}"] = ((wpt_dec if k == 0 else 0) + 2 * n4 + 1, 1)
    if wav:
        _, lens = wavedec_levels(N, L)
        names = [f"A{L}"] + [f"D{l}" for l in range(L, 0, -1)]
        for nm, ln in zip(names, lens):
            own[f"wav_energy_{nm}"] = (2 * ln, 0)
            own[f"wav_entropy_{nm}"] = (3 * ln, ln)
    for t in types:
        a, b = own[t]
        f += a
        s += b
    return f, s


def pair_cost(kinds, rate):
    """(flops, special_ops) per window for ONE channel pair; `kinds` subset of {'xcorr', 'bandcorr'} (bandcorr = 3 sub-bands, together)."""
    N, Nf, nb, _ = window_geometry(rate)
    f = s = 0.0
    if "xcorr" in kinds:
        f += 2 * N
        s += 2
    if "bandcorr" in kinds:
        f += 6 * nb
        s += 6
    return f, s


def extraction_cost(cols, channels_all, rate):
    """Operation counts per decision for the columns `cols` of an all-channel wide table (already restricted to the channel subset): returns dict(flops, special_ops, flops_per_s)."""
    per_ch, pair_kinds, share = {}, set(), False
    for c in cols:
        kind, t = type_of(c, channels_all)
        if kind == "chan":
            ch = next(ch for ch in sorted(channels_all, key=len, reverse=True) if c.startswith(ch + "_"))
            per_ch.setdefault(ch, set()).add(t)
        elif kind == "pair":
            pair_kinds.add(t)
        else:
            share = True
    flops = special = 0.0
    for ch, types in per_ch.items():
        f, s = per_channel_cost(types, rate, pair_fft="bandcorr" in pair_kinds)
        flops += f
        special += s
    nch = len(per_ch)
    n_pairs = nch * (nch - 1) // 2
    f, s = pair_cost(pair_kinds, rate)
    flops += n_pairs * f
    special += n_pairs * s
    if share:
        flops += nch
        special += nch
    return dict(flops=flops, special_ops=special, flops_per_s=flops * DECISIONS_PER_SECOND)


def model_cost(model, n_features, n_classes, **info):
    """Operation counts per decision for a fitted classifier (platform-independent). Standardisation of the inputs (subtract mean, multiply by 1/sd) is counted as n_features MACs for every model that scales.
    LDA: features x classes MACs. SVM (RBF, one-vs-one): n_sv x features MACs for the kernel plus n_sv x (classes - 1) MACs for the decision sums, n_sv exp. RandomForest: comparisons only,
    trees x deepest tree (an upper bound), plus trees x classes vote additions. RBFNN: centres x features + centres x classes MACs, centres exp. LSTM (1 layer, H units): per window step 4H(features + H) + H x classes MACs, 5H special."""
    F, C = n_features, n_classes
    if model == "LDA":
        return dict(macs=F + F * C, other_ops=0, special_ops=0)
    if model == "SVM":
        n_sv = info["n_sv"]
        return dict(macs=F + n_sv * F + n_sv * (C - 1), other_ops=0, special_ops=n_sv)
    if model == "RandomForest":
        return dict(macs=0, other_ops=info["trees"] * info["max_depth"] + info["trees"] * C, special_ops=0)
    if model == "RBFNN":
        k = info["centres"]
        return dict(macs=F + k * F + k * C, other_ops=0, special_ops=k)
    if model == "LSTM":
        H = info.get("hidden", 64)
        return dict(macs=F + 4 * H * (F + H) + H * C, other_ops=0, special_ops=5 * H)
    raise ValueError(model)
