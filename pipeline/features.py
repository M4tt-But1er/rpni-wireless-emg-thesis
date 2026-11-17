"""Per-window, per-channel feature extraction.

Windows are 200ms (200 samples @ 1kHz) with 50ms (50 sample) hop, computed
only within a trimmed segment (never crossing a segment boundary).

Feature families (24 features/channel x 8 channels = 192 features/window):

Time-domain (6): MAV, RMS, Waveform Length, Zero Crossings, Slope Sign
Changes, IEMG.

Frequency-domain (4): Mean Frequency, Median Frequency, Peak Frequency,
band power ratio (power in 100-300Hz / power in 300-499Hz -- the two
halves of our bandpass-filtered band; a common EMG spectral-shift feature).

Time-frequency (10): 4-level DWT (db4, the max safe level for a 200-sample
window per pywt.dwt_max_level) energy + Shannon entropy per sub-band
(D1..D4, A4) = 5 sub-bands x 2 stats.

Statistical (4): variance, skewness, kurtosis, Shannon entropy (of the
window's amplitude histogram).

ZC/SSC use a small deadband threshold (proportional to the window's own
std) to avoid counting sign flips from noise around zero -- standard
practice for these two features.
"""
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

from config import WINDOW_MS, HOP_MS, TARGET_FS

WINDOW_SAMPLES = int(round(WINDOW_MS * TARGET_FS / 1000))
HOP_SAMPLES = int(round(HOP_MS * TARGET_FS / 1000))

WAVELET = "db4"
WAVELET_LEVEL = 4

ZC_SSC_DEADBAND_FRAC = 0.02  # fraction of window std used as deadband


def window_starts(seg_len: int) -> np.ndarray:
    if seg_len < WINDOW_SAMPLES:
        return np.array([], dtype=int)
    last_start = seg_len - WINDOW_SAMPLES
    return np.arange(0, last_start + 1, HOP_SAMPLES)


def _sliding_windows(x: np.ndarray) -> np.ndarray:
    """(n_windows, WINDOW_SAMPLES) view via stride tricks."""
    starts = window_starts(len(x))
    if len(starts) == 0:
        return np.empty((0, WINDOW_SAMPLES), dtype=x.dtype)
    from numpy.lib.stride_tricks import sliding_window_view
    all_windows = sliding_window_view(x, WINDOW_SAMPLES)  # (len(x)-WINDOW_SAMPLES+1, WINDOW_SAMPLES)
    return all_windows[starts]


def _shannon_entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _amplitude_entropy(w: np.ndarray, n_bins: int = 16) -> float:
    hist, _ = np.histogram(w, bins=n_bins)
    p = hist / hist.sum()
    return _shannon_entropy_from_probs(p)


def _wavelet_features(w: np.ndarray, level: int = WAVELET_LEVEL) -> dict:
    coeffs = pywt.wavedec(w, WAVELET, level=level)  # [cA_level, cD_level, ..., cD1]
    labels = [f"A{level}"] + [f"D{lvl}" for lvl in range(level, 0, -1)]
    feats = {}
    for label, c in zip(labels, coeffs):
        energy = float(np.sum(c ** 2))
        p = (c ** 2)
        total = p.sum()
        p = p / total if total > 0 else p
        entropy = _shannon_entropy_from_probs(p)
        feats[f"wav_energy_{label}"] = energy
        feats[f"wav_entropy_{label}"] = entropy
    return feats


def extract_channel_features(x_windows: np.ndarray, fs: float = None, band: tuple = None) -> dict:
    """x_windows: (n_windows, window_samples). Returns dict of feature_name -> (n_windows,) array.

    `fs` defaults to the project-wide TARGET_FS (1kHz) -- every existing
    caller gets byte-identical behavior. Passed explicitly only by the
    bitrate-sweep experiment, which runs this same feature set at other
    sample rates; the actual window length always comes from the array
    itself (x_windows.shape[1]), never from the global WINDOW_SAMPLES
    constant, so this also works correctly for windows shorter/longer than
    the project-wide default.

    `band` = (low_hz, high_hz) is only used by the pass-band sweep: it sets the
    range the BandPowerRatio feature splits at its midpoint, so the feature
    follows the filter that was actually applied. None keeps the original
    behaviour (100 Hz to min(499, fs/2 - 1)), so every existing caller is unchanged.
    """
    if fs is None:
        fs = TARGET_FS
    n, window_samples = x_windows.shape
    feats = {}

    # --- time domain (vectorized) ---
    absx = np.abs(x_windows)
    feats["MAV"] = absx.mean(axis=1)
    feats["RMS"] = np.sqrt((x_windows ** 2).mean(axis=1))
    diffs = np.diff(x_windows, axis=1)
    feats["WL"] = np.abs(diffs).sum(axis=1)
    feats["IEMG"] = absx.sum(axis=1)

    win_std = x_windows.std(axis=1, keepdims=True)
    deadband = ZC_SSC_DEADBAND_FRAC * win_std
    # zero crossings: sign change between consecutive samples, both samples magnitude > deadband
    s1 = x_windows[:, :-1]
    s2 = x_windows[:, 1:]
    zc_mask = (s1 * s2 < 0) & (np.abs(s1) > deadband) & (np.abs(s2) > deadband)
    feats["ZC"] = zc_mask.sum(axis=1)

    # slope sign changes: diff[i] and diff[i+1] have opposite sign, magnitude of the slope change > deadband
    d1 = diffs[:, :-1]
    d2 = diffs[:, 1:]
    ssc_mask = (d1 * d2 < 0) & (np.abs(d1 - d2) > deadband)
    feats["SSC"] = ssc_mask.sum(axis=1)

    # --- frequency domain (vectorized FFT) ---
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / fs)
    fft_mag2 = np.abs(np.fft.rfft(x_windows, axis=1)) ** 2  # (n, n_freqs)
    total_power = fft_mag2.sum(axis=1)
    total_power_safe = np.where(total_power > 0, total_power, 1.0)

    feats["MNF"] = (fft_mag2 * freqs[None, :]).sum(axis=1) / total_power_safe
    feats["PeakFreq"] = freqs[np.argmax(fft_mag2, axis=1)]

    cumpower = np.cumsum(fft_mag2, axis=1)
    half = total_power[:, None] / 2.0
    mdf_idx = (cumpower >= half).argmax(axis=1)
    feats["MDF"] = freqs[mdf_idx]

    # band split at the midpoint of whatever band is actually present up to
    # this fs's Nyquist, rather than the hard-coded 100/300/499Hz split that
    # only makes sense when fs=1000 and the passband is 100-499Hz -- keeps
    # this feature meaningful (low half vs high half of the real content)
    # at any sample rate the bitrate sweep tests.
    band_lo, band_hi = band if band is not None else (100.0, min(499.0, fs / 2 - 1))
    band_mid = (band_lo + band_hi) / 2
    low_mask = (freqs >= band_lo) & (freqs < band_mid)
    high_mask = (freqs >= band_mid) & (freqs <= band_hi)
    low_power = fft_mag2[:, low_mask].sum(axis=1)
    high_power = fft_mag2[:, high_mask].sum(axis=1)
    feats["BandPowerRatio"] = low_power / np.where(high_power > 0, high_power, 1e-12)

    # --- statistical ---
    feats["Var"] = x_windows.var(axis=1)
    feats["Skew"] = skew(x_windows, axis=1)
    feats["Kurtosis"] = kurtosis(x_windows, axis=1)
    feats["ShannonEntropy"] = np.array([_amplitude_entropy(w) for w in x_windows])

    # --- wavelet (per-window loop; pywt has no native batched API) ---
    # Level is capped by what's actually safe for this window's length
    # (pywt.dwt_max_level) -- at fs=1000/200-sample windows this is exactly
    # the original fixed level 4, unchanged; the bitrate sweep's much
    # shorter windows (fewer samples at lower sample rates) legitimately
    # get fewer wavelet sub-bands (and therefore fewer total feature
    # columns for that config) rather than erroring or silently padding.
    safe_level = min(WAVELET_LEVEL, pywt.dwt_max_level(window_samples, pywt.Wavelet(WAVELET).dec_len))
    safe_level = max(safe_level, 1)
    wav_label_names = [f"wav_energy_A{safe_level}", f"wav_entropy_A{safe_level}"]
    for lvl in range(safe_level, 0, -1):
        wav_label_names += [f"wav_energy_D{lvl}", f"wav_entropy_D{lvl}"]

    if n > 0:
        wav_rows = [list(_wavelet_features(w, level=safe_level).values()) for w in x_windows]
        wav_arr = np.array(wav_rows)
        for j, name in enumerate(wav_label_names):
            feats[name] = wav_arr[:, j]
    else:
        for name in wav_label_names:
            feats[name] = np.array([])

    return feats


FEATURE_NAMES = [
    "MAV", "RMS", "WL", "ZC", "SSC", "IEMG",
    "MNF", "MDF", "PeakFreq", "BandPowerRatio",
    "wav_energy_A4", "wav_entropy_A4",
    "wav_energy_D4", "wav_entropy_D4",
    "wav_energy_D3", "wav_entropy_D3",
    "wav_energy_D2", "wav_entropy_D2",
    "wav_energy_D1", "wav_entropy_D1",
    "Var", "Skew", "Kurtosis", "ShannonEntropy",
]
