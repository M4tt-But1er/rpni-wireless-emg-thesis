"""Bandpass filtering + downsampling of the raw 30 kSps EMG.

Filtering is applied once over the *entire continuous recording* per channel
(not per-segment) to avoid edge-transient artifacts at every cue-change
boundary -- segments are only 2.5-5s long, too short to absorb filter
settling time cleanly on their own. Downsampling then follows by simple
decimation (stride), which is safe here because the signal is already
band-limited to <=499 Hz before the new Nyquist (500 Hz at 1 kSps).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import RAW_FS, TARGET_FS, BANDPASS_LOW, BANDPASS_HIGH


def bandpass_filter(x: np.ndarray, fs: int = RAW_FS,
                     low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
                     order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)


def downsample_factor(fs_in: int = RAW_FS, fs_out: int = TARGET_FS) -> int:
    factor = fs_in / fs_out
    if factor != int(factor):
        raise ValueError(f"Non-integer decimation factor: {factor}")
    return int(factor)


def preprocess_dataframe(df):
    """Bandpass-filter all EMG channels, then decimate time/cue/channels to
    TARGET_FS. Returns a new DataFrame at 1 kSps."""
    factor = downsample_factor()
    channels = [c for c in df.columns if c not in ("time_s", "cue")]

    out = {
        "time_s": df["time_s"].to_numpy()[::factor],
        "cue": df["cue"].to_numpy()[::factor],
    }
    for ch in channels:
        filtered = bandpass_filter(df[ch].to_numpy())
        out[ch] = filtered[::factor]

    import pandas as pd
    return pd.DataFrame(out)
