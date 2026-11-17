"""Cut a channel subset out of an all-channel wide table (grid_tables.py) for the bitrate grid.
Per-channel columns of the chosen channels are taken as they are; pair columns (xcorr, bandcorr) are taken when BOTH channels are in the subset (they do not depend on other channels);
rmsshare is recomputed over the subset from the subset's RMS columns (rmsshare_ch = RMS_ch / (sum of RMS over the subset + TINY), as wide_features.segment_columns defines it).
Verified against direct computation on the subset's windows by test_grid_slices.py (check V5)."""
import re

import numpy as np

from wide_features import TINY


def _pair(name, prefix_len):
    a, b = name[prefix_len:].split("__")
    return a, b


def column_channels(col, channels):
    """Channels a column depends on: ('chan', [ch]) for a per-channel column, ('pair', [a, b]), ('share', [ch])."""
    if col.startswith("xcorr_"):
        return "pair", list(_pair(col, len("xcorr_")))
    m = re.match(r"bandcorr\d_", col)
    if m:
        return "pair", list(_pair(col, m.end()))
    if col.startswith("rmsshare_"):
        return "share", [col[len("rmsshare_"):]]
    for ch in sorted(channels, key=len, reverse=True):
        if col.startswith(ch + "_"):
            return "chan", [ch]
    raise ValueError(f"cannot assign column to a channel: {col}")


def subset_columns(cols, channels, keep):
    keep = set(keep)
    return [c for c in cols if set(column_channels(c, channels)[1]) <= keep]


def slice_matrix(table, cols, channels, keep):
    """(X, names) for the columns `cols` restricted to the channel subset `keep`; rmsshare values recomputed over the subset."""
    names = subset_columns(cols, channels, keep)
    keep = list(keep)
    X = np.empty((len(table), len(names)), dtype=np.float64)
    rms = np.stack([table[f"{ch}_RMS"].to_numpy(dtype=np.float64) for ch in keep], axis=0)
    tot = rms.sum(axis=0) + TINY
    for j, c in enumerate(names):
        kind, chs = column_channels(c, channels)
        X[:, j] = rms[keep.index(chs[0])] / tot if kind == "share" else table[c].to_numpy(dtype=np.float64)
    return X, names
