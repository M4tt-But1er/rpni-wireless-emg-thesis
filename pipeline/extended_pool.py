"""The EXTENDED candidate pool: the 24 original per-channel features (cached tables, unchanged) plus 16 new per-channel features and
two cross-channel groups (built by build_extra_features.py into cache/<participant>_noRest_extra.parquet, row-aligned with the cached tables).

New per channel (16): DASDV, LogDet, WAMP, HjMob, HjCx, MaxAbs, AR1-AR4, CC1-CC4, SpecSpread, SpecFlat.
Cross-channel: xcorr_<chA>__<chB> (Pearson correlation of the two channels inside the window; every channel pair) and
rmsshare_<ch> (that channel's RMS divided by the sum of all channels' RMS in the window).
"""
import numpy as np
import pandas as pd

from config import CACHE_DIR
from features import FEATURE_NAMES

EXTRA_NAMES = ["DASDV", "LogDet", "WAMP", "HjMob", "HjCx", "MaxAbs", "AR1", "AR2", "AR3", "AR4",
               "CC1", "CC2", "CC3", "CC4", "SpecSpread", "SpecFlat"]
ALL_TYPE_NAMES = FEATURE_NAMES + EXTRA_NAMES + ["xcorr", "rmsshare"]
META = ["segment_id", "cue", "window_time_s"]


def extra_path(part):
    return CACHE_DIR / f"{part}_extra.parquet"


def feature_type(col):
    """Feature type of a column name (original 24, the 16 new ones, or the two cross-channel groups)."""
    if col.startswith("xcorr_"):
        return "xcorr"
    if col.startswith("rmsshare_"):
        return "rmsshare"
    for f in sorted(FEATURE_NAMES + EXTRA_NAMES, key=len, reverse=True):
        if col.endswith("_" + f):
            return f
    raise ValueError(col)


def load_extended(part):
    """Cached original table + the extra columns, after checking the two tables are row-for-row the same windows."""
    base = pd.read_parquet(CACHE_DIR / f"{part}.parquet")
    extra = pd.read_parquet(extra_path(part))
    assert len(base) == len(extra), (len(base), len(extra))
    assert (base["segment_id"].to_numpy() == extra["segment_id"].to_numpy()).all(), "segment_id mismatch"
    assert (base["cue"].to_numpy() == extra["cue"].to_numpy()).all(), "cue mismatch"
    assert np.allclose(base["window_time_s"].to_numpy(), extra["window_time_s"].to_numpy()), "window time mismatch"
    new_cols = [c for c in extra.columns if c not in META]
    assert not set(new_cols) & set(base.columns), "extra columns overlap the original ones"
    out = pd.concat([base.reset_index(drop=True), extra[new_cols].reset_index(drop=True)], axis=1)
    assert np.isfinite(out[new_cols].to_numpy(dtype=float)).all(), "non-finite value in the new features"
    return out
