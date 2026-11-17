"""Raw CSV loading for the RPNI EMG datasets.

Files have NO header row (verified by inspecting raw bytes) and are plain
ASCII/CRLF. Columns: time_s, cue, ch0..ch7.
"""
import numpy as np
import pandas as pd

from config import DATASETS


def load_raw(name: str) -> pd.DataFrame:
    """Load one raw dataset CSV into a DataFrame with proper dtypes/names."""
    spec = DATASETS[name]
    col_names = ["time_s", "cue"] + spec["channels"]
    dtypes = {"time_s": np.float64, "cue": np.int16}
    for ch in spec["channels"]:
        dtypes[ch] = np.float32

    df = pd.read_csv(
        spec["file"],
        header=None,
        names=col_names,
        dtype=dtypes,
        engine="c",
        memory_map=True,
    )
    return df
