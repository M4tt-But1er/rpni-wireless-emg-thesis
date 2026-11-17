"""Build rest-excluded ("movement-only") feature tables for both participants.

Research scope is movement classification specifically -- rest (cue=1) is not
a movement and is out of scope, so it's dropped entirely from the label set
(not just down-weighted, as the original class_weight='balanced' models did).

Reuses build_feature_matrix's existing exclude_cues mechanism (already used
by build_features_cuefiltered.py) -- rest is excluded from the FINAL feature
table only, *after* segmentation. This matters: onset detection still uses
adjacent rest segments internally as its baseline reference (see
segmentation.py), so removing rest here does not degrade onset-trimming
quality for the movement segments that remain.

Writes to new cache files (*_noRest.parquet) -- does not touch any existing
cached feature table.
"""
import pandas as pd

from config import CACHE_DIR, DATASETS, COMBINED_GROUPS, REST_CUE
from build_features import build_feature_matrix

SUFFIX = "_noRest"


def main():
    per_dataset = {}
    for name in DATASETS:
        label = f"{name}{SUFFIX}"
        fm = build_feature_matrix(name, exclude_cues=[REST_CUE], dataset_label=label)
        out_path = CACHE_DIR / f"{label}.parquet"
        fm.to_parquet(out_path, index=False)
        print(f"cached {label} -> {out_path} ({fm.shape})")
        per_dataset[name] = fm

    for combined_name, members in COMBINED_GROUPS.items():
        label = f"{combined_name}{SUFFIX}"
        combined = pd.concat([per_dataset[m] for m in members], axis=0, ignore_index=True)
        out_path = CACHE_DIR / f"{label}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"cached {label} -> {out_path} ({combined.shape})")
        print(f"  cues present: {sorted(combined['cue'].unique().tolist())} (should NOT include {REST_CUE})")


if __name__ == "__main__":
    main()
