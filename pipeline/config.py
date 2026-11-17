"""
Shared configuration for the RPNI EMG movement-classification pipeline.

Channel names, cue legend, and per-file cue sets are extracted verbatim from
DataLabellingAndProcessing.pdf (pages 2-4). Do NOT guess these; if the source
files change, re-check the PDF.
"""
import os
from pathlib import Path

# The repo folder itself, wherever it is cloned or moved to.
PIPELINE_DIR = Path(__file__).resolve().parent
# The folder that contains "Vaskov Data/" and "Wallace Data/". Defaults to the repo's parent
# folder (the layout used during development). Set the RPNI_DATA_DIR environment variable to
# point somewhere else. (The project folder was moved twice during development, which broke
# the old hard-coded absolute path each time.)
DATA_DIR = Path(os.environ.get("RPNI_DATA_DIR", PIPELINE_DIR.parent))
CACHE_DIR = PIPELINE_DIR / "cache"
FIGS_DIR = PIPELINE_DIR / "figs"

# Raw Vaskov CSVs/PDF were moved into their own subfolder (2026-09-17) when the
# user added the Wallace dataset alongside them. Only DATASETS' file paths
# below depend on this.
VASKOV_DATA_DIR = DATA_DIR / "Vaskov Data"

RAW_FS = 30000  # Hz, native sampling rate
TARGET_FS = 1000  # Hz, after downsampling
BANDPASS_LOW = 100.0  # Hz
BANDPASS_HIGH = 499.0  # Hz (kept <500 to stay safely under the post-downsample
                        # Nyquist of 500 Hz, matching the PDF's own Matlab example)

# Cue legend (PDF page 2). Note: 12, 15, 16, 17 are absent -- not a typo, the
# legend genuinely skips those numbers.
CUE_LEGEND = {
    1: "Rest",
    2: "Thumb flexion (IP joint)",
    3: "Index finger flexion",
    4: "Middle finger flexion",
    5: "Ring finger flexion",
    6: "Small finger flexion",
    7: "Fist",
    8: "Pinch",
    9: "Point",
    10: "Finger Extension",
    11: "Wrist Flexion",
    13: "Hand Abduction",
    14: "Hand Adduction",
    18: "Thumb opposition (MCP joint)",
}

REST_CUE = 1

# Columns 3-10 (0-indexed 2-9) channel names, per file, per PDF pages 3-4.
P1_CHANNELS = [
    "FDP_index",       # flexor digitorum profundus, index finger section
    "FCR",             # flexor carpi radialis
    "RPNI_ulnar",      # RPNI ulnar nerve
    "RPNI_median",     # RPNI median nerve
    "EDC",             # extensor digitorum communis
    "EPL",             # extensor pollicis longus
    "FDP_small",       # flexor digitorum profundus, small finger section
    "FPL",             # flexor pollicis longus
]

P2_CHANNELS = [
    "EPL",             # extensor pollicis longus
    "EDC",             # extensor digitorum communis
    "FPL",             # flexor pollicis longus
    "FDP_index",       # flexor digitorum profundus, index finger section
    "RPNI_median",     # RPNI median nerve
    "RPNI_ulnar1",     # RPNI ulnar nerve 1
    "RPNI_ulnar2",     # RPNI ulnar nerve 2
    "FCR",             # flexor carpi radialis
]

DATASETS = {
    "P1_1of12": dict(
        file=VASKOV_DATA_DIR / "P1_1of12.csv",
        participant="P1",
        channels=P1_CHANNELS,
        cues=[1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 18],
    ),
    "P1_Grasps": dict(
        file=VASKOV_DATA_DIR / "P1_Grasps.csv",
        participant="P1",
        channels=P1_CHANNELS,
        cues=[1, 6, 7, 8, 10, 11],
    ),
    "P2_1of10": dict(
        file=VASKOV_DATA_DIR / "P2_1of10.csv",
        participant="P2",
        channels=P2_CHANNELS,
        cues=[1, 2, 3, 4, 5, 6, 11, 13, 14, 18],
    ),
    "P2_Grasps": dict(
        file=VASKOV_DATA_DIR / "P2_Grasps.csv",
        participant="P2",
        channels=P2_CHANNELS,
        cues=[1, 7, 8, 9, 13],
    ),
}

# "combined" caches are per-participant (montage/channel layout is consistent
# within a participant across their two posture-set files, but differs between
# P1 and P2) -- see pipeline README notes for rationale.
COMBINED_GROUPS = {
    "P1_combined": ["P1_1of12", "P1_Grasps"],
    "P2_combined": ["P2_1of10", "P2_Grasps"],
}

# Onset detection parameters
# MAV window widened from an initial 50ms to 100ms after validation showed
# some genuine (visually obvious, sustained) contractions are physiologically
# bursty/oscillatory at ~150-250ms period (seen e.g. in P2_1of10 individual
# finger-flexion trials) -- a 50ms envelope followed those ripples down below
# threshold and broke the "sustained" run, causing false misses. 100ms still
# clearly separates rest from contraction but smooths through the ripple.
ONSET_MAV_WINDOW_MS = 100
ONSET_THRESHOLD_MULT = 4.0     # x rest-period MAV stdev (baseline computed on the MAV itself, see segmentation.py docstring)
ONSET_SUSTAIN_MS = 200         # must stay above threshold for this long

# Windowing parameters (applied to the 1 kHz downsampled, trimmed segments)
WINDOW_MS = 200
HOP_MS = 50


# =============================================================================
# Vaskov et al. 2020 (medRxiv / IEEE Trans Robot) benchmark-replication config.
# Additive only -- nothing above this line is touched by the Vaskov benchmark.
# See vaskov_preprocessing.py / vaskov_features.py / vaskov_models.py.
# =============================================================================

# Paper's own recipe: butter(2, 2*[100,499]/1e3, 'bandpass') then a single
# causal `filter()` pass (not filtfilt) -- reuses BANDPASS_LOW/HIGH above,
# which are already 100/499.
VASKOV_FILTER_ORDER = 2

# "7 increasing window lengths" -- paper states these exact values (Section
# VI-B): "10, 25, 50, 100, 150, 200, and 250ms of EMG history."
VASKOV_WINDOW_LENGTHS_MS = [10, 25, 50, 100, 150, 200, 250]

# Hop/update-rate: not stated for the Fig.8 sweep itself. Originally assumed
# a fixed 50ms hop (matching the paper's separate online "update rate" table
# entry), but that produced runaway accuracy approaching 100% at the larger
# window lengths under the paper's own *non-grouped* per-timestep 5-fold CV
# protocol -- a classic overlap-leakage signature (a 250ms window against a
# fixed 50ms hop is 80% overlapping with its neighbor, so near-duplicate
# windows from the same segment land in both train and test). Verified by
# comparing to hop == window length (no overlap): that version plateaus
# instead of running away, and lines up with the paper's own reference
# numbers far better (e.g. P1 1-of-10 NB: 86.9% @10ms, 95.3% @50ms vs the
# paper's ~90.5% reference sitting naturally in between). Switched to
# non-overlapping windows (hop = window length) for this reason -- this is a
# bug fix motivated by the leakage evidence itself, not by tuning to match
# the reference number.
VASKOV_HOP_MS = None  # unused; hop is set equal to window length, see vaskov_build_features.py

# Cue values present in the raw calibration file but excluded from the named
# paper posture set for that participant (see chat discussion / dataset PDF):
#   - P1_1of12 has 2 extra grasp cues (Fist=7, Pinch=8) beyond the "1 of 10"
#     set; excluding them yields exactly P2_1of10's cue set {1,2,3,4,5,6,11,13,14,18}.
#   - P1_Grasps has an extra Wrist Flexion (11) cue beyond the paper's 5-class
#     grasps set; P1's visual prompts substituted Finger Extension(10) for
#     abduction and Small Finger Flexion(6) for point (paper Section V-A text).
# P2's files already match the paper's sets exactly -- no exclusions needed.
VASKOV_EXCLUDED_CUES = {
    "P1_1of12": [7, 8],
    "P1_Grasps": [11],
}

# Maps a paper-benchmark posture-set name to the underlying raw dataset (and,
# via VASKOV_EXCLUDED_CUES, which cues to drop after segmentation).
VASKOV_DATASETS = {
    "P1_1of10": "P1_1of12",
    "P1_grasps": "P1_Grasps",
    "P2_1of10": "P2_1of10",
    "P2_grasps": "P2_Grasps",
}

VASKOV_CACHE_DIR = CACHE_DIR / "vaskov"
VASKOV_FIGS_DIR = FIGS_DIR / "vaskov"
