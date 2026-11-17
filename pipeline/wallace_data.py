"""Shared Wallace-dataset loading, generalized from wallace_pilot.py's
single-participant version to cover both S1 (8 implanted channels) and S2
(12 implanted channels).

Class set, per participant: every documented movement code (config.CUE_LEGEND,
i.e. Vaskov's own cue set) for which that participant has a dedicated
single-grip SNR-collection run with broadband EMG available. S1 has all 13
non-rest codes; S2 is missing only Point(9) -- no single-grip SNR run exists
for it (Point only appears mixed into S2's multi-grip online/calibration runs,
which use a different 2s/2s trial timing and were left out here to keep the
same 4s-rest/4s-hold protocol across both participants). This makes S2's
12-class set exactly equal to Vaskov's own P1_combined class composition, and
S1's 13-class set a strict superset of both P1_combined and P2_combined.

Rest is NOT excluded at the raw-loading stage -- each trial's 4s rest lead-in
is kept in the per-run cue array so segmentation.build_segments can compute
onset-detection baselines the normal way (same reasoning as Vaskov: an
adjacent rest segment's MAV stats). It's dropped after segmentation, same
place run_bitrate_sweep_round2.py/build_feature_matrix's exclude_cues does it.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

from config import DATA_DIR, REST_CUE, CUE_LEGEND

# docs/channel_mapping.md -- implanted channels only, in Broadband row order.
S1_IMPLANTED_CHANNELS = ["EPL", "EDC", "FPL", "FDPi", "Median1", "Ulnar1", "Ulnar2", "FCR"]
S2_IMPLANTED_CHANNELS = ["EDC", "EPL", "Supinator", "Radial", "Pronator", "Median2",
                         "Median4", "FDPs", "FDPi", "FPL", "Median3", "Median1"]

WALLACE_ROOT_S1 = DATA_DIR / "Wallace Data" / "data_S1" / "data" / "S1" / "visit2"
WALLACE_ROOT_S2 = DATA_DIR / "Wallace Data" / "data_S2_visit2_part1" / "data" / "S2" / "visit2"

# TaskNumber -> single-grip SNR run holding it, per participant (see module
# docstring for why S2 has no entry for 9/Point).
S1_RUN_FOR_TASK = {2: "Run-003", 3: "Run-004", 4: "Run-005", 5: "Run-006", 6: "Run-007",
                   7: "Run-008", 8: "Run-009", 9: "Run-010", 10: "Run-011", 11: "Run-021",
                   13: "Run-013", 14: "Run-014", 18: "Run-015"}
S2_RUN_FOR_TASK = {2: "Run-002", 3: "Run-003", 4: "Run-004", 5: "Run-005", 6: "Run-006",
                   7: "Run-007", 8: "Run-008", 10: "Run-010", 11: "Run-013", 13: "Run-011",
                   14: "Run-012", 18: "Run-017"}

PARTICIPANTS = {
    "Wallace_S1": dict(root=WALLACE_ROOT_S1, channels=S1_IMPLANTED_CHANNELS,
                       run_for_task=S1_RUN_FOR_TASK, file_stub="S1_visit2"),
    "Wallace_S2": dict(root=WALLACE_ROOT_S2, channels=S2_IMPLANTED_CHANNELS,
                       run_for_task=S2_RUN_FOR_TASK, file_stub="S2_visit2"),
}

assert set(S1_RUN_FOR_TASK) | {REST_CUE} == set(CUE_LEGEND), \
    "S1 task set should cover every documented CUE_LEGEND code"
assert set(S2_RUN_FOR_TASK) | {REST_CUE, 9} == set(CUE_LEGEND), \
    "S2 task set should cover every documented CUE_LEGEND code except Point(9)"


def load_wallace_run(participant: str, run_name: str) -> pd.DataFrame:
    """One SNR run -> (time_s, cue, <implanted channels>) DataFrame at the
    native 30kHz, trials concatenated in order -- same shape data_io.load_raw()
    produces for a Vaskov CSV, so the rest of the pipeline is unmodified."""
    spec = PARTICIPANTS[participant]
    channels = spec["channels"]
    mat_path = spec["root"] / run_name / f"Z_{spec['file_stub']}_{run_name}_wEMG.mat"
    S = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    z = S["zwEMG"]
    trials = z if hasattr(z, "__len__") else [z]

    n_implanted = len(channels)
    chan_chunks = {ch: [] for ch in channels}
    cue_chunks = []
    for t in trials:
        bb = np.asarray(t.Broadband, dtype=np.float64)[:n_implanted, :] / 4.0  # implanted only, uV
        n = bb.shape[1]
        rest_samples = int(round(float(t.RestHoldTime) / 1000.0 * 30000))
        rest_samples = min(rest_samples, n)
        cue = np.full(n, REST_CUE, dtype=np.int16)
        cue[rest_samples:] = int(t.TaskNumber)
        cue_chunks.append(cue)
        for i, ch in enumerate(channels):
            chan_chunks[ch].append(bb[i].astype(np.float32))

    cue_full = np.concatenate(cue_chunks)
    n_total = len(cue_full)
    out = {"time_s": np.arange(n_total, dtype=np.float64) / 30000.0, "cue": cue_full}
    for ch in channels:
        out[ch] = np.concatenate(chan_chunks[ch])
    return pd.DataFrame(out)
