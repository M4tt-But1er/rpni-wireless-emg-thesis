"""Segment identification and true-EMG-onset trimming.

A "segment" is a contiguous run of identical cue values in the (already
downsampled, 1 kHz) recording. For non-rest segments we trim the leading
edge to the true muscle-activation onset, since the cue-change timestamp
only marks when the *displayed* cue hand switched -- there is a genuine
reaction-time lag of several hundred ms before the participant's muscles
(phantom-limb attempt) actually start firing.

Onset rule (per user spec): first point where a channel's MAV envelope
exceeds ~4x that trial's rest-period standard deviation, sustained for
>=200ms. The "trial's rest period" is the nearest adjacent rest segment
(preceding, falling back to the following one for a segment at the very
start of the recording). Onset = earliest such crossing across all 8
channels (different muscles can lead/lag each other by tens of ms).

IMPORTANT implementation correction found during validation: the baseline
"rest-period standard deviation" must be computed on the *same MAV envelope*
being thresholded, not on the raw signal. Comparing a smoothed MAV value
against the raw (unsmoothed) signal's stdev is a scale mismatch -- raw EMG
noise has much higher variance than its own moving-average, so a literal
"4x raw std" threshold sits far above what a genuine, sustained-but-modest
contraction ever reaches (verified: real onsets visible by eye as a clear,
sustained step in nearly every channel around 300-400ms into the segment
were being missed this way, while checking the actual data). Threshold is
therefore: mean(MAV during adjacent rest) + 4x std(MAV during adjacent
rest) -- i.e. baseline stats of the rest-period MAV, which is the standard
formulation used in EMG onset detectors (e.g. Hodges & Bui-style baseline
mean + k*SD, applied to the same feature that's later thresholded).

Rest segments themselves are not onset-trimmed (there is no "activation" to
find), but we clip a fixed guard band off both ends to avoid contamination
from the decaying tail of the previous movement / anticipatory lead-in to
the next one. This is an engineering choice, not spelled out in the spec --
flagged here and in the final report.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import REST_CUE, TARGET_FS, ONSET_MAV_WINDOW_MS, ONSET_THRESHOLD_MULT, ONSET_SUSTAIN_MS

REST_GUARD_MS = 250  # fixed head/tail trim applied to rest segments only


@dataclass
class Segment:
    cue: int
    start: int          # index into the 1kHz dataframe, inclusive
    end: int             # exclusive
    trimmed_start: int   # start index after onset trimming (== start for rest, or if onset flagged)
    onset_lag_ms: Optional[float] = None
    onset_flagged: bool = False  # True if onset detection failed to find a crossing
    rest_ref_start: Optional[int] = None
    rest_ref_end: Optional[int] = None


def identify_segments(cue: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (cue_value, start_idx, end_idx) for contiguous runs."""
    change = np.flatnonzero(np.diff(cue) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(cue)]))
    return list(zip(cue[starts].tolist(), starts.tolist(), ends.tolist()))


def rolling_mav(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean absolute value. Output[i] = mean(|x[i-window+1:i+1]|).
    First `window-1` samples are computed over the partial (shorter) window
    rather than NaN, so onset search can start at index 0 of the segment."""
    absx = np.abs(x)
    csum = np.cumsum(absx)
    out = np.empty_like(absx)
    out[:window] = csum[:window] / (np.arange(window) + 1)
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def find_sustained_crossing(above: np.ndarray, sustain_samples: int) -> Optional[int]:
    """First index i such that above[i:i+sustain_samples] are all True."""
    if sustain_samples <= 1:
        idx = np.flatnonzero(above)
        return int(idx[0]) if len(idx) else None
    # run-length trick: convolve boolean with ones, find first full run
    n = len(above)
    if n < sustain_samples:
        return None
    csum = np.cumsum(np.concatenate(([0], above.astype(np.int32))))
    window_sums = csum[sustain_samples:] - csum[:-sustain_samples]
    hits = np.flatnonzero(window_sums == sustain_samples)
    return int(hits[0]) if len(hits) else None


def build_segments(dfp, dataset_name: str) -> list[Segment]:
    """dfp: preprocessed (1kHz) DataFrame for one file."""
    cue = dfp["cue"].to_numpy()
    channels = [c for c in dfp.columns if c not in ("time_s", "cue")]
    raw_segs = identify_segments(cue)

    mav_window = int(round(ONSET_MAV_WINDOW_MS * TARGET_FS / 1000))
    sustain_samples = int(round(ONSET_SUSTAIN_MS * TARGET_FS / 1000))
    guard_samples = int(round(REST_GUARD_MS * TARGET_FS / 1000))

    segments: list[Segment] = []
    for i, (cue_val, start, end) in enumerate(raw_segs):
        if cue_val == REST_CUE:
            trimmed_start = start + guard_samples if (end - start) > 2 * guard_samples else start
            trimmed_end_guard = end - guard_samples if (end - start) > 2 * guard_samples else end
            seg = Segment(cue=cue_val, start=start, end=trimmed_end_guard,
                          trimmed_start=trimmed_start)
            segments.append(seg)
            continue

        # find nearest adjacent rest segment for baseline stats
        rest_ref = None
        if i > 0 and raw_segs[i - 1][0] == REST_CUE:
            rest_ref = raw_segs[i - 1]
        elif i < len(raw_segs) - 1 and raw_segs[i + 1][0] == REST_CUE:
            rest_ref = raw_segs[i + 1]
        if rest_ref is None:
            # no adjacent rest -- fall back to whole-file rest stats
            rest_mask = cue == REST_CUE
            rest_start, rest_end = None, None
        else:
            _, rest_start, rest_end = rest_ref

        onset_idx_candidates = []
        for ch in channels:
            x = dfp[ch].to_numpy()[start:end]
            if rest_start is not None:
                rest_x = dfp[ch].to_numpy()[rest_start:rest_end]
            else:
                rest_x = dfp[ch].to_numpy()[rest_mask]
            rest_mav = rolling_mav(rest_x, mav_window)
            threshold = rest_mav.mean() + ONSET_THRESHOLD_MULT * rest_mav.std()
            mav = rolling_mav(x, mav_window)
            above = mav > threshold
            idx = find_sustained_crossing(above, sustain_samples)
            if idx is not None:
                onset_idx_candidates.append(idx)

        if onset_idx_candidates:
            onset_idx = min(onset_idx_candidates)
            trimmed_start = start + onset_idx
            onset_lag_ms = onset_idx * 1000.0 / TARGET_FS
            flagged = False
        else:
            trimmed_start = start
            onset_lag_ms = None
            flagged = True

        seg = Segment(
            cue=cue_val, start=start, end=end, trimmed_start=trimmed_start,
            onset_lag_ms=onset_lag_ms, onset_flagged=flagged,
            rest_ref_start=rest_start, rest_ref_end=rest_end,
        )
        segments.append(seg)

    # Fallback for segments where no channel produced a sustained crossing:
    # impute the file's median detected onset lag rather than leaving them
    # untrimmed (an untrimmed segment would include ~half a second of
    # pre-onset/rest-like signal mislabeled as the posture). These are
    # still marked onset_flagged=True for transparency/reporting.
    detected_lags = [s.onset_lag_ms for s in segments if s.onset_lag_ms is not None and not s.onset_flagged]
    if detected_lags:
        median_lag_ms = float(np.median(detected_lags))
        median_lag_samples = int(round(median_lag_ms * TARGET_FS / 1000))
        for seg in segments:
            if seg.cue != REST_CUE and seg.onset_flagged:
                seg.trimmed_start = min(seg.start + median_lag_samples, seg.end - 1)
                seg.onset_lag_ms = median_lag_ms

    return segments
