# EXPLORATORY: are RPNI signals more informative or more detailed than muscle signals? Declared protocol (14 Nov 2026, written before any result)

User request (14 Nov 2026): run options 1-4 of the earlier answer, as an exploratory study. Nothing in the thesis, the grid, the
channel-importance results or any existing cache file is replaced; all outputs go to `cache/rpni_vs_muscle/`.

**Channel type** (as in `channel_importance_declared_protocol.md`): RPNI = P1 RPNI_ulnar, RPNI_median; P2 RPNI_median, RPNI_ulnar1,
RPNI_ulnar2; S1 Median1, Ulnar1, Ulnar2; S2 Radial, Median1-4. All other implanted channels are intramuscular (muscle) channels.
13 RPNI and 23 muscle channels in 4 participants.

**Common settings for tests 1, 2 and 4 (fixed, nothing tuned).** Wide set W; a channel subset keeps the W columns whose channels are all in
the subset (`grid_slices.slice_matrix`); LDA, SVM and RF with the grid's settings (`run_grid.fit_predict_info`); grouped 5-fold CV of the
grid (`models.compute_cv_folds`); per-window predictions pooled over the folds, no vote. Every number is reported per model and as the
mean over the three models; conclusions use the mean. Check: single-channel and RPNI-only accuracies on the 30 kHz stream must equal the
values already in `cache/channel_importance/evals.json`.

## Test 1. Fair head-to-head at equal channel count
RPNI channels only, against every subset of the same number of muscle channels only (P1 2 of 6 = 15 subsets, P2 3 of 5 = 10, S1 3 of 5 = 10,
S2 5 of 7 = 21), on the 30 kHz / 16-bit stream (E1). Reported: RPNI-only accuracy, the muscle-only mean and range, and the RPNI-only rank
among all subsets. **Rule:** RPNIs are called more (less) informative than an equal number of muscle channels for a participant only if
RPNI-only lies above (below) the whole muscle range; otherwise "within the range".

## Test 2. Which movements each channel separates
For each channel alone (30 kHz stream), the per-movement recall (fraction of a movement's windows classified correctly), mean over the three
models. Reported per participant: (a) for every movement, the best single channel and whether it is an RPNI; (b) the best RPNI against the
best muscle channel per movement; (c) mean recall of RPNI and muscle channels per movement group: thumb and finger flexion (cues 2-6),
grasps (7 fist, 8 pinch, 9 point, 18 thumb opposition), extension and wrist/hand (10, 11, 13, 14). **Rule:** an RPNI is said to specialise
in a movement group in a participant if the best RPNI beats the best muscle channel by more than 5 points in most movements of that group.
Movements recur across participants, so a specialisation is called consistent only if it appears in at least 3 of the 4 participants.

## Test 3. Signal quality (closest to "detail")
From the raw 30 kHz recordings (the same files and runs as the tables), per channel:
- **Movement-to-rest ratio (dB), in band:** signal filtered as the pipeline does (100-499 Hz, zero-phase 4th-order Butterworth); segments
  from the pipeline's own segmentation (`bitrate_sweep.build_segments_variable_rate`: movement segments from the detected onset, rest
  segments with 250 ms trimmed at each end). Ratio = 20 log10(median movement-segment RMS / median rest-segment RMS), over all movements,
  and per movement (the best movement and the number of movements above 6 dB are reported).
- **Same ratio, broadband:** 20 Hz-7 kHz band instead of 100-499 Hz.
- **Frequency content of the movement signal:** Welch PSD (Hann, 0.1 s segments) on the broadband (20 Hz-7 kHz) signal, averaged over
  movement segments, minus the PSD averaged over rest segments (negative values set to 0), i.e. the movement-related excess power. From the
  excess PSD: median frequency, mean frequency, the frequency below which 95 % of the excess power lies, and the fraction of excess power
  above 500 Hz (the part that every stream of the thesis discards).
Caveats stated in advance: amplitudes depend on electrode type, placement and amplifier, which differ between Vaskov and Wallace, so
comparisons are made within a participant; the rest periods of Wallace runs precede a known grip and can carry anticipatory activity.

## Test 4. Does the information survive compression?
Single-channel accuracy for every channel on every grid table (5 sample rates 250, 500, 1000, 3000, 10 000 Hz x 3 bit depths 8, 12, 16;
`cache/grid/tables/`, Wallace with the fixed quantiser range) plus the 30 kHz / 16-bit table. Reported per channel and as the RPNI and muscle
means per participant: loss from rate = acc(1 kHz, 16 bit) - acc(250 Hz, 16 bit) and acc(1 kHz, 16 bit) - acc(500 Hz, 16 bit); loss from bit
depth = acc(1 kHz, 16 bit) - acc(1 kHz, 8 bit); combined loss = acc(30 kHz, 16 bit) - acc(250 Hz, 8 bit). Note: every stream is band-limited
to at most 499 Hz, so rates of 1 kHz and above carry the same band, and differences there are expected to be small.
**Rule:** RPNI information is said to sit in finer detail if the RPNI mean loss exceeds the muscle mean loss by more than 2 points in at
least 3 of the 4 participants. Because single-channel accuracy differs between channels, the loss is also reported relative to the channel's
accuracy above chance (loss / (acc - chance)).

## Statistics and reporting
13 RPNIs in 4 people: every test is descriptive. For tests 3 and 4, a within-participant permutation test of the channel-type labels
(all relabellings, mean RPNI minus mean muscle, two-sided) is reported per participant and, combined, as the sum of the per-participant
differences; p-values are indicative only (no correction, no pre-registration of an alpha). Each test reports a result even if it is null.

Scripts: `run_rpni_vs_muscle.py` (tests 1, 2, 4), `rpni_signal_quality.py` (test 3), `analyse_rpni_vs_muscle.py` (summary).
Outputs: `cache/rpni_vs_muscle/` (evals.json, signal_quality.json, summary.md); results doc `docs/rpni_vs_muscle_exploratory_results.md`.
