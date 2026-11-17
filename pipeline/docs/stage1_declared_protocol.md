# Stage 1 comparison: the wide feature ladder at two operating points. DECLARED BEFORE ANY RESULT (8 Nov 2026)

This file is written before the new features are built or any model is run on them. Nothing below may be changed after seeing accuracies. If a definition has to change (for example, a bug), the change is
recorded as a new labelled version (C2) and the original C results stay in the report.

## Question

In a raw-stream architecture (raw samples sent over the wireless link, classified on a body-worn unit), the feature count is free on the implant. So: (1) how much does a much wider feature set raise accuracy,
and (2) how much accuracy does the low-rate, low-bit operating point (500 Hz, 8 bit) give up against the reference (1 kHz, 16 bit), for each feature set?

## Operating points (both use the existing sweep signal path, `bitrate_sweep.build_one_config` / `wallace_bitrate.build_wallace_config`: filter at 30 kHz, decimate, requantise, window)

| Name | Sample rate | Bit depth | Pass-band | Window / hop | Channels | Noise |
|---|---|---|---|---|---|---|
| `ref` | 1000 Hz | 16 | 100–499 Hz | 200 ms / 50 ms | all (8, 8, 8, 12 for P1, P2, S1, S2) | none |
| `low` | 500 Hz | 8 | 100–249 Hz | 200 ms / 50 ms | all | none |

Quantiser: the sweep's own `quantize` (0.1–99.9 percentile range per channel per recording). Bit rate (assumed 8 channels): `ref` 128 kb/s, `low` 32 kb/s.
Note: the sweep path differs slightly from the cached feature tables used elsewhere (for example 3,152 against 3,153 windows for P1 at `ref`), so every number here is compared with other numbers from the same path only.

## Feature sets (nested: A inside B inside C). "Per channel" counts below.

**Set A** (24 per channel): the project's original types from `features.extract_channel_features(xw, fs, band)`, unchanged.

**Set B** = A + 16 per channel + 2 cross-channel groups, exactly as `build_extra_features.py` defines them (DASDV, LogDet, WAMP at 25 % of the window SD, Hjorth mobility and complexity, MaxAbs, AR1–AR4 by
Yule–Walker order 4, cepstral CC1–CC4 from the AR fit, spectral spread, spectral flatness) plus `xcorr` (Pearson correlation of every channel pair in the window) and `rmsshare` (channel RMS over the sum of RMS).
The only change: the spectral-flatness band is (100 Hz, min(499, fs/2 − 1)) instead of the fixed 100–499 Hz, so it follows the pass-band as the original code does. At 1 kHz it is identical (checked, below).

**Set C** = B + the following new families (26 per channel + band-limited correlation per pair), defined here once:

| Family | Definition (per window; passband = (100 Hz, min(499, fs/2 − 1)); P = |rfft(window)|², window unpadded) | Count |
|---|---|---|
| Sub-band powers `BP1..BP8` | log10(sum of P in the band + 1e-12 × total P + 1e-300); the passband split into 8 equal-width bands | 8 per channel |
| Wavelet-packet energies `WP1..WP8` | Daubechies-4 (`db4`, symmetric extension) packet tree to level 3 by repeated `pywt.dwt` on both branches (natural node order); log10(mean of squared coefficients + 1e-300) per node | 8 per channel |
| Shape statistics | time-domain skewness `Skew` and excess kurtosis `Kurt`; spectral skewness `SpecSkew` and excess kurtosis `SpecKurt` of P over the passband bins (0 where the variance is 0) | 4 per channel |
| Window-to-window changes `dMAV1, dMAV2, dRMS1, dRMS2, dWL1, dWL2` | the Set A MAV, RMS and WL of this window minus those of the window 1 and 2 hops (50 and 100 ms) earlier in the same segment; the lag is clipped at the segment start (so the first windows have lag 0 and a change of 0) | 6 per channel |
| Band-limited correlation `bandcorr1..3` | for every channel pair, the real part of the normalised cross-spectrum, Re(sum X_a conj(X_b)) / sqrt(sum |X_a|² sum |X_b|²), over each of 3 equal sub-bands of the passband (X = rfft) (this is a band-limited correlation, not coherence) | 3 per pair |

Resulting columns: Set A 192 (S2: 288); Set B 356 (558); Set C 648 (1,068) for 8 (12) channels. The changes use only earlier windows of the same segment, so they need no future samples.

### Amendment 1 (8 Nov 2026, made while testing the builder, before any model was run on any new table)

1. **Time-domain skewness and kurtosis are removed from Set C.** They already exist in Set A (`Skew`, `Kurtosis`), and my declaration above duplicated them (and the first test build overwrote Set A's `Skew`
   with the duplicate; that test table was deleted). Set C keeps only the spectral shape statistics `SpecSkew` and `SpecKurt`. So the shape statistics are 2 per channel, Set C adds 24 per channel (8 + 8 + 2 + 6) plus
   3 band-limited correlations per pair, and the column counts become Set C = 632 for 8 channels and 1,044 for 12.
2. **At 500 Hz Set A has 22 types per channel, not 24.** The original `extract_channel_features` caps the wavelet level at what the window length allows (level 3 for 100 samples, level 4 for 200 samples), so the
   `wav_*_A4/D4` pair becomes `A3/D3` and one level is lost. That is the original code's documented behaviour, not a change by me. Column counts at 500 Hz are 22 × channels for Set A and correspondingly fewer for B and C.
3. The extended-pool types `AR*`, `CC*`, etc. are unaffected.

### Amendment 2 (8 Nov 2026, user instruction "tune the RBFNN before reporting"; made before any model result on the new tables)

A tuned RBFNN (`RBFNN_tuned`) is run on every set and operating point, nested: for each outer fold, hyper-parameters are chosen by the grouped, stratified inner 3-fold CV on the outer-training part only
(`run_rbfnn_tuning.tune_rbfnn`: centres {16, 32, 64, 128, 256} × width multiplier {0.5, 1, 2, 4} × C {0.1, 1, 10, 100}, k-means fitted once per inner fold and centre count, ties go to fewer centres), then the model is refitted
on the whole outer-training part and scored on the untouched outer fold. The untuned RBFNN declared above is still run, as the reference. The report will not be written until the tuned runs are finished.

## Models, cross-validation, metrics

- LDA, SVM, RF with the project's fixed settings (`models.make_model_pipeline`); **RBFNN with its fixed untuned settings (64 centres)**. A properly tuned RBFNN is run later, only on the winning set.
- No feature selection, so nothing is nested. Same grouped, class-stratified 5-fold CV as everywhere (`models.compute_cv_folds` on each table). Sets A, B and C of one patient and operating point share rows, so their folds are identical.
- Report accuracy, weighted F1 and Cohen's kappa from pooled predictions, and per-fold accuracies.

## Comparisons to be reported (all of them, whatever the outcome)

1. Within each operating point, per patient and model: B − A, C − B, C − A, paired by fold, mean with a 95 % t-interval over 5 folds (4 df); "resolved" = interval excludes 0. Differences under about 2 points are treated as noise. 16 cells per comparison.
2. Per set and model: `low` − `ref`. The two operating points produce different tables (window counts differ slightly), so this is **unpaired**; I report the difference of pooled accuracies and the fold-wise difference as a rough guide only.
3. Which model is best at each set × operating point, with the same intervals (LDA against each of SVM, RF, RBFNN).

## Checks that must pass before any result is looked at

- V1: Set A columns from the new builder equal `build_one_config` / `build_wallace_config` output exactly (same shape, names, values) at `ref` and `low` for at least P1 and S1.
- V2: the Set B per-channel function with the fs-aware band equals `build_extra_features.extra_channel_features` at fs = 1000 on real windows (exact), and its recomputed RMS and WL equal Set A's columns.
- V3: every column finite in every table; constant columns are counted and reported, not removed.
- V4: Set B and C row order and metadata identical to Set A.

## Limits stated now

Four patients, five folds; the intervals are probably too narrow (folds share training data); one seed; untuned definitions (WAMP threshold, AR order, wavelet choice, number of bands). The new families were chosen from
the literature-style list the user approved, not from a search on these data; the choice of 8 bands and level 3 is a convenience, not a tuned value. The effective number of independent samples is closer to the
number of segments (59–96) than to the number of windows, so a result at 648–1,068 columns is in a regime where small differences should not be trusted.
