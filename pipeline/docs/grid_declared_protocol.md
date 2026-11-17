# Bitrate grid with FLOPs/MACs: DECLARED BEFORE ANY GRID RESULT (9 Nov 2026)

Part of `docs/plan_family_ablation_and_bitrate_grid.md`, steps 2 and 3. Nothing below may be changed after seeing grid results; a change is recorded as a labelled amendment at the end. The wide set W is fixed by the rule in
`docs/family_ablation_declared_protocol.md` (family ablation, step 1); the grid scripts read it from `cache/wide_set_W.json`.

## Question

How little data rate (sample rate × channels × bits) does the raw-stream architecture need for a given accuracy, with the original features (A) and with the reduced wide set (W), for four patients, and what do the classifiers cost per decision in FLOPs/MACs?

## Grid (as in the earlier exhaustive grid, `docs/14_experiments_since_august.md` section D)

- Sample rates 250, 500, 1000, 3000, 10000 Hz. Channels 2, 5, 8 (and 12 for S2): the top-N channels by the earlier channel ranking (`run_bitrate_search.get_ranking_and_grid`), identical for every configuration. Bit depths 8, 12, 16.
- Bit rate = rate × channels × bits, kept if ≤ 1.4 Mb/s. This gives the same 193 configurations as the earlier 1.28 Mb/s cap (P1 45, P2 45, S1 45, S2 58); the nearest excluded point is S2 at 10 kHz, 12 channels, 12 bit = 1.44 Mb/s.
- No added noise, no attenuation, no packet loss.
- Signal path = the sweep's own: filter at 30 kHz (100 Hz to min(499, fs/2 − 1)) → decimate → quantise → onset-based segmentation → 200 ms windows, 50 ms hop.
  Quantiser: Vaskov: range per channel per raw file (those files mix 4–11 grips each, checked 9 Nov 2026). **Wallace: one range per channel pooled over all of the participant's runs** (as a real ADC; the per-recording range leaks the grip class).
- Per-sample-rate notes (as found earlier): at 3 and 10 kHz the filter's upper edge stays at 499 Hz, so these rates carry the same 100–499 Hz band as 1 kHz, only oversampled; a flat curve above 1 kHz is expected by construction.
  At 250 and 500 Hz the pass-band is 100–124 and 100–249 Hz, so rate and band change together. The wavelet level is capped by the window length (A-set wavelet columns: 10 at 1 kHz and above, 8 at 500 Hz, 6 at 250 Hz).

### Deviations from the earlier grid (stated now)

1. **No feature selection.** Every model uses all columns of its feature set (A or W), as in Stage 1 and the raw-stream architecture; the earlier grid used Stage 1+2 selected features (MI ranking, correlation rejection, elbow). Numbers are **not directly comparable** with the earlier grid.
2. **One segmentation per (patient, rate, bits), shared by all channel subsets.** The earlier sweep re-ran onset detection on the chosen channels only (earliest onset among them), so each channel count had slightly different windows. Here the table is built once with all channels,
   and channel subsets are column slices of it (`grid_slices.py`; check V5: a slice equals direct computation on the subset's windows, per-channel, xcorr and bandcorr bit-identical, rmsshare equal to float32 rounding, tested 9 Nov 2026 at 250, 500 and 1000 Hz).
   Consequence: all channel counts share rows, labels and folds, so channel-count differences are paired by fold. Onset detection uses all channels (an offline labelling step, not part of inference).
3. The channel ranking was computed on all the data (not inside the cross-validation), as in the earlier grid; it is slightly optimistic for every channel count below the maximum.

## Feature sets

- **A**: the original types (`wide_features.set_columns(·, 'A')`), 24 per channel at 1 kHz and above (22 at 500 Hz, 20 at 250 Hz).
- **W**: the leaf families kept by the backward elimination of the ablation protocol, for every channel subset; cross-channel columns for the pairs inside the subset, `rmsshare` over the subset. W is chosen on Stage 1 data (4 patients, 1 kHz/16 bit and 500 Hz/8 bit),
  so W against A in the grid is **optimistic** at those operating points and a mild extrapolation at the others.

## Models and cross-validation

LDA, SVM, RF with the project settings (`models.make_model_pipeline`; RF 300 trees, `n_jobs` 2 per worker, which does not change the result), on the same grouped, class-stratified 5-fold CV (`models.compute_cv_folds` on the all-channel table, seed 0), pooled predictions.
Per cell (patient, rate, channels, bits, feature set, model) record: accuracy, weighted F1, Cohen's kappa, the pooled out-of-fold confusion matrix and class labels, the 5 fold accuracies, number of columns, number of windows, and from the fitted fold models
the support-vector count (SVM) and tree count and depth (RF). Each cell is run for sets A and W: 193 configurations × 2 sets × 3 models = 1,158 cells.
Execution order: sample rates up to 1 kHz before 3 and 10 kHz; within each group LDA, then SVM, then RF.

## FLOPs and MACs (analytic counts, not measurements)

Per patient, per cell, per decision (one 200 ms window; 20 decisions per second at the 50 ms hop). Implemented in `feature_cost.py`. Conventions: 1 MAC = 2 FLOPs; add, subtract, multiply, compare, abs, max = 1 FLOP; division, square root, log, exp are counted as
separate "special operations" (not folded into FLOPs); every intermediate (|x|, mean, variance, first difference, real FFT and power spectrum, wavelet decomposition) is computed once per window and channel and shared by the features that use it (charged only if a selected feature needs it);
each window is computed from scratch (no sliding-sum tricks); the figures describe a straightforward implementation (probably within a factor of about 2 of an efficient one) and compare configurations, they do not predict cycles or energy.
N = window samples, Nf = N/2 + 1 FFT bins, nb = FFT bins inside the pass-band, L = wavelet level of set A, DWT of n samples = 2 filters × ⌊(n + 7)/2⌋ outputs × 8 MACs.

| Item (per channel per window unless stated) | FLOPs | Special ops |
|---|---|---|
| shared: |x| (N); mean + centred variance (4N); first difference (N); real FFT + power spectrum (2.5 N log2 N + 3 Nf); A-set wavelet decomposition (DWTs of the approximations, 2 FLOPs per MAC) | as written | 1 (std) |
| MAV, RMS, WL, IEMG | N, 2N, 2N, 0 | 1, 2, 0, 0 |
| ZC, SSC | 6N, 8N | 0 |
| MNF, MDF, PeakFreq, BandPowerRatio | 2Nf, 2Nf, Nf, Nf | 1, 0, 0, 1 |
| Var, Skew, Kurtosis, ShannonEntropy (16-bin histogram) | 0, 4N, 4N, 3N + 64 | 0, 2, 2, 16 |
| wavelet energy / entropy per sub-band of length ℓ | 2ℓ / 3ℓ | 0 / ℓ |
| DASDV, LogDet, WAMP, MaxAbs | 2N, 3N, 3N, N | 1, N + 1, 0, 0 |
| HjMob, HjCx | 3N, 4N | 2, 3 |
| AR1–AR4 (Yule–Walker order 4: 5 autocorrelation lags, 4×4 solve), CC1–CC4 | 10N + 80 (charged to AR1), 40 (charged to CC1) | 0 |
| SpecSpread, SpecFlat | 6 Nf, 3 nb | 2, nb + 1 |
| BP1–BP8 (each) | nb/8 + 2 | 1 |
| WP1–WP8: level-3 wavelet packet (1 DWT of N, 2 of the next length, 4 of the next), energies | decomposition charged to WP1; each node 2ℓ₄ + 1 | 1 each |
| SpecSkew, SpecKurt | 10 nb, 4 nb | 4, 2 |
| dMAV1/2, dRMS1/2, dWL1/2 | 1 each | 0 |
| per channel pair: xcorr; bandcorr1–3 together | 2N; 6 nb | 2; 6 |
| rmsshare (per decision, all channels) | nch | nch |

Classifier cost per decision (counts from the fitted fold models, averaged over folds): standardisation of the inputs = F MACs for every model. LDA: F × C MACs. SVM (RBF, one-vs-one): n_SV × F MACs for the kernel + n_SV × (C − 1) MACs for the decision sums, n_SV exp.
RF: comparisons only, trees × (deepest tree) as an upper bound + trees × C vote additions, no MACs. RBFNN: centres × F + centres × C MACs, centres exp. LSTM (1 layer, H = 64, no extraction reuse): per window step 4H(F + H) + H × C MACs, 5H special.
(The SVM total adds the decision sums to the earlier convention of `model_cost_vs_features.py`, which counted n_SV × F only.)
Totals per decision = extraction FLOPs + 2 × model MACs + model comparison/vote operations (reported separately as well); per second = × 20.

## Analysis (all reported, whatever the outcome)

1. Per patient, model and feature set: the Pareto frontier of accuracy (and weighted F1, and kappa) against bit rate (`bitrate_pareto.pareto_frontier`; ties on equal bit rate and metric go to the first of (rate, channels, bits) ascending), its elbow on a log10-bit-rate axis
   (`bitrate_pareto.best_ratio_point`, the earlier max-distance-from-chord rule), and the cheapest configuration whose accuracy reaches 80%, 90%, 95%. The same against total FLOPs per decision and against model MACs.
2. Main effects over the grid as before: mean accuracy by channels, by sample rate, by bit depth, per patient, model and feature set, with the channel-count differences paired by fold (shared folds, section "Deviations").
3. W against A, paired by fold, per configuration and averaged over the grid, by patient and model (labelled optimistic, see above). LDA against SVM against RF.
4. For every patient and model: accuracy and F1 at the reference configuration (1000 Hz, 8 channels, 16 bit) and at the elbow, with the confusion matrices of both, and the FLOPs/MACs of both.
5. Fold-to-fold spread is reported next to every mean; differences under about 2 points are treated as noise (4 patients, 5 folds, intervals probably too narrow).

## Step 3: networks at the winners (rule fixed now)

Winners per patient and feature set (A and W): the distinct configurations among {accuracy-elbow, F1-elbow} × {LDA, SVM, RF} (frontiers of step 2), the cheapest configuration whose mean accuracy over LDA, SVM and RF reaches 95% (if one exists), and the reference configuration
(1000 Hz, 8 channels, 16 bit). At each winner, same table, same folds and metrics: RBFNN tuned (nested, centres {16, 32, 64, 128, 256} × width {0.5, 1, 2, 4} × C {0.1, 1, 10, 100}, inner 3-fold, as in Stage 1), LSTM with the headline setting (1 layer, 64 units, Adam 1e-3, 40 epochs,
batch 16) and the tuned LSTM (nested configuration and epoch choice, `run_lstm_training_check.part2`: 3 configurations × up to 400 epochs on a 20% validation split of the training sequences). The LSTM sees each segment as a sequence of window feature vectors (per-feature z-scoring on the
training folds), as in the earlier work, so it uses temporal context that the per-window classifiers do not. Its MACs are counted per window step as above (without the sequence context cost of anything but the recurrent state). Costs and confusion matrices as for the other models.

## Limits stated now

4 patients, 5 folds, one seed; intervals probably too narrow; no selection nested inside the CV for the channel ranking (optimistic); W chosen on the same data; one segmentation for all channel subsets (a deviation); 3 kHz and 10 kHz carry no new band by construction;
no noise, attenuation or packet loss; counts of operations are analytic, not measured on hardware, and say nothing about power or latency; the LSTM and RBFNN are run only at the winners (not on the whole grid).

## Amendments

1. (9 Nov 2026, found in the smoke test of the table builder at 250 Hz / 8 bit, before any model ran on a grid table.) A window whose quantised samples are all equal has zero variance, so the original set-A `Skew` and `Kurtosis` are NaN there (scipy).
   It happened for 6 windows of one weak Wallace channel (EDC) at 250 Hz / 8 bit. Non-finite feature values are set to 0 when a grid table is built (symmetric / normal-like), and the number of values and columns per table is printed in the build log (`cache/grid/tables.log`).
   No other column was non-finite in that test; any other column that turns up in a table log is reported in the results.
2. (9 Nov 2026. Made after the grid cells and a first analysis run existed, so **after seeing the as-coded elbows**; stated plainly.) While checking implausible-looking FLOPs elbows I found that `feature_selection.find_elbow` truncates its result to an integer, which on a log10 axis snaps the
   elbow to a whole number of decades (see `docs/elbow_truncation_note.md`; it also explains why the earlier grid found 10 kbps for every participant). The rule declared above is the geometric max-distance-from-chord rule, so the truncation is a bug, not a choice. The analysis therefore reports
   (a) the **geometric elbow** (no truncation, normalised axes) as the primary elbow and (b) the **as-coded elbow** (`bitrate_pareto.best_ratio_point`, truncated) for comparability with the earlier grid, and the step-3 winners contain the configurations of both. The model-cost axis is
   **model operations = MACs + comparison and vote operations** (a random forest has no MACs, so log10(MACs) was undefined for it). No grid cell was changed or re-run; this only changes the analysis (`analyse_grid.py`).
