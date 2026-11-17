# Family ablation: DECLARED BEFORE ANY RESULT (9 Nov 2026)

Part of `docs/plan_family_ablation_and_bitrate_grid.md`, step 1. Nothing below may be changed after seeing results; a change is recorded as a labelled amendment at the end.

## Question

Which feature families inside the wide set C carry the accuracy (what is useful), and which can be dropped without loss (what is dispensable)? And which reduced wide set W should the bitrate grid use?

## Data

The eight Stage 1 tables (4 patients × `ref` 1 kHz / 16 bit and `low` 500 Hz / 8 bit; all channels; 200 ms window, 50 ms hop). **Wallace S1/S2 use the fixed-range tables** (`cache/stage1_fixedrange/`, built with `STAGE1_TAG=fixedrange`),
P1/P2 the original tables (`cache/stage1/`). The script refuses to run without `STAGE1_TAG=fixedrange`. Same grouped, class-stratified 5-fold CV (`models.compute_cv_folds`) as Stage 1, so folds are identical across all feature subsets of a table.
Models: LDA and SVM with the project settings (`run_own_feature_selection.fit_predict`). RF is left out (slowest; no gain from the wide set in Stage 1). No feature selection, nothing tuned, so nothing is nested.

## Families (10 leaves in 5 branches; every one of the 632 columns of a Stage 1 table, at `ref`, belongs to exactly one leaf)

| Branch | Leaf | Members (per channel unless stated) |
|---|---|---|
| time_domain | amplitude | MAV, RMS, WL, IEMG, Var, MaxAbs, DASDV, LogDet |
| time_domain | shape_complexity | ZC, SSC, WAMP, HjMob, HjCx, Skew, Kurtosis, ShannonEntropy |
| frequency_domain | spectral_summary | MNF, MDF, PeakFreq, BandPowerRatio, SpecSpread, SpecFlat, SpecSkew, SpecKurt |
| frequency_domain | subband_power | BP1–BP8 |
| frequency_domain | wavelet | `wav_energy_*`, `wav_entropy_*` (A4, D4–D1; A3, D3–D1 at 500 Hz), WP1–WP8 |
| model_based | ar_cepstral | AR1–AR4, CC1–CC4 |
| temporal_context | window_change | dMAV1/2, dRMS1/2, dWL1/2 |
| cross_channel | pair_xcorr | `xcorr_*` (per channel pair) |
| cross_channel | rms_share | `rmsshare_*` (per channel) |
| cross_channel | pair_bandcorr | `bandcorr1..3_*` (per channel pair) |

Per channel at `ref`: amplitude 8, shape 8, spectral 8, subband 8, wavelet 10 + 8, AR/cepstral 8, change 6 = 64 (= A 24 + B 16 + C 24). A column is assigned by the type name at the end of its column name, as `wide_features.feature_group` does.

## Feature subsets (28 per table)

- `A` (original types) and `C` (everything): the references.
- `drop:<leaf>` for the 10 leaves: C without that leaf. `add:<leaf>`: A plus all columns of that leaf (the columns the leaf adds to A).
- Branch level, only for branches with more than one leaf (a one-leaf branch equals its leaf): `drop:<branch>` and `add:<branch>` for time_domain, frequency_domain, cross_channel.

28 subsets (2 + 10 + 10 + 3 + 3) × 2 models × 8 tables = 448 cells. Reported per cell: accuracy, weighted F1, Cohen's kappa (pooled predictions) and the 5 fold accuracies.

## Comparisons (all reported, whatever the outcome)

1. `drop` against C and `add` against A, paired by fold: mean difference in points with a 95% t-interval over 5 folds (4 df); "resolved" = the interval excludes 0; differences under about 2 points are treated as noise
   (as in Stage 1; the intervals are probably too narrow). Per table, and as a mean over the 8 tables per model; counts of resolved cells.
2. A family is called **dispensable** if the mean drop-one difference over the 8 tables is within ±0.3 points of 0 for both LDA and SVM, and **useful** if the mean add-one difference is at least +0.5 points for both. These labels are descriptive
   (rules fixed now); other cases are reported as mixed.
3. Per-leaf column counts, so that accuracy per column can be read off.

## Choosing the wide set W for the grid (rule fixed now)

Backward elimination over the 10 leaf families, LDA only, starting from C. Objective = mean pooled accuracy over the 8 tables. At each step, remove the family whose removal gives the highest objective (ties within 0.05 points: remove the one with more
columns per channel first). Stop before a removal if the objective after it would be more than 0.3 points below the objective of C. W = the families left. The first step reuses the drop-one cells above; later steps are LDA evaluations on the 8 tables,
cached in `cache/family_elimination_cache.json`. The order of removal and the objective at each step are reported in full.
Limits of this rule, stated now: W is chosen on the same data it is scored on, so W against A on these tables (and in the grid) is optimistic; greedy elimination can miss pairs of families that matter together; with 4 patients and 5 folds the
0.3-point threshold is far below the noise (1–3 points), so W will mostly reflect which families LDA does not need, not a resolved difference.

## Limits

4 patients, 5 folds, one seed; the intervals are probably too narrow; the families are my grouping (by what the feature measures), not a data-driven clustering; AR order, wavelet choice, number of bands are untuned; LDA and SVM only.

## Amendments

(none yet)
