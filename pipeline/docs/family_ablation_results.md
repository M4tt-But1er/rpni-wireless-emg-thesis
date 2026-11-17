# Family ablation results (9 Nov 2026)

Declared before any result in [family_ablation_declared_protocol.md](family_ablation_declared_protocol.md). All 448 cells are reported (28 feature subsets × LDA and SVM × 8 tables: 4 patients × `ref` 1 kHz / 16 bit and `low` 500 Hz / 8 bit).
Wallace S1 and S2 use the corrected fixed-range tables ([stage1_results_corrected.md](stage1_results_corrected.md)). Check V6 passed: the A and C cells equal the corrected Stage 1 cells exactly (same tables, folds and models).
Files: `cache/family_ablation_results.json` (cells), `cache/family_ablation_summary.txt` (all tables, including the per-table breakdown), `cache/family_elimination_summary.txt`, `cache/wide_set_W.json`.
Everything below is measured on 4 patients with 5 folds; differences under about 2 points are inside the noise, and the intervals are probably too narrow.

## In plain words

1. **No single family matters much once the others are present.** Dropping any one of the 10 leaf families from the full set C changes the mean accuracy by −0.28 to +0.04 points for LDA and −0.25 to +0.15 for SVM. In only 11 of 160 table-and-family-and-model cells is the change clearly
   resolved, and 8 of those are drops that hurt (3 are drops that helped). The families overlap a lot (for example sub-band powers and wavelet-packet energies both describe the spectrum).
2. **Four families carry most of the gain over the original features (set A):** sub-band powers (+0.9 LDA, +1.1 SVM when added to A), wavelet (+0.6, +1.0), channel-pair correlation `xcorr` (+1.0, +0.8) and band-limited pair correlation `bandcorr` (+0.7, +0.8).
   The other six add less than +0.5 for both models (and `shape_complexity`, `window_change` and `ar_cepstral` add about zero or slightly negative for SVM). At branch level, the whole frequency domain added to A gives +1.05 (LDA) and +1.48 (SVM), the cross-channel branch +0.88 and +0.88, the time-domain branch +0.30 and +0.46.
3. **Whole branches cost more than single families** when removed from C: frequency domain −0.92 (LDA) / −0.60 (SVM), cross-channel −0.65 / −0.42, time domain −0.55 / −0.28. Redundancy inside a branch hides each family's value, but a whole branch cannot be replaced.
4. **The wide set W for the grid** (rule fixed beforehand: LDA backward elimination, stop before a removal that costs more than 0.3 points against C): **W = amplitude, shape/complexity, sub-band powers, window-to-window change, `xcorr`**, 268 of the 632 columns at P1 `ref` (8 channels). The mean LDA accuracy over the 8 tables
   is 95.14 for W, 95.28 for C and 93.15 for A. W has no wavelet and no autoregressive families, which are the more expensive ones to compute.

## Drop one leaf family from C (mean difference against C over the 8 tables, points; resolved = interval excludes 0, "better/worse" counts of 8)

| Family | Columns (8 ch, P1) | LDA | resolved | SVM | resolved |
|---|---|---|---|---|---|
| amplitude | 64 | −0.28 | 0 / 1 | −0.25 | 0 / 1 |
| shape_complexity | 64 | −0.22 | 0 / 0 | −0.02 | 0 / 0 |
| spectral_summary | 64 | −0.13 | 0 / 1 | +0.15 | 1 / 0 |
| subband_power | 64 | −0.14 | 0 / 1 | −0.24 | 0 / 1 |
| wavelet | 144 | −0.04 | 1 / 0 | −0.08 | 0 / 0 |
| ar_cepstral | 64 | +0.04 | 0 / 0 | +0.03 | 1 / 0 |
| window_change | 48 | −0.19 | 0 / 2 | +0.06 | 0 / 0 |
| pair_xcorr | 28 | +0.00 | 0 / 0 | −0.00 | 0 / 0 |
| rms_share | 8 | −0.12 | 0 / 0 | −0.06 | 0 / 1 |
| pair_bandcorr | 84 | −0.03 | 0 / 0 | −0.03 | 0 / 0 |
| *branch: time_domain* | | −0.55 | 0 / 4 | −0.28 | 0 / 2 |
| *branch: frequency_domain* | | −0.92 | 0 / 1 | −0.60 | 0 / 2 |
| *branch: cross_channel* | | −0.65 | 0 / 1 | −0.42 | 0 / 3 |

(The "resolved" cells are written as better / worse for the version without the family against C; a "0 / 1" means that in 1 of 8 tables dropping the family clearly hurt.)

## Add one leaf family to A (mean difference against A, points)

| Family | LDA | resolved better / worse | SVM | resolved better / worse |
|---|---|---|---|---|
| amplitude | +0.21 | 3 / 0 | +0.47 | 1 / 0 |
| shape_complexity | +0.14 | 0 / 0 | −0.08 | 0 / 0 |
| spectral_summary | +0.29 | 0 / 0 | +0.11 | 1 / 0 |
| subband_power | **+0.93** | 4 / 0 | **+1.14** | 4 / 0 |
| wavelet | **+0.61** | 2 / 0 | **+0.99** | 3 / 0 |
| ar_cepstral | +0.21 | 0 / 0 | +0.01 | 0 / 2 |
| window_change | +0.28 | 2 / 0 | −0.10 | 0 / 0 |
| pair_xcorr | **+0.95** | 1 / 0 | **+0.76** | 2 / 0 |
| rms_share | −0.06 | 0 / 0 | +0.18 | 1 / 0 |
| pair_bandcorr | **+0.70** | 1 / 0 | **+0.77** | 2 / 0 |
| *branch: time_domain* | +0.30 | 1 / 0 | +0.46 | 1 / 0 |
| *branch: frequency_domain* | **+1.05** | 4 / 0 | **+1.48** | 4 / 0 |
| *branch: cross_channel* | **+0.88** | 0 / 0 | **+0.88** | 1 / 0 |

## Descriptive labels (rules fixed in the protocol)

"Dispensable" = mean drop within ±0.3 for LDA and SVM; "useful" = mean add at least +0.5 for LDA and SVM. **Useful: sub-band powers, wavelet, `xcorr`, `bandcorr`** (all four are also individually dispensable from C, because the others cover for them).
**Dispensable and not useful by the rule: amplitude, shape/complexity, spectral summary, AR/cepstral, window change, RMS share.** The labels are descriptions of 4 patients; with differences this small they are not proof that a family never matters.

## Backward elimination (LDA, objective = mean accuracy over the 8 tables)

| Step | Removed | Objective | Against C |
|---|---|---|---|
| start | (C, all 10) | 95.279 | |
| 1 | ar_cepstral | 95.320 | +0.042 |
| 2 | pair_bandcorr | 95.294 | +0.015 |
| 3 | wavelet | 95.309 | +0.031 |
| 4 | spectral_summary | 95.183 | −0.096 |
| 5 | rms_share | 95.139 | −0.140 |
| 6 (refused) | window_change | 94.960 | −0.318 (limit −0.3) |

W = amplitude, shape_complexity, subband_power, window_change, pair_xcorr. The first step reuses what the drop-one cells show (they are equal, LDA is deterministic); later steps were new LDA evaluations, cached in `cache/family_elimination_cache.json`.

## What this does and does not show

- It shows which groups of features carry the gain in this data, and that the wide set is highly redundant: about 40% of C's columns (W) keep 95.14 of C's 95.28 (LDA mean).
- It does **not** show that the wavelet family is useless: it was removed because the sub-band powers carry much of the same information, and wavelet is one of the four families that helps when added to A. Which of the two to keep is a choice by cost and by this rule, not a finding.
- The 0.3-point stopping threshold is far below the noise (1–3 points), and the last step was refused by 0.018 points. A threshold of 0.35 would also have removed `window_change`. W reflects which families LDA does not need on these tables, not a resolved difference.
- **W is not a superset of A:** it drops the A-set wavelet columns (10 per channel at 1 kHz) and A's spectral columns (MNF, MDF, PeakFreq, BandPowerRatio), because those belong to the wavelet and spectral-summary families. So W against A in the grid compares different families.
- W was chosen on the same 4 patients and 2 operating points it is evaluated on, so W looks better against A on those tables than it will on new data. The grid applies W at other rates and channel counts, where this is a mild extrapolation.
- LDA and SVM only (RF was not run); greedy elimination can miss families that matter only together; the families are my grouping by what a feature measures, not a data-driven clustering.
- No hardware cost was measured. By the analytic counts to be declared for the grid (`feature_cost.py`), W avoids the wavelet and AR computations.

## Reproduce

```
set STAGE1_TAG=fixedrange
python -W ignore -u family_ablation.py --workers 8            # 448 cells, about 5 minutes on this PC
python -W ignore compare_family_ablation.py > cache/family_ablation_summary.txt
python -W ignore -u family_backward_elimination.py --workers 8 > cache/family_elimination_summary.txt
```
