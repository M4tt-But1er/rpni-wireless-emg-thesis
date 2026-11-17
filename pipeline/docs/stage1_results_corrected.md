# Stage 1 corrected: Wallace S1 and S2 with a fixed quantiser range (9 Nov 2026)

Replaces the numbers of [stage1_results.md](stage1_results.md) for Wallace S1 and S2; that file is kept unchanged (with a correction banner). P1 and P2 are not affected (their results are byte-identical to the original file; their raw files mix 4–11 grips each,
checked 9 Nov 2026). Protocol: [stage1_declared_protocol.md](stage1_declared_protocol.md), unchanged apart from the quantiser (see "What was wrong, and what changed").

## What was wrong, and what changed

The first Stage 1 tables quantised each Wallace recording to its **own** 0.1–99.9 percentile range per channel. A Wallace recording is one grip, so the range itself carries the grip (the leak documented in
`docs/14_experiments_since_august.md`, section D, which I did not carry into Stage 1). The corrected tables use **one range per channel pooled over all of a participant's runs**, as a real ADC has
(`wallace_bitrate.global_quant_ranges` and `quantize_fixed`, switch `STAGE1_TAG=fixedrange` in `build_wide_tables.py`). Everything else is unchanged: same signal path, same features, same folds and models, same tuned-RBFNN grid.
New files only: `cache/stage1_fixedrange/` (4 tables), `cache/stage1_ops_results_fixedrange.json`, `cache/stage1_rbfnn_tuned_results_fixedrange.json` (P1/P2 entries copied unchanged from the originals), summaries `cache/stage1_ops_summary_fixedrange.txt` (20 cells),
`cache/stage1_ops_summary_fixedrange_declared4.txt` (16 cells) and `cache/stage1_headline_fixedrange.txt`. Checks V1–V4 were re-run on the corrected tables and passed (`cache/stage1_fixedrange_validate.log`);
window counts differ by 1–22 from the original tables because onset detection runs on the differently quantised signal.

## How large the effect was

Pooled accuracy, original -> corrected (change in points), Wallace only:

| Patient | Operating point | Model | Set A old -> corrected | Set B old -> corrected | Set C old -> corrected |
|---|---|---|---|---|---|
| S1 | ref 1 kHz / 16 bit | LDA | 93.1 -> 91.7 (-1.4) | 94.3 -> 93.2 (-1.1) | 94.7 -> 93.7 (-1.0) |
| S1 | ref 1 kHz / 16 bit | SVM | 93.8 -> 92.6 (-1.3) | 94.8 -> 93.4 (-1.4) | 95.4 -> 93.9 (-1.5) |
| S1 | ref 1 kHz / 16 bit | RandomForest | 91.4 -> 90.5 (-0.9) | 94.1 -> 91.0 (-3.0) | 93.4 -> 91.2 (-2.1) |
| S1 | low 500 Hz / 8 bit | LDA | 92.9 -> 90.9 (-2.0) | 94.8 -> 91.8 (-3.0) | 95.5 -> 92.6 (-2.9) |
| S1 | low 500 Hz / 8 bit | SVM | 94.1 -> 91.3 (-2.8) | 94.1 -> 92.2 (-1.9) | 95.2 -> 93.5 (-1.7) |
| S1 | low 500 Hz / 8 bit | RandomForest | 91.5 -> 90.5 (-1.0) | 93.6 -> 91.3 (-2.3) | 93.1 -> 91.6 (-1.5) |
| S2 | ref 1 kHz / 16 bit | LDA | 93.8 -> 92.1 (-1.7) | 94.8 -> 92.9 (-1.9) | 95.1 -> 93.7 (-1.4) |
| S2 | ref 1 kHz / 16 bit | SVM | 92.6 -> 91.0 (-1.6) | 94.1 -> 92.5 (-1.6) | 94.5 -> 93.4 (-1.2) |
| S2 | ref 1 kHz / 16 bit | RandomForest | 91.9 -> 90.7 (-1.2) | 94.4 -> 91.5 (-2.8) | 94.0 -> 91.7 (-2.3) |
| S2 | low 500 Hz / 8 bit | LDA | 89.9 -> 87.6 (-2.3) | 91.9 -> 90.3 (-1.6) | 93.8 -> 91.6 (-2.2) |
| S2 | low 500 Hz / 8 bit | SVM | 87.7 -> 85.9 (-1.8) | 89.1 -> 87.8 (-1.3) | 90.7 -> 89.9 (-0.8) |
| S2 | low 500 Hz / 8 bit | RandomForest | 85.6 -> 86.5 (+0.9) | 90.3 -> 87.3 (-3.0) | 89.5 -> 87.9 (-1.6) |

Mean change over S1 and S2, sets A–C, LDA/SVM/RF: **−1.65 points at `ref` (all 18 cells negative) and −1.83 points at `low`** (17 of 18 negative; the one rise is RF on S2 set A, +0.9).
**My earlier guess in the banner of `stage1_results.md`, that the 16-bit `ref` tables would be affected much less, was wrong:** the 16-bit tables moved as much as the 8-bit ones. The leak comes from the per-recording clipping range, not only from the coarse 8-bit step.

## Corrected results

Accuracy (%) per patient, pooled over 5 folds (weighted F1 and kappa: `cache/stage1_ops_summary_fixedrange.txt`, `cache/stage1_headline_fixedrange.txt`). Feature counts as before.

**ref 1 kHz / 16 bit**

| Patient | Model | Set A | Set B | Set C |
|---|---|---|---|---|
| P1 | LDA | 96.8 | 98.1 | 98.9 |
| P1 | SVM | 97.8 | 98.3 | 98.7 |
| P1 | RandomForest | 97.8 | 97.8 | 97.8 |
| P1 | RBFNN | 95.0 | 96.3 | 97.0 |
| P1 | RBFNN_tuned | 96.8 | 97.7 | 97.7 |
| P2 | LDA | 96.1 | 97.0 | 97.7 |
| P2 | SVM | 95.6 | 96.8 | 96.8 |
| P2 | RandomForest | 94.9 | 95.9 | 95.9 |
| P2 | RBFNN | 90.1 | 93.2 | 93.4 |
| P2 | RBFNN_tuned | 94.0 | 95.7 | 96.6 |
| S1 | LDA | 91.7 | 93.2 | 93.7 |
| S1 | SVM | 92.6 | 93.4 | 93.9 |
| S1 | RandomForest | 90.5 | 91.0 | 91.2 |
| S1 | RBFNN | 86.0 | 87.4 | 88.7 |
| S1 | RBFNN_tuned | 90.8 | 91.4 | 92.7 |
| S2 | LDA | 92.1 | 92.9 | 93.7 |
| S2 | SVM | 91.0 | 92.5 | 93.4 |
| S2 | RandomForest | 90.7 | 91.5 | 91.7 |
| S2 | RBFNN | 78.9 | 83.3 | 84.0 |
| S2 | RBFNN_tuned | 88.5 | 89.9 | 91.5 |

**low 500 Hz / 8 bit**

| Patient | Model | Set A | Set B | Set C |
|---|---|---|---|---|
| P1 | LDA | 97.2 | 97.7 | 98.3 |
| P1 | SVM | 97.8 | 98.3 | 98.6 |
| P1 | RandomForest | 97.5 | 97.3 | 97.5 |
| P1 | RBFNN | 94.2 | 95.4 | 96.2 |
| P1 | RBFNN_tuned | 96.4 | 97.3 | 97.4 |
| P2 | LDA | 92.7 | 94.4 | 95.7 |
| P2 | SVM | 92.2 | 94.9 | 95.4 |
| P2 | RandomForest | 94.5 | 95.2 | 95.5 |
| P2 | RBFNN | 89.2 | 91.4 | 91.9 |
| P2 | RBFNN_tuned | 91.5 | 92.9 | 94.1 |
| S1 | LDA | 90.9 | 91.8 | 92.6 |
| S1 | SVM | 91.3 | 92.2 | 93.5 |
| S1 | RandomForest | 90.5 | 91.3 | 91.6 |
| S1 | RBFNN | 83.4 | 84.7 | 86.2 |
| S1 | RBFNN_tuned | 90.1 | 91.1 | 91.5 |
| S2 | LDA | 87.6 | 90.3 | 91.6 |
| S2 | SVM | 85.9 | 87.8 | 89.9 |
| S2 | RandomForest | 86.5 | 87.3 | 87.9 |
| S2 | RBFNN | 75.2 | 76.5 | 77.4 |
| S2 | RBFNN_tuned | 85.1 | 86.2 | 88.3 |

Mean over the 4 patients (accuracy %), original in brackets:

| Operating point | Model | Set A | Set B | Set C |
|---|---|---|---|---|
| ref | LDA | 94.19 (94.97) | 95.28 (96.03) | 96.01 (96.61) |
| ref | SVM | 94.24 (94.96) | 95.24 (96.01) | 95.69 (96.36) |
| ref | RF | 93.50 (94.02) | 94.09 (95.56) | 94.16 (95.28) |
| ref | RBFNN untuned | 87.50 (88.85) | 90.04 (91.04) | 90.78 (91.61) |
| ref | RBFNN tuned | 92.52 (93.87) | 93.67 (94.59) | 94.60 (95.05) |
| low | LDA | 92.10 (93.18) | 93.56 (94.70) | 94.55 (95.84) |
| low | SVM | 91.81 (92.97) | 93.28 (94.07) | 94.34 (94.98) |
| low | RF | 92.25 (92.28) | 92.79 (94.11) | 93.11 (93.88) |
| low | RBFNN untuned | 85.50 (85.86) | 87.00 (87.10) | 87.91 (88.50) |
| low | RBFNN tuned | 90.80 (91.54) | 91.87 (92.45) | 92.83 (93.35) |

### The statements of the original report, one by one (corrected numbers; 16 cells = the four models declared before the tuned RBFNN was added)

1. **Wider features help, with diminishing returns: still true.** C minus A: **+1.79 at `ref`** (was +1.76), **+2.05 at `low`** (was +2.23); better in 16 of 16 cells at both; resolved better in 8 of 16 at `ref` (was 11) and 9 of 16 at `low` (was 10); none worse.
   B minus A: +1.30 and +1.23 (was +1.45, +1.42). C minus B: +0.49 and +0.82 (was +0.31, +0.81), resolved in 4 and 5 of 16 cells (was 2 and 3). Per model C minus A, `ref` / `low`: LDA +1.8 / +2.4, SVM +1.4 / +2.5, RF +0.7 / +0.9, untuned RBFNN +3.3 / +2.4.
2. **Low minus reference (unpaired, 16 cells): about the same as before on average, but the "wide features shrink the loss" idea is weaker.** Set A −1.94, set B −2.01, set C −1.68 (was −2.13, −2.16, −1.67). For LDA: −2.1 (A), −1.7 (B), −1.5 (C) (was −1.8, −1.3, −0.8);
   for SVM: −2.4, −2.0, −1.3 (was −2.0, −1.9, −1.4). The loss is still concentrated in P2 and S2 (S2 LDA −4.5 at set A, −2.1 at set C); P1 is within 0.6, S1 within 1.4. A rough Welch interval over 5 + 5 folds (`cache/stage1_headline_fixedrange.txt`) excludes 0 in none of the cells for the strong models.
   Both operating points of the Wallace tables moved by about the same amount, which is why the average low-minus-ref barely changed.
3. **Headline: original outline at `ref` (set A) against the wide set at `low` (set C).** Mean over 4 patients: **LDA +0.4, SVM +0.1, RF −0.4, tuned RBFNN +0.3** (was +0.9, 0.0, −0.1, −0.5). So the wide set at half the rate and 8 bit (32 kb/s against 128 kb/s for 8 channels) is about equal to the original
   features at the reference rate; per patient (LDA): P1 +1.5, P2 −0.4, S1 +0.9, S2 −0.5. Not a clear gain, not a clear loss.
4. **No model is clearly better than LDA: still true.** LDA minus SVM resolved in 0 of 16 cells; LDA minus RF resolved in 5 of 16 (4 for LDA, 1 for RF); LDA minus tuned RBFNN resolved in 4 of 16 (all for LDA). Mean LDA minus tuned RBFNN: +1.7 (`ref` A), +1.4 (`ref` C), +1.3 (`low` A), +1.7 (`low` C) (was +1.1, +1.6, +1.6, +2.5).
   Times within 0.5 point of the best model, 24 patient × operating point × set cases: LDA 17, SVM 17, RF 6, RBFNN (either) 0 (was 15, 14, 7, 0).
5. **Tuning the RBFNN helps a lot, but not enough to catch LDA, and the grid was too small: still true.** Tuned minus untuned +3.6 to +5.3 points on average (resolved in 20 of 24 cells). The inner search picked the top of the grid in 45–65% of folds for centres (256) and
   65–85% for C (100) (`cache/stage1_ops_summary_fixedrange.txt`), so the tuned RBFNN is probably still a lower bound.

## Limits (unchanged, plus one)

4 patients, 5 folds, one seed, intervals probably too narrow; the new feature definitions untuned; `low` minus `ref` is unpaired; no hardware measurements. Plus: the Wallace tables were built twice, and the corrected build is the only one to be used from now on. The grid (`docs/grid_declared_protocol.md`) and the
family ablation (`docs/family_ablation_declared_protocol.md`) use the corrected tables for Wallace.
