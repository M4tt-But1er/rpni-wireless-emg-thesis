# Bitrate grid results (9 Nov 2026)

Declared before any grid result in [grid_declared_protocol.md](grid_declared_protocol.md) (two amendments, both listed there and below). 193 configurations (rate × channels × bits ≤ 1.4 Mb/s: P1 45, P2 45, S1 45, S2 58) × feature sets A and W × LDA, SVM, RF = **1,158 cells**, all run, none skipped.
Per patient and cell the results hold accuracy, weighted F1, Cohen's kappa, the pooled out-of-fold confusion matrix, the five fold accuracies, FLOPs and MACs per decision (`cache/grid/grid_results.json`; one row per cell in `cache/grid/grid_cells.csv`).
Step 3 adds the tuned RBFNN and the LSTM at 28 winner cells (84 more cells, see below).
Everything is measured on 4 patients with 5 grouped folds and one seed; Wallace S1 and S2 use the corrected fixed quantiser range. Differences under about 2 points are inside the noise, and the intervals are probably too narrow.

## Checks that passed

- **V5** (`test_grid_slices.py`): a channel subset cut from an all-channel table equals direct computation on the subset's windows (per-channel, `xcorr`, `bandcorr` bit-identical; `rmsshare` equal to float32 rounding, below 1e-7), tested at 250, 500 and 1000 Hz.
- **V7** (`check_grid_vs_stage1.py`): the grid's set-A cells at 1 kHz / 16 bit and 500 Hz / 8 bit with all channels equal the corrected Stage 1 cells **exactly** (24 cells, down to the fold accuracies), although they go through different code.
- **V8** (`check_grid_integrity.py`): in all 1,158 cells the confusion matrix reproduces the accuracy and the number of windows; every table has one window count for all channel subsets; the column counts match the expected structure; bit rates are right; no non-finite metric; extraction cost rises with the channel count.

## Amendments

1. Skewness and kurtosis of a window whose quantised samples are all equal are NaN in the original code (scipy); 14 such values (7 windows of one weak channel, EDC, at 250 Hz / 8 bit: 12 values in Wallace S1, 2 in P2) were set to 0. No other table had a non-finite value.
2. The elbow rule. **The earlier grid's elbow ("10 kbps for every participant") was an artefact of an integer truncation in `find_elbow`** ([elbow_truncation_note.md](elbow_truncation_note.md)). I found this after the grid had run and a first analysis had printed the truncated elbows. The new analysis reports the geometric elbow
   (the rule that was declared) as the primary and the truncated one as "as coded" for comparison, and uses model operations (MACs plus comparisons) on the model-cost axis, because a random forest has no MACs.

## In plain words

1. **Channels matter most.** Paired by fold at the same rate, bits, feature set and model, going from 2 to 5 channels adds about 15 to 18 points for every patient (17.4, 14.9, 17.9, 17.0 for P1, P2, S1, S2), from 5 to 8 channels 4.6, 2.6 and 5.1 points for P1, P2, S1, and **18.7 points for S2** (S2 then gains another 3.9 points from 8 to 12 channels).
   S2 needs 8 to 12 channels; with 5 channels it stays near 65–68% on average.
2. **Sample rate matters up to 1 kHz, then it is flat.** Mean accuracy rises from 250 to 500 to 1,000 Hz (for example P1 set A 78.0 → 86.6 → 89.1; S2 set W 64.0 → 71.3 → 77.5) and is within about 1.5 points between 1, 3 and 10 kHz (S2 is 2 to 3 points lower at 10 kHz than at 3 kHz). This is expected by construction:
   the filter's upper edge stays at 499 Hz, so 3 and 10 kHz carry the same band as 1 kHz, only oversampled. It does not show that content above 500 Hz is useless.
3. **8 bits is enough, and is not worse.** With the original features (set A), 8 bit is 1.4 to 2.0 points above 16 bit on average (P1 87.9 against 86.2, P2 87.3 against 85.3, S1 80.9 against 79.1, S2 71.4 against 70.0); with W the three depths are within 0.8 points (S1: 8 bit is 0.2 below 16 bit).
   I did not test below 8 bit, and I did not investigate why 8 bit is slightly better for set A (an untested guess: the coarse step removes low-amplitude jitter that disturbs the zero-crossing and slope-change counts).
4. **Elbow of the accuracy-against-bit-rate frontier (geometric).** P1: 250 Hz, 8 channels, 8 bit = **16 kb/s** (93–97% for LDA, SVM, RF with either feature set). P2: 500 Hz, 5 channels, 8 bit = 20 kb/s (90–94%; SVM with W picks 10 kb/s, 88.6%). S1: 500 Hz, 8 channels, 8 bit = 32 kb/s (91–93.5%;
   the RF and W-LDA elbows sit at 10 and 16 kb/s with 82% and 88.5%). S2: 250 Hz, 8 channels, 8 bit = 16 kb/s, but only 74–80%. **The elbow is a geometric trade-off, not an accuracy target.** The truncated elbow of the earlier grid (10 kb/s for everyone) gave 5 to 22 points less accuracy in 7 of 8 earlier cases (the note), and in most cells here (compare tables T1 and T2).
5. **Cheapest stream for a target** (tables T3). For 90%: P1 10 kb/s with W (250 Hz, 5 ch, 8 b; 16 kb/s with A), P2 16–20 kb/s, S1 32 kb/s, S2 48 kb/s with W for LDA and SVM (500 Hz, 12 ch, 8 b; 96 kb/s for RF) and 96 kb/s with A. For 95%: P1 16 kb/s with W (32 kb/s with A for LDA and SVM), P2 32–40 kb/s with W (64 kb/s with A), **not reached for S1 or S2**.
6. **The wide set W beats the original features on average by +2.9 points over the 579 paired cells** (resolved better in 317, worse in 6), but almost only for LDA and SVM: +3.9 to +4.4 points for LDA and SVM per patient (resolved better in 31 to 40 of 45–58 configurations, never worse), against +0.2 to +1.1 for RF.
   The gain is largest at the cheap end: +5.1 at 250 Hz and +4.0 with 2 channels, against +1.7 at 10 kHz and +2.0 with 8 or 12 channels. **This is optimistic**: W was chosen on Stage 1 data (see the ablation). It is less of a worry for the cheap end, which is far from the 1 kHz and 500 Hz tables W was chosen on.
7. **Models.** Averaged over the whole grid, RF is best with set A (P1: RF 89.3, SVM 86.0, LDA 85.4) because it copes with the poor low-rate, few-channel configurations; with W, SVM and LDA catch up and pass RF (SVM best for P1, P2, S1; LDA 74.0 against SVM 73.7 and RF 71.9 for S2). At the reference configuration with set A the three are within about 2 points for every patient; with W, RF is 1 to 5 points behind LDA and SVM for P2, S1 and S2 (table T4).
8. **Cost per decision** (analytic counts, not measured). Extraction dominates for LDA and RF: with 8 channels at the reference configuration set A costs about 158 kFLOP and W 129 kFLOP (W is 14–18% cheaper at every rate, because it has no wavelet or autoregressive features); at 250 Hz both cost 31–36 kFLOP, at 10 kHz 1.4–1.7 MFLOP (table T6).
   The model itself is small for LDA (2.5 kOps with A, 3.6 with W) and RF (about 9–10 kOps of comparisons) and large for the SVM (about 315 kOps with A, 367 with W, set by the number of support vectors), so the SVM costs 4 to 8 times as many FLOPs in total as LDA for at most about 1 point of accuracy (table T4).
9. **The saving from the reference configuration (1 kHz, 8 channels, 16 bit = 128 kb/s) to the elbow** (table T4), for example P1 set W, LDA: 98.6% at 128 kb/s and 136 kFLOP per decision against 96.4% at 16 kb/s and 38 kFLOP: 8 times less data and 3.6 times fewer FLOPs for 2.2 points. For S2 the same step costs 12 points (91.9% to 79.5%): S2 loses a lot when the rate drops to 250 Hz (its 90% points are at 500 Hz with 12 channels, or 1 kHz).

## Tables (generated by `make_grid_tables.py`; accuracy in %, kappa on the 0–1 scale)

### T1. Geometric accuracy elbow of the Pareto frontier against bit rate (configuration, bit rate, accuracy %)

| Patient | Set | LDA | SVM | RF |
|---|---|---|---|---|
| P1 | A | 250 Hz, 8 ch, 8 b: 16 kb/s, 94.2 | 250 Hz, 8 ch, 8 b: 16 kb/s, 93.4 | 250 Hz, 8 ch, 8 b: 16 kb/s, 95.5 |
| P1 | W | 250 Hz, 8 ch, 8 b: 16 kb/s, 96.4 | 250 Hz, 8 ch, 8 b: 16 kb/s, 96.5 | 250 Hz, 8 ch, 8 b: 16 kb/s, 94.9 |
| P2 | A | 500 Hz, 5 ch, 8 b: 20 kb/s, 90.4 | 500 Hz, 5 ch, 8 b: 20 kb/s, 90.5 | 500 Hz, 5 ch, 8 b: 20 kb/s, 92.3 |
| P2 | W | 500 Hz, 5 ch, 8 b: 20 kb/s, 94.3 | 250 Hz, 5 ch, 8 b: 10 kb/s, 88.6 | 500 Hz, 5 ch, 8 b: 20 kb/s, 92.6 |
| S1 | A | 500 Hz, 8 ch, 8 b: 32 kb/s, 90.9 | 500 Hz, 8 ch, 8 b: 32 kb/s, 91.3 | 250 Hz, 5 ch, 8 b: 10 kb/s, 82.4 |
| S1 | W | 250 Hz, 8 ch, 8 b: 16 kb/s, 88.5 | 500 Hz, 8 ch, 8 b: 32 kb/s, 93.5 | 250 Hz, 5 ch, 8 b: 10 kb/s, 81.9 |
| S2 | A | 250 Hz, 8 ch, 8 b: 16 kb/s, 74.4 | 250 Hz, 8 ch, 8 b: 16 kb/s, 74.6 | 250 Hz, 8 ch, 8 b: 16 kb/s, 78.7 |
| S2 | W | 250 Hz, 8 ch, 8 b: 16 kb/s, 79.5 | 250 Hz, 8 ch, 8 b: 16 kb/s, 80.3 | 250 Hz, 8 ch, 8 b: 16 kb/s, 80.2 |

### T2. The same elbow as coded in the earlier grid (elbow truncated to a whole number of decades; see docs/elbow_truncation_note.md)

| Patient | Set | LDA | SVM | RF |
|---|---|---|---|---|
| P1 | A | 250 Hz, 5 ch, 8 b: 10 kb/s, 82.8 | 250 Hz, 5 ch, 8 b: 10 kb/s, 83.6 | 250 Hz, 5 ch, 8 b: 10 kb/s, 90.0 |
| P1 | W | 250 Hz, 5 ch, 8 b: 10 kb/s, 90.9 | 250 Hz, 5 ch, 8 b: 10 kb/s, 91.1 | 250 Hz, 5 ch, 8 b: 10 kb/s, 89.9 |
| P2 | A | 250 Hz, 5 ch, 8 b: 10 kb/s, 83.1 | 250 Hz, 5 ch, 8 b: 10 kb/s, 83.4 | 250 Hz, 5 ch, 8 b: 10 kb/s, 86.5 |
| P2 | W | 250 Hz, 5 ch, 8 b: 10 kb/s, 87.8 | 250 Hz, 5 ch, 8 b: 10 kb/s, 88.6 | 250 Hz, 5 ch, 8 b: 10 kb/s, 87.0 |
| S1 | A | 250 Hz, 5 ch, 8 b: 10 kb/s, 77.5 | 250 Hz, 5 ch, 8 b: 10 kb/s, 77.4 | 250 Hz, 5 ch, 8 b: 10 kb/s, 82.4 |
| S1 | W | 250 Hz, 5 ch, 8 b: 10 kb/s, 81.5 | 250 Hz, 5 ch, 8 b: 10 kb/s, 81.9 | 250 Hz, 5 ch, 8 b: 10 kb/s, 81.9 |
| S2 | A | 250 Hz, 5 ch, 8 b: 10 kb/s, 51.4 | 250 Hz, 5 ch, 8 b: 10 kb/s, 50.5 | 250 Hz, 5 ch, 8 b: 10 kb/s, 57.6 |
| S2 | W | 250 Hz, 5 ch, 8 b: 10 kb/s, 56.9 | 250 Hz, 5 ch, 8 b: 10 kb/s, 57.3 | 250 Hz, 5 ch, 8 b: 10 kb/s, 58.9 |

### T3. Cheapest configuration with accuracy >= 90% (bit rate, configuration; 'not reached' if none)

| Patient | Set | LDA | SVM | RF |
|---|---|---|---|---|
| P1 | A | 16 kb/s (250 Hz, 8 ch, 8 b, 94.2%) | 16 kb/s (250 Hz, 8 ch, 8 b, 93.4%) | 16 kb/s (250 Hz, 8 ch, 8 b, 95.5%) |
| P1 | W | 10 kb/s (250 Hz, 5 ch, 8 b, 90.9%) | 10 kb/s (250 Hz, 5 ch, 8 b, 91.1%) | 16 kb/s (250 Hz, 8 ch, 8 b, 94.9%) |
| P2 | A | 20 kb/s (500 Hz, 5 ch, 8 b, 90.4%) | 20 kb/s (500 Hz, 5 ch, 8 b, 90.5%) | 16 kb/s (250 Hz, 8 ch, 8 b, 90.2%) |
| P2 | W | 16 kb/s (250 Hz, 8 ch, 8 b, 91.7%) | 16 kb/s (250 Hz, 8 ch, 8 b, 92.8%) | 16 kb/s (250 Hz, 8 ch, 8 b, 90.6%) |
| S1 | A | 32 kb/s (500 Hz, 8 ch, 8 b, 90.9%) | 32 kb/s (500 Hz, 8 ch, 8 b, 91.3%) | 32 kb/s (500 Hz, 8 ch, 8 b, 90.5%) |
| S1 | W | 32 kb/s (500 Hz, 8 ch, 8 b, 92.6%) | 32 kb/s (500 Hz, 8 ch, 8 b, 93.5%) | 32 kb/s (500 Hz, 8 ch, 8 b, 91.0%) |
| S2 | A | 96 kb/s (1000 Hz, 12 ch, 8 b, 92.6%) | 96 kb/s (1000 Hz, 12 ch, 8 b, 91.2%) | 96 kb/s (1000 Hz, 12 ch, 8 b, 90.5%) |
| S2 | W | 48 kb/s (500 Hz, 12 ch, 8 b, 91.0%) | 48 kb/s (500 Hz, 12 ch, 8 b, 90.5%) | 96 kb/s (1000 Hz, 12 ch, 8 b, 90.4%) |

### T3. Cheapest configuration with accuracy >= 95% (bit rate, configuration; 'not reached' if none)

| Patient | Set | LDA | SVM | RF |
|---|---|---|---|---|
| P1 | A | 32 kb/s (500 Hz, 8 ch, 8 b, 97.2%) | 32 kb/s (500 Hz, 8 ch, 8 b, 97.8%) | 16 kb/s (250 Hz, 8 ch, 8 b, 95.5%) |
| P1 | W | 16 kb/s (250 Hz, 8 ch, 8 b, 96.4%) | 16 kb/s (250 Hz, 8 ch, 8 b, 96.5%) | 32 kb/s (500 Hz, 8 ch, 8 b, 97.1%) |
| P2 | A | 64 kb/s (1000 Hz, 8 ch, 8 b, 96.7%) | 64 kb/s (1000 Hz, 8 ch, 8 b, 96.5%) | 64 kb/s (1000 Hz, 8 ch, 8 b, 95.5%) |
| P2 | W | 32 kb/s (500 Hz, 8 ch, 8 b, 95.9%) | 40 kb/s (1000 Hz, 5 ch, 8 b, 96.4%) | 32 kb/s (500 Hz, 8 ch, 8 b, 95.3%) |
| S1 | A | not reached | not reached | not reached |
| S1 | W | not reached | not reached | not reached |
| S2 | A | not reached | not reached | not reached |
| S2 | W | not reached | not reached | not reached |

### T4. Reference configuration (1000 Hz, 8 ch, 16 b) against the geometric accuracy elbow: accuracy / weighted F1 / kappa, bit rate, FLOPs and model operations per decision

| Patient | Set | Model | Reference: acc / F1 / kappa | kb/s | kFLOP | model kOps | Elbow config | Elbow: acc / F1 / kappa | kb/s | kFLOP | model kOps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | A | LDA | 96.8 / 96.8 / 0.965 | 128 | 163 | 2.5 | 250 Hz, 8 ch, 8 b | 94.2 / 94.1 / 0.936 | 16 | 41 | 2.1 |
| P1 | A | SVM | 97.8 / 97.8 / 0.976 | 128 | 631 | 236.5 | 250 Hz, 8 ch, 8 b | 93.4 / 93.3 / 0.927 | 16 | 594 | 278.7 |
| P1 | A | RandomForest | 97.8 / 97.8 / 0.976 | 128 | 167 | 8.7 | 250 Hz, 8 ch, 8 b | 95.5 / 95.4 / 0.950 | 16 | 46 | 9.6 |
| P1 | W | LDA | 98.6 / 98.6 / 0.984 | 128 | 136 | 3.5 | 250 Hz, 8 ch, 8 b | 96.4 / 96.3 / 0.960 | 16 | 38 | 3.5 |
| P1 | W | SVM | 98.7 / 98.7 / 0.986 | 128 | 673 | 272.3 | 250 Hz, 8 ch, 8 b | 96.5 / 96.4 / 0.961 | 16 | 804 | 386.7 |
| P1 | W | RandomForest | 97.7 / 97.6 / 0.974 | 128 | 137 | 8.4 | 250 Hz, 8 ch, 8 b | 94.9 / 94.9 / 0.944 | 16 | 40 | 9.9 |
| P2 | A | LDA | 96.1 / 96.1 / 0.957 | 128 | 163 | 2.5 | 500 Hz, 5 ch, 8 b | 90.4 / 90.3 / 0.895 | 20 | 51 | 1.4 |
| P2 | A | SVM | 95.6 / 95.5 / 0.951 | 128 | 638 | 239.7 | 500 Hz, 5 ch, 8 b | 90.5 / 90.4 / 0.896 | 20 | 355 | 153.6 |
| P2 | A | RandomForest | 94.9 / 94.9 / 0.944 | 128 | 167 | 8.4 | 500 Hz, 5 ch, 8 b | 92.3 / 92.2 / 0.916 | 20 | 58 | 9.6 |
| P2 | W | LDA | 97.7 / 97.8 / 0.975 | 128 | 136 | 3.5 | 500 Hz, 5 ch, 8 b | 94.3 / 94.3 / 0.938 | 20 | 42 | 2.1 |
| P2 | W | SVM | 96.3 / 96.3 / 0.960 | 128 | 679 | 275.0 | 250 Hz, 5 ch, 8 b | 88.6 / 88.4 / 0.874 | 10 | 490 | 235.9 |
| P2 | W | RandomForest | 95.7 / 95.7 / 0.953 | 128 | 137 | 8.4 | 500 Hz, 5 ch, 8 b | 92.6 / 92.5 / 0.919 | 20 | 47 | 9.6 |
| S1 | A | LDA | 91.7 / 91.7 / 0.910 | 128 | 164 | 2.7 | 500 Hz, 8 ch, 8 b | 90.9 / 90.8 / 0.901 | 32 | 82 | 2.5 |
| S1 | A | SVM | 92.6 / 92.6 / 0.920 | 128 | 920 | 381.1 | 500 Hz, 8 ch, 8 b | 91.3 / 91.3 / 0.906 | 32 | 896 | 409.6 |
| S1 | A | RandomForest | 90.5 / 90.6 / 0.897 | 128 | 169 | 10.8 | 250 Hz, 5 ch, 8 b | 82.4 / 82.3 / 0.809 | 10 | 36 | 12.9 |
| S1 | W | LDA | 93.5 / 93.5 / 0.929 | 128 | 136 | 3.8 | 250 Hz, 8 ch, 8 b | 88.5 / 88.5 / 0.875 | 16 | 38 | 3.8 |
| S1 | W | SVM | 93.9 / 94.0 / 0.934 | 128 | 1001 | 436.2 | 500 Hz, 8 ch, 8 b | 93.5 / 93.5 / 0.930 | 32 | 1058 | 497.4 |
| S1 | W | RandomForest | 90.8 / 90.9 / 0.900 | 128 | 139 | 9.9 | 250 Hz, 5 ch, 8 b | 81.9 / 81.8 / 0.804 | 10 | 30 | 11.7 |
| S2 | A | LDA | 87.2 / 87.1 / 0.860 | 128 | 163 | 2.5 | 250 Hz, 8 ch, 8 b | 74.4 / 74.3 / 0.720 | 16 | 41 | 2.1 |
| S2 | A | SVM | 87.4 / 87.3 / 0.863 | 128 | 962 | 401.9 | 250 Hz, 8 ch, 8 b | 74.6 / 74.4 / 0.722 | 16 | 910 | 437.0 |
| S2 | A | RandomForest | 87.0 / 86.8 / 0.858 | 128 | 169 | 10.8 | 250 Hz, 8 ch, 8 b | 78.7 / 78.5 / 0.767 | 16 | 48 | 11.7 |
| S2 | W | LDA | 91.9 / 91.8 / 0.912 | 128 | 136 | 3.5 | 250 Hz, 8 ch, 8 b | 79.5 / 79.4 / 0.776 | 16 | 38 | 3.5 |
| S2 | W | SVM | 90.7 / 90.5 / 0.898 | 128 | 1094 | 482.8 | 250 Hz, 8 ch, 8 b | 80.3 / 80.2 / 0.785 | 16 | 1365 | 667.1 |
| S2 | W | RandomForest | 86.9 / 86.7 / 0.857 | 128 | 139 | 10.2 | 250 Hz, 8 ch, 8 b | 80.2 / 80.0 / 0.784 | 16 | 42 | 11.4 |

### T5. Main effects: mean accuracy (%) over the other settings and over LDA, SVM, RF, per patient and feature set


**Channels**

| Patient | Set | 2 | 5 | 8 | 12 |
|---|---|---|---|---|---|
| P1 | A | 73.1 | 90.9 | 96.7 |  |
| P1 | W | 77.2 | 94.2 | 97.7 |  |
| P2 | A | 74.5 | 90.3 | 93.4 |  |
| P2 | W | 79.3 | 93.1 | 95.3 |  |
| S1 | A | 65.6 | 84.3 | 89.6 |  |
| S1 | W | 69.7 | 86.9 | 91.7 |  |
| S2 | A | 47.9 | 65.1 | 83.7 | 87.4 |
| S2 | W | 51.1 | 67.8 | 86.6 | 89.4 |

**Sample rate (Hz)**

| Patient | Set | 250 | 500 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|---|
| P1 | A | 78.0 | 86.6 | 89.1 | 90.3 | 90.3 |
| P1 | W | 82.6 | 89.6 | 91.6 | 92.5 | 92.1 |
| P2 | A | 76.2 | 84.2 | 89.7 | 90.0 | 90.3 |
| P2 | W | 81.7 | 88.5 | 91.8 | 92.1 | 92.0 |
| S1 | A | 70.3 | 79.3 | 82.8 | 83.5 | 83.3 |
| S1 | W | 76.0 | 82.6 | 85.0 | 85.2 | 85.0 |
| S2 | A | 59.4 | 67.6 | 75.4 | 76.8 | 73.6 |
| S2 | W | 64.0 | 71.3 | 77.5 | 78.3 | 75.1 |

**Bit depth**

| Patient | Set | 8 | 12 | 16 |
|---|---|---|---|---|
| P1 | A | 87.9 | 86.5 | 86.2 |
| P1 | W | 89.9 | 89.6 | 89.5 |
| P2 | A | 87.3 | 85.6 | 85.3 |
| P2 | W | 89.6 | 89.1 | 89.0 |
| S1 | A | 80.9 | 79.5 | 79.1 |
| S1 | W | 82.6 | 82.9 | 82.8 |
| S2 | A | 71.4 | 69.9 | 70.0 |
| S2 | W | 73.7 | 72.9 | 72.9 |

### T6. Extraction cost per decision (kFLOP, analytic) of the two feature sets at selected configurations (8 channels; S2 also 12)

| Configuration | Set A | Set W |
|---|---|---|
| 250 Hz, 8 ch, 8 b (P1) | 36 (F = 160) | 31 (F = 268) |
| 500 Hz, 8 ch, 8 b (P1) | 77 (F = 176) | 63 (F = 268) |
| 1000 Hz, 8 ch, 16 b (P1) | 158 (F = 192) | 129 (F = 268) |
| 3000 Hz, 8 ch, 16 b (P1) | 480 (F = 192) | 402 (F = 268) |
| 10000 Hz, 8 ch, 16 b (P1) | 1653 (F = 192) | 1408 (F = 268) |

### T7. Model cost per decision at the reference configuration (kOps = MACs + comparisons): mean over the 4 patients

| Set | LDA | SVM | RF |
|---|---|---|---|
| A | 2.5 | 314.8 (SVM support vectors drive this) | 9.7 |
| W | 3.6 | 366.6 (SVM support vectors drive this) | 9.2 |


## Step 3: tuned RBFNN and LSTM at the 28 winner cells

Winners (rule fixed in the protocol, plus the as-coded elbows of Amendment 2): per patient and feature set, the distinct accuracy and F1 elbows of LDA, SVM and RF (geometric and as coded), the cheapest configuration reaching 95% mean accuracy over the three models (where one exists), and the reference configuration:
28 cells (A: 14, W: 14; table N3). At each one: the nested-tuned RBFNN, the LSTM with the headline setting (1 layer, 64 units, 40 epochs) and the tuned LSTM (configuration and epoch count chosen on a validation split of the training sequences), on the same folds, with the same metrics, confusion matrices and cost counts.
84 cells, all run (`cache/grid/networks_results.json`, summary `cache/grid/networks_summary.txt`, confusion matrices of every model at every winner `cache/grid/confusion_matrices.md`).

1. **The LSTM beats the per-window classifiers at the winner cells.** Mean accuracy over the 28 cells (%): set A: LDA 86.5, SVM 86.6, RF 88.3, tuned RBFNN 86.2, LSTM 89.7, tuned LSTM 90.2; set W: LDA 89.0, SVM 89.0, RF 87.8, tuned RBFNN 87.7, LSTM 92.3, tuned LSTM 92.2.
   Paired by fold over the 28 cells: tuned LSTM minus LDA **+3.45** points (resolved better in 13 cells, worse in none), minus SVM +3.37 (9 better), minus RF +3.10 (9 better), minus the best classical model of each cell +2.08 (7 better, none worse); the default LSTM: +3.25 (7 better), +3.17 (9), +2.90 (11), +1.88 (7).
2. **The gain is largest where the per-window classifiers are weakest:** the cheap configurations and S1 and S2 (table N2; for example S2 set W: LDA 76.1, LSTM 83.6; S1 set A: LDA 86.7, LSTM 91.5, tuned 92.6). At the reference configuration it is small or absent for P2 and S2 (P2 W: LDA 97.7, LSTM 96.3, tuned 96.7; S2 W: 91.9, 92.3, 92.4) and +2.3 to +2.6 for P1 with A (table N1).
3. **Tuning the LSTM did not help on average:** 90.2 against 89.7 with A, 92.2 against 92.3 with W. The tuning picked the higher learning rate (3e-3) in 96 of 140 folds and a median of 202 epochs (range 15 to 400).
4. **The tuned RBFNN is slightly below LDA** (−0.78 points on average, resolved worse in 4 of 28 cells, better in none) and 2.15 below the best classical model of each cell (worse in 10 cells). Its inner search again picked the top of the grid (256 centres in 86 of 140 folds, C = 100 in 89 of 140), so it is probably a lower bound on what a tuned RBFNN could do.
5. **Cost per decision at the reference configuration** (table N1): the LSTM needs 66.5 kMAC with A and 86.0 kMAC with W for the recurrent step, about 291 and 301 kFLOP in total with extraction, against 136–164 kFLOP for LDA, 137–169 for RF, 227–273 for the tuned RBFNN and 631–1,094 for the SVM. The LSTM also has to keep its recurrent state (64 cell and 64 hidden values) from one window to the next.

**Caveat on the LSTM comparison (important).** The LSTM reads each segment as a sequence of windows in time order, starting at the segment's onset, with its state reset at every segment start; a segment holds one movement, so the LSTM can use the persistence of the class over time.
The per-window classifiers get no such context, and I did **not** give them a causal vote or smoothing over the last few decisions, which would very likely lift them. The segment start is known here from the labels and the offline onset detection; a deployed system would need onset detection or a free-running state.
So the result is "a per-window classifier against a sequence model with segment context", not "neural network against classical machine learning on equal information". The comparison to run next, if you want it, is the classical models with a causal majority vote over the last k decisions.

> **Update, 9 Nov 2026 (later the same day): the fair comparison was run, and it reverses the ranking.** Details, protocol and tables: [fair_lstm_results.md](fair_lstm_results.md). With a causal majority vote over all decisions since the segment onset (no free parameter), LDA, SVM and RF gain +4.9, +5.6 and +5.8 points, and the best of the three is then **3.3 points above the tuned LSTM** on average (LSTM behind in 26 of 28 cells and in all four patients; the default LSTM 3.5 points behind). Points 1 and 2 above remain correct as statements about per-window classifiers **without history** (they reproduce exactly, +2.08 against the best classical model); they must not be quoted as "the LSTM beats the classical models". The LSTM here sits at about the level of a vote over the last 9 to 15 decisions.
>
> **Second update, 9 Nov 2026 (follow-up to the above):** a soft vote and stacked windows for the classical models, and a much larger LSTM search (64 random configurations in two rounds, nested per fold, 5-seed ensembles), did not change the ranking: the best classical pipeline (vote, no stacking) is 94.75% against 89.78% for the single-seed tuned LSTM (−5.0) and 92.14% for the 5-seed LSTM (−2.6), see [fair_followup_results.md](fair_followup_results.md).
> The result holds for this steady-state, onset-segmented benchmark with this LSTM and its limited tuning and data (47 to 78 training sequences per outer fold, from 59 to 96 segments per patient); it says nothing about transitions or online onset detection.

### Tables for step 3 (generated by `make_network_tables.py`)

#### N1. Reference configuration (1000 Hz, 8 ch, 16 b = 128 kb/s): all six models, accuracy / weighted F1 / kappa, model MACs and total FLOPs per decision (extraction + 2 x MACs + comparisons)

| Patient | Set | LDA | SVM | RandomForest | RBFNN_tuned | LSTM_default | LSTM_tuned |
|---|---|---|---|---|---|---|---|
| P1 | A | 96.8 / 96.8 / 0.965<br>2.5 kMAC, 163 kFLOP | 97.8 / 97.8 / 0.976<br>236.5 kMAC, 631 kFLOP | 97.8 / 97.8 / 0.976<br>0.0 kMAC, 167 kFLOP | 96.7 / 96.7 / 0.963<br>42.0 kMAC, 242 kFLOP | 99.4 / 99.4 / 0.993<br>66.5 kMAC, 291 kFLOP | 99.1 / 99.1 / 0.990<br>66.5 kMAC, 291 kFLOP |
| P1 | W | 98.6 / 98.6 / 0.984<br>3.5 kMAC, 136 kFLOP | 98.7 / 98.7 / 0.986<br>272.3 kMAC, 673 kFLOP | 97.7 / 97.6 / 0.974<br>0.0 kMAC, 137 kFLOP | 97.5 / 97.5 / 0.972<br>71.9 kMAC, 273 kFLOP | 99.6 / 99.6 / 0.995<br>86.0 kMAC, 301 kFLOP | 99.3 / 99.3 / 0.993<br>86.0 kMAC, 301 kFLOP |
| P2 | A | 96.1 / 96.1 / 0.957<br>2.5 kMAC, 163 kFLOP | 95.6 / 95.5 / 0.951<br>239.7 kMAC, 638 kFLOP | 94.9 / 94.9 / 0.944<br>0.0 kMAC, 167 kFLOP | 94.2 / 94.2 / 0.936<br>44.6 kMAC, 247 kFLOP | 95.8 / 95.8 / 0.954<br>66.5 kMAC, 291 kFLOP | 94.6 / 94.6 / 0.940<br>66.5 kMAC, 291 kFLOP |
| P2 | W | 97.7 / 97.8 / 0.975<br>3.5 kMAC, 136 kFLOP | 96.3 / 96.3 / 0.960<br>275.0 kMAC, 679 kFLOP | 95.7 / 95.7 / 0.953<br>0.0 kMAC, 137 kFLOP | 95.8 / 95.8 / 0.954<br>64.8 kMAC, 258 kFLOP | 96.3 / 96.3 / 0.960<br>86.0 kMAC, 301 kFLOP | 96.7 / 96.7 / 0.964<br>86.0 kMAC, 301 kFLOP |
| S1 | A | 91.7 / 91.7 / 0.910<br>2.7 kMAC, 164 kFLOP | 92.6 / 92.6 / 0.920<br>381.1 kMAC, 920 kFLOP | 90.5 / 90.6 / 0.897<br>0.0 kMAC, 169 kFLOP | 91.3 / 91.3 / 0.905<br>34.3 kMAC, 227 kFLOP | 94.6 / 94.6 / 0.942<br>66.6 kMAC, 291 kFLOP | 93.4 / 93.3 / 0.928<br>66.6 kMAC, 291 kFLOP |
| S1 | W | 93.5 / 93.5 / 0.929<br>3.8 kMAC, 136 kFLOP | 93.9 / 94.0 / 0.934<br>436.2 kMAC, 1001 kFLOP | 90.8 / 90.9 / 0.900<br>0.0 kMAC, 139 kFLOP | 92.1 / 92.1 / 0.914<br>65.0 kMAC, 259 kFLOP | 95.6 / 95.6 / 0.952<br>86.1 kMAC, 301 kFLOP | 94.7 / 94.7 / 0.943<br>86.1 kMAC, 301 kFLOP |
| S2 | A | 87.2 / 87.1 / 0.860<br>2.5 kMAC, 163 kFLOP | 87.4 / 87.3 / 0.863<br>401.9 kMAC, 962 kFLOP | 87.0 / 86.8 / 0.858<br>0.0 kMAC, 169 kFLOP | 85.4 / 85.2 / 0.841<br>44.6 kMAC, 247 kFLOP | 91.3 / 91.3 / 0.905<br>66.5 kMAC, 291 kFLOP | 91.6 / 91.6 / 0.908<br>66.5 kMAC, 291 kFLOP |
| S2 | W | 91.9 / 91.8 / 0.912<br>3.5 kMAC, 136 kFLOP | 90.7 / 90.5 / 0.898<br>482.8 kMAC, 1094 kFLOP | 86.9 / 86.7 / 0.857<br>0.0 kMAC, 139 kFLOP | 86.8 / 86.4 / 0.856<br>64.8 kMAC, 258 kFLOP | 92.3 / 92.2 / 0.916<br>86.0 kMAC, 301 kFLOP | 92.4 / 92.2 / 0.917<br>86.0 kMAC, 301 kFLOP |

#### N2. Mean accuracy over the winner cells of each patient and feature set (%), and the number of winner cells

| Patient | Set | cells | LDA | SVM | RandomForest | RBFNN_tuned | LSTM_default | LSTM_tuned |
|---|---|---|---|---|---|---|---|---|
| P1 | A | 4 | 92.8 | 93.2 | 95.2 | 93.0 | 95.1 | 95.9 |
| P1 | W | 3 | 95.3 | 95.4 | 94.2 | 94.5 | 98.0 | 97.6 |
| P2 | A | 4 | 91.6 | 91.5 | 92.3 | 90.2 | 93.2 | 93.7 |
| P2 | W | 4 | 93.9 | 93.4 | 92.6 | 91.9 | 94.2 | 94.2 |
| S1 | A | 3 | 86.7 | 87.1 | 87.8 | 86.6 | 91.5 | 92.6 |
| S1 | W | 4 | 89.0 | 89.4 | 87.6 | 88.0 | 92.6 | 92.8 |
| S2 | A | 3 | 71.0 | 70.8 | 74.4 | 71.3 | 75.9 | 75.5 |
| S2 | W | 3 | 76.1 | 76.1 | 75.3 | 75.1 | 83.6 | 83.3 |

#### N3. The winner configurations per patient and feature set

| Patient | Set | Winner configurations (rate, channels, bits) |
|---|---|---|
| P1 | A | 250 Hz x 5 ch x 8 b (10 kb/s); 250 Hz x 8 ch x 8 b (16 kb/s); 500 Hz x 8 ch x 8 b (32 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| P1 | W | 250 Hz x 5 ch x 8 b (10 kb/s); 250 Hz x 8 ch x 8 b (16 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| P2 | A | 250 Hz x 5 ch x 8 b (10 kb/s); 500 Hz x 5 ch x 8 b (20 kb/s); 1000 Hz x 8 ch x 8 b (64 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| P2 | W | 250 Hz x 5 ch x 8 b (10 kb/s); 500 Hz x 5 ch x 8 b (20 kb/s); 500 Hz x 8 ch x 8 b (32 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| S1 | A | 250 Hz x 5 ch x 8 b (10 kb/s); 500 Hz x 8 ch x 8 b (32 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| S1 | W | 250 Hz x 5 ch x 8 b (10 kb/s); 250 Hz x 8 ch x 8 b (16 kb/s); 500 Hz x 8 ch x 8 b (32 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| S2 | A | 250 Hz x 5 ch x 8 b (10 kb/s); 250 Hz x 8 ch x 8 b (16 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |
| S2 | W | 250 Hz x 5 ch x 8 b (10 kb/s); 250 Hz x 8 ch x 8 b (16 kb/s); 1000 Hz x 8 ch x 16 b (128 kb/s) |

#### N4. LSTM tuning: configuration and epoch count chosen per fold (counts over all winner cells x folds)

Configurations: base: 12, lr3: 96, reg: 32. Epochs chosen: median 202, range 15 to 400 (the headline LSTM uses 40).


## What this does and does not show

- It shows how accuracy changes with channels, sample rate and bit depth for these 4 patients with all features of a set, and what the cheapest configurations are for a stated target. All numbers are honest cross-validation: nothing is selected inside the folds except the channel ranking (see below).
- **Channel ranking:** computed on all the data (not inside the cross-validation), as in the earlier grid, so every channel count below the maximum is slightly optimistic.
- **Not comparable with the earlier grid:** no feature selection here (all columns of A or W), one shared segmentation for all channel subsets, corrected Wallace quantiser, geometric elbow.
- **Rate and band change together at 250 and 500 Hz** (pass-bands 100–124 Hz and 100–249 Hz), and 3 and 10 kHz add no band: the grid cannot separate sample rate from the pass-band.
- The FLOPs and MACs are analytic counts under the conventions of the protocol (one shared FFT per window, no sliding sums, divisions and transcendental functions counted separately), probably within a factor of about 2 of an efficient implementation. They do not measure power, latency or memory on a chip.
- W is chosen on the same patients; there is no noise, attenuation or packet loss; 4 patients, 5 folds, one seed.
- The networks were run only at the winner cells, not on the whole grid, and the LSTM has segment context that the other models lack (see the caveat in step 3).
- S1 and S2 never reach 95%, and S2's accuracy depends strongly on having 8 to 12 channels, so conclusions about "the cheapest stream" are patient-specific.

## Reproduce

```
python -W ignore -u grid_tables.py --workers 2 > cache/grid/tables.log           # 60 tables, about 10-15 minutes
python -W ignore test_grid_slices.py                                            # V5
set STAGE1_TAG=fixedrange
python -W ignore -u run_grid.py --workers 8 > cache/grid/run_grid.log           # 1,158 cells, about 20 minutes
python -W ignore check_grid_vs_stage1.py                                        # V7
python -W ignore check_grid_integrity.py                                        # V8
python -W ignore analyse_grid.py > cache/grid/grid_summary.txt                  # also grid_cells.csv, winners.json
python -W ignore make_grid_tables.py > cache/grid/grid_tables.md
python -W ignore check_elbow_truncation.py > cache/elbow_truncation_check.txt
python -W ignore -u run_grid_networks.py --workers 3 > cache/grid/run_networks.log  # step 3, about 1 hour
python -W ignore analyse_networks.py > cache/grid/networks_summary.txt          # also confusion_matrices.md
python -W ignore make_network_tables.py > cache/grid/network_tables.md
```
