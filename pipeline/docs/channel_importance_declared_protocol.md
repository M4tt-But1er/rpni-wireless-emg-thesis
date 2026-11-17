# Channel importance: declared protocol (12 Nov 2026, written before any result)

**Question.** Which implanted channels matter most for classification: which should be kept if channels must be dropped, and does the
existing channel order used by the bit-rate grid (best mutual-information rank of any feature of the channel) match their actual influence?
Secondary: do the RPNI channels carry more, less or the same information as the intramuscular channels?

**Data and models (fixed, nothing tuned).** The E1 stream: 30 kHz, all implanted channels, 16 bit, wide set W
(`cache/baseline30k/tables/<part>__30000Hz_16bit.parquet`, Wallace with the fixed per-channel quantiser range). A channel subset keeps only
the W columns whose channels are all in the subset (`grid_slices.slice_matrix`: cross-channel features are kept only if both channels
are kept, RMS shares are recomputed over the subset). LDA, SVM and RF with the grid's settings (`run_grid.fit_predict_info`), grouped
5-fold CV of the grid (`models.compute_cv_folds`), per-window accuracy pooled over folds, no vote. Check: the all-channel subset must
reproduce the E1 values in `cache/baseline30k/results.json`.

Channel type: RPNI = P1 RPNI_ulnar, RPNI_median; P2 RPNI_median, RPNI_ulnar1, RPNI_ulnar2; S1 Median1, Ulnar1, Ulnar2;
S2 Radial, Median1-4. All other channels are intramuscular electrodes in residual muscles.

**Measures** (each per participant, per model and as the mean over the three models):
1. **Leave one channel out (LOCO):** accuracy with all channels minus accuracy without channel c. The unique contribution of c
   (information no other channel provides). Uncertainty: the per-fold differences of the model mean, 95 % t-interval with 4 df, as
   elsewhere in the project. A drop is called resolved only if its interval excludes zero.
2. **Single channel:** accuracy with channel c alone. The information c carries on its own (redundancy not removed).
3. **Backward elimination:** from all channels, repeatedly remove the channel whose removal leaves the highest mean accuracy over the
   three models (ties: remove the channel later in the grid's channel order), down to one channel. The reverse of the removal order is
   the keep order; the curve is accuracy against the number of channels kept. The first step is measure 1.
   Caveat, stated in advance: the subset at each step is chosen on the same folds that score it, so the curve is optimistic and serves
   to order channels, not to estimate the accuracy of a deployed subset.
4. **Grid order:** accuracy of the first n channels of the existing grid order, n = 1 .. all, for comparison with measure 3.
5. **RPNI group test:** (a) all RPNI channels removed; (b) RPNI channels only; (c) control for (a): every way of removing the same number
   of intramuscular channels (P1 15, P2 10, S1 10, S2 21 subsets), reported as mean and range. RPNI channels are called more (less)
   informative than intramuscular channels only if the loss in (a) lies above (below) the whole range of (c).

**Reporting.** A single most-important channel is named only where the measures agree; where LOCO intervals include zero, the result is
reported as redundancy (no single channel is indispensable), not as a ranking. Channel identities differ between participants, so
conclusions about muscles are drawn only where they recur.

Script: `run_channel_importance.py`; results `cache/channel_importance/`; summary `cache/channel_importance/summary.md`.
