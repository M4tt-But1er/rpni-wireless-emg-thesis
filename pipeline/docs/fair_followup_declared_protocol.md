# Fair comparison, follow-up: richer history for LDA/SVM/RF and a properly tuned LSTM — declared protocol (written 9 Nov 2026, BEFORE any result of this follow-up)

Request (user, 9 Nov 2026): "try the soft vote and stacked windows for the classical models, and try tune the LSTM to its best feasible performance."
This extends `docs/fair_lstm_declared_protocol.md` and `docs/fair_lstm_results.md`. Nothing from the earlier run is overwritten; all new outputs go to `cache/grid/fair2/` and to new files. Changes to this protocol after the first result are labelled amendments at the end.

## What stays exactly as before

- The 28 winner cells of the grid (`cache/grid/winners.json`: A 14, W 14; P1 7, P2 8, S1 7, S2 6), the same features and channel slices (`run_grid_networks.prepare_cell`), the same 5 outer folds (`models.compute_cv_folds`, grouped by segment, class-stratified, seed 0).
- Metric: window accuracy pooled over the 5 outer test folds (also weighted F1 and kappa in the tables). Differences are paired by fold, 95% t-interval with 4 df. Differences below about 2 points are treated as noise.
- Reading rule for every declared comparison: |mean difference| < 1 point = "about equal"; otherwise "better / worse" only if the per-cell interval excludes 0 in at least one third of the cells; otherwise "unresolved". Sign counts (cells ahead/behind) are always shown next to it.
- Causality: every decision at window t uses windows 1..t of its own segment only (segment onset known, as in the earlier protocol).

## Part A. More history for LDA, SVM and RF

Estimators are exactly the grid's (LDA lsqr + auto shrinkage; SVM RBF C = 10, gamma "scale", balanced; RF 300 trees, balanced; z-scoring inside the fold for LDA/SVM). They are NOT re-tuned (see "Asymmetry" below). For the SVM, `probability=True` is added so that class probabilities exist (Platt scaling with internal 5-fold CV); `predict()` is not affected, so the hard decisions must equal the grid's (check V11a).

**A1. History in the features ("stacked windows"), four variants of the per-window feature matrix, all causal and all inside one segment.** Lags are counted in windows (hop 50 ms, so 4 windows = 200 ms = a non-overlapping window).
- `none`: the cell's features (the grid).
- `ST3`: [x(t), x(t-4), x(t-8)] (current window and the two non-overlapping windows before it, 600 ms of history).
- `ST5`: [x(t), x(t-4), x(t-8), x(t-12), x(t-16)] (1000 ms).
- `CUM`: [x(t), running mean of x(1..t)] (everything since the onset).
A lag that reaches before the first window of the segment is clamped to the first window (the same rule as the dMAV/dRMS/dWL features of set W). Dimension is 1x, 3x, 5x, 2x of the cell's feature count.

**A2. Ways to combine the decisions of one model over a causal span (22 candidates).** Spans k in {1, 3, 5, 9, 15, all since the onset}; the vote restarts at every onset.
- `hard_k`: majority of hard decisions (the earlier vote; ties to the most recent decision in the span).
- `mean_k`: argmax of the summed class probabilities over the span (soft vote).
- `log_k`: argmax of the summed log probabilities, each clipped below at 1e-3 (evidence accumulation; clipping stops one over-confident wrong window from dominating; 1e-3 is about the resolution of a 300-tree forest).
- `ema_a`: exponential average s(t) = a s(t-1) + (1-a) p(t), s(1) = p(1), argmax; a in {0.5, 0.8, 0.9, 0.95}.
Ties between classes in the soft candidates go to the class with the larger probability in the current window, then to the lower class index. `hard_k1` is the raw per-window decision.

**A3. Choosing the candidate (nested, no test data).** Per outer fold, the outer-training windows are split by a 3-fold grouped, class-stratified inner CV (`StratifiedGroupKFold(3, shuffle=True, random_state=0)`, as in the earlier nested k); the inner out-of-fold predictions give the accuracy of every candidate (and variant); the best is applied to the outer test fold. Ties go to the earlier candidate in the order hard, mean, log, ema and, within a family, the shorter span (smaller a).

**Pipelines reported, per model (LDA, SVM, RF):**
- P0 raw (variant `none`, `hard_k1`) = the grid.
- P1 hard-nested (`none`, nested over the 6 `hard_k`) = the earlier headline.
- P2 soft-nested (`none`, nested over the 16 soft candidates).
- P3 vote-nested (`none`, nested over all 22).
- P4 stack-only (each of ST3, ST5, CUM with `hard_k1`).
- P5 stack + vote (each of ST3, ST5, CUM with nested over 22).
- P6 full-nested (nested jointly over variant x 22 candidates = 88 candidates; ties to the earlier variant in the order none, ST3, ST5, CUM). This is the most flexible classical pipeline, fully nested.
A "best of three models per cell" is also shown for continuity with the earlier H1, but it is chosen after the fact and is labelled so.

**Declared comparisons (paired by fold, per model, over the 28 cells):**
- H4: P3 minus P1 (does a soft or exponential combination add to the hard vote?).
- H5: each stack-only variant minus P0 (does history in the features help without a vote?) and each stack + vote minus P3 (does it add anything on top of the best vote?).
- H6: P6 minus P1.

**Checks:** V11a variant `none`, `hard_k1`, equals the earlier grid predictions (`cache/grid/fair/classical/*.npz`, `pooled`) for all 84 (cell, model) pairs, and the nested hard-k choice equals the earlier `chosen_k`; V11b unit tests of the candidates (identity at k = 1 and a = 0, hand-computed cases, tie rule, `all` equals k = segment length, no leakage between segments, later decisions never change earlier ones); V11c the stacked features are causal (changing windows after t does not change row t; clamping; CUM equals the running mean); V11d row alignment of every stored array (labels, folds, order).

## Part B. The LSTM tuned as far as is feasible

**Why the earlier tuning was weak (facts from `cache/grid/networks_results.json`):** the earlier tuned LSTM chose 3 configurations by one validation split of 11 to 19 sequences; it picked the highest learning rate in its grid (3e-3) in 96 of 140 folds and hit the 400-epoch limit in about a quarter of the folds (median 202, 95th percentile 400). So that search was limited by the edge of its grid, which is what this part addresses.

**B1. Search space** (one configuration = one draw from the product): hidden size {32, 64, 128}; layers {1, 2}; input bottleneck {none, per-step Linear(F to 32) + ReLU}; dropout {0, 0.3, 0.5} (on the LSTM output, between the two layers if layers = 2, after the bottleneck if present); weight decay {0, 1e-3, 1e-2}; learning rate {1e-3, 3e-3, 1e-2}; Gaussian input noise on the z-scored inputs, training only, sigma {0, 0.3}; batch size {8, 16}. 1296 combinations. Fixed: unidirectional (causal), Adam, per-step cross-entropy with inverse-frequency class weights, state reset per segment, gradient-norm clip 1.0 for the new configurations (the three old ones have no clip, so that they reproduce the earlier training exactly), at most 400 epochs, validation every 5 epochs, stop after 10 validations (50 epochs) without a better validation loss.

**B2. Pilot (development; no outer test data touched).** 32 distinct random configurations (numpy seed 20261109; a draw equal to an old configuration is redrawn) plus the three old ones (`base`, `lr3`, `reg`) = 35. Pilot cells = the reference configuration (1000 Hz, 8 channels, 16 bit) of each of the 8 patient/set groups (8 cells, the widest inputs). For each (pilot cell, outer fold) the old inner split is used (a stratified hold-out of max(n_classes, ceil(0.2 n)) sequences of the outer-training set, `StratifiedShuffleSplit(random_state=0)`), every configuration is trained with early stopping, and the validation accuracy at the epoch of lowest validation loss, that loss, that epoch and the run time are stored. Only these inner-validation numbers are used. No outer-test prediction is made in the pilot.

**B3. Selection rule for the final candidate set (declared now):** rank the 32 random configurations by the mean over the 40 (cell, fold) pilot pairs of the validation accuracy at the loss-optimal epoch (ties by lower mean validation loss); the final set = the top 6 of them plus the three old ones = 9 configurations. No extension of the search space after the pilot (if the best configurations sit at an edge of the space, or hit 400 epochs often, that is reported as a limit).
Contamination, stated in advance: the pilot's validation sequences come from outer-training sets, which contain data that are test data in other folds, so the 8 pilot cells carry a small selection optimism. All LSTM results are therefore also shown for the 20 non-pilot cells.

**B4. Final nested run (all 28 cells x 5 outer folds).** Per outer fold, for each of the 9 configurations: inner validation with early stopping on the outer-training sequences only. Inner scheme: 3-fold stratified-by-movement K-fold over sequences (`StratifiedKFold(3, shuffle=True, random_state=0)`); fallback to the single old split if the time projected from the pilot (mean seconds per pilot run x 9 configurations x 3 inner folds x 140 cell-folds x 1.2 for retraining / number of workers) exceeds 3.5 hours; this rule uses pilot timings only. Configuration score = mean over inner folds of the validation accuracy at the loss-optimal epoch (ties by lower mean validation loss); epochs = round(mean of the inner best epochs), at least 5. The chosen configuration is retrained on all outer-training sequences for those epochs with seeds 0, 1, 2, 3, 4. Predictions kept per window: `LSTM_best` = seed 0 alone (primary, same compute as the earlier LSTM) and `LSTM_best_ens5` = mean of the 5 softmax outputs (secondary, 5x the compute; a different cost class). Softmax probabilities of the ensemble are stored.

**B5. Secondary (parameter-free, declared):** the same LSTM decisions smoothed by a causal vote over everything since the onset (`hard_all` on its decisions, `mean_all` on its probabilities). The classical side is allowed such post-processing, so the LSTM is too.

**Declared comparisons (28 cells, paired by fold):**
- H7: `LSTM_best` minus the earlier `LSTM_tuned` (what did the better tuning gain?).
- H8: `LSTM_best` minus each classical model's P3 and P6 (per model), and minus the best of three with P1 (continuity with the earlier H1 = -3.27) and with P6 (post hoc best of three).
- H9: the same with `LSTM_best_ens5`.
Each is read with the rule above, also on the 20 non-pilot cells.

**Checks:** V12a the new training code with the `base` configuration and 40 epochs reproduces the earlier `LSTM_default` fold accuracies exactly on three cells; V12b the stored outer-fold predictions map back onto the table rows (labels equal); V12c causality of the new network (output at window t unchanged when later windows are removed); V12d a rerun of a few tasks reproduces them bit for bit.

## Cost reporting
Analytic counts as before (`feature_cost.py`): the stacked classical models need the feature vectors of the earlier windows kept in memory (no recomputation), their model cost grows with the dimension (LDA: F x classes MACs; SVM: support vectors x F; RF unchanged); the vote candidates cost a few additions per class. LSTM MACs per window step for the chosen configurations: bottleneck F x b + layer 1: 4h(in + h) + layer 2 (if any) 4h(2h) + classifier h x classes; x5 for the ensemble.

## Order and time (estimates, not measurements)
One heavy job at a time, 6 to 7 single-thread workers (timing probe: one training epoch costs 0.05 to 0.09 s on one thread for the widest cells, and extra threads barely help). 1) Part A classical run, about 0.5 to 1.5 h. 2) LSTM pilot, about 1 h. 3) LSTM final, about 1 to 3 h depending on the inner scheme. 4) analysis and documents. All jobs are resumable per task and log to `cache/grid/fair2/`.

## Asymmetry stated in advance
The LSTM gets a large nested search (about 9 configurations x 3 inner folds per outer fold, after a 35-configuration pilot); LDA, SVM and RF keep the grid's fixed settings and only get history. If the classical models still win, this asymmetry does not weaken that finding; if the LSTM wins, it would, and a matching search for the classical models (SVM C and gamma, RF depth and max features, LDA shrinkage) would be needed before claiming it. This is not done here.

## Limits stated in advance
Steady-state segments with a known onset (the vote and the stacked CUM variant need it); 4 patients, 5 folds, one fold seed; intervals probably too narrow; the 28 cells are not independent; Wallace S1/S2 remain upper bounds because of the run signature; the LSTM search is a random search of 35 configurations, not exhaustive, and one architecture family (unidirectional LSTM on window features); more training data could help the LSTM more than the classical models.

## Amendments
- A2 (9 Nov 2026; written after the round-1 pilot and its selection, and after the round-1 final run was launched, but BEFORE any outer-test LSTM result of this follow-up had been looked at; it replaces the sentence "No extension of the search space after the pilot" in B3). **Reason:** the edge check of the round-1 pilot (`cache/grid/fair2/lstm_pilot_summary.md`) shows that the top 6 random configurations sit at an edge of the declared space on several axes: learning rate 0.01 (top edge) in 4 of 6, input noise 0.3 (top edge) in 5 of 6, batch 8 (lower edge) in 5 of 6, bottleneck 32 (top edge) in 5 of 6, hidden size 32 (lower edge) in 4 of 6; layers = 1 and weight decay = 0 (natural minima) in 5 and 4 of 6; dropout is interior (0.3). That is the same weakness as the earlier tuning, and the aim is the best feasible LSTM.
  **Rule (mechanical, fixed now):** an axis with at least 4 of the top 6 at one edge and a natural next step is extended by one step beyond that edge: hidden + 16, learning rate + 0.03, noise + 0.6, batch + 4, bottleneck + 64. Layers and weight decay are at their natural minimum and dropout is interior, so they stay. Round-2 space: hidden {16, 32, 64, 128}, layers {1, 2}, bottleneck {0, 32, 64}, dropout {0, 0.3, 0.5}, weight decay {0, 1e-3, 1e-2}, learning rate {1e-3, 3e-3, 1e-2, 3e-2}, noise {0, 0.3, 0.6}, batch {4, 8, 16} (7776 combinations).
  **Round 2:** 32 new distinct random configurations (numpy seed 20261110, never equal to a round-1 configuration; named s01 to s32; clip 1.0), pilot on the same 8 reference cells with the same inner splits and code; the ranking (B3 rule: mean validation accuracy at the loss-optimal epoch, ties by loss) runs over the 32 new ones plus the 6 round-1 carry-overs (their pilot numbers are reused, same cells, folds and splits); final set of round 2 = `base`, `lr3`, `reg` + the top 6 of those 38; the final nested run (B4) is repeated with it (same inner-scheme rule) into `cache/grid/fair2/lstm_r2/`. If no new configuration reaches the top 6 the final set equals round 1's and the round-1 files are reused. **No round 3** (if the round-2 top 6 again sit at the new edges, that is reported as a limit).
  **Names and primary comparator (fixed now):** round 1 (already running when this was written) = `LSTM_r1`, `LSTM_r1_ens5`; round 2 = `LSTM_best`, `LSTM_best_ens5`. The primary LSTM of H7 to H9 and of B5 is round 2 (`LSTM_best`, the more thorough search); round 1 is reported next to it (same comparisons, descriptive). Neither round is selected by outer-test results, and per-cell maxima over the two rounds are not used.
- A1 (9 Nov 2026, before the pilot or any LSTM result existed; clarification only): in B4 the "mean seconds per pilot run" in the time projection is the mean over the 9 selected configurations' pilot runs (not over all 35), with 6 workers assumed; `select_lstm_candidates.py` implements exactly this rule and prints the projection. Also, the Part A smoke test (one cell, 3 models) was run before the full classical run to time it; its raw accuracies were seen, nothing in this protocol was changed because of them.
