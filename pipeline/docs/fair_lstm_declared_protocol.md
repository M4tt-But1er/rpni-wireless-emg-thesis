# Fair LSTM comparison: protocol, declared before any result (9 Nov 2026)

Requested by the user on 9 Nov 2026 ("run the fair LSTM comparison"), as the follow-up to the caveat in `docs/grid_results.md` (step 3): the LSTM reads each segment from its onset with memory, while the per-window classifiers (LDA, SVM, RF) get no history. Nothing below has been run when this file is written. Changes only as labelled amendments at the end.

## Question

At the 28 winner cells of the grid, does the LSTM still beat LDA, SVM and RF when the classical models are given the same access to history (the windows since the segment onset), used in the simplest causal way, a majority vote?

## What the LSTM sees (read from the code, `lstm_model.py`, `run_lstm_training_check.py`)

One layer, 64 units, **unidirectional**, one output per window, so the decision at window t depends only on windows 1..t of the same segment (causal). The state is reset at every segment: the sequence starts at the detected onset. Training is per-timestep cross-entropy with inverse-frequency class weights on whole training segments. Inputs are the same feature columns as the classical models, standardised with training-fold statistics. A check that the output at t does not change when later windows are removed is part of the validation (V10d).
Both model families therefore use the segment onset as given by the earlier onset detector. This is the same for both and is **not** tested online here.

## What changes for the classical models

Same cells, same features (set A or W, same channel slice), same outer folds (`models.compute_cv_folds`, seed 0), same estimators as the grid (`run_grid.fit_predict_info`). Only the decision rule changes: the per-window predictions of one model, in time order within a segment, are combined by a **causal hard majority vote**, restarted at each segment onset.

| Variant | Rule |
|---|---|
| k1 | no vote (the grid's cell; must reproduce it exactly, V10a) |
| k3, k5, k9, k15 | majority of the last k decisions (fewer at the start of a segment) |
| all | majority of all decisions since the segment onset (no free parameter; the same information as the LSTM) |
| nested | k chosen from {1, 3, 5, 9, 15, all} separately in each outer fold, on the outer-training windows only, by 3-fold grouped, class-stratified inner CV (`StratifiedGroupKFold`, shuffle, seed 0); criterion = accuracy of the voted inner out-of-fold predictions; ties go to the smaller k |

Tie-break inside a vote: among the tied classes, the class of the most recent decision in the voting span. No weights, no probabilities (soft voting is out of scope, see below).
A decision at window t uses windows up to t only, so the vote adds no look-ahead. It does need the last k labels in memory.

## Cells and models

The 28 cells of `cache/grid/winners.json` (A 14, W 14; P1 7, P2 8, S1 7, S2 6). Classical: LDA, SVM, RF (3 models × 7 variants × 28 cells). LSTM: **rerun** for `LSTM_default` (headline setting, 40 epochs) and `LSTM_tuned` (nested config and epoch choice, exactly `run_grid_networks.lstm_tuned`), saving the per-window predictions, which the first run did not keep. The rerun is also a reproducibility check of the stored results (V10b). If the rerun differs from the stored numbers, both are reported and the fair analysis uses the rerun (it is the one with per-window predictions); the difference is stated.
Metrics per cell and variant: accuracy, weighted F1, Cohen's kappa, pooled out-of-fold confusion matrix, fold accuracies. All models score the same windows (every window of every test segment).

## Declared comparisons (all paired by fold, 4 df, 95% t-interval; the mean is over cells)

- **H1 (main):** LSTM_tuned minus the *best of the three classical models with nested vote*, per cell. "Best of three" is chosen per cell after the fact, which favours the classical side; this is the conservative direction for any claim that the LSTM wins.
- **H2 (information-matched, no free parameter):** LSTM_tuned minus each of LDA, SVM, RF with vote `all`.
- **H3:** the same two comparisons for LSTM_default.
- Reported separately for set A, set W and all 28 cells; per patient in a table. Also the vote's own effect: vote variant minus k1 for each classical model.
- **Reading rule, fixed now:** a mean difference over cells below 1.0 point in absolute value is called "about equal"; at 1.0 point or more it is called "better" or "worse" only if the per-cell interval excludes 0 in at least a third of the cells, otherwise "unresolved". The per-cell count of resolved cells (better, worse) is always shown.
- **Time-resolved accuracy**, for interpretation only: pooled accuracy per bin of window index since onset (1–2, 3–5, 6–10, 11–20, 21 and later; window i is decided 200 + 50 (i − 1) ms after the onset) for k1, `all`, LSTM_default, LSTM_tuned, mean over cells.
- **Cost** (analytic, as in `feature_cost.py`): per decision, base model cost plus the vote. Vote cost counted as: ring-buffer update and count update (2 integer ops) plus an argmax over C counters (C − 1 comparisons); `all` needs 1 count update and the argmax. Memory: k stored labels plus C counters for the vote; 2 × 64 state values for the LSTM. These are counts under stated conventions, not measurements.

## Checks (must pass before any number is reported)

- **V10a:** variant k1 reproduces `cache/grid/grid_results.json` for every cell and model (accuracy, confusion matrix, fold accuracies).
- **V10b:** the LSTM rerun against `cache/grid/networks_results.json` (accuracy per cell; reported, not required to be identical, because multi-threaded float sums can differ).
- **V10c:** unit tests of the vote on toy sequences (k = 1 is the identity; hand-computed cases; the tie rule; `all` equals k = T; segments do not leak into each other; changing a later decision never changes an earlier voted one).
- **V10d:** LSTM causality: logits at window t are unchanged when the sequence is cut after t.
- **V10e:** the row order used to map predictions to windows is the table order (labels, segment ids and time order checked).

## Out of scope (possible follow-ups, not run)

Soft voting (mean probabilities) or an exponential average; classical models given stacked or differenced features of the last m windows; a bidirectional LSTM; other fold seeds; other configurations than the 28 cells; online use, where the onset is not known in advance; hardware measurements.

## Known limits, stated in advance

One seed, 4 patients, 5 folds (intervals probably too narrow). The LSTM is tuned on a validation split of segments; the vote's k is tuned by inner CV; neither family is tuned beyond that. The classical side is best of three per cell (post hoc); the LSTM is a single method per cell. A vote is a weak way to use history; if the LSTM still wins after it, that is consistent with the LSTM using temporal dynamics (early versus sustained windows) that a vote does not, but this protocol does not test that explanation.

## Amendments

**Amendment 1 (9 Nov 2026, made after a first look at the analysis output; the cells, models, variants, comparisons H1 to H3 and the reading rule are unchanged).** The first run of `analyse_fair.py` showed that the per-cell intervals (5 folds, 4 df) resolve very few cells even when the mean difference is several points (for example mean −3.3 points but resolved worse in only 3 of 28 cells), so the declared labels read "unresolved" for a difference that has the same sign in most cells.
Added to the printout, descriptive only and with no change to any label: (i) the number of cells in which the LSTM is ahead or behind by the sign of the mean fold difference; (ii) section 5b, the paired difference per patient, because the cells of one patient share data and are not independent. Also a column-width fix in the per-patient table. No result was changed or removed.
