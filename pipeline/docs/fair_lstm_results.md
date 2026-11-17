# Fair LSTM comparison: results (9 Nov 2026)

**Follow-up (9 Nov 2026, later): a soft vote, stacked windows and a much larger LSTM search were tried and did not change the ranking, see [fair_followup_results.md](fair_followup_results.md). The numbers below are unchanged.**

Protocol, declared before any result: [fair_lstm_declared_protocol.md](fair_lstm_declared_protocol.md) (one amendment, made after the first look at the output and described there). Follows from the caveat in [grid_results.md](grid_results.md), step 3.
Everything below is computed from stored per-window predictions by `analyse_fair.py` and `make_fair_tables.py`; the tables at the end are generated, not typed. Run time: about 4 minutes for the classical side, about 22 minutes for the LSTM rerun.

## The answer in plain words

Earlier I reported that the LSTM beat LDA, SVM and RF by about 2 to 3 points at the 28 winner cells, with the warning that the LSTM sees the windows since the segment onset and the others see one window. I have now given LDA, SVM and RF the same history, in the simplest causal way: a majority vote over their own recent decisions, restarted at each onset.

**With the same history the classical models are ahead of the LSTM, not behind it.**
- The vote over all decisions since the onset lifts LDA by +4.9, SVM by +5.6 and RF by +5.8 points (mean over the 28 cells; never worse in any cell, resolved better in 18 to 22 cells).
- The best classical model of each cell with the nested vote is **3.3 points above the tuned LSTM** (−3.27 for the LSTM; the LSTM is behind in 26 of 28 cells, and in every cell of S1 and S2). Against the default LSTM it is 3.5 points. Without the vote the same comparison gives +2.08 for the LSTM, exactly the number I reported before.
- This does not depend on picking the best of three models per cell: the weakest of them, LDA with the vote, is still 1.4 points above the tuned LSTM (the LSTM is ahead in 9 cells and behind in 19).
- By patient, the tuned LSTM minus the best classical model with the nested vote: P1 −1.6, P2 −1.9, S1 −2.5, S2 −7.9 points. Without the vote it was +1.3, +0.4, +3.5, +3.6.
- The LSTM reaches about the accuracy of a classical vote over the last 9 to 15 decisions (a span of 600 to 900 ms), not of a vote over the whole history.

So the earlier ranking came from the unequal information, not from the model family. I should not say "the LSTM beats the classical models" in the thesis.

**How firmly.** By the reading rule I declared in advance, the main comparison is labelled "unresolved", because the per-cell 95% intervals (5 folds, 4 df) resolve only 3 of 28 cells although the mean is −3.3 points; a strict reader should say "behind, not formally resolved cell by cell". What supports the direction is its consistency: 26 of 28 cells, all four patients, both feature sets, both LSTM variants, and the single-model comparisons. One seed, 4 patients, intervals probably too narrow.

## What was done

- **Same cells, same features, same folds.** The 28 winner cells of the grid (A 14, W 14). LDA, SVM and RF refitted with the grid's own function (`run_grid.fit_predict_info`), keeping the per-window out-of-fold predictions. Variant "no vote" reproduces the grid exactly in all 84 (cell, model) pairs (V10a).
- **The vote.** Hard majority of a model's last k decisions inside a segment, restarted at each onset, ties to the most recent decision in the span: k = 3, 5, 9, 15, "all since the onset" (no free parameter), and "nested k", chosen per outer fold from those by a 3-fold grouped inner CV on the outer-training windows only. A decision at window t uses windows up to t only (tested).
- **How much history each side has (added 9 Nov 2026 after a question from the user).** The classical models get history only through the vote over their own past hard decisions; the feature vectors, scores or probabilities of earlier windows are not used. The vote never reaches back before the segment onset. A vote over k windows covers 200 + 50 (k − 1) ms of signal: k = 5 → 400 ms, k = 9 → 600 ms, k = 15 → 900 ms, "all" → up to about 2 s (4 to 38 windows per segment, median 34, so about 1.85 s). One exception: set W contains the window-change family (dMAV, dRMS, dWL: the change of MAV, RMS and WL against the window 1 and 2 hops, i.e. 50 and 100 ms, earlier in the same segment; for the first windows of a segment the earlier index is clamped to the first window, so the change is 0). In the 14 W cells every per-window decision therefore already sees up to 100 ms of earlier signal through those three features; set A has none. Consecutive windows overlap by 150 ms (200 ms window, 50 ms hop), so neighbouring decisions are strongly correlated and the vote is not 30 independent opinions. This does not change the comparison (the LSTM sees the same segment from the onset), but it means the W cells are not strictly "one window, no history" even at k = 1.
- **The LSTM** (one layer, 64 units, unidirectional, one output per window, state reset at each segment) was rerun with exactly the earlier code and its per-window predictions kept. The rerun reproduces the stored first run in every one of the 56 cells (identical accuracy), so the first run is reproducible (V10b).
- **Same windows, same metrics** for every method: accuracy, weighted F1, Cohen's kappa, the pooled out-of-fold confusion matrix (`cache/grid/fair/fair_confusion_matrices.md`), fold accuracies. All differences are paired by fold.

## Checks (all passed; logs in `cache/grid/fair/`)

V10a classical k1 equals the grid in accuracy, confusion matrix and fold accuracies for 84 of 84. V10b LSTM rerun equals the stored run for 56 of 56 cells. V10c 11 unit tests of the vote (identity at k = 1, hand-computed cases, tie rule, "all" equals k = segment length, no leakage between segments in any order, later decisions never change earlier ones, nested choice behaves as intended). V10d the LSTM is causal (its output at window t does not change when later windows are removed). V10e predictions are aligned with table rows (fold ids, labels, order; 140 files).

## Results

1. **The vote helps every classical model, and more history helps more** (table F1, F2): no vote 87.7 / 87.8 / 88.1 (LDA / SVM / RF), k = 5 +1.4 to +2.0, k = 15 +3.4 to +4.3, all since the onset 92.60 / 93.43 / 93.85. The nested choice picked the whole history in 387 of 420 fold choices (92%), k = 15 in 28 and k = 9 in 5 (table F6), so "nested" and "all" are nearly the same result.
2. **Main comparison (H1, table F2):** tuned LSTM minus the best classical model with the nested vote −3.27 points (set A −4.09, with the LSTM behind in all 14 cells; set W −2.45, behind in 12 of 14). The default LSTM −3.47. The comparison without the vote (the first run's) gives +2.08 and +1.88, as before (21 and 22 cells ahead).
3. **One model at a time, vote over all windows (H2, H3):** tuned LSTM minus LDA −1.42, minus SVM −2.24, minus RF −2.67 points.
4. **Per patient (table F3):** the LSTM is behind the best voted classical model in every patient; the gap is largest for S2 (−7.9; LSTM 79.4% against 84.1 to 87.0% for the voted classical models, and S2 has the fewest training sequences, 47 to 48 per fold).
5. **Where in time (table F4).** 61% of all windows lie 1.2 s or more after the onset, so the pooled accuracy mostly measures how well a method settles on one class during a sustained hold. In the first 5 windows (decided within 400 ms of the onset, 9.8% of windows) the LSTM is 0.2 to 3 points below every classical model; in windows 6 to 10 it is level with LDA (0.2 to 0.4 points above it) and 0.1 to 2.8 points below SVM and RF. From window 11 the LSTM is ahead of the classical models without vote (92.8 against 87.7 at window 21 and later) but behind the vote over all windows (90.5 against 91.1 to 93.2 at windows 11 to 20, 92.8 against 94.8 to 95.7 from window 21).
   There is no time range in which the LSTM is clearly ahead of the voted classical models; its only lead is 0.4 points over LDA at windows 6 to 10, which is within noise.
6. **Cost (table F7, analytic counts, not measurements).** LDA with a vote costs 87 kFLOP per decision with set A and 66 with set W, against 196 and 213 for the LSTM, 93 and 71 for RF and 649 and 838 for the SVM. The vote adds about 13 integer operations (0.02% of the LDA total) and needs a few stored labels or a class counter; the LSTM has to keep 128 state values.
7. **Hyper-parameters.** The tuned LSTM is 0.20 points above the default LSTM on average (better in no cell, worse in none, by the interval), as in the first run.

## What this does and does not show

- **Shown:** on this benchmark (sustained postures, onset-trimmed segments, known onset, one class per segment) and with this LSTM, a causal vote over the decisions of a per-window LDA, SVM or RF is at least as accurate as the LSTM and about 2 to 3 times cheaper to compute than it for LDA and RF. The earlier LSTM advantage was an information advantage.
- **Not shown:**
  - **Transitions.** The vote over the whole history cannot change class inside a segment, which suits sustained holds and not changes of intent. A short span (k = 5) gives +1.4 to +2.0 points and reacts within about 0.4 s; a recurrent model could in principle follow transitions, but this benchmark has none, so it cannot test that.
  - **Online onset detection.** Both families use the segment onset as given; a deployed system would need to detect it, with errors and delay, and I did not test that.
  - **A better LSTM.** The LSTM here is one layer of 64 units, tuned over three settings and the epoch count on a validation split, with 47 to 78 training sequences per fold. More data, another size, regularisation or a different training signal could change its result; I did not search. The correct statement is "this LSTM, as tuned, does not beat a vote", not "no recurrent model could".
  - **Other ways to use history for the classical side** (mean probabilities, exponential averaging, features stacked from the last windows): not run. The vote is the simplest and was declared first.
- The "best of three per cell" is chosen after the fact, which favours the classical side; the single-model comparisons agree in sign.
- The per-window accuracy of all three classical models is lower in the latest windows (88.9% at windows 11 to 20, 87.7% from window 21); I did not investigate why.
- Wallace S1 and S2 results remain upper bounds because of the run signature (see the defence guide); this affects all models, but a long vote could amplify a run-specific signature, so S1 and S2 differences between methods should be read with that in mind.
- One seed, 4 patients, 5 folds; intervals probably too narrow; the 28 cells are not independent (cells of a patient share data), which is why the per-patient view is given.

## Reproduce

```
python -W ignore test_fair_vote.py                       # V10c, V10d
python -W ignore -u run_fair_classical.py --workers 6    # 84 tasks, about 4 min; cache/grid/fair/classical/
python -W ignore -u run_fair_lstm.py --workers 3         # 56 tasks, about 22 min; cache/grid/fair/lstm/
python -W ignore check_fair_v10.py                       # V10a, V10b, V10e
python -W ignore analyse_fair.py > cache/grid/fair/fair_summary.txt
python -W ignore make_fair_tables.py                     # cache/grid/fair/fair_tables.md (the tables below)
```

Files: `cache/grid/fair/fair_results.json` (every method, variant and cell: accuracy, F1, kappa, confusion matrix, fold accuracies, nested k), `fair_cells.csv` (one row per cell and method), `fair_confusion_matrices.md` (LSTM tuned and the three classical models with nested vote at every cell), `fair_summary.txt`, `fair_tables.md`, `check_v10cd.log`, `check_v10abe.log`.

---

## Tables (generated by `make_fair_tables.py`)

### F1. Mean accuracy (%) over the winner cells, by model and vote variant (the vote is causal and restarts at each segment onset; k = number of last decisions)

**all 28 cells**

| Model | no vote | k=3 | k=5 | k=9 | k=15 | all since onset | nested k |
|---|---|---|---|---|---|---|---|
| LDA | 87.72 | 88.45 | 89.09 | 90.06 | 91.09 | 92.60 | 92.49 |
| SVM | 87.80 | 88.94 | 89.80 | 91.09 | 92.10 | 93.43 | 93.35 |
| RandomForest | 88.08 | 88.88 | 89.48 | 90.59 | 91.58 | 93.85 | 93.86 |
| best of three, per cell | 89.09 | 89.92 | 90.63 | 91.84 | 92.82 | 94.51 | 94.46 |

LSTM (no vote, it has its own memory): LSTM_default 90.97, LSTM_tuned 91.19

**set A, 14 cells**

| Model | no vote | k=3 | k=5 | k=9 | k=15 | all since onset | nested k |
|---|---|---|---|---|---|---|---|
| LDA | 86.45 | 87.35 | 88.10 | 89.10 | 90.06 | 91.90 | 91.75 |
| SVM | 86.61 | 88.16 | 89.21 | 90.62 | 91.69 | 93.00 | 92.90 |
| RandomForest | 88.33 | 89.12 | 89.68 | 90.82 | 91.72 | 93.84 | 93.83 |
| best of three, per cell | 88.76 | 89.73 | 90.44 | 91.72 | 92.68 | 94.33 | 94.26 |

LSTM (no vote, it has its own memory): LSTM_default 89.66, LSTM_tuned 90.19

**set W, 14 cells**

| Model | no vote | k=3 | k=5 | k=9 | k=15 | all since onset | nested k |
|---|---|---|---|---|---|---|---|
| LDA | 88.99 | 89.54 | 90.07 | 91.03 | 92.12 | 93.30 | 93.24 |
| SVM | 89.00 | 89.71 | 90.40 | 91.56 | 92.51 | 93.85 | 93.80 |
| RandomForest | 87.83 | 88.63 | 89.28 | 90.37 | 91.43 | 93.87 | 93.88 |
| best of three, per cell | 89.42 | 90.11 | 90.81 | 91.96 | 92.96 | 94.69 | 94.66 |

LSTM (no vote, it has its own memory): LSTM_default 92.29, LSTM_tuned 92.20

### F2. Paired differences in points (mean over cells; in brackets: cells where the first model is ahead / behind by the sign of the mean fold difference; cells resolved better / worse by the 95% interval)

| Comparison | all 28 cells | set A (14) | set W (14) | Reading (all 28 cells, declared rule) |
|---|---|---|---|---|
| LSTM_tuned minus best classical, no vote (the first run's comparison) | +2.08 (21/7; 7/0) | +1.40 (10/4; 3/0) | +2.76 (11/3; 4/0) | unresolved (LSTM better ahead, resolved in 7 of 28 cells) |
| LSTM_tuned minus best classical, nested vote (H1) | -3.27 (2/26; 0/3) | -4.09 (0/14; 0/3) | -2.45 (2/12; 0/0) | unresolved (classical better ahead, resolved in 3 of 28 cells) |
| LSTM_tuned minus LDA, vote over all windows since the onset (H2/H3) | -1.42 (9/19; 0/0) | -1.73 (5/9; 0/0) | -1.10 (4/10; 0/0) | unresolved (classical better ahead, resolved in 0 of 28 cells) |
| LSTM_tuned minus SVM, vote over all windows since the onset (H2/H3) | -2.24 (4/24; 0/2) | -2.83 (1/13; 0/2) | -1.65 (3/11; 0/0) | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| LSTM_tuned minus RandomForest, vote over all windows since the onset (H2/H3) | -2.67 (5/23; 0/3) | -3.67 (2/12; 0/3) | -1.67 (3/11; 0/0) | unresolved (classical better ahead, resolved in 3 of 28 cells) |
| LSTM_tuned minus best of three, vote over all windows (H2/H3) | -3.32 (2/26; 0/3) | -4.16 (0/14; 0/3) | -2.49 (2/12; 0/0) | unresolved (classical better ahead, resolved in 3 of 28 cells) |
| LSTM_default minus best classical, no vote (the first run's comparison) | +1.88 (22/6; 7/0) | +0.90 (10/4; 2/0) | +2.86 (12/2; 5/0) | unresolved (LSTM better ahead, resolved in 7 of 28 cells) |
| LSTM_default minus best classical, nested vote (H1) | -3.47 (1/27; 0/6) | -4.59 (0/14; 0/4) | -2.36 (1/13; 0/2) | unresolved (classical better ahead, resolved in 6 of 28 cells) |
| LSTM_default minus LDA, vote over all windows since the onset (H2/H3) | -1.62 (6/22; 0/0) | -2.23 (3/11; 0/0) | -1.01 (3/11; 0/0) | unresolved (classical better ahead, resolved in 0 of 28 cells) |
| LSTM_default minus SVM, vote over all windows since the onset (H2/H3) | -2.44 (5/23; 1/7) | -3.33 (1/13; 0/4) | -1.55 (4/10; 1/3) | unresolved (classical better ahead, resolved in 7 of 28 cells) |
| LSTM_default minus RandomForest, vote over all windows since the onset (H2/H3) | -2.87 (7/21; 0/4) | -4.17 (3/11; 0/3) | -1.58 (4/10; 0/1) | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| LSTM_default minus best of three, vote over all windows (H2/H3) | -3.53 (1/27; 0/5) | -4.66 (0/14; 0/4) | -2.40 (1/13; 0/1) | unresolved (classical better ahead, resolved in 5 of 28 cells) |

Effect of the vote itself (variant minus no vote, same model):

| Model | variant | all 28 cells | set A (14) | set W (14) |
|---|---|---|---|---|
| LDA | k=5 | +1.37 (20/0) | +1.65 (11/0) | +1.08 (9/0) |
| LDA | k=15 | +3.36 (21/0) | +3.60 (12/0) | +3.12 (9/0) |
| LDA | all since onset | +4.87 (18/0) | +5.44 (11/0) | +4.29 (7/0) |
| LDA | nested k | +4.76 (19/0) | +5.29 (12/0) | +4.23 (7/0) |
| SVM | k=5 | +1.99 (25/0) | +2.59 (14/0) | +1.40 (11/0) |
| SVM | k=15 | +4.29 (24/0) | +5.07 (13/0) | +3.50 (11/0) |
| SVM | all since onset | +5.60 (22/0) | +6.38 (12/0) | +4.83 (10/0) |
| SVM | nested k | +5.52 (22/0) | +6.27 (12/0) | +4.77 (10/0) |
| RandomForest | k=5 | +1.40 (19/0) | +1.35 (10/0) | +1.45 (9/0) |
| RandomForest | k=15 | +3.50 (20/0) | +3.39 (10/0) | +3.61 (10/0) |
| RandomForest | all since onset | +5.77 (20/0) | +5.50 (10/0) | +6.04 (10/0) |
| RandomForest | nested k | +5.77 (20/0) | +5.50 (10/0) | +6.05 (10/0) |

### F3. Per patient (mean over that patient's cells, sets A and W together): accuracy % / weighted F1 % / kappa

| Patient (cells) | LDA, no vote | SVM, no vote | RF, no vote | LDA, nested vote | SVM, nested vote | RF, nested vote | LSTM default | LSTM tuned |
|---|---|---|---|---|---|---|---|---|
| P1 (7) | 93.8 / 93.8 / 0.932 | 94.1 / 94.0 / 0.935 | 94.7 / 94.7 / 0.942 | 96.9 / 96.7 / 0.965 | 97.0 / 96.8 / 0.966 | 97.6 / 97.5 / 0.973 | 96.3 / 96.2 / 0.959 | 96.6 / 96.5 / 0.963 |
| P2 (8) | 92.7 / 92.7 / 0.920 | 92.5 / 92.4 / 0.917 | 92.5 / 92.4 / 0.917 | 95.4 / 95.4 / 0.949 | 95.1 / 95.0 / 0.946 | 95.1 / 95.0 / 0.946 | 93.7 / 93.6 / 0.931 | 94.0 / 93.9 / 0.934 |
| S1 (7) | 88.0 / 88.0 / 0.870 | 88.4 / 88.4 / 0.875 | 87.7 / 87.7 / 0.867 | 92.0 / 92.0 / 0.913 | 94.7 / 94.6 / 0.942 | 94.6 / 94.6 / 0.942 | 92.1 / 92.1 / 0.915 | 92.7 / 92.7 / 0.921 |
| S2 (6) | 73.5 / 73.5 / 0.711 | 73.5 / 73.3 / 0.710 | 74.9 / 74.7 / 0.726 | 84.1 / 84.0 / 0.827 | 85.2 / 85.1 / 0.839 | 87.0 / 86.6 / 0.858 | 79.7 / 79.5 / 0.779 | 79.4 / 79.0 / 0.775 |

LSTM tuned minus the best classical model of each cell, by patient (points; cells where the LSTM is ahead / behind):

| Patient | no vote | nested vote | vote over all windows |
|---|---|---|---|
| P1 | +1.26 (6/1) | -1.62 (1/6) | -1.63 (1/6) |
| P2 | +0.43 (3/5) | -1.90 (1/7) | -2.01 (1/7) |
| S1 | +3.52 (7/0) | -2.52 (0/7) | -2.59 (0/7) |
| S2 | +3.56 (5/1) | -7.91 (0/6) | -7.91 (0/6) |

### F4. Time-resolved accuracy (%): pooled per cell by window index since the onset, then the mean over the 28 cells (window i is decided 200 + 50 (i-1) ms after the onset)

| Window index since onset | 1-2 | 3-5 | 6-10 | 11-20 | 21+ |
|---|---|---|---|---|---|
| Time of decision after onset (ms) | 200-250 | 300-400 | 450-650 | 700-1150 | 1200+ |
| Share of all windows | 3.9% | 5.9% | 9.8% | 19.4% | 61.0% |
| LDA, no vote | 82.2 | 84.4 | 87.5 | 88.9 | 87.7 |
| LDA, vote over all | 82.2 | 84.4 | 87.4 | 91.1 | 94.8 |
| SVM, no vote | 82.7 | 85.8 | 87.9 | 89.2 | 87.7 |
| SVM, vote over all | 82.7 | 86.4 | 89.8 | 92.7 | 95.2 |
| RF, no vote | 83.3 | 86.7 | 89.4 | 89.7 | 87.7 |
| RF, vote over all | 83.3 | 86.0 | 90.5 | 93.2 | 95.7 |
| LSTM default | 80.3 | 84.0 | 87.6 | 90.5 | 92.6 |
| LSTM tuned | 81.2 | 84.2 | 87.8 | 90.5 | 92.8 |

### F5. Every winner cell: accuracy % (weighted F1 %, kappa) without vote, with the nested vote, and the LSTM

| Patient | Set | Configuration | kb/s | LDA no vote | SVM no vote | RF no vote | Best classical, nested vote | LSTM default | LSTM tuned | LSTM tuned minus best nested vote |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | A | 250 Hz x 5 ch x 8 b | 10 | 82.8 (82.5, 0.808) | 83.6 (83.1, 0.818) | 90.0 (89.9, 0.888) | RF: 95.4 (95.2, 0.949) | 84.3 (84.1, 0.825) | 86.8 (86.6, 0.853) | -8.6 |
| P1 | A | 250 Hz x 8 ch x 8 b | 16 | 94.2 (94.1, 0.936) | 93.4 (93.3, 0.927) | 95.5 (95.4, 0.950) | RF: 98.7 (98.7, 0.985) | 97.4 (97.3, 0.971) | 98.0 (98.0, 0.978) | -0.7 |
| P1 | A | 500 Hz x 8 ch x 8 b | 32 | 97.2 (97.2, 0.969) | 97.8 (97.8, 0.976) | 97.5 (97.5, 0.972) | SVM: 99.7 (99.7, 0.996) | 99.2 (99.2, 0.991) | 99.7 (99.7, 0.996) | -0.0 |
| P1 | A | 1000 Hz x 8 ch x 16 b | 128 | 96.8 (96.8, 0.965) | 97.8 (97.8, 0.976) | 97.8 (97.8, 0.976) | SVM: 99.5 (99.5, 0.994) | 99.4 (99.4, 0.993) | 99.1 (99.1, 0.990) | -0.4 |
| P1 | W | 250 Hz x 5 ch x 8 b | 10 | 90.9 (90.9, 0.899) | 91.1 (91.0, 0.901) | 89.9 (89.8, 0.888) | LDA: 96.3 (95.8, 0.959) | 95.7 (95.1, 0.952) | 94.6 (93.8, 0.939) | -1.8 |
| P1 | W | 250 Hz x 8 ch x 8 b | 16 | 96.4 (96.3, 0.960) | 96.5 (96.4, 0.961) | 94.9 (94.9, 0.944) | LDA: 98.5 (98.4, 0.983) | 98.8 (98.8, 0.987) | 99.0 (99.0, 0.989) | +0.5 |
| P1 | W | 1000 Hz x 8 ch x 16 b | 128 | 98.6 (98.6, 0.984) | 98.7 (98.7, 0.986) | 97.7 (97.6, 0.974) | SVM: 99.6 (99.6, 0.996) | 99.6 (99.6, 0.995) | 99.3 (99.3, 0.993) | -0.3 |
| P2 | A | 250 Hz x 5 ch x 8 b | 10 | 83.1 (82.9, 0.814) | 83.4 (83.1, 0.817) | 86.5 (86.3, 0.851) | SVM: 91.4 (91.3, 0.905) | 86.9 (86.6, 0.856) | 90.4 (90.2, 0.895) | -1.0 |
| P2 | A | 500 Hz x 5 ch x 8 b | 20 | 90.4 (90.3, 0.895) | 90.5 (90.4, 0.896) | 92.3 (92.2, 0.916) | SVM: 94.1 (93.9, 0.935) | 94.0 (94.1, 0.935) | 93.1 (93.3, 0.925) | -0.9 |
| P2 | A | 1000 Hz x 8 ch x 8 b | 64 | 96.7 (96.7, 0.964) | 96.5 (96.4, 0.961) | 95.5 (95.5, 0.951) | LDA: 98.0 (98.0, 0.978) | 96.0 (96.0, 0.956) | 96.7 (96.6, 0.964) | -1.4 |
| P2 | A | 1000 Hz x 8 ch x 16 b | 128 | 96.1 (96.1, 0.957) | 95.6 (95.5, 0.951) | 94.9 (94.9, 0.944) | LDA: 98.0 (98.0, 0.978) | 95.8 (95.8, 0.954) | 94.6 (94.6, 0.940) | -3.4 |
| P2 | W | 250 Hz x 5 ch x 8 b | 10 | 87.8 (87.6, 0.865) | 88.6 (88.4, 0.874) | 87.0 (86.8, 0.857) | SVM: 92.4 (92.4, 0.917) | 89.7 (89.5, 0.886) | 92.5 (92.4, 0.918) | +0.0 |
| P2 | W | 500 Hz x 5 ch x 8 b | 20 | 94.3 (94.3, 0.938) | 94.2 (94.1, 0.936) | 92.6 (92.5, 0.919) | LDA: 97.1 (97.1, 0.968) | 95.2 (95.2, 0.948) | 93.8 (93.7, 0.932) | -3.2 |
| P2 | W | 500 Hz x 8 ch x 8 b | 32 | 95.9 (95.9, 0.955) | 94.7 (94.6, 0.941) | 95.3 (95.3, 0.948) | LDA: 97.4 (97.5, 0.972) | 95.5 (95.4, 0.950) | 93.8 (93.6, 0.932) | -3.6 |
| P2 | W | 1000 Hz x 8 ch x 16 b | 128 | 97.7 (97.8, 0.975) | 96.3 (96.3, 0.960) | 95.7 (95.7, 0.953) | LDA: 98.5 (98.5, 0.983) | 96.3 (96.3, 0.960) | 96.7 (96.7, 0.964) | -1.8 |
| S1 | A | 250 Hz x 5 ch x 8 b | 10 | 77.5 (77.5, 0.756) | 77.4 (77.4, 0.755) | 82.4 (82.3, 0.809) | RF: 92.3 (92.2, 0.916) | 86.4 (86.3, 0.853) | 90.0 (89.9, 0.891) | -2.4 |
| S1 | A | 500 Hz x 8 ch x 8 b | 32 | 90.9 (90.8, 0.901) | 91.3 (91.3, 0.906) | 90.5 (90.6, 0.897) | RF: 95.8 (95.8, 0.955) | 93.5 (93.5, 0.930) | 94.4 (94.4, 0.940) | -1.4 |
| S1 | A | 1000 Hz x 8 ch x 16 b | 128 | 91.7 (91.7, 0.910) | 92.6 (92.6, 0.920) | 90.5 (90.6, 0.897) | SVM: 97.4 (97.3, 0.971) | 94.6 (94.6, 0.942) | 93.4 (93.3, 0.928) | -4.0 |
| S1 | W | 250 Hz x 5 ch x 8 b | 10 | 81.5 (81.5, 0.799) | 81.9 (81.9, 0.804) | 81.9 (81.8, 0.804) | SVM: 92.0 (91.9, 0.913) | 88.6 (88.5, 0.876) | 89.9 (89.9, 0.890) | -2.1 |
| S1 | W | 250 Hz x 8 ch x 8 b | 16 | 88.5 (88.5, 0.875) | 88.4 (88.4, 0.874) | 86.7 (86.7, 0.856) | RF: 94.8 (94.8, 0.944) | 91.7 (91.6, 0.910) | 92.4 (92.3, 0.917) | -2.6 |
| S1 | W | 500 Hz x 8 ch x 8 b | 32 | 92.6 (92.6, 0.920) | 93.5 (93.5, 0.930) | 91.0 (91.1, 0.903) | SVM: 96.8 (96.8, 0.965) | 94.6 (94.6, 0.941) | 94.3 (94.3, 0.939) | -2.5 |
| S1 | W | 1000 Hz x 8 ch x 16 b | 128 | 93.5 (93.5, 0.929) | 93.9 (94.0, 0.934) | 90.8 (90.9, 0.900) | SVM: 97.4 (97.4, 0.972) | 95.6 (95.6, 0.952) | 94.7 (94.7, 0.943) | -2.7 |
| S2 | A | 250 Hz x 5 ch x 8 b | 10 | 51.4 (51.3, 0.469) | 50.5 (50.4, 0.459) | 57.6 (57.4, 0.537) | RF: 73.0 (72.0, 0.705) | 52.1 (51.2, 0.477) | 52.5 (51.1, 0.481) | -20.4 |
| S2 | A | 250 Hz x 8 ch x 8 b | 16 | 74.4 (74.3, 0.720) | 74.6 (74.4, 0.722) | 78.7 (78.5, 0.767) | SVM: 90.7 (90.7, 0.898) | 84.1 (84.0, 0.827) | 82.5 (82.3, 0.809) | -8.3 |
| S2 | A | 1000 Hz x 8 ch x 16 b | 128 | 87.2 (87.1, 0.860) | 87.4 (87.3, 0.863) | 87.0 (86.8, 0.858) | RF: 95.9 (95.9, 0.955) | 91.3 (91.3, 0.905) | 91.6 (91.6, 0.908) | -4.3 |
| S2 | W | 250 Hz x 5 ch x 8 b | 10 | 56.9 (57.0, 0.529) | 57.3 (57.2, 0.533) | 58.9 (58.8, 0.552) | RF: 76.1 (75.3, 0.739) | 71.3 (71.3, 0.687) | 69.6 (69.4, 0.668) | -6.4 |
| S2 | W | 250 Hz x 8 ch x 8 b | 16 | 79.5 (79.4, 0.776) | 80.3 (80.2, 0.785) | 80.2 (80.0, 0.784) | RF: 91.8 (91.8, 0.911) | 87.3 (87.2, 0.861) | 87.8 (87.7, 0.867) | -4.0 |
| S2 | W | 1000 Hz x 8 ch x 16 b | 128 | 91.9 (91.8, 0.912) | 90.7 (90.5, 0.898) | 86.9 (86.7, 0.857) | LDA: 96.4 (96.4, 0.961) | 92.3 (92.2, 0.916) | 92.4 (92.2, 0.917) | -4.0 |

### F6. Nested choice of the vote span (counts over cells x outer folds)

| Model | k=1 | k=3 | k=5 | k=9 | k=15 | all since onset |
|---|---|---|---|---|---|---|
| LDA | 0 | 0 | 0 | 1 | 15 | 124 |
| SVM | 0 | 0 | 0 | 3 | 8 | 129 |
| RandomForest | 0 | 0 | 0 | 1 | 5 | 134 |

### F7. Cost per decision (analytic counts, mean over the cells; total = extraction + 2 x model MACs + comparisons; the vote adds C + 1 integer operations, C = number of classes)

| Set | Extraction kFLOP | LDA total | SVM total | RF total | LSTM total | Vote (integer ops) |
|---|---|---|---|---|---|---|
| set A, 14 cells | 82.7 | 86.7 | 648.7 | 93.0 | 195.6 | 13 |
| set W, 14 cells | 60.2 | 66.3 | 837.6 | 70.6 | 212.5 | 13 |
