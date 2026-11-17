# Fair comparison, follow-up: soft vote and stacked windows for LDA/SVM/RF, and a tuned LSTM — results (9 Nov 2026)

Protocol, declared before any result: [fair_followup_declared_protocol.md](fair_followup_declared_protocol.md) (two amendments, both described there; A2 was made before any outer-test LSTM result was seen).
Request (user): "try the soft vote and stacked windows for the classical models, and try tune the LSTM to its best feasible performance." Continues [fair_lstm_results.md](fair_lstm_results.md); nothing from that run was changed.
Everything below is computed from stored per-window predictions by `analyse_fair2.py`; the tables at the end are generated (copied from `cache/grid/fair2/fair2_tables.md`), not typed. Same 28 winner cells, same features, same 5 outer folds, same metric as before.

## The answer in plain words

**1. Soft voting and stacked windows do not beat the simple vote.**
- **Soft vote.** Adding probability-based votes (mean probability, summed log-probability, exponential averaging) to the earlier hard vote lifts the best classical pipeline only by +0.2 (LDA), +0.5 (SVM) and +0.5 (RF) points (all 28 cells; sign counts 15/11, 17/10, 20/8). By the declared rule that is "about equal". The best classical result is now 92.70 / 93.79 / 94.35 (LDA / SVM / RF) and 94.75 for the best of the three per cell, against 94.46 before.
- **Stacked windows without a vote** (features of the last 3 or 5 non-overlapping windows, or the running mean since the onset) help a lot compared with the raw per-window decision (+2.2 to +4.6 points, better in 23 to 28 of 28 cells), but are worse than the vote alone (best stack, running mean: 92.34 / 91.56 / 92.23 against 92.70 / 93.79 / 94.35).
- **Stacked windows together with a vote** are worse than the vote alone for every model. For RF clearly: −1.5 (3 lags), −3.6 (5 lags) and −2.8 (running mean) points, worse in all 28 cells for each of the three, and the interval rule resolves the last two (15 and 11 cells). For LDA and SVM the sign is the same (−0.5 to −2.4 points), but the reading is "about equal" or "unresolved".
- **Choosing everything per fold** (stack variant and vote, 88 candidates, fully nested) is about equal to the earlier hard vote (−0.4, +0.2, +0.1) and mostly picks "no stacking" (81, 120 and 125 of 140 folds). Extra flexibility only adds selection noise.
- So the best classical pipeline is still a per-window model plus a vote over its own decisions since the onset (a soft or log-probability vote that accumulates over the whole history, picked in 59 / 59 / 62 of 140 folds by LDA / SVM / RF). Stacking costs more and gains nothing on top of it: the SVM with 5 stacked windows costs 3,566 kFLOP per decision against 743 without stacking.

**2. A much more thorough LSTM search did not help; the LSTM is still behind.**
- Two rounds of random search (64 random configurations plus the three earlier ones) gave a tuned LSTM, `LSTM_best`, whose accuracy is **89.78%** with one training seed (28 cells). The earlier hand-tuned LSTM scored 91.19%, round 1 of this search 90.30%. So the larger search ended **1.4 points below the earlier LSTM** (behind in 19 of 28 cells, ahead in 9; "unresolved" by the interval rule).
- Averaging 5 training seeds (5 times the compute) gives `LSTM_best_ens5` = **92.14%**, which is only 0.07 above round 1 and 0.95 above the earlier single LSTM. Round 2 did not improve on round 1 in either form.
- Against the classical models with the best vote (P3): single-seed LSTM −2.9 (LDA), −4.0 (SVM), −4.6 (RF) points; 5-seed LSTM −0.6, −1.7, −2.2. Against the best of the three per cell (P1, the earlier headline): −4.7 single seed (behind in 26 of 28 cells), −2.3 with 5 seeds (behind in 22 of 28). The 5-seed LSTM is about equal to LDA with a vote (−0.6; +0.03 against LDA with the full nested choice), behind SVM and RF.
- Per patient the 5-seed LSTM is behind the best voted classical pipeline in P1 (−1.4 points), P2 (−2.3), S1 (−1.3) and above all S2 (−6.0; single seed −11.1). S2 has the fewest training sequences.
- The extra search did not remove the gap, so the earlier conclusion stands: **with the same history, this kind of LSTM does not beat per-window LDA, SVM or RF with a vote**, and I should not write "the LSTM beats the classical models" in the thesis.

**How firmly.** The reading rule I declared (per-cell 95% intervals with 4 df) calls almost every LSTM comparison "unresolved", although the mean gaps are −2 to −5 points and the LSTM is behind in 22 to 27 of 28 cells. A strict reader should say "behind in sign and size, not formally resolved cell by cell". One seed of the data split, 4 patients, intervals probably too narrow, cells not independent. The two rounds differ from each other by less than the seed noise (the single-seed number moves by about 0.5, the 5-seed ensemble by 0.07, while a 5-seed ensemble gains +2.35 over a single seed).

## What was done

**Part A (classical models, run first).**
- Per cell and model (LDA, SVM, RF, estimators exactly as in the grid, not re-tuned), predictions were made for four feature variants: no stacking, ST3 = [x(t), x(t−4), x(t−8)] (600 ms of history), ST5 = five lags (1000 ms), CUM = [x(t), mean of x(1..t)]. Lags before the segment start are clamped to the first window. The SVM uses `probability=True` (Platt scaling); its hard decisions are unchanged (check V11a).
- 22 vote candidates per variant: hard vote over the last k = 1, 3, 5, 9, 15 or all decisions since the onset; mean probability; summed log-probability (clipped at 1e-3); exponential averaging with a = 0.5, 0.8, 0.9, 0.95. All causal, restarted at each onset.
- The candidate (and the variant) was chosen per outer fold by a 3-fold grouped inner CV on the outer-training windows only (nested). Pipelines P0 to P6 as declared.
- Run time of the classical run (measured): about 36 minutes of wall-clock time (84 of 84 tasks; 3.2 hours of summed task time on 6 workers).

**Part B (LSTM).**
- A tunable causal LSTM (unidirectional, one output per window, state reset per segment) with hidden size, 1 or 2 layers, an optional per-step input bottleneck, dropout, weight decay, learning rate, Gaussian input noise, batch size; early stopping on validation loss. The three earlier settings (`base`, `lr3`, `reg`) are reproduced exactly (V12a).
- **Round 1 (declared before any result).** Pilot of 32 random configurations plus the three earlier ones on 8 reference cells (1 kHz, 8 channels, 16 bit; one per patient and feature set), 40 inner validations each, using only outer-training data. The top 6 plus the three earlier settings formed the final set of 9.
- **Amendment A2 (declared after the round-1 edge check, before any outer-test LSTM result was looked at).** The top 6 of round 1 sat at the edges of the search space on several axes (learning rate, noise, batch size, bottleneck, hidden size). Rule: extend each such axis by one step; round 2 = 32 new random configurations in the extended space, ranked together with the round-1 carry-overs; no round 3.
- **Round-2 pilot result.** The best pilot configurations stay at the same level (mean inner validation accuracy 98.16% for r19 from round 1; the two new ones that reached the top 6 score 98.06% and 97.99%). The extension did not find anything clearly better. In the round-2 top 6 the learning rate (0.01 in 5 of 6, 0.003 in one), noise, batch size, bottleneck and hidden size are no longer all at the edge of the space; layers = 1 and weight decay = 0 (both natural minima) in 5 and 4 of 6. So the search was no longer limited by the edge of its grid (this is the limit I had stated for the earlier tuning), and the plateau is real.
- **Final nested run (both rounds).** Per (cell, outer fold): 3-fold inner validation of the 9 configurations on the outer-training sequences only; the best by mean validation accuracy is retrained on all outer-training sequences for the mean best epoch with seeds 0 to 4. `LSTM_best` = seed 0 alone; `LSTM_best_ens5` = mean softmax of the 5 seeds (5 times the compute). Round 1 is stored as `LSTM_r1`, `LSTM_r1_ens5`, round 2 as `LSTM_best`, `LSTM_best_ens5` (primary for H7 to H9 as declared). No per-cell maxima over rounds are used.
- The chosen configurations differ by cell (table F15): round 2 chose r28 (128 units, bottleneck 32, dropout 0.5, lr 0.01, noise 0.3, batch 8) in 56 of 140 folds, s25 in 28, s09 in 24, r19 in 18; the three earlier settings in 9 folds. The retrains used a median of 52 epochs (5th to 95th percentile 18 to 103, none near the 400 limit).
- Run times (measured, 6 single-thread workers): round 1 pilot 17 minutes and final run 31 minutes; round 2 pilot about 1.5 hours and final run about 2.25 hours. The round-2 projection in `lstm_pilot_summary_r2.md` was 1.1 hours for the final run, so it was too low by about a factor of two; it only decided the inner scheme, which stayed 3-fold (limit 3.5 hours). Part of the extra time is that the round-2 space contains slower configurations (batch size 4, 128 units); I did not investigate the rest.

## Checks

V11a classical variant "none", k = 1 equals the grid in 84 of 84 (cell, model) pairs and the nested hard-k choice equals the earlier `chosen_k`; V11b 9 unit tests of the candidates and V11c stacking causality (`test_fair_history.py`); V11d row alignment of every stored array (`check_fair2_v11.py`; logs `check_v11ad.log`, `check_v11bc.log`).
V12a the new training code reproduces the earlier LSTM_default fold accuracies exactly on three cells and `reg` exactly; V12c the new network is causal (`check_lstm_v12.py`, `check_v12ac.log`). V12b the stored outer-fold predictions map back onto the table rows: asserted inside every round-2 final-run task (labels and fold ids of the stored rows; a task that failed would not have written its file; 140 of 140 exist), and `analyse_fair2.py` also stops if any table row is not covered. V12d passed: a rerun of two round-2 final tasks (P1 at 250 Hz / 5 ch / 8 bit, set A, fold 4, and Wallace S2 at the same operating point, set A, fold 0; 1 thread each as in the run) is bit-identical in the stored predictions, ensemble probabilities, chosen configuration, epochs and inner scores (`check_lstm_v12d.py`, `check_v12d.log`; this checks reproducibility on two cheap tasks, not all 140).
All logs are in `cache/grid/fair2/`.

## Results, in the order of the declared comparisons

1. **H4, soft vote adds to the hard vote?** (table F9) P3 − P1: LDA +0.21, SVM +0.45, RF +0.49 points; about equal (resolved better in 0, 2 and 4 of 28 cells, never worse). Which candidate is picked (F11): `log_all` (summed log-probabilities over the whole history) in 59 / 59 / 62 of 140 folds, `mean_all` 19 / 35 / 55, `hard_all` 33 / 24 / 13. Fixed candidates (F10): for RF `log_all` 94.60 and `mean_all` 94.47 against `hard_all` 93.85; for LDA `log_all` 92.94 against 92.60. Exponential averaging is never better than the whole-history vote (best a = 0.9 or 0.95: 91.2 / 92.6 / 93.0).
2. **H5a, stacking without a vote** (F9, F10): ST3 +2.83 / +2.64 / +2.22, ST5 +2.90 / +3.05 / +2.28, running mean +4.61 / +3.73 / +4.16 points over the raw per-window decision, all "first better". Running-mean stacking is nearly a vote already, but it is worse in the first windows (windows 1 to 2: 75.0 for the SVM against 82.7 raw; F13). A plausible reason, not tested: in those windows the running mean equals the current window, a case the model sees rarely in training.
3. **H5b, stacking on top of the best vote** (F9): ST3 −0.53 / −1.15 / −1.54, ST5 −1.91 / −2.32 / −3.56, running mean −0.57 / −2.41 / −2.81 points. Stacking does not add to a vote; for RF it is clearly worse. Why is not tested here. A plausible reading (not verified) is that a model trained on the bigger input overfits the small number of training sequences and its errors are more strongly correlated from window to window, so the vote has less to average out.
4. **H6, everything chosen per fold** (F9): P6 − P1 = −0.40 / +0.24 / +0.14 points, about equal.
5. **H7, did the better tuning help the LSTM?** (F16) `LSTM_best` − `LSTM_tuned` (earlier) = −1.42 points (9 ahead, 19 behind). Round 1 minus the earlier LSTM = −0.88. No: on the 20 non-pilot cells the earlier hand-tuned LSTM (89.59%) beats both rounds (87.87%, 87.44%). On the 8 pilot cells the new ones are ahead of the old one (96.37 and 95.63 against 95.22), which is what the pilot optimised.
6. **H8, single-seed LSTM against the classical models** (F16): `LSTM_best` minus P3: LDA −2.93, SVM −4.02, RF −4.58; minus P6: −2.32, −3.81, −4.23; minus the best of three with P1 / P3 / P6: −4.69 / −4.97 / −4.70 (behind in 26, 27, 26 of 28 cells). Round 1: −4.15 / −4.43 / −4.16.
7. **H9, 5-seed LSTM** (F16): `LSTM_best_ens5` minus P3: −0.58 (LDA, about equal), −1.67 (SVM), −2.23 (RF); minus the best of three with P1 / P3 / P6: −2.33 / −2.62 / −2.35 (behind in 22, 24, 23 of 28). Round 1: −2.38 / −2.67 / −2.40. The 5-seed LSTM is a different cost class: 603 kFLOP per decision against 77 for LDA, 82 for RF and 743 for the SVM (F14).
8. **Pilot cells against the others** (the pilot cells carry some selection optimism; declared). On the 8 pilot cells: `LSTM_best` 95.63, `LSTM_best_ens5` 96.50, against 96.95 (LDA), 97.07 (SVM), 97.09 (RF) with the vote P3, so the gap there is small (0.5 to 0.6 points for the 5-seed LSTM, 1.3 to 1.5 for the single seed). On the 20 other cells: `LSTM_best` 87.44, `LSTM_best_ens5` 90.39, against 90.99 / 92.48 / 93.26, a gap of 0.6 to 2.9 points (5-seed) and 3.6 to 5.8 points (single seed). The inner validation accuracy of the chosen configuration was 95.2% on the pilot cells (outer 95.6%) and 90.6% on the others (outer single seed 87.4%), so on the cheaper cells the selection is 3.1 points optimistic.
9. **B5, vote on top of the LSTM** (F17): hurts. `LSTM_best` 89.78 → 88.85 (hard vote over all windows) and 89.00 (mean probability); `LSTM_best_ens5` 92.14 → 90.76 / 90.78. The LSTM has already integrated the history, and a vote over its own decisions adds the delay of the first windows without adding information. This differs from the classical models, for which the vote is where almost all of the gain comes from.
10. **Per patient** (F12, F18): `LSTM_best_ens5` against the best voted classical pipeline (P3): P1 −1.44, P2 −2.29, S1 −1.28, S2 −5.99 points (single seed: −2.63, −3.61, −3.65, −11.06). The 5-seed LSTM is above LDA with a vote in S1 (93.71 against 92.04) and below it in S2 (82.28 against 85.10).
11. **Where in time** (F13). In the first 2 windows (decided within 250 ms of the onset, 3.9% of windows) the 5-seed LSTM is level with the classical models (82.1 against 82.4 to 83.1 for the voted ones); from window 21 on (61% of windows) it is behind the voted ones (94.0 against 94.9 for LDA, 95.5 SVM, 96.2 RF; single seed 91.4). The LSTM has no advantage in any time range.
12. **Cost** (F14, analytic counts, not measurements). `LSTM_best`: 178 kFLOP per decision on average over the chosen configurations (the earlier LSTM: 196 to 213; the 5-seed version 603); LDA 77, RF 82, SVM 743 without stacking. The vote itself adds a few additions per class. So LDA and RF with a vote cost about 2.3 times less than one LSTM and about 8 times less than the 5-seed LSTM; the SVM costs more than the single LSTM and about as much as the 5-seed one.

## What this does and does not show

- **Shown:** on this benchmark (steady postures, onset-trimmed segments, known onset), a causal vote over the decisions or probabilities of a per-window LDA, SVM or RF is at least as good as the best LSTM I could find with a large but finite search; LDA and RF with a vote also cost about 2.3 times less than one LSTM (8 times less than the 5-seed version), while the SVM does not. History stacked into the features does not improve on the vote. A soft vote is slightly better than a hard one (about +0.5 points), not meaningfully so.
- **Not shown:**
  - **No LSTM could do better.** The search was random over a fixed family (unidirectional LSTM on window features, 47 to 78 training sequences per fold, one data split). Not tried: other architectures (GRU, temporal convolution), pre-training or pooling across patients, data augmentation beyond Gaussian noise, per-cell hyper-parameters, more training data, class-conditional tricks. The honest statement is "this LSTM, as searched, does not beat a vote".
  - **The asymmetry.** I declared in advance that the LSTM gets a large search and the classical models keep fixed settings. Because the classical side still wins, the asymmetry does not weaken the finding; it would only matter if the LSTM had won. The classical models could probably gain a little from tuning (SVM C and gamma, RF depth, LDA shrinkage); that was not done, and it would make their lead larger, not smaller.
  - **Why the search did not transfer.** Measured: the configurations that win the pilot (large learning rate 0.01, small batches, input noise) are better on the wide, easy pilot cells and no better, in several cases worse, on the cheap cells (the earlier hand-tuned LSTM wins on the non-pilot cells), and the inner validation accuracy overstates the outer accuracy by about 3 points there. Not tested (hypotheses): that these settings suit wide inputs only; that 3-fold validation on 47 to 78 sequences is too noisy to pick between 9 configurations; that early stopping on a validation split with few sequences is unreliable. A pilot over more diverse cells (including cheap ones) would test the first.
  - **Seed variance.** One seed of the LSTM is noisy (a 5-seed ensemble gains +2.35 points on average, with a spread of about 4.7 points across cell-folds). Single-seed comparisons between settings of the LSTM (round 1 against round 2, against the earlier LSTM) therefore have a noise of about this size; the declared H7 reading ("unresolved") reflects that.
  - **Transitions and online onset detection:** as before, the benchmark has none; the vote and the CUM stack need a known onset.
- Wallace S1 and S2 stay upper bounds because of the run signature (see the defence guide); a long vote could amplify a run-specific signature, so method differences on S1 and S2 should be read with that in mind.
- The per-cell "resolved" counts are low because a 5-fold paired interval has only 4 degrees of freedom; the reading rule is conservative; sign counts are shown next to every comparison for that reason.
- The "best of three per cell" is chosen after the fact and favours the classical side; the single-model comparisons agree in sign.

## Reproduce

```
python -W ignore test_fair_history.py                            # V11b, V11c (9 unit tests)
python -W ignore -u run_fair_classical2.py --workers 6           # Part A, 84 tasks, about 36 min; cache/grid/fair2/classical/
python -W ignore check_fair2_v11.py                              # V11a, V11d
python -W ignore check_lstm_v12.py                               # V12a, V12c
python -W ignore -u run_lstm_pilot.py --round 1 --workers 6      # round-1 pilot, cache/grid/fair2/lstm_pilot/
python -W ignore select_lstm_candidates.py --round 1 --workers 6
python -W ignore -u run_lstm_final.py --round 1 --workers 6      # cache/grid/fair2/lstm/ (LSTM_r1)
python -W ignore -u run_lstm_pilot.py --round 2 --workers 6      # amendment A2, cache/grid/fair2/lstm_pilot_r2/
python -W ignore select_lstm_candidates.py --round 2 --workers 6
python -W ignore -u run_lstm_final.py --round 2 --workers 6      # cache/grid/fair2/lstm_r2/ (LSTM_best), about 2.3 h
python -W ignore check_lstm_v12d.py                              # V12d, reruns two tasks into cache/grid/fair2/v12d_rerun/
python -W ignore analyse_fair2.py > cache/grid/fair2/fair2_summary_round2.txt
```

Files in `cache/grid/fair2/`: `fair2_tables.md` (the tables below), `fair2_results.json` (every method and cell), `fair2_cells.csv` (one row per cell and method, with a `pilot` column), `lstm_pilot_summary.md` / `lstm_pilot_summary_r2.md` (pilot rankings, edge checks, projections), `lstm_candidates.json` / `lstm_candidates_r2.json`, `fair2_summary_partA.txt`, `fair2_summary_round1.txt`, `fair2_summary_round2.txt`, run logs and check logs.
The earlier round-1 pilot, selection and final files are kept unchanged (`lstm_pilot/`, `lstm/`).

---

## Tables (generated by `analyse_fair2.py`; contents of `cache/grid/fair2/fair2_tables.md`)

# Fair comparison follow-up: generated tables (analyse_fair2.py; every number is computed from stored predictions)

## F8. Part A: mean window accuracy (%) over the cells, by pipeline and model (P0..P6 as in the protocol)

**all 28 cells**

| pipeline | description | LDA | SVM | RF | best of three (post hoc) / LSTM |
|---|---|---|---|---|---|
| P0 | raw per-window (grid) | 87.72 | 87.80 | 88.08 | 89.09 |
| P1 | hard vote, nested k (earlier headline) | 92.49 | 93.35 | 93.86 | 94.46 |
| P2 | soft/EMA vote, nested | 92.74 | 93.80 | 94.38 | 94.75 |
| P3 | hard + soft vote, nested (22 candidates) | 92.70 | 93.79 | 94.35 | 94.75 |
| stack:ST3 | stack ST3, no vote | 90.55 | 90.44 | 90.29 | 91.65 |
| stack:ST5 | stack ST5, no vote | 90.62 | 90.86 | 90.36 | 91.93 |
| stack:CUM | stack CUM, no vote | 92.34 | 91.56 | 92.23 | 93.77 |
| stackvote:ST3 | ST3 + vote nested (22) | 92.17 | 92.65 | 92.81 | 93.70 |
| stackvote:ST5 | ST5 + vote nested (22) | 90.78 | 91.48 | 90.79 | 92.37 |
| stackvote:CUM | CUM + vote nested (22) | 92.12 | 91.38 | 91.53 | 93.28 |
| P6 | full nested (4 variants x 22) | 92.09 | 93.58 | 94.00 | 94.47 |
|  | LSTM_default (earlier) |  |  |  | 90.97 |
|  | LSTM_tuned (earlier) |  |  |  | 91.19 |
|  | LSTM_r1 |  |  |  | 90.30 |
|  | LSTM_r1_ens5 |  |  |  | 92.07 |
|  | LSTM_r1+hard_all |  |  |  | 89.26 |
|  | LSTM_r1+mean_all |  |  |  | 89.36 |
|  | LSTM_r1_ens5+hard_all |  |  |  | 90.88 |
|  | LSTM_r1_ens5+mean_all |  |  |  | 90.92 |
|  | LSTM_best |  |  |  | 89.78 |
|  | LSTM_best_ens5 |  |  |  | 92.14 |
|  | LSTM_best+hard_all |  |  |  | 88.85 |
|  | LSTM_best+mean_all |  |  |  | 89.00 |
|  | LSTM_best_ens5+hard_all |  |  |  | 90.76 |
|  | LSTM_best_ens5+mean_all |  |  |  | 90.78 |

**set A (14)**

| pipeline | description | LDA | SVM | RF | best of three (post hoc) / LSTM |
|---|---|---|---|---|---|
| P0 | raw per-window (grid) | 86.45 | 86.61 | 88.33 | 88.76 |
| P1 | hard vote, nested k (earlier headline) | 91.75 | 92.90 | 93.83 | 94.26 |
| P2 | soft/EMA vote, nested | 92.08 | 93.54 | 94.41 | 94.71 |
| P3 | hard + soft vote, nested (22 candidates) | 92.04 | 93.49 | 94.36 | 94.67 |
| stack:ST3 | stack ST3, no vote | 89.72 | 89.60 | 90.48 | 91.54 |
| stack:ST5 | stack ST5, no vote | 90.08 | 89.90 | 90.58 | 91.89 |
| stack:CUM | stack CUM, no vote | 91.63 | 90.45 | 92.60 | 93.57 |
| stackvote:ST3 | ST3 + vote nested (22) | 91.51 | 92.18 | 92.69 | 93.61 |
| stackvote:ST5 | ST5 + vote nested (22) | 90.42 | 90.73 | 90.91 | 92.36 |
| stackvote:CUM | CUM + vote nested (22) | 91.49 | 90.39 | 91.78 | 93.11 |
| P6 | full nested (4 variants x 22) | 91.22 | 93.44 | 93.98 | 94.39 |
|  | LSTM_default (earlier) |  |  |  | 89.66 |
|  | LSTM_tuned (earlier) |  |  |  | 90.19 |
|  | LSTM_r1 |  |  |  | 89.86 |
|  | LSTM_r1_ens5 |  |  |  | 91.97 |
|  | LSTM_r1+hard_all |  |  |  | 88.62 |
|  | LSTM_r1+mean_all |  |  |  | 88.74 |
|  | LSTM_r1_ens5+hard_all |  |  |  | 90.71 |
|  | LSTM_r1_ens5+mean_all |  |  |  | 90.70 |
|  | LSTM_best |  |  |  | 88.78 |
|  | LSTM_best_ens5 |  |  |  | 91.56 |
|  | LSTM_best+hard_all |  |  |  | 87.61 |
|  | LSTM_best+mean_all |  |  |  | 87.88 |
|  | LSTM_best_ens5+hard_all |  |  |  | 90.14 |
|  | LSTM_best_ens5+mean_all |  |  |  | 90.00 |

**set W (14)**

| pipeline | description | LDA | SVM | RF | best of three (post hoc) / LSTM |
|---|---|---|---|---|---|
| P0 | raw per-window (grid) | 88.99 | 89.00 | 87.83 | 89.42 |
| P1 | hard vote, nested k (earlier headline) | 93.24 | 93.80 | 93.88 | 94.66 |
| P2 | soft/EMA vote, nested | 93.41 | 94.06 | 94.36 | 94.79 |
| P3 | hard + soft vote, nested (22 candidates) | 93.35 | 94.10 | 94.35 | 94.82 |
| stack:ST3 | stack ST3, no vote | 91.37 | 91.29 | 90.10 | 91.77 |
| stack:ST5 | stack ST5, no vote | 91.15 | 91.81 | 90.14 | 91.98 |
| stack:CUM | stack CUM, no vote | 93.04 | 92.66 | 91.86 | 93.98 |
| stackvote:ST3 | ST3 + vote nested (22) | 92.83 | 93.11 | 92.94 | 93.79 |
| stackvote:ST5 | ST5 + vote nested (22) | 91.13 | 92.23 | 90.66 | 92.38 |
| stackvote:CUM | CUM + vote nested (22) | 92.74 | 92.37 | 91.28 | 93.44 |
| P6 | full nested (4 variants x 22) | 92.97 | 93.72 | 94.02 | 94.55 |
|  | LSTM_default (earlier) |  |  |  | 92.29 |
|  | LSTM_tuned (earlier) |  |  |  | 92.20 |
|  | LSTM_r1 |  |  |  | 90.74 |
|  | LSTM_r1_ens5 |  |  |  | 92.17 |
|  | LSTM_r1+hard_all |  |  |  | 89.91 |
|  | LSTM_r1+mean_all |  |  |  | 89.97 |
|  | LSTM_r1_ens5+hard_all |  |  |  | 91.05 |
|  | LSTM_r1_ens5+mean_all |  |  |  | 91.15 |
|  | LSTM_best |  |  |  | 90.79 |
|  | LSTM_best_ens5 |  |  |  | 92.71 |
|  | LSTM_best+hard_all |  |  |  | 90.09 |
|  | LSTM_best+mean_all |  |  |  | 90.12 |
|  | LSTM_best_ens5+hard_all |  |  |  | 91.39 |
|  | LSTM_best_ens5+mean_all |  |  |  | 91.56 |

**20 non-pilot cells**

| pipeline | description | LDA | SVM | RF | best of three (post hoc) / LSTM |
|---|---|---|---|---|---|
| P0 | raw per-window (grid) | 85.13 | 85.27 | 86.24 | 86.91 |
| P1 | hard vote, nested k (earlier headline) | 90.64 | 91.79 | 92.59 | 93.11 |
| P2 | soft/EMA vote, nested | 91.05 | 92.48 | 93.30 | 93.64 |
| P3 | hard + soft vote, nested (22 candidates) | 90.99 | 92.48 | 93.26 | 93.63 |
| stack:ST3 | stack ST3, no vote | 88.42 | 88.38 | 88.88 | 89.86 |
| stack:ST5 | stack ST5, no vote | 88.55 | 88.92 | 88.99 | 90.25 |
| stack:CUM | stack CUM, no vote | 90.72 | 89.70 | 91.12 | 92.54 |
| stackvote:ST3 | ST3 + vote nested (22) | 90.51 | 91.22 | 91.61 | 92.43 |
| stackvote:ST5 | ST5 + vote nested (22) | 88.88 | 89.98 | 89.39 | 90.97 |
| stackvote:CUM | CUM + vote nested (22) | 90.50 | 89.65 | 90.51 | 91.93 |
| P6 | full nested (4 variants x 22) | 90.19 | 92.26 | 92.80 | 93.22 |
|  | LSTM_default (earlier) |  |  |  | 89.11 |
|  | LSTM_tuned (earlier) |  |  |  | 89.59 |
|  | LSTM_r1 |  |  |  | 87.87 |
|  | LSTM_r1_ens5 |  |  |  | 90.23 |
|  | LSTM_r1+hard_all |  |  |  | 86.73 |
|  | LSTM_r1+mean_all |  |  |  | 86.79 |
|  | LSTM_r1_ens5+hard_all |  |  |  | 88.87 |
|  | LSTM_r1_ens5+mean_all |  |  |  | 88.90 |
|  | LSTM_best |  |  |  | 87.44 |
|  | LSTM_best_ens5 |  |  |  | 90.39 |
|  | LSTM_best+hard_all |  |  |  | 86.42 |
|  | LSTM_best+mean_all |  |  |  | 86.59 |
|  | LSTM_best_ens5+hard_all |  |  |  | 88.79 |
|  | LSTM_best_ens5+mean_all |  |  |  | 88.80 |

## F9. Declared comparisons H4-H6 (classical pipelines; paired by fold; mean difference in points over cells; ahead/behind = cells where the first is ahead/behind by the sign of the mean fold difference; res = cells resolved better/worse by the 95% interval)

**all 28 cells**

| id | comparison | model | mean diff (pts) | ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|---|
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | LDA | +0.21 | 15/11 | 0/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | SVM | +0.45 | 17/10 | 2/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | RF | +0.49 | 20/8 | 4/0 | about equal |
| H5a | stack ST3, no vote, minus P0 | LDA | +2.83 | 26/2 | 22/0 | first better |
| H5a | stack ST3, no vote, minus P0 | SVM | +2.64 | 28/0 | 18/0 | first better |
| H5a | stack ST3, no vote, minus P0 | RF | +2.22 | 24/4 | 15/0 | first better |
| H5a | stack ST5, no vote, minus P0 | LDA | +2.90 | 25/3 | 14/0 | first better |
| H5a | stack ST5, no vote, minus P0 | SVM | +3.05 | 26/2 | 13/0 | first better |
| H5a | stack ST5, no vote, minus P0 | RF | +2.28 | 23/5 | 13/0 | first better |
| H5a | stack CUM, no vote, minus P0 | LDA | +4.61 | 26/2 | 18/0 | first better |
| H5a | stack CUM, no vote, minus P0 | SVM | +3.73 | 28/0 | 14/0 | first better |
| H5a | stack CUM, no vote, minus P0 | RF | +4.16 | 27/1 | 12/0 | first better |
| H5b | ST3 + vote minus P3 | LDA | -0.53 | 8/20 | 0/1 | about equal |
| H5b | ST3 + vote minus P3 | SVM | -1.15 | 4/24 | 0/5 | unresolved (second better ahead, resolved in 5 of 28 cells) |
| H5b | ST3 + vote minus P3 | RF | -1.54 | 0/28 | 0/3 | unresolved (second better ahead, resolved in 3 of 28 cells) |
| H5b | ST5 + vote minus P3 | LDA | -1.91 | 2/26 | 0/3 | unresolved (second better ahead, resolved in 3 of 28 cells) |
| H5b | ST5 + vote minus P3 | SVM | -2.32 | 1/27 | 0/5 | unresolved (second better ahead, resolved in 5 of 28 cells) |
| H5b | ST5 + vote minus P3 | RF | -3.56 | 0/28 | 0/15 | second better |
| H5b | CUM + vote minus P3 | LDA | -0.57 | 10/18 | 0/1 | about equal |
| H5b | CUM + vote minus P3 | SVM | -2.41 | 1/27 | 0/6 | unresolved (second better ahead, resolved in 6 of 28 cells) |
| H5b | CUM + vote minus P3 | RF | -2.81 | 0/28 | 0/11 | second better |
| H6 | P6 minus P1 | LDA | -0.40 | 7/20 | 0/0 | about equal |
| H6 | P6 minus P1 | SVM | +0.24 | 15/13 | 1/0 | about equal |
| H6 | P6 minus P1 | RF | +0.14 | 16/12 | 1/0 | about equal |

**set A (14)**

| id | comparison | model | mean diff (pts) | ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|---|
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | LDA | +0.30 | 8/5 | 0/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | SVM | +0.60 | 8/6 | 1/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | RF | +0.52 | 11/3 | 2/0 | about equal |
| H5a | stack ST3, no vote, minus P0 | LDA | +3.27 | 14/0 | 13/0 | first better |
| H5a | stack ST3, no vote, minus P0 | SVM | +2.99 | 14/0 | 11/0 | first better |
| H5a | stack ST3, no vote, minus P0 | RF | +2.15 | 11/3 | 7/0 | first better |
| H5a | stack ST5, no vote, minus P0 | LDA | +3.63 | 14/0 | 8/0 | first better |
| H5a | stack ST5, no vote, minus P0 | SVM | +3.29 | 13/1 | 5/0 | first better |
| H5a | stack ST5, no vote, minus P0 | RF | +2.24 | 12/2 | 6/0 | first better |
| H5a | stack CUM, no vote, minus P0 | LDA | +5.18 | 14/0 | 10/0 | first better |
| H5a | stack CUM, no vote, minus P0 | SVM | +3.81 | 14/0 | 7/0 | first better |
| H5a | stack CUM, no vote, minus P0 | RF | +4.28 | 14/0 | 6/0 | first better |
| H5b | ST3 + vote minus P3 | LDA | -0.54 | 3/11 | 0/1 | about equal |
| H5b | ST3 + vote minus P3 | SVM | -1.31 | 1/13 | 0/0 | unresolved (second better ahead, resolved in 0 of 14 cells) |
| H5b | ST3 + vote minus P3 | RF | -1.68 | 0/14 | 0/2 | unresolved (second better ahead, resolved in 2 of 14 cells) |
| H5b | ST5 + vote minus P3 | LDA | -1.63 | 2/12 | 0/1 | unresolved (second better ahead, resolved in 1 of 14 cells) |
| H5b | ST5 + vote minus P3 | SVM | -2.77 | 1/13 | 0/2 | unresolved (second better ahead, resolved in 2 of 14 cells) |
| H5b | ST5 + vote minus P3 | RF | -3.45 | 0/14 | 0/6 | second better |
| H5b | CUM + vote minus P3 | LDA | -0.55 | 5/9 | 0/0 | about equal |
| H5b | CUM + vote minus P3 | SVM | -3.12 | 1/13 | 0/3 | unresolved (second better ahead, resolved in 3 of 14 cells) |
| H5b | CUM + vote minus P3 | RF | -2.57 | 0/14 | 0/5 | second better |
| H6 | P6 minus P1 | LDA | -0.53 | 2/11 | 0/0 | about equal |
| H6 | P6 minus P1 | SVM | +0.55 | 8/6 | 1/0 | about equal |
| H6 | P6 minus P1 | RF | +0.14 | 8/6 | 1/0 | about equal |

**set W (14)**

| id | comparison | model | mean diff (pts) | ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|---|
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | LDA | +0.11 | 7/6 | 0/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | SVM | +0.30 | 9/4 | 1/0 | about equal |
| H4 | P3 minus P1 (does a soft/EMA candidate add to the hard vote?) | RF | +0.47 | 9/5 | 2/0 | about equal |
| H5a | stack ST3, no vote, minus P0 | LDA | +2.38 | 12/2 | 9/0 | first better |
| H5a | stack ST3, no vote, minus P0 | SVM | +2.29 | 14/0 | 7/0 | first better |
| H5a | stack ST3, no vote, minus P0 | RF | +2.28 | 13/1 | 8/0 | first better |
| H5a | stack ST5, no vote, minus P0 | LDA | +2.16 | 11/3 | 6/0 | first better |
| H5a | stack ST5, no vote, minus P0 | SVM | +2.80 | 13/1 | 8/0 | first better |
| H5a | stack ST5, no vote, minus P0 | RF | +2.31 | 11/3 | 7/0 | first better |
| H5a | stack CUM, no vote, minus P0 | LDA | +4.04 | 12/2 | 8/0 | first better |
| H5a | stack CUM, no vote, minus P0 | SVM | +3.65 | 14/0 | 7/0 | first better |
| H5a | stack CUM, no vote, minus P0 | RF | +4.04 | 13/1 | 6/0 | first better |
| H5b | ST3 + vote minus P3 | LDA | -0.52 | 5/9 | 0/0 | about equal |
| H5b | ST3 + vote minus P3 | SVM | -0.98 | 3/11 | 0/5 | about equal |
| H5b | ST3 + vote minus P3 | RF | -1.41 | 0/14 | 0/1 | unresolved (second better ahead, resolved in 1 of 14 cells) |
| H5b | ST5 + vote minus P3 | LDA | -2.20 | 0/14 | 0/2 | unresolved (second better ahead, resolved in 2 of 14 cells) |
| H5b | ST5 + vote minus P3 | SVM | -1.86 | 0/14 | 0/3 | unresolved (second better ahead, resolved in 3 of 14 cells) |
| H5b | ST5 + vote minus P3 | RF | -3.68 | 0/14 | 0/9 | second better |
| H5b | CUM + vote minus P3 | LDA | -0.59 | 5/9 | 0/1 | about equal |
| H5b | CUM + vote minus P3 | SVM | -1.70 | 0/14 | 0/3 | unresolved (second better ahead, resolved in 3 of 14 cells) |
| H5b | CUM + vote minus P3 | RF | -3.05 | 0/14 | 0/6 | second better |
| H6 | P6 minus P1 | LDA | -0.27 | 5/9 | 0/0 | about equal |
| H6 | P6 minus P1 | SVM | -0.07 | 7/7 | 0/0 | about equal |
| H6 | P6 minus P1 | RF | +0.14 | 8/6 | 0/0 | about equal |

## F10. Part A: mean accuracy (%) of every FIXED candidate (no choice, applied to the stored per-window outputs), all 28 cells, by model and stack variant

| candidate | LDA none | LDA ST3 | LDA ST5 | LDA CUM | SVM none | SVM ST3 | SVM ST5 | SVM CUM | RF none | RF ST3 | RF ST5 | RF CUM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hard_k1 | 87.72 | 90.55 | 90.62 | 92.34 | 87.80 | 90.44 | 90.86 | 91.56 | 88.08 | 90.29 | 90.36 | 92.23 |
| hard_k3 | 88.45 | 90.78 | 90.74 | 92.14 | 88.94 | 90.89 | 91.09 | 91.49 | 88.88 | 90.39 | 90.29 | 92.07 |
| hard_k5 | 89.09 | 90.93 | 90.71 | 91.98 | 89.80 | 90.99 | 91.11 | 91.40 | 89.48 | 90.49 | 90.22 | 91.97 |
| hard_k9 | 90.06 | 91.11 | 90.61 | 91.73 | 91.09 | 91.16 | 90.99 | 91.31 | 90.59 | 90.68 | 90.11 | 91.83 |
| hard_k15 | 91.09 | 91.36 | 90.53 | 91.53 | 92.10 | 91.49 | 90.90 | 91.19 | 91.58 | 90.93 | 90.03 | 91.76 |
| hard_all | 92.60 | 92.25 | 90.73 | 91.15 | 93.43 | 92.60 | 91.17 | 90.87 | 93.85 | 92.66 | 90.50 | 91.54 |
| mean_k1 | 87.72 | 90.55 | 90.62 | 92.34 | 87.86 | 90.34 | 90.75 | 91.48 | 88.08 | 90.29 | 90.36 | 92.23 |
| mean_k3 | 88.58 | 90.77 | 90.73 | 92.14 | 89.43 | 90.87 | 91.09 | 91.60 | 89.35 | 90.47 | 90.31 | 92.05 |
| mean_k5 | 89.26 | 90.92 | 90.71 | 91.97 | 90.34 | 91.01 | 91.03 | 91.59 | 90.22 | 90.53 | 90.27 | 91.97 |
| mean_k9 | 90.25 | 91.08 | 90.60 | 91.68 | 91.53 | 91.27 | 90.98 | 91.50 | 91.29 | 90.75 | 90.23 | 91.89 |
| mean_k15 | 91.15 | 91.33 | 90.48 | 91.46 | 92.45 | 91.70 | 90.91 | 91.44 | 92.26 | 91.18 | 90.29 | 91.84 |
| mean_all | 92.60 | 92.18 | 90.66 | 91.06 | 93.76 | 92.92 | 91.39 | 91.21 | 94.47 | 93.00 | 90.97 | 91.77 |
| log_k1 | 87.72 | 90.55 | 90.62 | 92.34 | 87.86 | 90.34 | 90.75 | 91.48 | 88.08 | 90.29 | 90.36 | 92.23 |
| log_k3 | 88.71 | 90.77 | 90.77 | 92.14 | 89.50 | 90.84 | 91.07 | 91.54 | 89.36 | 90.45 | 90.31 | 92.06 |
| log_k5 | 89.43 | 90.90 | 90.71 | 91.97 | 90.44 | 90.97 | 91.00 | 91.52 | 90.24 | 90.50 | 90.25 | 91.96 |
| log_k9 | 90.40 | 91.07 | 90.64 | 91.69 | 91.61 | 91.26 | 90.93 | 91.40 | 91.38 | 90.73 | 90.21 | 91.88 |
| log_k15 | 91.34 | 91.35 | 90.56 | 91.46 | 92.61 | 91.71 | 90.87 | 91.28 | 92.41 | 91.13 | 90.24 | 91.81 |
| log_all | 92.94 | 92.32 | 90.83 | 91.05 | 93.99 | 92.96 | 91.32 | 90.98 | 94.60 | 92.94 | 90.95 | 91.81 |
| ema_0.5 | 88.67 | 90.84 | 90.78 | 92.28 | 89.66 | 90.95 | 91.16 | 91.75 | 89.61 | 90.60 | 90.39 | 92.15 |
| ema_0.8 | 90.29 | 91.16 | 90.75 | 91.71 | 91.59 | 91.45 | 91.14 | 91.61 | 91.40 | 90.98 | 90.50 | 92.00 |
| ema_0.9 | 91.21 | 91.23 | 90.45 | 90.96 | 92.58 | 91.82 | 91.03 | 91.27 | 92.65 | 91.53 | 90.60 | 91.73 |
| ema_0.95 | 90.83 | 90.42 | 89.20 | 89.10 | 92.59 | 91.61 | 90.35 | 90.53 | 93.04 | 91.38 | 89.95 | 91.12 |

(hard_k1 of `none` is the raw grid result; for the SVM the soft candidates use Platt probabilities, so mean_k1 differs from hard_k1.)

## F11. Which candidate the nested choice picked (counts over 28 cells x 5 outer folds = 140 choices per model)

**P3** (hard + soft vote, nested (22 candidates))

| model | family counts | top 5 candidates |
|---|---|---|
| LDA | hard 39, mean 19, log 67, ema 15 | log_all 59, hard_all 33, mean_all 19, ema_0.95 9, log_k15 7 |
| SVM | hard 26, mean 36, log 65, ema 13 | log_all 59, mean_all 35, hard_all 24, log_k15 6, ema_0.95 5 |
| RF | hard 13, mean 55, log 63, ema 9 | log_all 62, mean_all 55, hard_all 13, ema_0.95 7, log_k15 1 |

**P6** (full nested (4 variants x 22))

| model | family counts | variant counts | top 5 candidates |
|---|---|---|---|
| LDA | hard 61, mean 16, log 45, ema 18 | none 81, ST3 21, ST5 6, CUM 32 | log_all 39, hard_all 38, hard_k1 16, mean_all 15, ema_0.5 8 |
| SVM | hard 27, mean 37, log 62, ema 14 | none 120, ST3 6, ST5 3, CUM 11 | log_all 56, mean_all 35, hard_all 21, log_k15 6, ema_0.95 5 |
| RF | hard 17, mean 54, log 59, ema 10 | none 125, ST3 2, ST5 0, CUM 13 | log_all 57, mean_all 54, hard_all 16, ema_0.95 8, log_k15 2 |

## F12. Per patient: mean accuracy (%) over that patient's cells (sets A and W together)

| patient | LDA/P0 | LDA/P1 | LDA/P3 | LDA/P6 | SVM/P0 | SVM/P1 | SVM/P3 | SVM/P6 | RF/P0 | RF/P1 | RF/P3 | RF/P6 | LSTM_tuned | LSTM_r1 | LSTM_r1_ens5 | LSTM_best | LSTM_best_ens5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 (n=7) | 93.85 | 96.89 | 97.10 | 96.53 | 94.14 | 97.00 | 97.29 | 97.11 | 94.74 | 97.61 | 98.00 | 97.72 | 96.64 | 96.13 | 96.91 | 95.85 | 97.06 |
| P2 (n=8) | 92.74 | 95.35 | 95.11 | 95.05 | 92.46 | 95.10 | 95.42 | 95.31 | 92.48 | 95.09 | 95.35 | 95.12 | 93.95 | 93.30 | 94.44 | 92.53 | 93.85 |
| S1 (n=7) | 88.02 | 92.01 | 92.04 | 91.28 | 88.44 | 94.65 | 94.49 | 94.30 | 87.69 | 94.61 | 94.83 | 94.83 | 92.72 | 91.81 | 93.98 | 91.31 | 93.71 |
| S2 (n=6) | 73.53 | 84.10 | 85.10 | 83.92 | 73.46 | 85.24 | 86.73 | 86.34 | 74.88 | 86.95 | 88.21 | 87.20 | 79.39 | 77.73 | 81.03 | 77.25 | 82.28 |

## F13. Time-resolved accuracy (%): pooled per cell by window index since the onset, then the mean over the 28 cells (window i is decided 200 + 50 (i-1) ms after the onset)

| method | 1-2 | 3-5 | 6-10 | 11-20 | 21+ |
|---|---|---|---|---|---|
| share of windows | 3.9% | 5.9% | 9.8% | 19.4% | 61.0% |
| LDA/P0 | 82.2 | 84.4 | 87.5 | 88.9 | 87.7 |
| LDA/P3 | 82.4 | 84.3 | 87.4 | 91.0 | 94.9 |
| LDA/stack:CUM | 78.7 | 83.2 | 87.7 | 91.7 | 94.4 |
| LDA/P6 | 81.9 | 84.5 | 87.1 | 90.6 | 94.2 |
| SVM/P0 | 82.7 | 85.8 | 87.9 | 89.2 | 87.7 |
| SVM/P3 | 83.1 | 86.8 | 90.5 | 93.0 | 95.5 |
| SVM/stack:CUM | 75.0 | 80.4 | 87.5 | 91.2 | 93.9 |
| SVM/P6 | 82.9 | 86.1 | 89.9 | 92.8 | 95.4 |
| RF/P0 | 83.3 | 86.7 | 89.4 | 89.7 | 87.7 |
| RF/P3 | 82.9 | 87.0 | 91.0 | 93.7 | 96.2 |
| RF/stack:CUM | 79.6 | 85.0 | 89.3 | 91.9 | 93.9 |
| RF/P6 | 82.7 | 86.8 | 90.7 | 93.3 | 95.8 |
| LSTM_tuned | 81.2 | 84.2 | 87.8 | 90.5 | 92.8 |
| LSTM_r1 | 80.4 | 83.8 | 87.3 | 89.7 | 91.9 |
| LSTM_r1_ens5 | 82.3 | 85.0 | 88.8 | 90.9 | 93.8 |
| LSTM_best | 80.1 | 83.4 | 86.7 | 89.2 | 91.4 |
| LSTM_best_ens5 | 82.1 | 85.1 | 88.5 | 90.9 | 94.0 |

## F14. Analytic cost per decision (kFLOP = extraction + 2 x model MACs + comparisons; mean over the 28 cells; stacked models keep the earlier windows' feature vectors in memory, no recomputation; RF comparisons are not recomputed for stacked forests (depth not stored))

| model | variant | kFLOP per decision |
|---|---|---|
| LDA | none | 76.5 |
| LDA | ST3 | 86.7 |
| LDA | ST5 | 96.8 |
| LDA | CUM | 81.6 |
| SVM | none | 743.2 |
| SVM | ST3 | 2154.0 |
| SVM | ST5 | 3565.9 |
| SVM | CUM | 974.5 |
| RF | none | 81.8 |
| RF | ST3 | 81.8 |
| RF | ST5 | 81.8 |
| RF | CUM | 81.8 |

LSTM_r1: mean 177.3 kFLOP per decision (chosen configurations averaged over folds); LSTM_r1_ens5: 600.9.

LSTM_best: mean 177.8 kFLOP per decision (chosen configurations averaged over folds); LSTM_best_ens5: 603.1.

For comparison the earlier LSTM_tuned (64 units, 1 layer) cost about 196 (set A) / 213 (set W) kFLOP.

## F15. Part B: chosen LSTM configurations (counts over 28 cells x 5 outer folds) and epochs

**LSTM_r1** (round 1). Chosen: r28 58, r07 34, r19 22, r20 9, r32 6, lr3 5, r25 4, base 1, reg 1

Epochs of the retrain: median 57, 5th-95th percentile 18-125, max 337, share at the 400 limit 0.0%.

**LSTM_best** (round 2). Chosen: r28 56, s25 28, s09 24, r19 18, lr3 6, r25 3, base 2, r32 2, reg 1

Epochs of the retrain: median 52, 5th-95th percentile 18-103, max 123, share at the 400 limit 0.0%.

## F16. Declared comparisons H7-H9 (LSTM; paired by fold; mean difference in points; LSTM ahead/behind = cells where the LSTM is ahead/behind; res = resolved better/worse). Primary LSTM = LSTM_best (round 2, amendment A2); LSTM_r1 (round 1) is the same comparison, descriptive

**all 28 cells**

| id | comparison | mean diff (pts) | LSTM ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|
| H7 | LSTM_r1 minus LSTM_tuned (earlier) | -0.88 | 11/17 | 0/1 | about equal |
| H7/H8 | LSTM_r1 minus LDA/P3 | -2.39 | 5/23 | 0/1 | unresolved (classical better ahead, resolved in 1 of 28 cells) |
| H7/H8 | LSTM_r1 minus SVM/P3 | -3.48 | 5/23 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H7/H8 | LSTM_r1 minus RF/P3 | -4.04 | 4/24 | 0/3 | unresolved (classical better ahead, resolved in 3 of 28 cells) |
| H7/H8 | LSTM_r1 minus LDA/P6 | -1.79 | 8/20 | 0/0 | unresolved (classical better ahead, resolved in 0 of 28 cells) |
| H7/H8 | LSTM_r1 minus SVM/P6 | -3.27 | 6/22 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H7/H8 | LSTM_r1 minus RF/P6 | -3.69 | 4/24 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H7/H8 | LSTM_r1 minus best of three/P1 (post hoc) | -4.15 | 3/25 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H7/H8 | LSTM_r1 minus best of three/P3 (post hoc) | -4.43 | 2/26 | 0/5 | unresolved (classical better ahead, resolved in 5 of 28 cells) |
| H7/H8 | LSTM_r1 minus best of three/P6 (post hoc) | -4.16 | 3/25 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P3 | -0.63 | 12/16 | 2/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P3 | -1.72 | 9/19 | 0/1 | unresolved (classical better ahead, resolved in 1 of 28 cells) |
| H9 | LSTM_r1_ens5 minus RF/P3 | -2.28 | 5/23 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P6 | -0.02 | 16/12 | 1/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P6 | -1.51 | 10/18 | 0/0 | unresolved (classical better ahead, resolved in 0 of 28 cells) |
| H9 | LSTM_r1_ens5 minus RF/P6 | -1.93 | 6/22 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P1 (post hoc) | -2.38 | 6/22 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P3 (post hoc) | -2.67 | 4/24 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P6 (post hoc) | -2.40 | 5/23 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H7 | LSTM_best minus LSTM_tuned (earlier) | -1.42 | 9/19 | 0/2 | unresolved (earlier LSTM better ahead, resolved in 2 of 28 cells) |
| H7/H8 | LSTM_best minus LDA/P3 | -2.93 | 4/24 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H7/H8 | LSTM_best minus SVM/P3 | -4.02 | 4/24 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H7/H8 | LSTM_best minus RF/P3 | -4.58 | 4/24 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H7/H8 | LSTM_best minus LDA/P6 | -2.32 | 7/21 | 0/2 | unresolved (classical better ahead, resolved in 2 of 28 cells) |
| H7/H8 | LSTM_best minus SVM/P6 | -3.81 | 6/22 | 0/3 | unresolved (classical better ahead, resolved in 3 of 28 cells) |
| H7/H8 | LSTM_best minus RF/P6 | -4.23 | 4/24 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H7/H8 | LSTM_best minus best of three/P1 (post hoc) | -4.69 | 2/26 | 0/5 | unresolved (classical better ahead, resolved in 5 of 28 cells) |
| H7/H8 | LSTM_best minus best of three/P3 (post hoc) | -4.97 | 1/27 | 0/5 | unresolved (classical better ahead, resolved in 5 of 28 cells) |
| H7/H8 | LSTM_best minus best of three/P6 (post hoc) | -4.70 | 2/26 | 0/6 | unresolved (classical better ahead, resolved in 6 of 28 cells) |
| H9 | LSTM_best_ens5 minus LDA/P3 | -0.58 | 13/15 | 1/1 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P3 | -1.67 | 7/21 | 0/1 | unresolved (classical better ahead, resolved in 1 of 28 cells) |
| H9 | LSTM_best_ens5 minus RF/P3 | -2.23 | 6/22 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H9 | LSTM_best_ens5 minus LDA/P6 | +0.03 | 14/14 | 1/0 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P6 | -1.46 | 8/20 | 0/1 | unresolved (classical better ahead, resolved in 1 of 28 cells) |
| H9 | LSTM_best_ens5 minus RF/P6 | -1.88 | 7/21 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H9 | LSTM_best_ens5 minus best of three/P1 (post hoc) | -2.33 | 6/22 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H9 | LSTM_best_ens5 minus best of three/P3 (post hoc) | -2.62 | 4/24 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |
| H9 | LSTM_best_ens5 minus best of three/P6 (post hoc) | -2.35 | 5/23 | 0/4 | unresolved (classical better ahead, resolved in 4 of 28 cells) |

**set A (14)**

| id | comparison | mean diff (pts) | LSTM ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|
| H7 | LSTM_r1 minus LSTM_tuned (earlier) | -0.30 | 6/8 | 0/0 | about equal |
| H7/H8 | LSTM_r1 minus LDA/P3 | -2.18 | 3/11 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus SVM/P3 | -3.62 | 3/11 | 0/4 | unresolved (classical better ahead, resolved in 4 of 14 cells) |
| H7/H8 | LSTM_r1 minus RF/P3 | -4.48 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_r1 minus LDA/P6 | -1.35 | 5/9 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus SVM/P6 | -3.57 | 3/11 | 0/4 | unresolved (classical better ahead, resolved in 4 of 14 cells) |
| H7/H8 | LSTM_r1 minus RF/P6 | -4.11 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P1 (post hoc) | -4.39 | 2/12 | 0/3 | unresolved (classical better ahead, resolved in 3 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P3 (post hoc) | -4.80 | 1/13 | 0/4 | unresolved (classical better ahead, resolved in 4 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P6 (post hoc) | -4.52 | 2/12 | 0/4 | unresolved (classical better ahead, resolved in 4 of 14 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P3 | -0.08 | 6/8 | 1/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P3 | -1.52 | 5/9 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_r1_ens5 minus RF/P3 | -2.39 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P6 | +0.75 | 9/5 | 1/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P6 | -1.47 | 5/9 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_r1_ens5 minus RF/P6 | -2.01 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P1 (post hoc) | -2.29 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P3 (post hoc) | -2.70 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P6 (post hoc) | -2.42 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7 | LSTM_best minus LSTM_tuned (earlier) | -1.42 | 4/10 | 0/2 | unresolved (earlier LSTM better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus LDA/P3 | -3.30 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_best minus SVM/P3 | -4.74 | 3/11 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus RF/P3 | -5.61 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus LDA/P6 | -2.47 | 5/9 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_best minus SVM/P6 | -4.69 | 4/10 | 0/3 | unresolved (classical better ahead, resolved in 3 of 14 cells) |
| H7/H8 | LSTM_best minus RF/P6 | -5.23 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P1 (post hoc) | -5.51 | 1/13 | 0/3 | unresolved (classical better ahead, resolved in 3 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P3 (post hoc) | -5.92 | 0/14 | 0/3 | unresolved (classical better ahead, resolved in 3 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P6 (post hoc) | -5.64 | 1/13 | 0/4 | unresolved (classical better ahead, resolved in 4 of 14 cells) |
| H9 | LSTM_best_ens5 minus LDA/P3 | -0.51 | 7/7 | 1/1 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P3 | -1.95 | 4/10 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_best_ens5 minus RF/P3 | -2.82 | 4/10 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus LDA/P6 | +0.31 | 8/6 | 1/0 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P6 | -1.90 | 5/9 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_best_ens5 minus RF/P6 | -2.44 | 4/10 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P1 (post hoc) | -2.73 | 4/10 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P3 (post hoc) | -3.14 | 3/11 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P6 (post hoc) | -2.86 | 3/11 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |

**set W (14)**

| id | comparison | mean diff (pts) | LSTM ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|
| H7 | LSTM_r1 minus LSTM_tuned (earlier) | -1.46 | 5/9 | 0/1 | unresolved (earlier LSTM better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_r1 minus LDA/P3 | -2.61 | 2/12 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_r1 minus SVM/P3 | -3.34 | 2/12 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus RF/P3 | -3.60 | 2/12 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_r1 minus LDA/P6 | -2.22 | 3/11 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus SVM/P6 | -2.97 | 3/11 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus RF/P6 | -3.28 | 2/12 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P1 (post hoc) | -3.91 | 1/13 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P3 (post hoc) | -4.06 | 1/13 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_r1 minus best of three/P6 (post hoc) | -3.80 | 1/13 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P3 | -1.18 | 6/8 | 1/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_r1_ens5 minus SVM/P3 | -1.92 | 4/10 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus RF/P3 | -2.17 | 2/12 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P6 | -0.80 | 7/7 | 0/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P6 | -1.55 | 5/9 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_r1_ens5 minus RF/P6 | -1.85 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P1 (post hoc) | -2.48 | 3/11 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P3 (post hoc) | -2.64 | 1/13 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P6 (post hoc) | -2.37 | 2/12 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7 | LSTM_best minus LSTM_tuned (earlier) | -1.41 | 5/9 | 0/0 | unresolved (earlier LSTM better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_best minus LDA/P3 | -2.56 | 1/13 | 0/1 | unresolved (classical better ahead, resolved in 1 of 14 cells) |
| H7/H8 | LSTM_best minus SVM/P3 | -3.30 | 1/13 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_best minus RF/P3 | -3.56 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus LDA/P6 | -2.18 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus SVM/P6 | -2.93 | 2/12 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H7/H8 | LSTM_best minus RF/P6 | -3.23 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P1 (post hoc) | -3.86 | 1/13 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P3 (post hoc) | -4.02 | 1/13 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H7/H8 | LSTM_best minus best of three/P6 (post hoc) | -3.76 | 1/13 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus LDA/P3 | -0.64 | 6/8 | 0/0 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P3 | -1.38 | 3/11 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_best_ens5 minus RF/P3 | -1.64 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus LDA/P6 | -0.26 | 6/8 | 0/0 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P6 | -1.01 | 3/11 | 0/0 | unresolved (classical better ahead, resolved in 0 of 14 cells) |
| H9 | LSTM_best_ens5 minus RF/P6 | -1.31 | 3/11 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P1 (post hoc) | -1.94 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P3 (post hoc) | -2.10 | 1/13 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |
| H9 | LSTM_best_ens5 minus best of three/P6 (post hoc) | -1.84 | 2/12 | 0/2 | unresolved (classical better ahead, resolved in 2 of 14 cells) |

**20 non-pilot cells**

| id | comparison | mean diff (pts) | LSTM ahead/behind | res better/worse | reading |
|---|---|---|---|---|---|
| H7 | LSTM_r1 minus LSTM_tuned (earlier) | -1.69 | 4/16 | 0/1 | unresolved (earlier LSTM better ahead, resolved in 1 of 20 cells) |
| H7/H8 | LSTM_r1 minus LDA/P3 | -3.11 | 1/19 | 0/1 | unresolved (classical better ahead, resolved in 1 of 20 cells) |
| H7/H8 | LSTM_r1 minus SVM/P3 | -4.59 | 3/17 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H7/H8 | LSTM_r1 minus RF/P3 | -5.37 | 2/18 | 0/3 | unresolved (classical better ahead, resolved in 3 of 20 cells) |
| H7/H8 | LSTM_r1 minus LDA/P6 | -2.31 | 4/16 | 0/0 | unresolved (classical better ahead, resolved in 0 of 20 cells) |
| H7/H8 | LSTM_r1 minus SVM/P6 | -4.37 | 3/17 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H7/H8 | LSTM_r1 minus RF/P6 | -4.92 | 2/18 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H7/H8 | LSTM_r1 minus best of three/P1 (post hoc) | -5.23 | 1/19 | 0/3 | unresolved (classical better ahead, resolved in 3 of 20 cells) |
| H7/H8 | LSTM_r1 minus best of three/P3 (post hoc) | -5.74 | 0/20 | 0/5 | unresolved (classical better ahead, resolved in 5 of 20 cells) |
| H7/H8 | LSTM_r1 minus best of three/P6 (post hoc) | -5.34 | 1/19 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P3 | -0.77 | 8/12 | 2/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P3 | -2.25 | 6/14 | 0/1 | unresolved (classical better ahead, resolved in 1 of 20 cells) |
| H9 | LSTM_r1_ens5 minus RF/P3 | -3.02 | 3/17 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H9 | LSTM_r1_ens5 minus LDA/P6 | +0.04 | 12/8 | 1/0 | about equal |
| H9 | LSTM_r1_ens5 minus SVM/P6 | -2.03 | 6/14 | 0/0 | unresolved (classical better ahead, resolved in 0 of 20 cells) |
| H9 | LSTM_r1_ens5 minus RF/P6 | -2.57 | 4/16 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P1 (post hoc) | -2.88 | 4/16 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P3 (post hoc) | -3.40 | 2/18 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H9 | LSTM_r1_ens5 minus best of three/P6 (post hoc) | -2.99 | 3/17 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H7 | LSTM_best minus LSTM_tuned (earlier) | -2.14 | 4/16 | 0/2 | unresolved (earlier LSTM better ahead, resolved in 2 of 20 cells) |
| H7/H8 | LSTM_best minus LDA/P3 | -3.57 | 2/18 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H7/H8 | LSTM_best minus SVM/P3 | -5.05 | 3/17 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H7/H8 | LSTM_best minus RF/P3 | -5.82 | 2/18 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H7/H8 | LSTM_best minus LDA/P6 | -2.76 | 4/16 | 0/2 | unresolved (classical better ahead, resolved in 2 of 20 cells) |
| H7/H8 | LSTM_best minus SVM/P6 | -4.83 | 3/17 | 0/3 | unresolved (classical better ahead, resolved in 3 of 20 cells) |
| H7/H8 | LSTM_best minus RF/P6 | -5.37 | 2/18 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H7/H8 | LSTM_best minus best of three/P1 (post hoc) | -5.68 | 1/19 | 0/5 | unresolved (classical better ahead, resolved in 5 of 20 cells) |
| H7/H8 | LSTM_best minus best of three/P3 (post hoc) | -6.20 | 0/20 | 0/5 | unresolved (classical better ahead, resolved in 5 of 20 cells) |
| H7/H8 | LSTM_best minus best of three/P6 (post hoc) | -5.79 | 1/19 | 0/6 | unresolved (classical better ahead, resolved in 6 of 20 cells) |
| H9 | LSTM_best_ens5 minus LDA/P3 | -0.62 | 9/11 | 1/1 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P3 | -2.10 | 4/16 | 0/1 | unresolved (classical better ahead, resolved in 1 of 20 cells) |
| H9 | LSTM_best_ens5 minus RF/P3 | -2.88 | 4/16 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H9 | LSTM_best_ens5 minus LDA/P6 | +0.18 | 10/10 | 1/0 | about equal |
| H9 | LSTM_best_ens5 minus SVM/P6 | -1.88 | 4/16 | 0/1 | unresolved (classical better ahead, resolved in 1 of 20 cells) |
| H9 | LSTM_best_ens5 minus RF/P6 | -2.43 | 5/15 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H9 | LSTM_best_ens5 minus best of three/P1 (post hoc) | -2.74 | 4/16 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H9 | LSTM_best_ens5 minus best of three/P3 (post hoc) | -3.26 | 2/18 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |
| H9 | LSTM_best_ens5 minus best of three/P6 (post hoc) | -2.85 | 3/17 | 0/4 | unresolved (classical better ahead, resolved in 4 of 20 cells) |

## F17. Part B secondary (parameter-free): the LSTM decisions smoothed by a causal vote over everything since the onset, against the classical pipeline P6 (best of three, post hoc)

| method | all 28 cells | set A (14) | set W (14) | 20 non-pilot cells | minus best-of-three P6 (all 28) |
|---|---|---|---|---|---|
| LSTM_r1 | 90.30 | 89.86 | 90.74 | 87.87 | -4.16 |
| LSTM_r1_ens5 | 92.07 | 91.97 | 92.17 | 90.23 | -2.40 |
| LSTM_r1+hard_all | 89.26 | 88.62 | 89.91 | 86.73 | -5.20 |
| LSTM_r1+mean_all | 89.36 | 88.74 | 89.97 | 86.79 | -5.11 |
| LSTM_r1_ens5+hard_all | 90.88 | 90.71 | 91.05 | 88.87 | -3.59 |
| LSTM_r1_ens5+mean_all | 90.92 | 90.70 | 91.15 | 88.90 | -3.54 |
| LSTM_best | 89.78 | 88.78 | 90.79 | 87.44 | -4.70 |
| LSTM_best_ens5 | 92.14 | 91.56 | 92.71 | 90.39 | -2.35 |
| LSTM_best+hard_all | 88.85 | 87.61 | 90.09 | 86.42 | -5.64 |
| LSTM_best+mean_all | 89.00 | 87.88 | 90.12 | 86.59 | -5.49 |
| LSTM_best_ens5+hard_all | 90.76 | 90.14 | 91.39 | 88.79 | -3.72 |
| LSTM_best_ens5+mean_all | 90.78 | 90.00 | 91.56 | 88.80 | -3.71 |

## F18. Per patient: LSTM minus the best of three classical models, by pipeline (mean over that patient's cells; cells of one patient are not independent)

| LSTM | classical pipeline | P1 | P2 | S1 | S2 |
|---|---|---|---|---|---|
| LSTM_r1 | P1 | -2.11 (2/5 of 7) | -2.55 (0/8 of 8) | -3.40 (1/6 of 7) | -9.53 (0/6 of 6) |
| LSTM_r1 | P3 | -2.33 (2/5 of 7) | -2.85 (0/8 of 8) | -3.15 (0/7 of 7) | -10.49 (0/6 of 6) |
| LSTM_r1 | P6 | -2.04 (3/4 of 7) | -2.92 (0/8 of 8) | -3.04 (0/7 of 7) | -9.61 (0/6 of 6) |
| LSTM_r1_ens5 | P1 | -1.36 (4/3 of 7) | -1.41 (1/7 of 8) | -1.24 (1/6 of 7) | -6.21 (0/6 of 6) |
| LSTM_r1_ens5 | P3 | -1.58 (3/4 of 7) | -1.71 (0/8 of 8) | -0.99 (1/6 of 7) | -7.17 (0/6 of 6) |
| LSTM_r1_ens5 | P6 | -1.29 (4/3 of 7) | -1.77 (0/8 of 8) | -0.88 (1/6 of 7) | -6.29 (0/6 of 6) |
| LSTM_best | P1 | -2.41 (1/6 of 7) | -3.31 (0/8 of 8) | -3.90 (1/6 of 7) | -10.10 (0/6 of 6) |
| LSTM_best | P3 | -2.63 (1/6 of 7) | -3.61 (0/8 of 8) | -3.65 (0/7 of 7) | -11.06 (0/6 of 6) |
| LSTM_best | P6 | -2.34 (2/5 of 7) | -3.68 (0/8 of 8) | -3.54 (0/7 of 7) | -10.18 (0/6 of 6) |
| LSTM_best_ens5 | P1 | -1.22 (4/3 of 7) | -1.99 (1/7 of 8) | -1.53 (1/6 of 7) | -5.03 (0/6 of 6) |
| LSTM_best_ens5 | P3 | -1.44 (3/4 of 7) | -2.29 (0/8 of 8) | -1.28 (1/6 of 7) | -5.99 (0/6 of 6) |
| LSTM_best_ens5 | P6 | -1.15 (4/3 of 7) | -2.36 (0/8 of 8) | -1.17 (1/6 of 7) | -5.11 (0/6 of 6) |

