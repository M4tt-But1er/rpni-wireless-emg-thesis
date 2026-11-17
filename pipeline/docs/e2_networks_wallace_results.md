# E2 networks on S1 and S2 (12 Nov 2026)

Fills the empty S1/S2 cells of the E2 table (appendix table `tab:res_e1`: engineered cascade set, 1 kHz stream).
Scripts: `pipeline/run_e2_networks_wallace.py`, `pipeline/run_e2_lstm_defaultthreads.py`. Logs: `pipeline/e2_networks_wallace.log`, `e2_lstm_dt_S1.log`, `e2_lstm_dt_S2.log`.
Results: `cache/e2_networks_wallace.json` (one-thread run), `cache/e2_networks_wallace/defaultthreads__*.json` (LSTM rows, default threads).

Same code as the P1/P2 values, imported unchanged: tuned RBFNN = `run_rbfnn_tuning.tune_rbfnn` nested per fold on the default RBFNN's
per-fold Stage-3 sets (`cascade_baseline_wallace_lda_rbfnn.json`); LSTM = `lstm_model` headline setting (40 epochs, 64 units) on the Stage-2 set;
tuned LSTM = `run_lstm_training_check.part2` (base / lr3 / reg, early stopping). Same folds and metrics.

| | S1 | S2 |
|---|---|---|
| RBFNN, tuned | 89.30 (F1 89.33, kappa 0.884) | 86.25 (F1 86.13, kappa 0.850) |
| LSTM | 88.90 (F1 88.98, kappa 0.880) | 83.30 (F1 83.23, kappa 0.818) |
| LSTM, tuned | 87.46 (F1 87.67, kappa 0.864) | 84.14 (F1 83.31, kappa 0.827) |

Table values: LSTM rows from the default-thread run (as P1/P2 were made); tuned RBFNN from the one-thread parallel run.

## Reproduction check on P1 (same scripts, P1's own inputs)

| | published | one thread per task | default threads (10) |
|---|---|---|---|
| RBFNN, tuned | 96.76 | 96.89 | not rerun |
| LSTM | 94.58 | 94.83 | 94.58 (all 5 folds identical) |
| LSTM, tuned | 97.40 | 98.51 | not rerun |

The thread count changes floating-point summation order and so the training path. With the default thread count the P1 headline LSTM is
reproduced exactly. On one thread, the tuned LSTM chose a different setting in one P1 fold (lr3 instead of base: 98.6 vs 93.6 % on that fold),
which moved P1 by 1.1 points; the tuned RBFNN chose different settings in 2 of 5 folds (within 0.6 points per fold, 0.13 overall).
On S1/S2 the thread count changed the LSTM results by at most 0.13 points (one thread: S1 88.90 / 87.51, S2 83.30 / 84.27).
So the tuned networks carry run-to-run variation of up to about 1 point from the selection step alone.

## Observations
- On S1 and S2 neither network beats the SVM (91.9 / 89.1) on the engineered set; the tuned RBFNN is the best network (89.3 / 86.2).
- Tuning the LSTM did not help on S1 (87.5 vs 88.9) and helped little on S2 (84.1 vs 83.3), unlike P1 (+2.8).
