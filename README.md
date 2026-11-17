# Wireless Communication and Movement Classification for Regenerative Peripheral Nerve Interfaces: thesis code

This repository is the code behind the honours thesis *Wireless Communication and Movement Classification for Regenerative Peripheral Nerve Interfaces* (Bachelor of Engineering (Honours) – Biomedical Engineering, The University of Sydney, 2026). It contains:

- the offline analysis pipeline (data preparation, stream emulation, feature sets, classifiers, data-rate grid, channel studies, packet-loss study, classifier comparisons, link budget and figures);
- the ESP32 Bluetooth Low Energy firmware and PC scripts used for the gel-phantom tests;
- the declared protocols, written before each experiment's results were seen, with the matching results notes.

It holds the scripts that produce the thesis results, together with every module they import. Exploratory and superseded scripts from development are not included.

## Data (not included)

The recordings are public, from the University of Michigan RPNI early-feasibility trial (NCT03260400), held in Deep Blue Data:

- **Vaskov et al.** (participants P1, P2): <https://deepblue.lib.umich.edu/data/concern/data_sets/nv9353227>
- **Wallace et al.** (participants S1, S2): <https://deepblue.lib.umich.edu/data/concern/data_sets/zp38wd87k>

Download them into one folder containing `Vaskov Data/` and `Wallace Data/`, then set:

```
set RPNI_DATA_DIR=C:\path\to\that\folder        (Windows)
export RPNI_DATA_DIR=/path/to/that/folder        (macOS / Linux)
```

If the variable is not set, the pipeline looks in the folder above this repository (`pipeline/config.py`). Feature tables, cross-validation results and saved models are written to `pipeline/cache/`, which is not included (about 2 GB). The scripts rebuild it.

## Setup

Python 3.14 (Windows 11) was used. Older versions will probably work.

```
pip install -r pipeline/requirements.txt
pip install -r esp32_ble_link/requirements.txt     # only for the phantom tests
```

## Where each thesis result comes from

Section numbers refer to the final thesis (Chapter 3 Methods, Chapter 4 Results). Run the scripts from inside `pipeline/`. Each `run_*` script writes to `pipeline/cache/`, and the matching `analyse_*` / `make_*` script turns that output into the thesis tables.

| Thesis section | What | Scripts (in `pipeline/`) | Protocol / results notes (in `pipeline/docs/`) |
|---|---|---|---|
| 3.1.1 Data, preparation and stream emulation | Band-pass filtering, onset detection, 200 ms windows, rest exclusion; reduced sample rate, channel count and bit depth | `config.py`, `data_io.py`, `preprocessing.py`, `segmentation.py`, `build_features.py`, `build_features_norest.py`, `wallace_data.py`, `bitrate_sweep.py`, `wallace_bitrate.py` | — |
| 3.1.2 Feature sets and classifiers | ES (24 features per channel), WS (ten families), OS (five families); LDA, SVM, RF, RBFNN, LSTM; grouped 5-fold cross-validation; temporal voting | `features.py`, `wide_features.py`, `build_extra_features.py`, `build_wide_tables.py`, `families.py`, `models.py`, `nested_cv.py`, `hyperparameter_tuning.py`, `run_rbfnn_tuning.py`, `lstm_model.py`, `lstm_tuning.py`, `fair_vote.py`, `fair_history.py`; checks: `run_hyperparameter_check_clean.py`, `run_lstm_training_check.py` | — |
| 3.1.3 / 4.1 Baseline on the raw 30 kHz stream (Table 4.1) | Five classifiers at 3.84 Mb/s vs the 1 kHz reference | `run_baseline_30k.py`, `analyse_baseline_30k.py` | `three_experiments_declared_protocol.md`, `three_experiments_results.md` |
| 3.2.1 / 4.2.1 Data-rate grid (Figs 4.1, 4.2; Table 4.2) | 193 configurations of sample rate × channels × bit depth; Pareto frontier and elbow; computational cost | `run_bitrate_search.py` (channel ranking), `run_grid.py`, `analyse_grid.py`, `analyse_grid_W.py`, `grid_slices.py`, `grid_tables.py`, `make_grid_tables.py`, `bitrate_pareto.py`, `compute_complexity_and_latency.py`, `feature_cost.py`; checks: `check_grid_integrity.py`, `check_elbow_truncation.py`, `test_grid_slices.py` | `grid_declared_protocol.md`, `grid_results.md`, `elbow_truncation_note.md` |
| 3.2.1 / 4.2.1 Check of the channel order (Fig 4.3) | Leave-one-channel-out, single channels, backward-elimination keep order | `run_channel_importance.py`, `analyse_channel_importance.py`, `run_grid_keeporder.py`, `analyse_grid_keeporder.py` | `channel_importance_declared_protocol.md`, `channel_importance_results.md`, `grid_keeporder_exploratory_results.md` |
| 3.2.2 / 4.2.2 RPNI vs residual-muscle channels, exploratory (Tables 4.3–4.4, Fig 4.4) | Equal channel count, per-movement recall, signal quality, sensitivity to the stream | `run_rpni_vs_muscle.py`, `analyse_rpni_vs_muscle.py`, `rpni_signal_quality.py` | `rpni_vs_muscle_declared_protocol.md`, `rpni_vs_muscle_exploratory_results.md` |
| 3.3.2 / 4.3.1 Link budget (Fig 4.5, Appendix tables) | Friis transmission with absorption in muscle; packet error rate from margin; maximum depth | `link_budget.py`, `link_budget_per.py`, `ble_link_budget_depth.py` | — |
| 3.3.3 / 4.3.2 Gel phantom: RSSI, throughput, delay, backlog (Table 4.5) | Real BLE link between two XIAO ESP32-S3 boards through PVP-agar gel | `esp32_ble_link/` (see below); packet tally: `gel_packet_tally.py` | `three_experiments_declared_protocol.md` (delay test) |
| 3.4.1 / 4.4.1 Feature-set development (Appendix tables) | Compact (selected) sets vs ES; ES vs WS; family ablation and the elimination that chose OS | `run_feature_selection_norest.py`, `feature_selection.py`, `run_models_norest.py`, `run_wallace_norest.py`, `run_own_feature_selection.py`, `run_fully_nested_check.py`, `extended_pool.py`, `run_stage1_ops.py`, `run_stage1_rbfnn_tuned.py`, `compare_stage1_ops.py`, `compare_stage1_headline.py`, `family_ablation.py`, `family_backward_elimination.py`, `compare_family_ablation.py`, `family_ablation_rf.py`, `analyse_family_ablation_rf.py` | `stage1_declared_protocol.md`, `stage1_results_corrected.md`, `family_ablation_declared_protocol.md`, `family_ablation_results.md`, `family_ablation_rf_exploratory_results.md` |
| 3.4.2 / 4.4.2 Classifiers with equal access to history (Table 4.6, Fig 4.6) | Temporal voting vs tuned and searched LSTM at the 14 candidate streams; RBFNN | `run_grid_networks.py`, `analyse_networks.py`, `make_network_tables.py`, `run_fair_classical.py`, `run_fair_lstm.py`, `analyse_fair.py`, `make_fair_tables.py`, `run_fair_classical2.py`, `run_lstm_pilot.py`, `select_lstm_candidates.py`, `run_lstm_final.py`, `analyse_fair2.py`, `analyse_fair_W.py`, `run_e2_networks_wallace.py`; checks: `check_fair_v10.py`, `check_fair2_v11.py`, `test_fair_vote.py`, `test_fair_history.py` | `fair_lstm_declared_protocol.md`, `fair_lstm_results.md`, `fair_followup_declared_protocol.md`, `fair_followup_results.md`, `e2_networks_wallace_results.md` |
| 3.4.3 / 4.4.3 Artificially introduced packet loss (Table 4.7) | Isolated, 200 ms and 1 s burst loss at 2–20%; three decision rules | `run_loss_pattern_test.py` | `three_experiments_declared_protocol.md`, `three_experiments_results.md` |
| 3.4.4 / 4.4.4 End-to-end check (Appendix table) | Streams received through the phantom classified with the offline models | `gel_classification_retest.py` | `gel_retest_declared_protocol.md` |
| Figures | Thesis figures from stored results (no model is rerun) | `make_thesis_figures.py`, `make_thesis_figures_W.py`, `make_thesis_figures_v3.py`, `make_fig_frontier_2x2.py`, `ble_link_budget_depth.py`; Fig 2.2 (RPNI formation): `rpni_figures/make_rpni_figure_scientific.py` | — |

A few modules (`noise_injection.py`, `run_noise_round2_tuned_pilot.py`, `save_models.py`, `vaskov_features.py`) are included only because the scripts above import them.

## ESP32 BLE link (`esp32_ble_link/`)

- **`firmware/emg_ble_link/emg_ble_link.ino`:** Arduino firmware flashed on both boards. The PC chooses which board transmits (the emulated implant) and which receives.
- **`ble_link_test.py`, `run_phantom_campaign.py`:** stream each participant's emulated EMG at set data rates and transmit powers, recording RSSI, sequence numbers, checksums and timestamps.
- **`link_timing.py`, `link_timing_analyse.py`:** delay and backlog test (transit and delivery delay, refused sends, projected buffer overflow).
- **`evaluate_received.py`, `link_diagnostics.py`, `rssi_*.py`, `power_cal*.py`, `live_classify.py`:** checks of received data, RSSI and transmit-power calibration.
- **`README.md`, `unpolarised_test_protocol.md`:** hardware setup, test procedures and run log.

## Notes on the analysis

- **When each choice was fixed.** Every experiment has a declared protocol, written before its results were seen. Changes made later are recorded as labelled amendments. The thesis Methods (Section 3, Table 3.1) states which choices were fixed in advance, which were made within the training folds, and which were made on the same data used to report performance.
- **Seeds.** All evaluations use grouped, class-stratified 5-fold cross-validation (`models.compute_cv_folds`) with fixed random seeds.
