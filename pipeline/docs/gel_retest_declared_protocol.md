# Re-test of the current classifiers on data received through the gel: DECLARED BEFORE ANY RESULT (9 Nov 2026)

User's request (9 Nov 2026): "re-test the ML + NN of the throughput data from the polarised gel run on a sample of segments and confirm they have the same accuracy."

## Question

Do the current models give the same predictions, and therefore the same accuracy, on data that crossed the gel-phantom BLE link as on the same data processed offline? The current models are set W features with LDA, SVM, RF (with and without the causal vote), the tuned RBFNN and the tuned LSTM.

## Data (all from the polarised, aligned-antenna orientation; nothing new is transmitted)

Every gel run that saved its received stream: 20 streams.
- `esp32_ble_link/results/phantom_campaign_20261104_141740`: 0 dBm setting; Vaskov P1, P2 and Wallace S1, S2; 10, 32, 128 and 384 kb/s (16 streams). The 960 kb/s streams were not recorded.
- `esp32_ble_link/results/phantom_campaign_20261104_160046`: -24 dBm setting (the lowest); P1 at 10, 128 and 384 kb/s (3 streams).
- `esp32_ble_link/results/20261104_134557_stream_vaskov-P1_1of12`: 0 dBm pilot, P1 at 128 kb/s (1 stream).

Configurations: 10 kb/s = 250 Hz x 5 ch x 8 bit; 32 = 500 x 8 x 8; 128 = 1000 x 8 x 16; 384 = 3000 x 8 x 16. These give 16 distinct (participant, configuration) cells.

**The sample of segments:** in each stream, the movement segments whose every window lies inside the transmitted 37 to 60 s slice.

## Procedure

1. Decode the sent and received bytes to samples with the stream's own quantiser range. This is the same formula as the grid's quantiser.
2. Take the grid table of the same participant, sample rate and bit depth (`cache/grid/tables`). Use the stream's channel subset and the set-W columns, with the grid's grouped 5-fold CV (`compute_cv_folds`, seed 0).
3. For the sample windows, recompute the set-W features from the decoded sent stream and from the decoded received stream, with the grid's feature function (`wide_features.segment_columns`, whole segments, so the window-change features see the same earlier windows).
4. For each fold:
   - train each model on the training rows of the offline table;
   - predict that fold's test rows;
   - for sample windows in that fold, also predict from the sent and the received features.
   - Every sample window is therefore predicted by a model that never saw its segment.
5. Models, with the settings used in the grid and its step 3:
   - LDA, SVM, RF;
   - LDA, SVM, RF each followed by the causal hard vote over all windows since the onset (`fair_vote.causal_vote`, k = None);
   - the tuned RBFNN (nested, `run_grid_networks.rbfnn_tuned` procedure);
   - the tuned LSTM (three-setting nested tuning, `run_grid_networks.lstm_tuned` procedure). For the LSTM the received features replace the sample segments' test sequences; scaling is fitted on the training fold as usual.

## Checks (reported, whatever the outcome)

- **V-a.** The received bytes equal the sent bytes. This was already recorded at run time as a matched CRC and is rechecked here byte by byte.
- **V-b.** Features from the decoded sent stream equal the grid table's features for the sample windows (largest absolute difference reported). This checks the decoding and the window alignment.
- **V-c.** The overall CV accuracy of LDA, SVM and RF on the offline table equals `cache/grid/grid_results.json` for that cell, where the stream's channels equal the grid's top-N channels.

## Outcome and reading rule

Per stream and model:
- accuracy on the sample windows, offline, from the sent stream and from the received stream;
- the share of sample windows whose received prediction equals the offline prediction.

"Same accuracy" means the received predictions equal the offline predictions on 100 % of sample windows. Any difference is reported with its cause.

The sample accuracies themselves are from one short slice per stream (a few to about a dozen segments). They are noisy and are not the participant's accuracy. Only the offline-against-received comparison is the result. Wallace slices hold a single grip each.

## Amendment 1 (9 Nov 2026, after the smoke test, for cost)

**What the smoke test showed.** It ran one cell: P1, 250 Hz x 5 ch x 8 bit, with 2 streams.
- It took 938 s, about three times the estimate.
- Running the networks on all 16 cells would therefore take about 3 h.
- Its result was 100 % agreement for every model. I have seen this result, so the amendment is not blind to it.

**The reason for the change** is the cost alone. The `--check` run, made before any model, had already shown that the received features are bit-identical to the sent and offline features for all 20 streams.

**Changed:**
- LDA, SVM and RF, with and without the vote, run on all 16 cells (all 20 streams).
- The tuned RBFNN and the tuned LSTM run on one cell per participant, a sample: P1 at 250 Hz x 5 x 8 (the smoke cell, kept), and P2, S1 and S2 at the reference stream, 1000 Hz x 8 x 16.

Everything else is unchanged.

## Cost (estimate, not measured)

16 cells, each with a tuned RBFNN (about 4 min) and a tuned LSTM (about 2 min). That is about 1.5 h of task time, run with 3 workers. Results go to `cache/link/gel_retest/` (one JSON per cell, resumable). Code: `gel_classification_retest.py`.
