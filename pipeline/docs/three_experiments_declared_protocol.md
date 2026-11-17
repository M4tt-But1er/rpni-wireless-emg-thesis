# Three experiments requested on 11 Nov 2026: declared protocol

Written before any of the three was run. Requested by the user on 11 Nov 2026:
1. "the initial baseline was meant to be with 30kHz sampling = 3.28Mbs, not 1kHz and on the wide feature set";
2. "measure time for lost packets / backlog under unpolarised conditions for discussion";
3. "whether burst loss matters more than an equivalent amount of isolated loss".
The user also asked that the wide feature set W be used for everything, except where feature sets themselves are compared (the feature-optimisation results, including how the LDA/SVM ranking depends on the features).

**Note on the bit rate.** 30 kHz × 8 channels × 16 bit = **3.84 Mb/s** (S2, 12 channels: 5.76 Mb/s). The figure 3.28 Mb/s in the request does not match any participant; 3.84 Mb/s is used and the discrepancy is reported to the user.

---

## X1. Baseline on the full-rate stream (30 kHz) with the wide set W

**Question.** How accurately can the movements be classified from the raw stream, before any reduction, with the receiver's wide feature set? This replaces the old E1 baseline (1 kHz, engineered set) as the reference for every later loss. The old E1 stays in the thesis as part of the feature-optimisation results.

**Stream.** 30 kHz, all implanted channels (8 for P1, P2 and S1, 12 for S2), 16 bit. The grid's own signal path (`grid_tables.decimated_sources` and `build_table`) with rate 30000. Band-pass 100 to 499 Hz, zero-phase (the same band as every grid rate of 1 kHz or more), no decimation, the grid's quantiser (Vaskov: range per channel per raw file; Wallace: one range per channel pooled over the participant's runs). Onset trimming and windows (200 ms, 50 ms hop = 6000 and 1500 samples) exactly as the grid does at its own rates. This is the grid's procedure at one more rate, so the baseline is directly comparable with the grid cells.

**Features.** Set W (`cache/wide_set_W.json`: amplitude, shape and complexity, sub-band power, window change, pair correlation), all channels, no selection.

**Models and folds.** The grid's grouped 5-fold cross-validation (`compute_cv_folds`, seed 0).
- LDA, SVM, RF with the grid's fixed settings (`run_grid.fit_predict_info`);
- RBFNN with its default setting (64 centres, C = 1) and tuned (nested, `run_grid_networks.rbfnn_tuned`);
- LSTM default (40 epochs) and tuned (`run_grid_networks.lstm_default`, `lstm_tuned`).
Secondary: LDA, SVM and RF with the causal hard vote over all decisions since the onset (`fair_vote.causal_vote`, k = all). This span is fixed, not chosen by inner cross-validation, because the inner choice picked it in 92 % of folds in E5.

**Analysis.**
- Accuracy, weighted F1 and Cohen's kappa per model and participant.
- Paired fold-wise difference between the two best per-window models (95 % t-interval, 4 df), read with the thesis rule.
- Comparison with the 1 kHz reference (1 kHz, all channels, 16 bit, W) of the grid and of the network runs at the winner cells. The two rates produce slightly different windows (onset detection runs at each rate), so this comparison is unpaired: pooled accuracies are compared, with the fold spread.
- Extraction cost per decision at 30 kHz (`feature_cost.extraction_cost`).

**Checks.** Window counts within 2 % of the 10 kHz table of the same participant. LDA at 30 kHz is reported next to 10 kHz and 1 kHz (same bit depth, W, all channels) as a plausibility check. Above 1 kHz the band is the same, so large differences are not expected.

---

## X2. Delay and backlog of the link in the unpolarised (worst-case) orientation

**Question.** In the unpolarised orientation the link refused send attempts and fell behind real time at 960 kb/s, and its maximum throughput fell with the transmit setting. How much delay do retransmissions and backlog add to the data, at the rates of interest and at the rate where the link struggled? No packet has been lost so far, so "time for lost packets" is measured as the delay that retransmission and queueing add.

**Setup.** The phantom, unpolarised: the receiving antenna turned 90° on the surface above the implanted transmitter (3.5 cm gel), as on 9 Nov 2026. XIAO ESP32-S3 boards, firmware 1.2, 2M PHY, 7.5 ms connection interval, 240-byte payloads (as in the campaign). The user sets up the hardware and confirms the orientation before the run.

**Runs.** Vaskov P1 slice (`vaskov:P1_1of12`), 60 s each, streamed in real time: the PC acts as the sensor and hands each frame to USB at its production time, never earlier (as `live_classify.py`). Rates 32, 128, 384 and 960 kb/s, at the 0 dBm and -24 dBm labels: 8 runs. No classifier runs during the stream, so the PC's compute cannot add to the timing.

**Measured per packet (one PC clock, `perf_counter`).**
- `t_src`: when the last byte of the packet was produced by the sensor (ideal sensor clock);
- `t_fed`: when that byte was handed to the transmitter's USB port;
- `t_rx`: when the receiver delivered the packet to the PC.
From these: **total delay** = t_rx - t_src (packet fill + backlog + link); **link transit** = t_rx - t_fed (USB + BLE, including link-layer retransmissions and the transmitter's queue); **sensor backlog** = t_fed - t_src (data that was due but could not be handed over because the transmitter's buffer was full).

**Reported per run.**
- Median, p95, p99 and maximum of total delay and link transit.
- Backlog over time: the slope of a straight-line fit of total delay against time (ms per s), and the delay in the last 5 s.
- Stalls: receiver inter-arrival gaps longer than 3 connection intervals (22.5 ms): their number, total time and longest.
- Refused send attempts (transmitter buffer full), packets lost, RSSI and link loss, integrity (three-way CRC).
- Throughput.

**Reading.** Descriptive; one run per setting. The rig's USB hops add under 1 ms (measured by PING in every run, as before). A link that keeps up has a total delay that does not grow over the run (slope near 0). A run that falls behind shows a positive slope, and the delay at the end is the backlog.

**Limits stated in advance.** One run per setting, one geometry. The rig's sensor is ideal and the PC holds data the transmitter cannot accept. In an implant that data would sit in the implant's buffer, so the total delay is the right measure of backlog, wherever it sits.

**Amendment 1 (11 Nov 2026, during the 0 dBm run, before any stress-run result).** The user clarified the aim: "I want to find out about packet loss, retransmission, queue times". The rates 32 to 384 kb/s do not load the link, so the 0 dBm runs at 32, 128, 384 and 960 kb/s are kept only as the no-load reference. A **stress set** is added that drives the link to and beyond its capacity: 640, 768, 960 and 1280 kb/s (5, 6, 7.5 and 10 kHz × 8 ch × 16 bit), 60 s each, at the −24 dBm label (the weakest; the unpolarised capacity there was 867 kb/s) and at 0 dBm. The −24 dBm runs at 32 to 384 kb/s are dropped.

Changes to the measures:
- **Retransmission and queue time.** The ESP32 does not report link-layer retransmissions; only a BLE sniffer could count them. Each retransmission or wait for a later connection event adds at least one connection interval (7.5 ms). The excess link transit above the run's fastest 5 % of packets is therefore reported in connection intervals: the share of packets delayed by at least 1, 2 and 4 intervals, and the most. This is retransmission and transmitter queueing combined; the two cannot be separated without a sniffer.
- **Queue (backlog).** Data produced by the sensor but not yet at the PC, at every packet arrival, in kB: median, maximum, final value, and the time at which it first exceeds 4, 8, 16, 24 and 64 kB. An implant with a buffer of that size would start to lose data at that moment. This is the packet-loss mechanism of a connected BLE link, because the link layer itself does not drop packets. Loss can also occur if the connection drops (2 s supervision timeout); that is reported if it happens.
- **Stalls** are redefined as receiver gaps longer than the packet spacing by more than 3 connection intervals. The first definition counted every packet at low rates.

All runs, including the 0 dBm reference, are analysed by the same code, `esp32_ble_link/link_timing_analyse.py`, from each run's `packets.csv`.

---

## X3. Burst loss against an equal amount of isolated loss

**Question.** At the same share of lost packets, does losing them in bursts cost more classification accuracy than losing them one at a time?

**Operating points (all channels, set W).**
- the recommended stream, 500 Hz × 8 bit (32 kb/s for 8 channels, 48 kb/s for S2);
- the reference stream, 1 kHz × 16 bit (128 / 192 kb/s).
The grid's signal path and feature function are used.

**Packets.** 240-byte payloads, samples packed frame by frame (all channels of one instant together), packets numbered from the start of each recording file or run, which is one transmission. One packet holds 30 frames (60 ms) at 500 Hz × 8 ch × 8 bit, 20 frames (40 ms) for S2. At 1 kHz × 16 bit it holds 15 frames (15 ms), 10 frames (10 ms) for S2.

**Loss patterns.** The number of lost packets is set per transmission as K = round(p × n_packets), with p ∈ {2, 5, 10, 20} %.
- **Isolated:** K packets, no two adjacent (each lost packet has received packets on both sides).
- **Short bursts:** runs of L = max(2, round(200 ms / packet duration)) consecutive packets, round(K / L) runs. That is 3 packets (180 ms) at 500 Hz, 5 (200 ms) for S2, and 13 (195 ms) or 20 (200 ms) at 1 kHz.
- **Long bursts:** as above with 1 s: L = round(1000 ms / packet duration). That is 17 packets (1.02 s) or 25 (1.0 s) at 500 Hz, and 67 (1.005 s) or 100 (1.0 s) at 1 kHz.
Runs are placed uniformly at random, never overlapping or touching. Burst patterns can lose slightly more or fewer packets than K because of rounding; the realised loss is reported. Three random placements (repeats) per pattern and level, with fixed seeds.

**Receiver.** It knows which packets are missing (sequence numbers) and fills their samples with zeros (the filtered stream has zero mean). This is the same rule as `evaluate_received.decode(..., "zero")`. Segments and windows are taken from the clean signal, so every condition has the same rows. Only sample values inside lost packets change. Features are recomputed for every segment that contains a lost sample. The other segments keep their clean features, which are identical by construction.

**Models.** LDA, SVM and RF, fixed settings, trained on the CLEAN training folds (grid folds) and tested on the lossy held-out fold. The model never sees loss during training. This is the harsh case, and the realistic one for a model calibrated on a good link.
- Per-window decisions (primary);
- the causal hard vote over all decisions since the onset (secondary);
- the same vote with damaged windows skipped (secondary): the receiver knows which windows contain lost samples, leaves them out of the vote, and keeps showing the current vote. A window with no intact decision yet in its segment counts as an error.

A window is **damaged** if any of its W features differ from the clean ones (exact comparison). This includes windows whose window-change features use an earlier damaged window.

**Primary analysis.** Per participant, operating point, model and loss level: accuracy averaged over the 3 repeats for isolated, short-burst and long-burst loss. The differences short − isolated and long − isolated are paired by fold (fold-wise accuracy averaged over repeats, 95 % t-interval over 5 folds). The thesis rule is used: a mean below 1 point is "about equal"; otherwise better or worse if the interval excludes zero in at least a third of the cells (participant × model × level), else "unresolved". The number of cells in which bursts are worse is always given.

**Secondary.** The same comparison with the vote and with vote-and-skip. The share of damaged windows. Accuracy on undamaged windows.

**Check.** The clean condition must reproduce the grid's accuracy for the same cell exactly (LDA, SVM, RF, W, all channels).

**Expectation, stated before running.** Without a vote, isolated loss may cost MORE than bursts at the same loss. Each isolated 60 ms gap damages every overlapping 200 ms window (up to 5) and, through the window-change features, the next two windows. A burst damages fewer windows for the same number of lost samples. With a vote, bursts may cost more, because they corrupt consecutive decisions that a vote cannot outvote. Either result is reported.
