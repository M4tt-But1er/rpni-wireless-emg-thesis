# Three experiments of 11 Nov 2026: results

Protocol, declared before any result: [three_experiments_declared_protocol.md](three_experiments_declared_protocol.md). It includes amendment 1, which redirected X2 to packet loss, retransmission and queue times at the user's request, before the stress runs.

In the thesis these are E1 (X1), the delay-and-backlog part of the phantom test (X2) and E6 (X3). The chapters were then rewritten so that the wide set W is used everywhere except where feature sets are compared. Experiments are renumbered in chapter order: E1 baseline; E2 engineered set (old E1); E3 wider sets (old E2); E4 ablation (old E3); E5 grid (old E4); E6 packet loss (new); E7 temporal context (old E5).

## X1. Baseline on the raw stream (30 kHz, all channels, 16 bit, W)

- **Code and output.** Code: `run_baseline_30k.py` (tables + models), `analyse_baseline_30k.py` (declared analysis). Output: `cache/baseline30k/summary.md`; results in `results.json`.
- **Bit rate.** 3.84 Mb/s (8 ch), 5.76 Mb/s (S2). The user's figure of 3.28 Mb/s matches no participant.
- **Accuracy (%).**

| | LDA | SVM | RF | RBFNN default / tuned | LSTM default / tuned | LDA / SVM / RF + vote |
|---|---|---|---|---|---|---|
| P1 | 98.6 | 98.7 | 97.4 | 96.6 / 98.4 | 99.6 / 99.5 | 99.7 / 99.6 / 98.8 |
| P2 | 96.8 | 96.0 | 96.2 | 91.3 / 94.8 | 96.7 / 96.3 | 98.0 / 95.9 / 96.9 |
| S1 | 93.8 | 94.6 | 90.6 | 89.6 / 92.8 | 95.1 / 94.7 | 95.8 / 97.4 / 97.0 |
| S2 | 93.7 | 92.3 | 89.9 | 81.1 / 90.0 | 93.2 / 94.4 | 97.1 / 95.1 / 96.0 |

- **No gain over 1 kHz.** Against the 1 kHz reference (unpaired), LDA, SVM and RF averaged −0.12 points (range −0.9 to +0.6). The first 30-fold reduction costs nothing, because both rates carry the same 100–499 Hz band.
- **Extraction cost.** At 30 kHz, feature extraction costs 4,411 kFLOP per decision (8 ch), against 129 at 1 kHz.
- **Checks.** Window counts are within 0.3 % of the 10 kHz tables. LDA at 30 kHz equals 10 kHz to within 0.2 points.

## X2. Delay and backlog of the link, unpolarised orientation (11 Nov 2026)

- **Code and output.** Code: `esp32_ble_link/link_timing.py` (runs) and `link_timing_analyse.py` (metrics). Table: `esp32_ble_link/results/link_timing_summary_20261111/`. Runs: `results/link_timing_0dBm_20261111_203841` (reference), `link_timing_-24dBm_20261111_204333` and `link_timing_0dBm_20261111_204846` (stress).
- **No loss.** 245,000 packets were sent and none was lost; every stream arrived bit-exact.
- **Signal.** RSSI was −56 to −58 dBm at the 0 dBm label and about −82 dBm at −24 dBm. The link loss of 56–58 dB is about 4 dB more than on 9 Nov, because the receiving antenna was placed again.
- **Up to 768 kb/s, at both labels.** Link transit was 4.6–9.1 ms (median), with a 99th percentile of 7.5–13.0 ms. At most 4.4 % of packets waited an extra 7.5 ms connection interval.
- **960 kb/s.** At 0 dBm, 12 % of sends were refused and the 99th-percentile delay was 62 ms. At −24 dBm, 48 % were refused, the 99th-percentile delay was 0.5 s and the backlog reached 60 kB. A 24 kB buffer would overflow after 43 s.
- **1280 kb/s (over capacity).** The link delivered 995–1023 kb/s. The delay grew by 0.25–0.29 s per second, and a 24 kB buffer would overflow within 0.6–0.7 s.
- **Limits.** Retransmissions are not counted by the ESP32. "Extra intervals" combines retransmission and queueing; a BLE sniffer would be needed to separate them. Each setting was run once.

## X3. Burst against isolated packet loss

- **Code and output.** Code: `run_loss_pattern_test.py` (`--summarise` gives the declared analysis). Output: `cache/loss_pattern/summary.md`; results in `results.json`.
- **Check passed.** The clean condition reproduced the grid exactly in all 8 cells, for LDA, SVM and RF.
- **Per window, bursts are worse than isolated loss at equal loss.**
  - At 500 Hz × 8 bit: −2.5 points (200 ms bursts; 48/48 cells worse) and −2.7 (1 s bursts; 40/48).
  - At 1 kHz × 16 bit: −5.0 and −5.7.
  - This is the opposite of the declared expectation.
- **With the vote.** Short bursts are about equal to isolated loss. Long bursts are 4.8–4.9 points worse.
- **Skipping damaged windows in the vote.** This helps against bursts but harms isolated loss, which damages most windows.

## Re-summaries with W only (nothing re-run)

| Script | Output | What it covers |
|---|---|---|
| `analyse_grid_W.py` | `cache/grid/grid_summary_W.txt` | Main effects, paired channel steps, costs at the reference stream |
| `analyse_fair_W.py` | `cache/grid/fair2/fair_W_summary.md` | E7 on the 14 W winner cells: accuracies, comparisons, time course, costs |
| `make_thesis_figures_W.py` | `docs/thesis_sim/figures/fig_main_effects_W.pdf`, `fig_temporal_W.pdf` | The two W-only figures; the original figures are kept |
