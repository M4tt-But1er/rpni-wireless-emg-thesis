# Channel importance: results (12 Nov 2026)

Protocol: `channel_importance_declared_protocol.md` (written before the run). Full tables: `cache/channel_importance/summary.md`.
Figure: `docs/thesis_sim/figures/fig_channel_importance.pdf` (accuracy against channels kept, backward elimination vs grid order).
Scripts: `run_channel_importance.py`, `analyse_channel_importance.py`. Check passed: the all-channel subsets reproduce the E1 values exactly.

## Findings

1. **Mostly redundancy, with one exception.** In P1, P2 and S1, removing any single channel costs at most 1.7 points (mean of LDA, SVM, RF),
   and most losses are within the fold noise. No channel in these three participants is indispensable. Resolved losses: P1 FDP_small
   (+1.7 ± 1.6); S1 FCR (+1.5 ± 0.9), Ulnar2 (+0.9 ± 0.6), FPL (+0.9 ± 0.5), Ulnar1 (+0.8 ± 0.4); S2 Supinator (+0.6 ± 0.6).
   **The exception is S2's Median2 RPNI:** removing it alone costs 6.4 ± 3.8 points, with all three models agreeing (+4.6 to +7.6).
2. **RPNI vs muscle.** In P1, P2 and S1, removing all RPNI channels costs about as much as removing the same number of muscle channels
   (inside the control range). In S2 it costs 11.4 points against 1.7–3.9 for any equal-size set of muscle channels, so the S2 RPNIs
   are more informative, mainly through Median2. The RPNIs rank first in the keep order of S1 (Median1, Ulnar2) and S2 (Median2), and
   second and third in P2.
3. **How well a channel does alone is a poor guide to whether to keep it.** Examples: S1 FCR scores 23.7 % alone but costs a resolved 1.5 points when removed; P1
   FDP_index scores 57.4 % alone but is almost fully redundant (+0.2). The grid's channel order ranks channels by their best single
   feature (mutual information), i.e. by stand-alone relevance, and so does not capture what a channel adds to the others.
4. **The grid's channel order was suboptimal, most of all for S2.** At equal channel counts, the grid order is up to 10 points below the
   backward-elimination order in P1, P2 and S1 (mostly 1–4), and 13–21 points below in S2 at 1–7 channels, because Median2 is 8th in the grid order.
   Adding Median2 as the 8th grid channel raises S2 from 77.1 to 89.2 %. The backward-elimination curve is optimistic (chosen on the
   scoring folds), but that bias is far smaller than these gaps.
5. **About 5–6 channels suffice** (descriptive, optimistic): within 1 point of all channels with 5 (P1, P2) or 6 (S1, S2) channels in the
   backward-elimination order.

Keep orders (first = most important to keep):
- P1: FDP_small, FPL, FCR, EDC, RPNI_ulnar, EPL, FDP_index, RPNI_median
- P2: FPL, RPNI_median, RPNI_ulnar2, EDC, FCR, FDP_index, EPL, RPNI_ulnar1
- S1: Median1, Ulnar2, EDC, FCR, FPL, EPL, Ulnar1, FDPi
- S2: Median2, FDPs, EDC, FDPi, Median1, Supinator, Median3, EPL, Pronator, FPL, Median4, Radial

No muscle recurs at the top across all participants. FPL and EDC are in the top five in three of the four.

## Consequences for the thesis text (not edited)

- Results E5 and Discussion: "reducing from 5 to 2 channels cost 13–17 points" and "S2 needed all 12 channels / 8–12 channels" were
  measured with the grid order. On the 30 kHz stream the 5→2 loss is 11–15 points in the grid order but 7–14 points in the
  backward-elimination order (optimistic), and S2's need for 8+ channels is
  mostly an artefact of Median2 being 8th. **Channel count still matters far more than sample rate or bit depth**, so the main conclusion
  stands, but the channel losses are overstated, especially for S2.
- The Discussion's limitation about the channel ranking can now be made concrete: the ranking used stand-alone relevance, and an
  order based on accuracy keeps the same accuracy with fewer channels.
- The Discussion's claim that RPNIs and muscle electrodes give complementary information is supported. A stronger claim that RPNIs are
  more informative holds only for S2.

Not done (would need its own declared protocol): a fully nested estimate of the accuracy of a channel subset chosen on training data
only; rerunning the bit-rate grid's channel axis with the new order.
