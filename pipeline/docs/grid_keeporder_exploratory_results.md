# EXPLORATORY: bit-rate grid with the channel keep order (12 Nov 2026)

User request: rerun the grid's channel cells with the keep order of the channel-importance study, as an exploratory study,
replacing nothing. Nothing in `cache/grid/`, the thesis figures or the thesis text was changed.

- Scripts: `run_grid_keeporder.py` (cells), `analyse_grid_keeporder.py` (summary). Log: `grid_keeporder.log`.
- Outputs: `cache/grid_keeporder/results.json` (810 re-run cells), `summary.md` (full tables), `fig_frontier_keeporder.pdf`.
- Re-run cells: 2 and 5 channels for all participants, plus 8 channels for S2 (sets A and W, LDA/SVM/RF, every rate and bit depth).
  Other cells keep the same channel set, so they are identical by construction. Run time 17 min on 12 workers.
- Checks:
  - With the grid order, the script reproduces 20 published cells exactly (three participants, all models).
  - With the grid order, the summary reproduces the published E5 channel losses (5→2: 13.9–17.2; 8→5: 2.1–4.8, S2 18.8;
    S2 12→8: 3.3, published 3.4).
- Caveat: the keep order was chosen on the same participants (30 kHz, set W), as the grid order was. The comparison is therefore
  a sensitivity check of the channel order, not an unbiased accuracy estimate.

## Results (set W)

Mean accuracy by channel count (over rates, bits and models), grid order → keep order:

| | 2 ch | 5 ch | 8 ch | 12 ch |
|---|---|---|---|---|
| P1 | 77.2 → 85.4 | 94.2 → 96.9 | 97.7 | |
| P2 | 79.3 → 78.7 | 93.1 → 92.7 | 95.3 | |
| S1 | 69.7 → 69.9 | 86.9 → 88.1 | 91.7 | |
| S2 | 51.1 → 73.7 | 67.8 → 85.9 | 86.6 → 89.8 | 89.4 |

Paired loss (points) when removing channels, grid → keep:

| | 5 → 2 | 8 → 5 | 12 → 8 |
|---|---|---|---|
| P1 | 17.0 → 11.5 | 3.5 → 0.8 | |
| P2 | 13.9 → 13.9 | 2.1 → 2.6 | |
| S1 | 17.2 → 18.1 | 4.8 → 3.6 | |
| S2 | 16.7 → 12.2 | 18.8 → 3.9 | 3.3 → 0.1 |

## What this means

1. **S2 changes a lot.**
   - The large S2 losses in the grid were an artefact of the channel order: its best channel, Median2 (RPNI), was 8th.
   - With Median2 kept, 5 channels lose 3.9 points instead of 18.8. 8 channels match all 12 (89.8 vs 89.4).
   - S2 reaches 90 % at 32 kb/s (500 Hz, 8 channels, 8 bits, SVM) instead of 48 kb/s with all 12 channels.
   - The best accuracy at ≤32 kb/s rises from 86.7 to 90.3 %.
2. **P1 improves at low channel counts** (2 ch +8.2, 5 ch +2.7 points). Its 90 % and 95 % thresholds barely move (RF 95 %: 20 instead of 32 kb/s).
3. **P2 and S1 do not improve.**
   - Means are within ±1.2 points.
   - At the lowest budgets (≤4 kb/s, 2 channels) the keep order is worse, by 12–14 points, because the last two channels of a
     greedy backward elimination at 30 kHz are not the best pair at 250 Hz and 8 bits.
   - The keep order is not uniformly better. Its main value is in not dropping a channel that carries unique information.
4. **The main E5 conclusion stands:** channel count still costs far more than sample rate or bit depth. Going from 5 to 2 channels
   still costs 11.5–18.1 points with the keep order.
5. **Thesis statements that the exploratory result contradicts or qualifies (not edited):**
   - "S2 needed all 12 channels (48 kb/s at 500 Hz)" and "8 to 5 channels cost 18.8 points for S2, which also lost 3.4 between 12 and 8":
     both depend on the channel order. With Median2 kept, 8 channels suffice for S2 and 90 % is reached at 32 kb/s.
   - "S2, which needed 8–12 channels, illustrates that this dependence varies between individuals" (Discussion): rather, S2 depends on
     one specific channel.
   - The limitation sentence on the channel ranking can be made concrete with these numbers.

Not changed by this study: the grid's winner cells, E6 and E7 (all run at the published winners), and all figures and tables in the thesis.
