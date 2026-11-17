# EXPLORATORY: E4 family ablation with Random Forest (13 Nov 2026)

User request: run RF for E4 as an exploratory study, replacing nothing. The published LDA/SVM ablation
(`cache/family_ablation_results.json`), the wide set W, the thesis text and the figures are unchanged.

- Scripts: `family_ablation_rf.py` (cells), `analyse_family_ablation_rf.py` (summary). Log: `family_ablation_rf.log`.
- Outputs: `cache/family_ablation_rf/results.json` (224 RF cells) and `summary.md` (full tables).
- Run time: 10 min on 8 workers (81 core-minutes).
- Checks:
  - The script reproduces two published LDA cells exactly.
  - RF sets A and C equal the Stage-1 RF cells in all 16 cases, so RF on 2 threads gives the same trees as on all cores.

## Results (mean over the 8 tables = 4 participants x 1 kHz reference and 500 Hz/8-bit low streams)

| | LDA | SVM | RF |
|---|---|---|---|
| Set A (192 features for 8 channels) | 93.1 | 93.0 | 92.9 |
| Set C (632 features) | 95.3 | 95.0 | 93.6 |
| Gain from A to C | +2.2 | +2.0 | +0.7 |

Adding one family to A (points):

| Family | LDA | SVM | RF |
|---|---|---|---|
| Sub-band power | +0.93 | +1.14 | +0.09 |
| Wavelet | +0.61 | +0.99 | -0.18 |
| Pairwise cross-correlation | +0.95 | +0.76 | +0.23 |
| Pairwise band correlation | +0.70 | +0.77 | +0.33 |

Adding a whole branch to A (points):

| Branch | LDA | SVM | RF |
|---|---|---|---|
| Frequency domain | +1.05 | +1.48 | +0.11 |
| Cross-channel | +0.88 | +0.88 | +0.71 |

## What this means

1. **RF gains little from the wider feature sets.** RF gains +0.7 from A to C, against +2.0 to +2.2 for LDA and SVM.
   - No single family adds 0.5 points or more for RF, and dropping any family changes RF by at most 0.5 points.
   - The only consistent gain for RF is the cross-channel branch (+0.71, resolved in 2 of 8 tables).
2. **Amplitude features slightly hurt RF.**
   - Dropping the amplitude family from C *raises* RF by 0.47 (resolved in 2 tables).
   - Adding amplitude to A lowers RF by 0.19.
   - A likely reason is that set A already holds RMS-type features, so the extra amplitude columns are redundant and dilute RF's random feature choice. This is a hypothesis, not tested.
3. **The spectral families help LDA and SVM, not RF.** Sub-band power and wavelet are the families LDA and SVM benefit from most.
   For RF they are neutral, probably because a tree ensemble can already build non-linear combinations of the set-A features.
4. **Consequences for the thesis (nothing edited):**
   - E4's conclusion ("no single family was essential once the others were present") also holds for RF.
   - The statement that wider sets raised accuracy for every model holds for RF, but its gain is small.
   - W was chosen by an LDA-only rule. RF's indifference to most families means W is not a disadvantage for RF, but it is not where RF's
     accuracy comes from either. The Methods' reason for leaving RF out of E4 (cost) can be complemented by this result if the user wishes.
