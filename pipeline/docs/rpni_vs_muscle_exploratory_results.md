# EXPLORATORY: are RPNI signals more informative or more detailed than muscle signals? (14 Nov 2026)

User request: "run options 1-4" (from the earlier answer on RPNI vs muscle channels), as an exploratory study. Nothing in the thesis, the
grid, the channel-importance results or any existing cache file was changed.

- **Protocol** (written before any result): `docs/rpni_vs_muscle_declared_protocol.md`.
- **Scripts:**
  - `run_rpni_vs_muscle.py`: tests 1, 2 and 4. 1,908 evaluations; log `rpni_vs_muscle.log`.
  - `rpni_signal_quality.py`: test 3, from the raw 30 kHz recordings; log `rpni_signal_quality.log`.
  - `analyse_rpni_vs_muscle.py`: summary.
- **Outputs:** `cache/rpni_vs_muscle/`, containing:
  - `evals.json` and `signal_quality.json`;
  - `summary.md`, with every table per participant and channel;
  - `summary.json`;
  - `fig_rpni_vs_muscle.pdf/png`.
- **Run time:** about 1 h 50 min. Windows slowed the background workers for part of the run.
- **Check:** every single-channel and RPNI-only accuracy at 30 kHz is identical to `cache/channel_importance/evals.json`.
- **Sample:** 13 RPNIs and 23 muscle channels in 4 people. All p-values are indicative only.

## 1. Equal channel count: RPNI channels only vs every equal-size set of muscle channels only (30 kHz, mean of LDA/SVM/RF)

| | RPNIs | RPNI-only | muscle-only mean (range) | RPNI rank | verdict |
|---|---|---|---|---|---|
| P1 | 2 | 87.5 | 78.9 (60.7-91.0) | 2 of 16 | within range |
| P2 | 3 | 88.1 | 88.9 (83.9-91.7) | 9 of 11 | within range |
| S1 | 3 | 81.9 | 80.3 (76.8-82.7) | 2 of 11 | within range |
| S2 | 5 | 85.8 | 75.5 (71.5-78.2) | 1 of 22 | **more** |

At equal channel count, the RPNIs beat every muscle set only in S2.
- In P1 and S1 they are near the top of the muscle range.
- In P2 they are below the muscle mean.

This agrees with the removal test of the channel-importance study.

## 2. Which movements each channel separates (channel alone, per-movement recall)

There is **no consistent specialisation.**
- The rule was: the best RPNI beats the best muscle channel by more than 5 points in most movements of a group, in at least 3 of 4 participants.
- It was met only for thumb and finger flexion in S2, and for no group anywhere else.

How often an RPNI was the best single channel, compared with the RPNIs' share of the channels:

| | RPNI is the best single channel | RPNIs' share of the channels |
|---|---|---|
| P1 | 3 of 12 movements | 2 of 8 |
| P2 | 3 of 12 movements | 3 of 8 |
| S1 | 4 of 13 movements | 3 of 8 |
| S2 | 10 of 12 movements | 5 of 12 |

In S2, Median2 alone is the best channel for 7 movements, for example index flexion (84% against 44% for the best muscle channel).

The median RPNI is the best channel for thumb flexion in P1 and P2. In S1 the best channel for thumb flexion is an ulnar RPNI, and in S2 it is a median RPNI again. These are suggestive, not a pattern that meets the rule.

## 3. Signal quality (raw 30 kHz recordings)

**RPNI signals are not stronger and do not contain more high-frequency content than the muscle signals.**

| Measure (sum over participants of RPNI mean minus muscle mean) | difference | p | participants with RPNI higher |
|---|---|---|---|
| Movement-to-rest ratio, 100-499 Hz | -2.4 dB | 0.65 | 2 of 4 |
| Movement-to-rest ratio, 20 Hz-7 kHz | -7.8 dB | 0.20 | 0 of 4 |
| Ratio for the channel's strongest movement | -22.9 dB | **0.02** | 0 of 4 |
| Movements above 6 dB | +2.5 | 0.34 | 4 of 4 |
| Median frequency of the movement-related power | -4 Hz | 0.95 | 1 of 4 |
| Frequency below which 95 % of that power lies | -77 Hz | 0.54 | 1 of 4 |
| Fraction of that power above 500 Hz | -4 % | 0.43 | 1 of 4 |

- **Peak response:** each RPNI's response to its strongest movement is weaker than a muscle channel's, in all four participants.
- **Breadth:** RPNIs respond above 6 dB to slightly more movements in all four participants, but the difference is small.
- **Spectra:** they do not differ. The movement-related power of both channel types lies mostly below 500 Hz (median frequency 140-340 Hz), and 1-15 % of it lies above 500 Hz.
- **Signal size does not predict usefulness:** S2's most informative channel, Median2, has one of the weakest signals (4.5 dB above rest). What made it valuable was which movements it responded to, not how large its signal was.

## 4. Does the information survive compression? (each channel alone, all grid tables)

Mean loss in points (RPNI minus muscle); positive means RPNIs lose more:

| Loss | P1 | P2 | S1 | S2 | participants > 2 points | combined p (absolute / relative to accuracy above chance) | verdict |
|---|---|---|---|---|---|---|---|
| 1 kHz to 250 Hz (16 bit) | +2.8 | +3.8 | +9.8 | +2.4 | 4 of 4 | 0.07 / **0.007** | finer detail |
| 1 kHz to 500 Hz (16 bit) | +3.7 | +6.1 | +3.6 | -0.2 | 3 of 4 | **0.01** / **0.008** | finer detail |
| 16 to 8 bit (1 kHz) | -2.8 | +2.4 | +0.1 | -0.5 | 1 of 4 | 0.76 / 0.42 | not shown |
| 30 kHz/16 bit to 250 Hz/8 bit | +1.0 | +5.5 | +8.7 | +4.1 | 3 of 4 | 0.07 / **0.02** | finer detail |

**RPNIs lose more accuracy than muscle channels when the sample rate falls below 1 kHz, but not when the bit depth falls.**
- Every stream is band-limited: 100-499 Hz from 1 kHz upwards, 100-249 Hz at 500 Hz, and 100-124 Hz at 250 Hz. So "finer detail" here means the upper part of the 100-499 Hz band.
- Examples:
  - P2: RPNI_median falls from 70.6% to 46.2% at 500 Hz, while the muscle channels fall by about 11 points.
  - S1: the three RPNIs lose 19.5-32.9 points at 250 Hz, against 4.6-24.9 for the muscles.
- Test 3 found no extra high-frequency *power* in RPNIs. So the difference is in where the RPNIs' *discriminating* information lies, not in how much power they have there. This is an interpretation, not tested.

## Overall answer

1. **More informative?** Not as a rule. RPNIs beat an equal number of muscle channels only in S2, because of one median RPNI. Elsewhere they are within the muscle range.
2. **Stronger or richer signals?** No. Their signals are no larger, their response to their strongest movement is smaller, and their spectra are the same as the muscles'.
3. **More detailed?** In one specific sense, yes. More of the RPNIs' discriminating information lies between about 125 and 500 Hz, so they lose more than muscle channels at 250-500 Hz sampling. Bit depth affects both types equally.

## What it could mean for the thesis (nothing edited)

- **Channel order (E5):** this supports and refines the existing conclusion that sample rate matters up to about 1 kHz. The channels that lose most below 1 kHz are, on average, the RPNIs.
  - This is relevant to any design that lowers the rate on some channels.
  - A mixed-rate scheme (RPNIs at 1 kHz, muscles at 500 Hz) was not tested.
- **Discussion sentence:** "Except in S2, the RPNI channels were neither more nor less informative than the same number of muscle channels" holds. Test 1 confirms it at equal channel count.
- **Possible additions to the thesis:**
  - The RPNIs' information was more sensitive to sample rate than the muscles' (test 4).
  - A channel's usefulness was not related to its signal strength (test 3).
- **Limits:**
  - 4 participants and 13 RPNIs.
  - The channel types are confounded with nerve and electrode site.
  - Vaskov and Wallace use different electrodes and amplifiers, so comparisons are made within each participant.
  - p-values are uncorrected and indicative only.
