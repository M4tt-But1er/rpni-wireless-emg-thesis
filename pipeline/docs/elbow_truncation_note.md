# Note: the earlier bitrate "knee" (10 kbps) comes from an integer truncation (found 9 Nov 2026, during the new grid)

**Status (updated 9 Nov 2026, later the same day): found, checked on the earlier grid's own data, and now APPLIED to the older documents at the user's request** (see "What was changed" at the end). The first version of this note said nothing in older files had been changed; that was true when it was written.

## What happens

`bitrate_pareto.best_ratio_point` finds the elbow of the accuracy-against-bit-rate Pareto frontier on a log10 bit-rate axis by calling `feature_selection.find_elbow`. That function is documented as a Kneedle-style rule (the point of maximum perpendicular
distance from the chord joining the first and last point, on normalised axes), but it ends with `return int(k_grid[elbow_idx])`, written for the integer K grid of the feature-selection curve. When the axis is log10(bit rate), the returned elbow
(for example 4.2) is cut to a whole number (4), and `best_ratio_point` then snaps it to the nearest frontier point. A log10 value of 4.0 is exactly 10 kb/s, which is 250 Hz × 5 channels × 8 bit. So for these frontiers the rule returns the point nearest 10 kb/s
whenever the true elbow lies between 10 kb/s and 100 kb/s (log10 between 4 and 5), whatever the data are.

## Check on the earlier grid's own data (`check_elbow_truncation.py`, output `cache/elbow_truncation_check.txt`)

Same data (`bitrate_grid_full_results.json`, Wallace from the fixed-range rerun), same selection of the best configuration at each bit rate, accuracy as the metric; "as coded" = what was used then, "geometric" = the rule the docstring describes, no truncation:

| Participant, model | As coded (earlier result) | Geometric elbow |
|---|---|---|
| P1 RF | 10 kb/s, 250 Hz × 5 ch × 8 b, 89.1% | 16 kb/s, 250 Hz × 8 ch × 8 b, 94.4% |
| P1 SVM | 10 kb/s, 83.1% | 16 kb/s, 250 Hz × 8 ch × 8 b, 93.9% |
| P2 RF | 10 kb/s, 86.9% | 16 kb/s, 91.3% |
| P2 SVM | 10 kb/s, 81.3% | 16 kb/s, 89.1% |
| S1 RF | 10 kb/s, 81.4% | 10 kb/s (same), 81.4% |
| S1 SVM | 10 kb/s, 78.2% | 16 kb/s, 84.7% |
| S2 RF | 10 kb/s, 56.3% | 16 kb/s, 78.9% |
| S2 SVM | 10 kb/s, 55.5% | 16 kb/s, 76.2% |

The two rules agree in 1 of 8 cases. In the other 7 the geometric elbow is the next point up the frontier (16 kb/s) with 5 to 22 points more accuracy (quick-method numbers; the "as coded" column are the cached tier-1 scores, not the nested winners).
The elbow of a frontier is a geometric trade-off, not an accuracy target, so neither is "wrong" as a trade-off; but the earlier statement that the **elbow rule found 10 kbps for every participant is an artefact of the truncation**, not a property of the data.

## What is and is not affected

Affected statements (the elbow claim, as written): `README.md` line 40; `docs/12_defence_guide.md` lines 30, 369, 441, 461, 493; `docs/14_experiments_since_august.md` lines 112, 126 and 287 (section D, "Winners");
`run_networks_at_bitrate.py` and `run_packet_loss_test.py` describe 250 Hz × 5 ch × 8 bit as "the knee" in comments. `cache/bitrate_grid_winners_results.json` holds real nested results for the 10 kbps configuration; they stay valid as results **for that configuration**, which is not the elbow.

Not affected: the exhaustive grid scores themselves; the table "cheapest configuration reaching 80 / 90 / 95%" in `docs/14` (no elbow rule involved); the main-effect tables (channels, sample rate, bit depth); the networks-at-bitrate and packet-loss results, which are results at specific configurations.

## What the new grid does

`docs/grid_declared_protocol.md`, Amendment 2: the new grid reports **both** elbows, the geometric one as the primary and the as-coded (truncated) one for comparability with the earlier grid, and includes both sets of configurations among the step-3 winners.
`feature_selection.find_elbow` and `bitrate_pareto.py` are not edited (they are earlier files); the new analysis has its own elbow function.

## Same truncation elsewhere: the SNR shoulder (checked, immaterial)

`run_snr_shoulder.py` also calls `find_elbow` on a non-integer axis (achieved SNR in dB), so its stored shoulders are whole dB truncated toward zero. Recomputed from `cache/snr_shoulder_results.json` without the truncation (the stored values reproduce exactly):
P1 −26.13 (stored −26), P2 −25.62 (−25), S1 −30.11 (−30), S2 −22.26 (−22). The largest shift is 0.62 dB, small against a collapse that spans about 15 dB, so the "shoulder at −22 to −30 dB" statements stand. The K elbow of feature selection (integer K grid) is not affected.

## What was changed (9 Nov 2026, after the user said "yes, correct the older docs")

Inline corrections, each marked "corrected 9 Nov" or "Correction of 9 Nov 2026"; the original wording was kept in place where it was a claim to qualify, replaced only where it stated the false claim as fact:
- `README.md`: the data-rate bullet about the "best accuracy-per-bit trade-off" now states that 10 kbps is not the elbow and gives both grids' elbows.
- `docs/12_defence_guide.md`: a banner after the title; sections 0 (extensions), 9.2 (table row), 13 (limitations row on quick-method sweeps), 14 (question 10) and 15 (numbers to remember); notes on claim 3 and question 7 (spectral features), which the wide-feature work qualifies.
- `docs/14_experiments_since_august.md`: a correction box at the start of section D; the "Winners" paragraph; and the sentence in section J that said the 10 kbps knee "stands".
- `esp32_ble_link/README.md`: two labels "the 10 kb/s knee" now say "the 10 kb/s point (formerly called the knee)". The 10 kb/s rate remains a valid test point of the link study (the radio only sees bits per second); only the label was wrong.
- `feature_selection.find_elbow` and `bitrate_pareto.best_ratio_point`: a WARNING in the docstring (comment only; the code and all earlier results are unchanged).

- `docs/13_code_walkthrough_extended.md` section 13: a correction box and a note on `best_ratio_point` (found later; my first list of files missed it).

Left unchanged on purpose: `docs/15_methods_section.tex` and its two siblings (protected). **Line 138 of the tex does describe the log-axis elbow as the way the best configuration was chosen**, so it needs a hand correction; a drop-in replacement is in [suggested_fix_methods_tex_bitrate_elbow.md](suggested_fix_methods_tex_bitrate_elbow.md). The tex's other "elbow" mentions are the Stage-2 K elbow (lines 28, 71) and the SNR shoulder (line 127), which are fine.
Also unchanged: code comments "the knee" in `run_networks_at_bitrate.py` and `run_packet_loss_test.py` (they label the configuration those scripts were run at); result files and logs of the 10 kb/s runs (valid results for that configuration); `docs/04`, `08`, `10`, `11` and `feature_choice_audit.md` (their "elbow" is the Stage-2 K elbow on the integer grid, not affected).
The thesis results chapter (not in this folder) has not been checked.
