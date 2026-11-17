# Gel phantom, unpolarised (worst-case) orientation: protocol and commands

Written 9 Nov 2026, before the test, at the user's request. The aim is to repeat the polarised measurements (README sections 13.2, 13.5, 13.6) with the antennas in the worst-case orientation, so that the two sets of results compare one to one.

**Scope.** This is the worst case for *this antenna pair* (Seeed A-02, flat) only. Antenna selection matters for an implant, but it is not reviewed in this thesis.

## Setup

- **Phantom and implanted antenna:** the same as before, untouched. The transmitter (COM5) sits between the two gel blocks, 3.5 cm below the top surface.
- **Receiving antenna (COM6):** flat on the top surface, directly above the implanted antenna as before, but **turned 90 degrees in the horizontal plane**. Its long axis is then at right angles to the implanted antenna's long axis (cross-polarised).
- Write down exactly how it was turned and take a photo, for the thesis figure.
- Keep the cables and the boards where they were. Only the receiving antenna moves.
- Check that both U.FL connectors are pressed fully home.

## Measurements (same as the polarised runs)

Run these in PowerShell in `esp32_ble_link`. Put the same note on each, so the runs are labelled in the results.

```
$note = "gel phantom, UNPOLARISED: RX antenna turned 90 deg on the surface above the implanted TX (3.5 cm gel), A-02 antennas"
```

1. **RSSI check, 60 s at 0 dBm.** This gives a stable reading before anything else.
```
python rssi_monitor.py --tx-port COM5 --rx-port COM6 --txpwr 0 --seconds 60 --note $note
```
2. **Power sweep:** 5 s bursts at six settings, as on 4 Nov (README 13.2).
```
python ble_link_test.py --tx-port COM5 --rx-port COM6 --mode burst --burst-seconds 5 --txpwr=9,0,-6,-12,-18,-24 --note $note
```
3. **Streaming campaign at 0 dBm:** four datasets at 10, 32, 128, 384 and 960 kb/s, 60 s each (README 13.5). It takes about 25 min.
```
python run_phantom_campaign.py --tx-port COM5 --rx-port COM6 --txpwr 0 --note $note
```
4. **Lowest power:** P1 at 10, 128, 384 and 960 kb/s, 30 s each (README 13.6).
```
python run_phantom_campaign.py --tx-port COM5 --rx-port COM6 --txpwr=-24 --sources vaskov:P1_1of12 --rates 10,128,384,960 --seconds 30 --note $note
```

If the boards do not connect at a setting, the sweep records it in `failed_to_connect.json` and carries on. That is a result: note the setting.

## Analysis (declared now)

- **PER, RSSI and link loss per setting:** the same table as the polarised case (thesis Table "Gel phantom, polarised orientation"). The PER bound uses the rule of three.
- **Orientation loss:** the difference in RSSI between the unpolarised and the polarised runs at each setting.
- **Classification:** the received streams are re-tested with `pipeline/gel_classification_retest.py` (its list of folders is extended to the new runs). The reading rule is the same: "same accuracy" means the received predictions equal the offline predictions on 100 % of sample windows.

**Expectation, stated before the test.**
- Turning a linearly polarised antenna through 90 degrees typically costs a further 10 to 25 dB or more.
- At 0 dBm the polarised link had 55 to 60 dB of margin, so no loss is expected there.
- At -24 dBm the margin was about 30 dB, so a large orientation loss could bring the signal close to the PER cliff (about -90 dBm for 2M). That is the condition this test is designed to find.
