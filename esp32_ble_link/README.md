# ESP32 BLE link tester

Two ESP32 boards, both plugged into this PC, connect to each other over Bluetooth LE. One plays the implant
(transmitter), the other the external receiver. The PC feeds the transmitter a stream built from the Vaskov or
Wallace recordings, and reports throughput, lost and retried packets, transfer time, RSSI and link loss.

Your boards (detected 3 Nov 2026): **COM3** and **COM4**, both ESP32-C3 (BLE 5, 2M PHY, up to +20 dBm). If Windows
renumbers them, run `python ble_link_test.py --list-ports`.

**Since 4 Nov 2026 the boards in use are two Seeed Studio XIAO ESP32-S3 with their supplied external U.FL antennas, on COM5 and COM6**
(section 12). Everything from sections 1 to 11 was measured on the ESP32-C3 pair (COM3, COM4) and stays valid for that hardware; the
C3 power calibration (section 11) does not apply to the S3 boards.

```
esp32_ble_link/
  firmware/emg_ble_link/emg_ble_link.ino   same firmware for BOTH boards (role is chosen by the PC at run time)
  ble_link_test.py                         PC script: controls both boards, sends the data, prints and saves results
  evaluate_received.py                     feeds the recorded received samples back into the classifiers
  rssi_monitor.py                          live RSSI while you move or rotate the boards (finds what dominates the link loss)
  rssi_segments.py                         stable reading per labelled position from a monitor CSV (the 10 s after each label)
  rssi_distance.py                         link loss against distance and orientation from a monitor CSV (labels like 30cm-r90)
  run_phantom_campaign.py                  datasets x bitrates through the link, then classify what arrives (section 13)
  live_classify.py                         real-time stream, classify each window as it arrives, measure the latency of each decision (section 14)
  link_diagnostics.py                      fixed checks on the measurement method itself (section 10)
  power_calibration.py                     measures the real power staircase of the transmit-power labels (section 11)
  power_cal.py                             applies a calibration: effective power, and the label that gives a wanted power
                                           (a table belongs to one board; the scripts warn if the transmitter's MAC differs)
  requirements.txt                         pyserial, numpy, matplotlib, bleak
  results/                                 one folder per run (created automatically)
```

## 1. Flash the firmware (once per board)

Arduino IDE 2.3 (already installed, ESP32 core 3.3.12 already installed):

1. Open `firmware/emg_ble_link/emg_ble_link.ino`.
2. Tools > Board > esp32 > **ESP32C3 Dev Module**.
3. Tools > **USB CDC On Boot > Enabled** (without this the sketch talks on the wrong serial port and the PC gets no reply).
4. Tools > Port > **COM3**, click Upload.
5. Tools > Port > **COM4**, click Upload again (same sketch).
6. Close the IDE's Serial Monitor on both ports before running the script.

If an upload cannot start: hold the board's BOOT button, tap RESET, release BOOT, and upload again (the COM number may change while it is in bootloader mode).

The sketch was compile-tested for the ESP32-C3 with `arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc`.
Both boards were flashed and the burst test was run on 3 Nov 2026 (section 8).

## 2. Run both boards at the same time

Each board is its own COM port, so there is nothing special to do: **one command opens both ports** (one reader
thread per port) and drives them together.

```
pip install -r requirements.txt
python ble_link_test.py --list-ports
python ble_link_test.py --tx-port COM3 --rx-port COM4 --source vaskov:P1_1of12 --seconds 30
```

Only one program can hold a COM port at a time, so keep the Arduino Serial Monitor (and anything else) closed while it runs.
To test the other direction, swap the two port names.

### Useful runs

| Goal | Command |
|---|---|
| Maximum BLE throughput (synthetic data, 10 s) | `--mode burst --burst-seconds 10` |
| Vaskov data at the thesis configuration (128 kb/s), real time | `--source vaskov:P1_1of12 --fs 1000 --channels 8 --bits 16 --seconds 30 --realtime` |
| The 10 kb/s point (called "the knee" before 9 Nov 2026; that was an elbow-rule artefact, see pipeline/docs/elbow_truncation_note.md; the geometric elbow is about 16 kb/s, 250 Hz x 8 ch x 8 bit) | `--source vaskov:P2_1of10 --fs 250 --channels 5 --bits 8 --realtime` |
| Wallace data | `--source wallace:S1` (or `wallace:S2`, optional `:task` index) |
| Transmit power sweep (link loss vs power) | `--mode burst --txpwr=-24,-12,0,9,20` |
| 1M instead of 2M PHY | `--phy 1` |
| Longer connection interval | `--ci 30` |
| Repeat a run | `--repeat 5` |

Without `--realtime` the data is sent as fast as the link accepts it, which measures how long the transfer takes.
With `--realtime` it is paced at the sampling rate, which shows whether the link keeps up with the sensor.
Loading a Vaskov CSV takes 20 to 80 s before the test starts.

## 3. What is reported

| Statistic | Where it comes from |
|---|---|
| **Throughput** (kb/s, kB/s) | payload bytes received, divided by first-to-last-packet time on the receiver's clock |
| **Transfer time** | receiver first-to-last packet; also the host wall time including the USB feed |
| **Packets sent / received / lost** | each notification carries a sequence number; the receiver counts gaps |
| **Duplicates, out-of-order, bad payload** | same sequence check; bad payload = burst pattern mismatch |
| **Retries** | notifications the transmitter's BLE stack refused (buffer full) and that were sent again |
| **Inter-arrival gaps** | histogram and maximum gap at the receiver (jitter / stalls) |
| **TX power** | the setting you asked for and the level the radio applied (3 dB steps, -24 to +20 dBm) |
| **RSSI** | measured once a second on both boards (idle and during transfer) |
| **Link loss (dB)** | TX power minus mean RSSI at the receiver; includes both antennas, mismatch and the boards' own errors |
| **Margin (dB)** | mean RSSI minus an assumed sensitivity (`--sensitivity`, default -94 dBm) |
| **Integrity** | CRC32 of the whole stream at the host, transmitter and receiver must match |

Each run writes `summary.json`, one `timeseries_*.csv` per power level, `sweep.csv` and `throughput_rssi.png` (for
sweeps), and the raw `tx_serial.log` / `rx_serial.log` in `results/<timestamp>_.../`.

## 4. Limits you should know

* **Over-the-air retransmissions are invisible.** BLE retransmits at the link layer until a packet is acknowledged, and
  the ESP32 does not expose that count. "Lost" therefore means lost after those retransmissions (it should be 0 unless
  the connection drops or the receiver is overloaded). "Retries" are the transmitter's software buffer being full, which
  is a sign the link is the bottleneck. To count real retransmissions you need a BLE sniffer.
* **Transmit power is a setting, not a measurement.** The firmware sets the chip's power level; the radiated power also
  depends on the board's antenna. The RSSI-based link loss at 10 to 30 cm is mostly antenna and near-field effects, so
  it is only comparable between runs on the same desk. Increase the distance or lower the power to see losses.
* **RSSI is a coarse reading** (about 1 dB steps, a few dB of error) and the receiver front end saturates at high input
  levels, so use low power or distance if the readings sit near the top of the range.
* **Latency** is not reported: the two boards have unsynchronised clocks and USB adds its own delay.
* **The USB feed is not the limit** unless you ask for more than a few hundred kb/s in real-time mode (USB full speed carries
  far more, but the transmitter buffer is 24 kB).
* Bits are packed sample-major (all channels at one instant, then the next instant), not aligned to packet boundaries.
  12-bit samples are packed two per three bytes.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `does not answer PING` | firmware not flashed, or USB CDC On Boot was not Enabled when flashing |
| `could not open port` | another program (Serial Monitor) has the port open |
| Boards do not connect | keep them within a metre, make sure only one script is running, retry |
| COM numbers changed | `python ble_link_test.py --list-ports` |
| Many `EV LINKSETUP ... rc` errors | the PHY/data-length request was refused; the link still works at 1M |
| `the board REBOOTED during the run` | the firmware crashed and restarted (it then has no role). Firmware 1.0 did this after repeated scans (section 11.1); flash 1.1. The crash dump is in the run's `*_serial.log` |

## 6. The received data is recorded, and can be fed back into the classifiers

In stream mode the receiver board forwards every payload it gets to the PC (`B <seq> <len>` records on its USB serial
port), and the script saves, in `results/<run>/`:

| File | Content |
|---|---|
| `sent_stream.bin` | exactly the bytes sent (bit-packed samples, all channels interleaved) |
| `received_stream_<P>dBm.bin` | the bytes the receiver delivered (zeros where a packet never arrived) |
| `received_missing_<P>dBm.npy` | per-byte mask of what is missing |
| `stream_meta.json` | sample rate, channels (names), bits, quantiser range per channel, where the slice sits in the recording |

Then:

```
python evaluate_received.py results/<run folder>
```

It decodes both streams, trains RF and SVM with the thesis pipeline at the same configuration on the rest of the
dataset (every movement segment that overlaps the transmitted slice is left out), and classifies the slice three ways:
`offline` (the sweep pipeline), `sent` (my decoder on the sent bytes) and `received` (the link's output). `sent` must equal
`offline`; `received` differs from `sent` only through packets the link lost. It also reports accuracy on windows touched by
lost data versus intact windows.

The samples are produced exactly as in the sweeps: the whole recording is band-pass filtered at 30 kHz, decimated and
quantised with the same percentile range (Wallace: the same pooled fixed range), and the transmitted slice is cut out of that.
Use `--no-record` to skip the recording (burst mode never records).

A 30 s slice holds only a handful of movement segments (about 190 windows in 6 segments for Vaskov P1), so accuracy from one
slice is noisy: check the `sent = received` agreement, and use several slices or longer ones for accuracy numbers.
Without hardware, the whole chain was tested with a simulated receiver: a perfect link gave identical predictions on all three
paths (RF 86.8 %, SVM 90.5 % on that slice), and removing 20 % of packets changed 6 to 12 % of the predictions.

## 7. Which bitrates to transmit (method)

Reduce the data BEFORE the radio, then test a handful of rates. Do not transmit the maximum rate and reduce afterwards.

* **The thesis claim is about what the radio carries.** Down-sampling, fewer bits and fewer channels happen on the implant,
  so the reduced stream is what goes over the air. That is what this tester sends (`--fs`, `--channels`, `--bits`).
* **The maximum stream cannot be sent over BLE in real time.** The recordings are 30 kHz; 8 channels x 16 bit x 30 kHz is
  3.84 Mb/s, well above what one BLE link carries (measure your ceiling with `--mode burst`). Sending it anyway only
  measures the link failing.
* **Reducing after reception adds no information.** A reliable link is bit-exact (the CRC check shows it), so classifying
  "full rate, then reduce" gives the same numbers as the offline sweeps you already have. The hardware's job is to show
  that each reduced rate really is delivered intact and in time, and what it costs (loss, retries, RSSI).
* **You do not need all 193 combinations.** The radio only sees bits per second (and packet size), not whether they came
  from sample rate, channels or bit depth. Test a ladder that spans the thesis range, for example 10, 32, 128 and 384 kb/s plus
  the burst ceiling, and run `evaluate_received.py` on a few of them (the 10 kb/s point, formerly called the knee, see the correction in pipeline/docs/elbow_truncation_note.md; the 32 kb/s and 128 kb/s points).
* A cheaper variant is to send one rate the link sustains and derive lower channel counts and bit depths from the received
  data, but a new sample rate needs the filter re-run, so it is only an approximation of sending at that rate.

| Rate | Command |
|---|---|
| 10 kb/s | `--fs 250 --channels 5 --bits 8 --realtime` |
| 32 kb/s | `--fs 500 --channels 8 --bits 8 --realtime` |
| 128 kb/s | `--fs 1000 --channels 8 --bits 16 --realtime` |
| 384 kb/s | `--fs 3000 --channels 8 --bits 16 --realtime` |
| ceiling | `--mode burst --burst-seconds 10` |

## 8. First measurements on your boards (3 Nov 2026)

Burst test, ESP32-C3 to ESP32-C3, +9 dBm, 2M PHY, 7.5 ms connection interval, MTU 247, 240-byte payloads, COM3 -> COM4:

| | Run 1 (10 s) | Run 2 (5 s) |
|---|---|---|
| Application throughput | **1,021 kb/s** (127.6 kB/s) | 961 kb/s (120.2 kB/s) |
| Packets sent / received / lost | 5,328 / 5,328 / 0 | 2,518 / 2,518 / 0 |
| Duplicates, out of order, bad payload | 0, 0, 0 | 0, 0, 0 |
| Stream CRC (tx = rx) | match | match |
| Longest gap between packets | 10.8 ms | 43.6 ms |
| Mean RSSI at the receiver | -34.4 dBm | -65.4 dBm |
| Link loss (TX power - RSSI) | 43.4 dB | 74.4 dB |
| Notify attempts refused by the BLE stack | 25,873 (83 %) | 12,480 (83 %) |

* The ceiling with these settings is about 1.0 Mb/s: the link carries about four 240-byte notifications per 7.5 ms connection
  event (4 x 240 B / 7.5 ms = 1.02 Mb/s). Longer intervals (`--ci 15`, `--ci 30`) or 1M PHY (`--phy 1`) change that and have
  not been tried yet. The thesis grid's top setting (8 ch x 10 kHz x 16 bit = 1.28 Mb/s) is above it; the 10, 32, 128 and
  384 kb/s rungs are well inside it.
* The refused-attempt count is the burst test saturating the link on purpose (the transmitter keeps trying). It is back-pressure,
  not over-the-air retransmission.
* The two runs differ by 31 dB in RSSI because the boards were not in the same place; the weaker run lost about 6 % throughput and
  had a longer worst-case gap (consistent with link-layer retransmissions), yet still delivered every packet. Record the distance
  and orientation when you report link loss.
* `EV TXPWR ... conn=-999` in the early logs was the old firmware calling a getter that is only valid after a connection exists.
  The firmware now reads back the level the controller reports for every state, and section 10 confirms it equals the requested
  level. That is the configured level, not a measurement of the radiated power.

### Transmit-power sweep (3 Nov 2026, boards about 5 cm apart in air)

16 transmit levels from -24 to +20 dBm, 4 s burst each, 2M PHY, 7.5 ms interval. Results and plot:
`results/20261103_113954_burst_vaskov-P1_1of12/` (`summary.json`, `sweep.csv`, `rssi_vs_power.png`).

* **Every level delivered every packet** (0 lost, CRC match, 16 of 16).
* **RSSI follows the setting at about 1 dB per dB from -15 to +9 dBm**, with a link loss of 59.5 dB, steady to within 1 dB.
  Free space at 5 cm would be about 14 dB, so about 45 dB is not explained by distance. Section 10 reviews the possible causes;
  the absolute value is a property of this placement and these boards, not of the radio alone.
* **RSSI flattens near -79 dBm at -24 and -21 dBm, and jumps 9 dB between +9 and +12 dBm** (loss falls from 58 to 52 dB for the
  higher levels). The jump appears on both boards, so it is systematic: either the labelled levels do not match the radiated power
  step for step, or the RSSI reading is non-linear at strong signals. It needs an external reference to settle (section 10).
* **Throughput falls before packets are lost**: 1,000 to 1,024 kb/s down to about -72 dBm, then 969, 928, 860 and 914 kb/s at
  -75 to -79 dBm, with longer gaps (up to 18 ms): link-layer retransmissions absorbing the weaker signal.
* The earlier two runs at +9 dBm read -34 dBm, -65 dBm and (here) -49 dBm for three different placements, so at 5 cm the
  reading depends strongly on how the boards sit.

### Real-time stream test (3 Nov 2026): Vaskov P1, 8 ch x 1 kHz x 16 bit = 128 kb/s, 30 s, +9 dBm

`results/20261103_115432_stream_vaskov-P1_1of12/`. Delivered 128.5 kb/s: 2,000 of 2,000 packets, 0 refused by the BLE stack,
CRC identical at the PC, transmitter and receiver, and the recording is a bit-exact copy of what was sent. The link carried it with
about eight times spare capacity. `evaluate_received.py` on the recording: offline, sent and received predictions identical
(Random Forest 86.8 %, SVM 90.5 % on 189 windows in 6 segments). 11.7 % of the gaps between packets were 20 ms or longer and the
longest was 123 ms; that includes the PC's paced USB feed, so it is an upper bound on the link's own jitter.

### Why the link loss is so high

See section 10 for the full review. In short: 52 to 60 dB at the best placement and 80 to 90 dB at the worst, against about 14 dB for
free space at 5 cm. The measurement chain checks out (section 10.3), so the remaining candidates are the antennas and how the boards
sit (orientation and polarisation, antenna efficiency, the USB cables and the PC), plus an unknown offset in the power labels or RSSI
of up to about 7 dB.

## 9. Live RSSI monitor

`rssi_monitor.py` connects the two boards, transfers nothing, and prints the signal strength 4 times a second (`--rate-ms`,
50 ms or more) so you can see what moving or rotating a board does. It needs the firmware from 3 Nov 2026 12:08 or later
(the `MON` command); both boards were re-flashed with it.

```
python rssi_monitor.py --tx-port COM3 --rx-port COM4 --txpwr 0
```

Type a label and press Enter at each new position or orientation (`0deg`, `90deg`, `flat`, `30cm`, `battery`). Ctrl+C (or
`--seconds`) prints one row per label with mean, min, max RSSI and the implied link loss, plus the spread between rows, and saves
every sample to `results/rssi_monitor_<time>.csv`. Type each label just after you reach a position: the start of the segment is then the
new position and its end contains the move to the next, so the whole-segment mean in that table is dragged by the moves. For the stable
reading per position run `python rssi_segments.py <csv>` (the 10 s after each label; add `--series` to see the moves).

Suggested protocol to find what dominates the loss (keep the transmit power fixed, hold each position for about 10 s):

| Step | What to change | A swing larger than about 10 dB means |
|---|---|---|
| 1 | rotate one board in 90 degree steps about each axis | antenna pattern or polarisation |
| 2 | move one board to 5, 15, 30, 60, 100 cm | near-field coupling (loss does not follow 20 log d at short range) |
| 3 | touch or move the USB cables, then add ferrites | cable and ground coupling |
| 4 | power one board from a power bank (its serial link is then lost, so read the other side) | the PC is part of the antenna |
| 5 | lift both boards 20 cm above the desk, away from metal | surroundings |

Check on 3 Nov 2026 (boards moved since the sweep): at 0 dBm the monitor read about -88 dBm (loss 88 dB), against -60 dBm
(loss 60 dB) in the transmit-power sweep earlier the same day, for the same setting, so placement alone moved it by about 28 dB.

## 10. Method review: could the measurement itself inflate the link loss?

Prepared for supervisor review, 3 Nov 2026. Hardware: two ESP32-C3 boards (chip revision v0.4), Arduino-ESP32 core 3.3.12 (NimBLE
host), firmware `emg_ble_link` 1.0, both boards on USB cables to one Windows 11 PC, 2M PHY unless stated.

### 10.1 What is measured, and what it is not

* **Link loss L = configured transmit power (dBm) - RSSI at the other board (dBm).** The power is the level the controller is
  configured to use (steps of 3 dB, -24 to +20 dBm). RSSI is the controller's estimate for the last packet received on the
  connection (HCI Read RSSI), read every 200 ms to 1 s.
* L is a **coupling loss of the whole setup**: free-space path, both antennas' gain and efficiency, polarisation and pattern
  mismatch, impedance mismatch, coupling to cables, PC and desk, and any error in the power label or the RSSI calibration. It is not
  a path loss and it is not the antenna loss term of the thesis link budget (which used 0 dB for the antennas).

### 10.2 Audit of parameters and method

| # | Item | Could it inflate L? | Evidence | Status |
|---|---|---|---|---|
| 1 | Transmit-power configuration. The first firmware set the connection power before connecting with the older API and ignored return codes; the core documents that connection power can only be set after connecting | Yes, if the setting did not take effect | The controller now reports the requested level for default, advertising, scan, initiating and connection states, before and after an explicit post-connection call (return code 0), at -24, -12, 0 and +9 dBm on both boards. RSSI rises about 1 dB per dB | **Ruled out** as a configuration error. Firmware fixed and now logs return codes and read-backs |
| 2 | Configured versus radiated power | Yes, up to about 7 dB either way | The core documents that the applied power can be 0 to 2 dBm below the request. RSSI jumps 9 dB for the 3 dB step from +9 to +12 dBm, on both boards, and flattens at -24 and -21 dBm | **Partly resolved in 11.4** (relative staircase measured with three readings; the 9 dB jump was not reproduced). The absolute level still needs a spectrum analyser, a calibrated receiver or a conducted (coax and attenuator) measurement |
| 3 | RSSI calibration and statistics | Yes, a few dB | The link dropped when the average RSSI fell to about -93 dBm, close to the assumed -94 dBm sensitivity, so the scale is plausible at the low end. Per-packet scatter is 1.0 to 1.4 dB (sd) at 0 dBm and 2 to 3 dB at +9 dBm. RSSI exists only for packets that arrive (biased high near the limit), freezes at its last value when packets stop (12 identical readings before a drop), and single readings near the limit can be 10 dB or more off | **Partly checked.** An uncertainty of about 5 dB is an estimate, not a measurement |
| 4 | Idle connection versus data traffic | Yes | In the 16-level sweep, idle RSSI matched RSSI during transfer within 1.5 dB at every level | **Ruled out** |
| 5 | Connection interval (7.5 ms versus 30 ms) | Unlikely | A 7.5 ms monitor run read -78 to -80 dBm at its start, the same as the earlier 30 ms runs (-80, -88), while the 7.5 ms sweep read -60 dBm: the difference tracks placement, not interval | **Ruled out** |
| 6 | PHY (1M versus 2M) | Unlikely | 1M read 2.1 dB higher, inside the 2.7 dB reconnection spread (10.3 E) | **Ruled out** |
| 7 | Direction and board asymmetry | Yes, if one board's transmitter were weak | TX at +9 dBm and RX at -24 dBm: RX heard -42.6, TX heard -77.2, a 34.6 dB difference for a 33 dB power difference. Implied loss 51.6 dB (TX to RX) and 53.2 dB (RX to TX) | **Ruled out.** The channel is reciprocal and each board's setting governs its own transmissions |
| 8 | Reconnection scatter | Small | Three connections at 0 dBm without moving anything: -55.1, -55.6, -57.8 dBm (spread 2.7 dB) | **Measured**, about +/-1.4 dB |
| 9 | Placement, distance and orientation | **Yes, the largest factor** | At +9 dBm the same boards read -34, -65, -49, -55 and -43 to -47 dBm in different placements; at 0 dBm -58, -60 (sweep) and -80 to -88 dBm (monitor). Distances and orientations were not recorded until now | **Not controlled.** Likely dominant |
| 10 | The boards' antennas | **Yes, possibly large** | The board model and antenna type (PCB antenna, ceramic, or a u.FL connector with no antenna) were not recorded | **Unknown.** Record the board model and a photograph |
| 11 | USB cables and the PC as part of the antenna system | Yes, several dB or more | Both boards are on USB cables to the same PC | **Not tested** |
| 12 | Surroundings (desk, metal, people) | Yes | Not varied | **Not tested** |
| 13 | Near-field coupling at 5 cm (0.4 wavelengths) | Not in the direction assumed | Free space does not apply at 0.4 wavelengths, but near-field coupling between antennas normally reduces loss, so it does not explain extra loss. An earlier version of this README listed it as a cause; that was wrong | **Corrected** |
| 14 | Interference (Wi-Fi, PC Bluetooth) | No | It causes retransmissions and lower throughput, not a lower RSSI on packets that arrive | **Not a cause of L**; may explain throughput dips |
| 15 | Power supply or brownout | Possible | No reboot or brownout text appears in any run's serial logs. The build enables a PHY power reduction after brownout (`CONFIG_ESP_PHY_REDUCE_TX_POWER`); there is no sign it triggered | **No evidence** |
| 16 | The -94 dBm sensitivity used for margins | Affects the margin, not L | Measured working limit about -93 dBm in one link-failure event | **Plausible** |

### 10.3 Diagnostic experiments (`link_diagnostics.py`, 3 Nov 2026 13:15, boards left as they were)

`results/link_diagnostics_20261103_131515/diagnostics.json`; each row holds 3 s of readings at 5 per second.

| Experiment | TX / RX power | RX hears (dBm) | TX hears (dBm) | Loss TX to RX |
|---|---|---|---|---|
| B, old method (no post-connection call), -12 dBm | -12 / -12 | -69.7 (sd 1.4) | -71.2 | 57.7 dB |
| B, with the call, -12 dBm | -12 / -12 | -70.1 (1.3) | -71.1 | 58.1 dB |
| B, old method, 0 dBm | 0 / 0 | -57.9 (1.4) | -58.6 | 57.9 dB |
| B, with the call, 0 dBm | 0 / 0 | -57.8 (1.2) | -58.6 | 57.8 dB |
| B, old method, +9 dBm | +9 / +9 | -47.3 (2.1) | -48.0 | 56.3 dB |
| B, with the call, +9 dBm | +9 / +9 | -43.1 (3.1) | -45.8 | 52.1 dB |
| C, 1M PHY, 0 dBm | 0 / 0 | -55.7 (0.9) | -56.7 | 55.7 dB |
| D, direction test | +9 / -24 | -42.6 (3.1) | -77.2 | 51.6 dB |
| E, reconnect 2 and 3, 0 dBm | 0 / 0 | -55.1, -55.6 | -56.7, -56.2 | 55.1, 55.6 dB |

* **A, read-back:** the controller reported the requested level index for all five states in every run (for example index 4 for
  -12 dBm, 8 for 0 dBm, 11 for +9 dBm, 0 for -24 dBm) and the explicit connection call returned success.
* **B:** adding the post-connection call changed RSSI by -0.4 dB (-12 dBm), +0.1 dB (0 dBm) and +4.2 dB (+9 dBm). The read-back is
  identical in both cases, and the +9 dBm runs scatter more (sd 2 to 3 dB), so the 4.2 dB is most likely reconnection variability at the
  stronger level. It is not resolved and is a reason to repeat the experiment with more connections.
* **Loss in this placement is 52 to 58 dB**, in line with the sweep (59.5 dB) and well below the 80 to 90 dB seen when the boards were
  arranged differently.

### 10.4 What the evidence supports

1. **The configuration and measurement chain behaves as intended**: the radio reports the requested power, each board's setting
   governs its own transmissions, the PHY and connection interval do not matter, reconnection scatter is about 1.4 dB, idle and
   traffic readings agree, and the RSSI scale is plausible where the link fails.
2. **The high and variable loss therefore comes from outside the firmware**: mainly placement and orientation (a 30 dB range between
   placements) and the antenna systems of these boards, with a further uncertainty of up to about 7 dB from the power labels and RSSI
   calibration (items 2 and 3).
3. **How to quote the numbers.** The loss is a coupling loss of this setup, with about +/-8 dB calibration uncertainty, +/-1.5 dB
   reconnection scatter and 30 dB or more between placements. It must not be used as the antenna loss term of the thesis link budget.
   What can be trusted: relative changes (about 1 dB per dB of transmit power from -15 to +9 dBm), the failure threshold of about
   -93 dBm RSSI (one event), the fall in throughput below about -75 dBm, and the data integrity results.

### 10.5 Recommended follow-up, in priority order

1. Record the board model and antenna type with a photograph (item 10).
2. Run the `rssi_monitor.py` protocol in section 9 (rotate in 90 degree steps, change distance, touch the cables, power one board from a
   battery), using `--note` to record each position.
3. Measure at 0.5, 1 and 2 m on a fixed jig and fit L(d) = L0 + 20 log10(d); free space gives a slope of 20 dB per decade.
4. Calibrate: measure the radiated power and RSSI scale with a spectrum analyser or another calibrated receiver, or run a conducted
   test through coax and attenuators (items 2 and 3).
5. Repeat the power sweep after steps 3 and 4 with at least 10 connections per level (item 2, and the +9 dBm result in 10.3 B).
6. For the thesis link budget, measure the actual implant antenna in the phantom gel instead of using these boards.

### 10.6 Changes made to the method because of this review

* Firmware: the connection power is now set after the connection exists, with return codes and read-backs logged (`PWR?` prints them
  on demand; `SET cpwr=0` restores the old behaviour for comparison).
* Every result can now carry a placement note: `--note "5 cm, flat, antennas facing"` on `ble_link_test.py`, `rssi_monitor.py` and
  `link_diagnostics.py`.
* `rssi_monitor.py` shows the median of 5 readings, reports the last good signal when the link drops, and defaults to the same 7.5 ms
  interval as the other tests.
* Statements corrected in this README: near-field coupling is not a cause of extra loss (item 13), the `conn=-999` explanation, and the
  RSSI floor remark in the sweep.

To repeat the audit: `python link_diagnostics.py --tx-port COM3 --rx-port COM4 --note "<where the boards are>"` (about 75 s; keep the
boards still).

## 11. Power calibration (what a "spectral" calibration can and cannot be on an ESP32-C3)

### 11.1 What is possible

The ESP32-C3 radio is not a measuring instrument. It reports one RSSI number per received packet, and its transmit power is one of 16
labels (-24 to +20 dBm in 3 dB steps) whose real output is not reported back. So:

| Wanted | Possible here? |
|---|---|
| The real power curve relative to one label (how many dB stronger is +12 than 0?) | **Yes**, with two independent receivers (this tool) |
| A correction table, and choosing the label that gives a wanted effective power | **Yes** (`power_cal.py`, `--cal`, `--effective-dbm`), in 3 dB steps |
| Absolute power in dBm, or a spectrum (spurs, harmonics, occupied bandwidth, centre frequency) | **No.** Needs a spectrum analyser or power meter, for example a TinySA-class device (about 100 to 150 GBP) |
| Continuous (non-3 dB) power adjustment | **No.** Only the 16 levels exist; the finer step would have to be an external attenuator |

The table is therefore **relative to label 0 dBm, which is assumed to radiate 0 dBm**. This is saved in every calibration file and in
every result that uses it.

### 11.2 Method (`power_calibration.py`)

For every label (16), in random order, repeated `--reps` times (default 2), the transmitter advertises at that label while three
readings are taken:

1. **PC adapter, advertisements.** The PC's own Bluetooth adapter (Windows BLE API through `bleak`) records the RSSI of each
   advertisement for 8 s: an independent receiver of a different make, antenna and RSSI calibration.
2. **Second ESP32, advertisements.** The RSSI of the advertisement the second board finds when it scans (its own scan result; one
   reading per measurement).
3. **Second ESP32, connection.** The board then connects and records the RSSI of the link for 3 s (200 ms samples). This is the mode
   the data is sent in.

For each reading, the staircase is the mean RSSI at the label minus the mean RSSI at label 0. The departure from the ideal 1 dB per dB
is `dev = (RSSI(label) - RSSI(0)) - label`. Readings 1 and 2 are two receivers of the same advertising-mode power: if they agree
(within 3 dB), the second board's RSSI scale can be trusted at that label. The correction for the **connected link** is then reading 3's
`dev`. The rules, all shown in the verdict column:

* a label is corrected only if readings 1 and 2 agree **and** the repetitions agree within 5 dB (a larger difference means something
  moved);
* a label heard by only one advertising-mode receiver, or never connected, is left at its nominal value and marked "not verified";
* the correction makes the computed link loss independent of the label, which is what a path-loss number should be. It does not say
  which side (transmitter or receiver) causes a departure; see 11.4.

The boards and the PC are not moved during a run (about 12 minutes for 3 repetitions). Record where they are with `--note`.

```
python power_calibration.py --tx-port COM3 --rx-port COM4 --reps 3 --note "5 cm apart, flat on desk, PC 1 m away"
python power_calibration.py --tx-port COM3 --rx-port COM4 --reps 3 --phy 1     # connection at 1M PHY (see 11.4, open question)
python power_calibration.py --tx-port COM4 --rx-port COM3          # calibrate the other board (the table belongs to one board)
python power_calibration.py --tx-port COM3 --rx-port COM4 --resume results/power_cal_<time>   # continue an interrupted run
python power_cal.py latest                                          # print the newest table
```

Using the table (any of the test scripts):

```
python ble_link_test.py ... --cal latest --txpwr=-12,0,12 --effective-dbm   # "I want these effective powers": the nearest labels are sent
python ble_link_test.py ... --cal latest --txpwr=-12,0,9                    # send these labels, report their effective power and calibrated loss
python rssi_monitor.py ... --cal latest --txpwr 0                           # loss computed from the effective power
```

Results then carry `tx_power_effective_dbm`, `tx_power_label_verified`, `link_loss_calibrated_db`, and the calibration file name, next to
the original uncorrected `link_loss_db`.

### 11.3 Firmware 1.1: a crash found by the calibration, and how it was fixed

The first calibration attempts crashed the receiver board. The reboot was visible only as `READY` appearing again in the serial log and the
board having no role afterwards. Decoding the crash dump (`addr2line`) gave an abort in `BLEAdvertisedDevice::parseAdvertisement`, a
`std::vector` growing, `operator new` throwing `bad_alloc`. Cause: the Arduino BLE library keeps **every advertiser it hears for the whole
30 s scan**, and a room full of phones and earbuds filled the 120 kB heap after a few scans; the code also allocated a new callback object
on every `SCAN` command. This was a firmware bug, not a radio effect. The earlier throughput, sweep, stream and diagnostic runs
show no reboot in their serial logs, and in them the receiver found the transmitter within about a second of starting the scan (for
example 0.1 s in the 16-level sweep), so they never ran a long scan; the leak was present in firmware 1.0 and is removed in 1.1. The
`EV SCAN_TIMEOUT` lines that appear right after `EV FOUND` in the old logs are a harmless misprint of 1.0 (the callback ran before the
found flag was set); 1.1 prints that line only for a real timeout.

Fix (firmware 1.1): scanning in 2 s windows with the library's result list cleared after each, one static callback object, ownership of the
found device handled in one place, `doConnect` set before `stop()`, and free heap reported in `PING` (`heap=`, `heap_min=`). Host side: a
board that restarts during a run now stops the run with a clear message instead of producing wrong numbers.

Stress test after the fix (3 Nov 2026): a full 30 s scan with no transmitter timed out as designed and the heap returned to its starting
value (minimum 99.5 kB during the scan, 120.2 kB afterwards); 12 of 12 repeated connections succeeded with no restarts and the heap stayed
at 116.5 kB.

One more behaviour, handled on the host: if the receiver finds a transmitter that is too weak to connect (below about -95 dBm), the
library's `connect()` blocks for up to 30 s and the board cannot answer commands meanwhile. The calibration now waits up to 40 s for the
reply to `STOP` instead of stopping the run.

### 11.4 Results (3 Nov 2026, run `results/power_cal_20261103_231003`)

Firmware 1.1, 16 labels x 3 repetitions = 48 measurements, about 12 minutes, no crashes. Fixed placement chosen by the user; at the 0 dBm
label the PC adapter read -82 dBm and the second ESP -68 dBm (checked stable before the run). The earlier attempts (a firmware crash, and a
run in which the boards were moved) are archived in `results/_archive_power_cal_attempts_20261103/` with a note and are not used.
Repetitions agreed closely: the second ESP's connection readings differ by 0.1 to 0.6 dB between repetitions at most labels (3.2 dB at
+18 and 1.8 dB at +20), the PC adapter by 0.2 to 1.5 dB (3.0 dB at the 0 dBm anchor).

Departure from 1 dB per dB, in dB (positive = the signal rose more than the label step), and the correction used for the connected link:

| Label (dBm) | PC adapter, advertisements | Second ESP, advertisements | Second ESP, connection | Correction | Effective (relative) |
|---:|---:|---:|---:|---:|---:|
| -24 | not heard | +1.7 | +2.9 | 0 (not verified) | -24.0 |
| -21 | not heard | -1.0 | -0.1 | 0 (not verified) | -21.0 |
| -18 | +1.4 | -1.0 | -0.1 | -0.1 | -18.1 |
| -15 | +0.6 | -0.3 | +0.4 | +0.4 | -14.6 |
| -12 | -0.3 | -0.7 | +0.4 | +0.4 | -11.6 |
| -9 | -0.7 | -1.0 | -0.3 | -0.3 | -9.3 |
| -6 | -0.7 | -1.3 | -0.4 | -0.4 | -6.4 |
| -3 | -1.0 | -1.7 | -0.3 | -0.3 | -3.3 |
| 0 | 0 (anchor) | 0 (anchor) | 0 (anchor) | 0 | 0 |
| +3 | -0.5 | -1.3 | +0.1 | +0.1 | +3.1 |
| +6 | -1.2 | -1.3 | -0.1 | -0.1 | +5.9 |
| +9 | 0.0 | -0.7 | +1.4 | +1.4 | +10.4 |
| +12 | +0.6 | 0.0 | +2.0 | +2.0 | +14.0 |
| +15 | +1.7 | +1.3 | +3.3 | +3.3 | +18.3 |
| +18 | +0.9 | +0.3 | +2.3 | +2.3 | +20.3 |
| +20 | +1.6 | +1.3 | **+6.5** | +6.5 | **+26.5** (above the chip's maximum) |

(`results/power_cal_20261103_231003/power_staircase.png` plots the three views.)

What this shows:

1. **In advertising mode the labels are accurate.** Two independent receivers (the PC adapter and the second ESP) agree with each other
   within 2.4 dB at every label they both heard, and with 1 dB per dB within about 1.7 dB over -18 to +20 dBm. Nothing here supports the
   9 dB jump between +9 and +12 dBm seen in the earlier sweep: here it is 3.6 dB (PC, advertising) and 3.6 dB (connection) for the 3 dB
   step. The earlier sweep was taken before the connection power was set explicitly after connecting (10.6) and in a different placement;
   the cause of that jump is **not determined**.
2. **In connected mode the signal rises faster than the label above +6 dBm**: +1.4, +2.0, +3.3, +2.3 and +6.5 dB at +9, +12, +15, +18
   and +20 dBm. Below +6 dBm connection and advertising agree within about 1 dB.
3. **The cause of 2 is open.** For the same receiver at the same label, the connection reading minus the advertisement reading is 0.2 dB
   at 0 dBm (-68.1 against -68.3 dBm) and 5.4 dB at +20 dBm (-41.6 against -47.0 dBm). Either the transmitter's connection power differs from its advertising power at the top labels, or the
   receiver reads connection packets high at strong signals (its connection is 2M PHY, the advertisements are heard on 1M, and the
   reading is about -42 dBm). Two receivers cannot separate these, because only the second ESP can measure the connection. The chip's
   limit points towards a readout effect at the top: taken at face value, +26.5 dBm is not a power this chip can radiate. To test it, repeat
   with the connection at 1M: `python power_calibration.py --tx-port COM3 --rx-port COM4 --reps 3 --phy 1`. If the jump vanishes, it is the
   2M receive path.
4. **The two lowest labels give the same signal** (-89.2 and -89.1 dBm in connection, -90.7 and -90.3 dBm in advertisements): -24 dBm
   behaves like -21 dBm, about 3 dB more than labelled. Only the second ESP heard them (the PC adapter floor is about -98 dBm), so they are
   marked not verified.
5. **The anchor assumption is visibly strained.** The connection staircase spans 26.5 dB between the 0 and +20 dBm labels. If +20 really
   is about +20 dBm, the 0 dBm label radiates about -6.5 dBm in connected mode, and every effective value moves down by that much. This
   is an inference from the chip's maximum, not a measurement. One absolute measurement of one label (spectrum analyser or power meter)
   would settle the whole table; until then read "effective" as relative to label 0.
6. **Spread between receivers.** The second ESP reads 13 to 15 dB stronger than the PC adapter at the same label in this placement (in the
   placement of the earlier, discarded attempt it read about 17 dB weaker): which receiver reads higher depends on where it is. An
   RSSI-based link loss is therefore a property of the placement, as stated in 10.4.

Check of the table in use (`ble_link_test.py --mode burst --cal latest --effective-dbm --txpwr=-10,10`, same placement): wanted -10 and
+10 dBm were mapped to the labels -9 and +9 dBm (effective -9.3 and +10.4 dBm). Loss at label -9 was 68.5 dB uncorrected and 68.2 dB corrected,
at label +9 it was 67.8 dB uncorrected and 69.2 dB corrected; all four agree with the 68 dB from the calibration at 0 dBm to within about 1 dB, which is the noise level, so no improvement
is claimed at these two powers. The correction matters at the top labels (at +20 dBm the uncorrected loss would read about 6 dB low).

The table and its raw data are in `calibration.json` (the first version of the table, which blended advertising and connection readings,
is kept as `calibration_v1_blended_BACKUP.json`; `raw_partial.json` is the untouched raw data).

### 11.5 To repeat or extend

Firmware 1.1 is flashed on both boards (COM3 and COM4), the scripts compile, and nothing is running. Open items, in order of value:

1. `--phy 1` run (item 3 above): tells whether the high-label correction is a transmitter effect or a 2M receive-path readout effect.
2. One absolute measurement of one label with a spectrum analyser or power meter: fixes the anchor (item 5).
3. The same run with the boards swapped (`--tx-port COM4 --rx-port COM3`): the table belongs to one board.

Before any run:

1. Fix the two boards and the PC in place, on a non-metal surface, with the boards a known distance apart and away from the PC's
   Bluetooth antenna. Do not touch them, the cables or the desk for the 12 minutes of the run. Write the arrangement in `--note`.
2. Check the placement first: at the 0 dBm label the PC adapter should read about -50 to -85 dBm and the second ESP should connect (this
   run: -82 and -68 dBm). The PC adapter loses the labels below about -15 dBm when it reads -82 dBm at 0 dBm, and the second ESP needs
   a loss low enough to connect at the lowest labels.
3. Run `python power_calibration.py --tx-port COM3 --rx-port COM4 --reps 3 --note "<arrangement>"` (3 repetitions give the repetition
   check something to compare). If it stops, add `--resume <its folder>`.
4. Read the verdict column: a label is corrected only where both advertising-mode receivers agree and the repetitions agree. Then
   `python ble_link_test.py ... --cal latest`. A new run in a different placement is a new table: `--cal latest` uses the newest.

For an absolute calibration, a conducted setup (coax and fixed attenuators between the two boards) or a spectrum analyser removes the
placement, antenna and RSSI-scale problems at once; see 11.1.

## 12. XIAO ESP32-S3 boards with external antennas (4 Nov 2026)

### 12.1 Why, and what changed

The high and placement-dependent link loss of the ESP32-C3-Zero boards (sections 8 to 10) was suspected to come from their ceramic chip
antennas (10.4, and 11.4 item 6). The test pair is now two Seeed Studio **XIAO ESP32-S3** boards with the supplied external antennas on
the U.FL connector (COM5 transmitter, COM6 receiver). `esptool` identifies both as ESP32-S3 (QFN56, revision v0.2, 8 MB PSRAM); Bluetooth
MACs `ac:27:6e:a5:b0:d1` (COM5) and `28:84:85:bb:78:95` (COM6).

**No code change was needed.** The firmware (1.1) and every host script ran unchanged: both use the same BLE library (NimBLE), the same
USB serial interface, 2M PHY and data-length extension. Only the build setting differs (12.2).

### 12.2 Build setting: the board file labels the option the wrong way round

On the XIAO ESP32-S3 the command-line option values are reversed: `CDCOnBoot=default` means **Enabled** and `CDCOnBoot=cdc` means
**Disabled**. The ESP32-C3 command in section 1 (`CDCOnBoot=cdc`) therefore builds firmware whose `Serial` is the hardware UART pins and
not the USB port: the boards enumerate but are silent, `PING` gets no reply, and writes time out. This cost one flash cycle on 4 Nov 2026.
Correct commands:

```
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default firmware/emg_ble_link
arduino-cli upload  -p COM5 --fqbn esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default firmware/emg_ble_link
```

In the Arduino IDE choose Board "XIAO_ESP32S3", USB CDC On Boot "Enabled", USB Mode "Hardware CDC and JTAG". Check a build by looking for
`-DARDUINO_USB_CDC_ON_BOOT=1` and `-DARDUINO_USB_MODE=1` in its `compile_commands.json`.

### 12.3 Antennas

Seeed's documentation says the supplied antenna must be attached for the radio to work reliably. Keep it connected whenever the board
transmits, and start at a low power (the first checks here used the 0 dBm label). At close range the signal is now much stronger: at the
0 dBm label the boards hear -41 dBm, so at +20 dBm they would hear about -21 dBm; use some distance for high-power runs.

### 12.4 First results (4 Nov 2026; same placement for both; distance and orientation not recorded)

| Check | Result |
|---|---|
| RSSI monitor, 14 s, 0 dBm label (`results/rssi_monitor_20261104_s3_first_check.csv`) | receiver hears -41.5 dBm (range -43 to -40), transmitter hears -41.1 dBm: the same in both directions, link loss 41.5 dB |
| Full-speed burst, 5 s (`results/20261104_121300_burst_vaskov-P1_1of12`) | 1,015 kb/s, 2,656 of 2,656 packets received, CRC matched, 2M PHY, 7.5 ms interval, MTU 247; RSSI -42.0 dBm, link loss 42.0 dB |

For comparison the C3 pair gave 68 dB in its best settled placement, 52 to 60 dB in earlier good placements, and 80 to 90 dB in the worst
(section 10.2 item 9). The burst throughput is the same as on the C3 (about 1.02 Mb/s, set by 4 packets of 240 bytes per 7.5 ms event),
as expected: the radio link is no longer the limit at this distance.

This is consistent with the C3 chip antennas having been the main problem, but **it is not yet shown**: the placement was not recorded
and the C3 numbers came from other placements. The like-for-like test is the rotation and distance protocol of section 9.

### 12.5 Rotation test (4 Nov 2026, 12:34): antennas 5 cm apart, 0 dBm label, receiving board turned

Raw data: `results/rssi_monitor_20261104_123406.csv` (505 s, eight labelled positions). The transmitting board stayed fixed; the
receiving board (COM6) was turned between positions. Labels were typed after reaching each position, so the **stable reading of a
position is the median of the 10 s after its label** (`python rssi_segments.py <csv>`); the whole-segment means printed by the monitor
are dragged by the moves between positions and are not used.

| Position (label as typed) | Receiver hears (dBm) | Transmitter hears (dBm) | Link loss (dB) |
|---|---:|---:|---:|
| start (282 s, nothing moved) | -33 | -32 | 33 |
| base | -32 | -31 | 32 |
| rx90 | -43 | -42 | 43 |
| rx180 | -28 | -26 | 28 |
| rx270 | -22 | -20 | 22 |
| rx-edge | -27 | -27 | 27 |
| rx-end | -41 | -40 | 41 |
| base2 (back at the start) | -32 | -31 | 32 |

What this shows:

1. **Orientation still moves the signal by 21 dB** (-22 to -43 dBm) at a fixed 5 cm distance. By the criterion of section 9 (more than
   10 to 15 dB), antenna pattern and polarisation matter strongly even with the external antennas. At 5 cm (0.4 wavelengths) the two
   antennas are in each other's near field, where coupling depends on geometry, so a large swing was expected there. The distance
   series (12.6) then found that the swing does not shrink with distance, so near-field coupling is not the explanation.
2. **The measurement is repeatable.** Over the 282 s at the start the receiver stayed within -34 to -30 dBm, and after the whole
   sequence of turns the return to the start position read the same as before (-32 and -32 dBm).
3. **The link is reciprocal**: the transmitter hears within 1 to 2 dB of the receiver in every position, so the two boards and
   antennas behave alike and neither is weak.
4. **Compared with the C3 pair at about the same 5 cm distance** (loss 52 to 60 dB in the best C3 placements and 80 to 90 dB in the worst,
   section 10.2 item 9), the worst S3 position (43 dB) is better than the best C3 placement. This supports the chip antennas of the C3 boards
   as the main cause of their high loss. It is not a controlled comparison: the C3 orientations were not recorded.

Limits of this test: the label names are all that records how the board was turned (the axes and directions were not noted); `rx-edge` and
`rx-end` were still settling in the first seconds of their windows (their readings in the window ranged -36 to -24 and -57 to -35 dBm,
so their medians are good to about 2 dB); the part of the `rx270` segment after its first 20 s (about -42 to -30 dBm) is movement
that was not labelled and is not used; the transmitting board was not turned, and the cable and distance steps of section 9 were not done.

Two changes to the tools came out of this: `rssi_monitor.py` no longer reads old log lines at start-up (it printed a false "LINK LOST"
because the disconnect that its own initial `STOP` causes was still in the log, and that would also have hidden a real drop later), and
`rssi_segments.py` was added to take the stable window after each label.

### 12.6 Distance series (4 Nov 2026, 13:13): loss against distance and orientation

**Method** (`rssi_monitor.py` at the 0 dBm label, analysed by `rssi_distance.py`). The transmitting board stayed fixed and only the
receiving board was moved, to 5, 15, 30 and 60 cm (antenna to antenna, the same reference points each time). At each distance, three
orientations of the receiving board, turned together with its antenna, flat on the desk, cable left as it was: `base` (as at the
start of the 5 cm test), `r90` (turned 90 degrees clockwise seen from above) and `r270` (90 degrees counter-clockwise). Labels were typed
after reaching each position; each position's reading is the median of the 10 s after its label. Command:
`python rssi_distance.py results/rssi_monitor_20261104_131334_relabelled.csv --txpwr 0`.

Link loss in dB at the 0 dBm label (transmitter and receiver agree within 1 to 3 dB in every position):

| Distance (cm) | base | r90 | r270 | Swing between orientations | Mean | Free space, two 0 dBi antennas |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 28 | 33 | 31 | 5 | 31 | 14 (near field) |
| 15 | 46 | 48 | 41 | 7 | 45 | 24 (near field) |
| 30 | 40 | 46 | 57 | **17** | 48 | 30 |
| 60 | 59 | 69.5 | 65 | 10 | 64.5 | 36 |
| 100 | not recorded | | | | | 40 |

What this shows:

1. **The swing between orientations does not shrink with distance**: 5, 7, 17 and 10 dB at 5, 15, 30 and 60 cm. At 30 cm it is above the
   10 to 15 dB criterion of section 9. The idea that the large orientation effect of 12.5 was near-field coupling at 5 cm is therefore
   **not supported**. The flat rotations here changed the 5 cm loss by only 5 dB, so the 21 dB of 12.5 came from the other orientations
   used there (standing the board on its edge or end), whose axes were not recorded.
2. **The loss is far above free space for ideal antennas**: 10 to 27 dB above at 30 cm and 23 to 34 dB above at 60 cm (mean 18 and 29 dB).
   It also rises faster than 20 dB per decade: about 20 dB between 15 and 60 cm (an exponent of about 3.2); per orientation the fitted
   exponents are 2.2 (`base`), 4.0 (`r270`) and 3.6 (`r90`), each from three distances only. Even allowing for the RSSI scale uncertainty
   of 8 dB or more found in sections 10 and 11, the 60 cm loss is higher than two working antennas would give.
3. **The cause is not identified; this series does not test any of the candidates.** In order of what the data points to: a polarisation
   mismatch between the two antennas (only the receiver was turned, and in 12.5 standing it on its edge changed the loss by 6 dB or more);
   antenna pattern nulls along the line joining the boards; a U.FL connector that is not fully seated; coupling through the USB cables
   to the PC; and the RSSI scale itself.

Limits: one pass, no repeats, so the spread from moving a board a few centimetres (several dB indoors) is not measured; at 60 cm the
positions were still settling in the first seconds (the `r90` reading was -70 dBm for 5 s, then -65 to -66 dBm for 20 s, so its median
is good to about 4 dB); **100 cm was not recorded** (the label `1m-base` was typed but never entered); the transmitting board was not turned.

Corrections to the data, with the originals kept: the first attempt (`rssi_monitor_20261104_125208.csv`) ended when the transmitter's USB
cable was unplugged and holds no labelled data. In the second (`rssi_monitor_20261104_131334.csv`) the label `60cm-r90` was typed twice
and `60cm-r270` never. The user confirmed that the second `60cm-r90` is r90 and that the two were swapped, so
`rssi_monitor_20261104_131334_relabelled.csv` renames the first one `60cm-r270` (the note file beside it says how). The times at which
the labels were typed were recovered from the terminal history by matching the displayed readings to the CSV; they agree with the
file's own segment boundaries to about 0.3 s.

Tool changes made for this run: `rssi_monitor.py` now says "NO READINGS" instead of repeating a frozen value when the link or a USB cable
drops, and reports "USB SERIAL ... DROPPED"; `rssi_distance.py` has `--relabel OLD=NEW,...` for fixing labels without editing a file.

### 12.7 What is not yet done for the S3 pair

1. **Repeat the series with the two antennas given the same polarisation** (for example both boards standing on their edge so both
   antennas point the same way, the U.FL connectors pressed fully home so they click, the USB cables routed away from the antennas), at
   15, 30, 60 and 100 cm, and enter every label (`100cm-...`, not `1m-...`). If the 60 and 100 cm loss then falls towards the free-space
   value (36 and 40 dB, less the antenna gains), the antennas are fine and the geometry was the problem. If it stays 20 dB or more above, check the RSSI
   scale with an independent receiver (the PC adapter, as in `power_calibration.py`) and then the antennas themselves.
   Turning the transmitting board is also still to do.
2. **A new power calibration.** The table of section 11 belongs to a C3 board and must not be used; the scripts warn when the transmitter's
   MAC differs. The S3's power labels are untested: `applied 0 dBm` was read back at label 0 only, and the S3's output at the other labels
   may not follow the C3 staircase. Run `python power_calibration.py --tx-port COM5 --rx-port COM6 --reps 3 --note "<arrangement>"`; with
   a 27 dB lower loss, the PC adapter will hear more of the low labels and the second ESP the whole range.
3. The C3 results (sections 8 to 11) stay as the record of that hardware.

## 13. Gel phantom experiment (4 Nov 2026)

### 13.1 Setup

A gel phantom made by the user to mimic muscle tissue (no fat or skin layers), wider than it is thick. **Correction (4 Nov 2026):** it is
3.5 cm each way, read as 7 cm thick in total (the user wrote "7mm", taken as a typo for 7 cm; to be confirmed): the transmitter's antenna is
implanted in the middle, 3.5 cm below the top surface and 3.5 cm above the bottom, and the other antenna lies on the top surface directly
above it, so there are **3.5 cm of gel between the two antennas** (an earlier version of this section said 3.5 cm thick and about 1.75 cm deep,
which was wrong). Both antennas have the same orientation, so the two are stacked vertically. XIAO ESP32-S3 pair with the supplied U.FL antennas (Seeed Studio 2.4 GHz antenna A-02; the implanted one is slightly insulated, per the user, and its
exact insulation is not recorded). The board on COM5 is the transmitter and is the implanted one
(confirmed by the user); the surface antenna is on COM6, the receiver. The link is reciprocal, so the numbers do not depend on which is which. The phantom's electrical properties
were not measured here; "muscle-like" is the user's statement.

### 13.2 Link characterisation: full-speed bursts at six power labels (`results/20261104_134402_burst_vaskov-P1_1of12`)

| TX power label (dBm) | Signal at receiver (dBm) | Link loss (dB) | Throughput (kb/s) | Packets lost |
|---:|---:|---:|---:|---:|
| +9 | -26.0 | 35.0 | 1,024 | 0 of 2,680 |
| 0 | -34.2 | 34.2 | 1,023 | 0 of 2,677 |
| -6 | -40.2 | 34.2 | 1,021 | 0 of 2,673 |
| -12 | -46.0 | 34.0 | 1,024 | 0 of 2,679 |
| -18 | -53.4 | 35.4 | 1,019 | 0 of 2,666 |
| -24 | -58.2 | 34.2 | 1,010 | 0 of 2,645 |

* Through the phantom the link loss is 34 to 35 dB at every power, and the CRC matched at every power. The ceiling is about 1.02 Mb/s,
  set by the connection settings (4 packets of 240 bytes per 7.5 ms event), not by the tissue.
* Even at the lowest label the signal is about 36 dB above the -94 dBm working limit, so no packet loss could be produced with power alone.
  A degraded link would need extra attenuation (for example a fixed attenuator in the antenna line, or a greater separation).
* The S3 power labels are not calibrated (section 11 covers a C3 board). Relative to each other the receiver readings follow the label
  steps within about 2 dB. The transmitter's own reading during a burst was erratic (-22.8 to -58 dBm, up to 13 dB from the receiver's)
  and is not used.
* For scale: the same boards in air read 28 to 33 dB at 5 cm and 40 to 57 dB at 30 cm (section 12.6), so the 3.5 cm of gel between the antennas is in the
  same range as a few centimetres of air. This is one phantom and one geometry and is not interpreted further here.

### 13.3 Pilot: Vaskov P1, 128 kb/s (`results/20261104_134557_stream_vaskov-P1_1of12`)

8 channels x 1 kHz x 16 bit = 128 kb/s, 60 s, paced in real time, 0 dBm label: 4,000 of 4,000 packets received, CRC matched at the
transmitter, receiver and host, the recorded copy bit-exact, no stalls, link loss 33.6 dB. `evaluate_received.py` on the received
slice (379 test windows, 0 touched by lost data): RF 90.0 %, SVM 92.1 %, **identical on the offline pipeline, the sent bytes and the
received bytes**. A reliable link is bit-exact, so classifier performance through the phantom equals the offline sweep result; the
hardware shows that each reduced rate really is delivered intact and in time.

### 13.4 Campaign driver (`run_phantom_campaign.py`)

Sends several datasets at several bitrates through the link, classifies what arrives, and collects one table
(`campaign_summary.csv` and `.md`). The rate ladder (reduced before the radio, section 7): 10 kb/s (250 Hz x 5 ch x 8 bit), 32 (500 Hz x 8 x 8),
128 (1 kHz x 8 x 16), 384 (3 kHz x 8 x 16) and 960 kb/s (7.5 kHz x 8 x 16, 94 % of the ceiling). Streams above 400 kb/s are not recorded
(forwarding every packet over the receiver's USB port would become the limit, not the radio) but are checked by the three-way CRC and
reported for throughput, stalls and loss. It loads each recording once, saves progress after every run (`--resume` continues) and
`--dry-run` prints the plan.

Validation run (1 dataset, 10 and 128 kb/s, 20 s each): both bit-exact, classified in about 30 s each (RF 90.3 % and 79.5 %, SVM 87.3 % and
85.2 %: 20 s slices hold too few windows for accuracy, which is why the campaign uses 60 s). Campaign: Vaskov P1 and P2, Wallace S1
and S2 at the five rates, 60 s each, 20 runs (results in 13.5).

Tool changes for this experiment: `ble_link_test.py` sweeps now carry on past a power at which the boards never connect (listed in
`failed_to_connect.json`) instead of aborting, and cache a loaded recording within one process.

### 13.5 Campaign results (4 Nov 2026, `results/phantom_campaign_20261104_141740`)

Four datasets (Vaskov P1 and P2, Wallace S1 and S2) x five rates, 60 s each, real-time, 0 dBm label, through the phantom. All 20
transfers completed.

**The link.** In every run: 0 packets lost, 0 retries, 0 s of stalls, the transmitter, receiver and host CRCs matched, and the throughput
equalled the offered rate. Link loss 33.6 to 34.4 dB in all 20 runs (mean 34.1), the same as the power sweep of 13.2.

| Offered rate | Throughput received (four datasets) | Packets lost | Recorded copy bit-exact |
|---:|---:|---:|---|
| 10 kb/s | 10.2 to 10.4 | 0 | yes |
| 32 kb/s | 32.2 to 32.4 | 0 | yes |
| 128 kb/s | 128.2 to 128.4 | 0 | yes |
| 384 kb/s | 384.1 to 384.3 | 0 | yes |
| 960 kb/s | 960.0 to 960.1 | 0 | not recorded (CRC-checked); 94 % of the 1.02 Mb/s ceiling |

So **about 1 Mb/s passes through 3.5 cm of gel between the antennas without a bottleneck**; 960 kb/s was sustained in real time with no stall. (The
longest gap between packets, 1,631, 506, 123, about 37 and 14 to 18 ms at the five rates, is **not a radio property**: it equals the time
the test harness takes to feed one 2,048-byte USB write at that rate, 2048 B divided by the rate. The delay of a real implant is the time to
fill one 240-byte packet, 192, 60, 15, 5 and 2 ms, if it waits for a full packet; that was not measured here.)

**The classifiers on the received data** (RF and SVM trained with the thesis pipeline at each configuration on the rest of the dataset,
tested on the transmitted slice):

| Dataset | Rate | Test windows | RF (offline = sent = received) | SVM (offline = sent = received) | Sent vs received predictions |
|---|---:|---:|---:|---:|---:|
| Vaskov P1 | 10 kb/s | 395 | 83.5 % | 80.0 % | 100 % identical |
| Vaskov P1 | 32 kb/s | 377 | 90.5 % | 92.6 % | 100 % |
| Vaskov P1 | 128 kb/s | 379 | 90.0 % | 92.1 % | 100 % |
| Vaskov P1 | 384 kb/s | 379 | 90.0 % | 91.3 % | 100 % |
| Vaskov P2, Wallace S1, Wallace S2 | 10, 32, 128, 384 kb/s | 246 to 458 | **not defined** (see below) | | 100 % identical in all 12 runs |

The 960 kb/s runs are link tests only (not recorded, not classified); delivery is bit-exact, so their classifier result would equal the offline one.

**What this shows.** Because every stream arrived bit-exact, **classifier performance through the phantom equals the offline pipeline's at the
same configuration**, on all three paths and in all 16 classified runs; the predictions made from the received data never differed from
those made from the sent data. The hardware shows each reduced rate is delivered intact and in time; it does not change the classification.

**Why accuracy is only given for Vaskov P1.** For P2 and the Wallace runs the 60 s slice contains every segment of the classes in it (P2
has two classes in the slice, each Wallace run one), and the method trains on everything outside the slice, so those classes have no
training windows and the accuracy is 0 whatever the link does. The first pass printed those zeros; `evaluate_received.py` now flags the
condition (`classes_missing_from_training`, `accuracy_meaningful: false`) and the table leaves the accuracy blank. P1 is valid because its
slice holds 11 classes that also occur in other recordings. The absolute accuracies of P2 and Wallace at each configuration are the
thesis's cross-validated sweep numbers, which this link does not change. Getting a hardware-derived number for them would need either
shorter slices that leave some segments of each class for training (for P2; within one Wallace run that would reintroduce the run-signature
confound) or transmitting whole datasets. The single-slice P1 accuracies are noisy (a few points) and the lower value at 10 kb/s
(83.5 %, 80.0 %) is for one slice only.

**Not covered.** The link never degraded (36 dB of margin at the lowest power label, 34 dB loss), so there is no packet-loss regime in
these data; one phantom, one geometry, one 60 s slice per run, one pass.

Bugs found and fixed during the campaign (the originals are kept in the manifest backup `manifest_before_reevaluation_BACKUP.json`):
the transfer watchdog assumed at least 17 kb/s and aborted the two 60 s streams at 10 kb/s (they were redone), and `evaluate_received.py`
printed 0 % where the test classes had no training data. The summary table now also lists the sent-versus-received agreement.

### 13.6 The lowest power label (4 Nov 2026, 16:00, `results/phantom_campaign_20261104_160046`)

Vaskov P1, 30 s per run, real-time, at the **-24 dBm label, the lowest the ESP32-S3 can set**, through the phantom:

| Offered rate | Throughput received | Packets lost | Notify attempts refused (back-pressure) | Stalls | Signal at receiver | Link loss | CRC |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 kb/s | 10.5 | 0 | 0.0 % | 0 s | -59.3 dBm | 35.3 dB | OK |
| 128 kb/s | 128.5 | 0 | 0.0 % | 0 s | -59.4 | 35.4 | OK |
| 384 kb/s | 384.3 | 0 | 0.0 % | 0 s | -59.7 | 35.7 | OK |
| 960 kb/s | 959.9 | 0 | **2.3 %** | 0 s | -59.5 | 35.5 | OK |

Every rate, up to 960 kb/s, is carried in real time with no loss at the lowest power the radio can set. The first sign of the limit is at
960 kb/s: 2.3 % of notify attempts were refused by the BLE stack (the sender's buffer pushing back), which did not happen at the 0 dBm label,
and the burst ceiling at this label was 1,010 kb/s against 1,024 at +9 dBm (13.2), so 960 kb/s is 95 % of the capacity here.

**The hardware floor is not the feasible minimum.** At -24 dBm the receiver still reads about -59.5 dBm, about 35 dB above the roughly
-94 dBm at which a Bluetooth link fails, so the true minimum is below what this radio can set. Finding it needs extra loss in the path, either
calibrated U.FL or SMA attenuators in the receiving antenna line, or lifting the receiving antenna away from the surface in steps. The
quantity to measure is then the maximum sustainable throughput at each signal level (a link that is too weak first loses capacity, so
the sender's buffer overflows and packets are lost, and only then drops the connection; it does not corrupt data, because Bluetooth acknowledges
and retransmits). For each offered rate the weakest signal that still carries it gives the minimum transmit power for this phantom as that signal
level plus the 35 dB path loss. An indicative figure from the C3 measurements of section 8, which is a different radio and not measured here:
throughput began to fall below about -72 to -75 dBm and the link failed near -93 dBm, which would put the minimum near -40 dBm for 1 Mb/s and
near -58 dBm for low rates, for this aligned geometry; misalignment (section 12) and a real implant antenna would raise both.

## 14. Live classification and latency (5 Nov 2026)

Section 13 sent the data first and classified the received copy afterwards. This section classifies **while the data arrives** and measures how
long each decision takes. Script `live_classify.py`; results `results/live_classification_20261105_090141` (8 runs, summary in `live_summary.md`,
chart `latency_breakdown.png`).

### 14.1 Method

- **The sensor is the PC.** It produces one frame (all channels) every 1/fs seconds and hands it to the transmitter's USB port at that moment, never
  earlier, so nothing can be delivered ahead of time. Frame *f* exists at `t0 + f/fs` on the PC clock.
- **The chain** is the one of section 13: PC, USB, transmitter (in the gel), Bluetooth LE through 3.5 cm of gel, receiver (on the surface), USB, PC.
  0 dBm label, 2M PHY, 7.5 ms connection interval, Vaskov P1, 60 s of the recording (the same slice as section 13.5).
- **The classifier** runs in a worker thread and sees only bytes that have arrived. Each time the packet that completes a window arrives, it
  decodes the window, extracts the thesis features (the same function as the offline pipeline), and runs the Random Forest (300 trees) and the SVM.
  Both were trained beforehand on everything outside the transmitted slice, with the same configuration (sample rate, channels, bits) as the stream,
  exactly as `evaluate_received.py` does. Every class in the slice has training data (11 classes, 377 to 395 test windows), so the accuracy is meaningful here
  (unlike P2 and Wallace, section 13.5). The number of test windows differs slightly between rates because the windows are cut at each rate's own sample grid.
- **Which windows:** the same 200 ms windows with 50 ms hop as the offline evaluation (movement only; rest is not a class in the thesis pipeline).
- **Latency** of one decision = (time the prediction is ready) minus (time the window's last sample was produced), all on one PC clock (`perf_counter`).
  It is split into parts that add up exactly:

  | Part | What it is |
  |---|---|
  | packet fill wait | the transmitter sends only **full** packets, so the window's last sample waits for the rest of its packet: payload size / data rate at worst |
  | feed lag (rig) | this script handing the byte to USB later than its ideal time; measured so it can be seen to be small |
  | link transit | USB to the transmitter, Bluetooth, USB from the receiver, until the PC has the packet |
  | queue wait | the decoder was still busy with an earlier window |
  | feature extraction, prediction | PC compute |

  The 200 ms window itself comes on top: its oldest sample is 200 ms old when the decision is made. "Controller delay" in the prosthesis literature is
  usually window plus processing, so both are shown.
- **Rig overhead:** a `PING` through each board's own USB port takes about 0.5 ms round trip (median 0.4 to 0.6 ms in all runs), so the two USB hops of this
  rig add well under 1 ms; a real implant has neither. The feed lag was 0.3 to 0.8 ms median, at most 3.2 ms.
- **Random Forest evaluator.** sklearn's `predict()` for one window costs 11 to 18 ms (95 ms with its default `n_jobs=-1`), almost all Python overhead and not the
  trees, which would hide the link latency. `FastForest` walks the same fitted trees with numpy (0.3 ms). It is checked on every run before the stream
  starts: same predictions and probabilities equal on all 377 to 395 test windows (largest difference 0.0). sklearn's single call time is saved
  in each `summary.json` (`standalone_timings`) for reference.
- **Checks on every run:** the stream arrived bit-exact (three-way CRC); the live predictions equal sklearn's batch predictions on the received stream
  (features identical, largest difference 0); and they equal the predictions from features that never went through the link.

### 14.2 Results (Vaskov P1, gel phantom, 60 s per run, each run done once)

All 8 runs: every window classified (377 to 395 of 377 to 395), 0 packets lost, 0 refused sends, 0 USB forwarding drops, integrity OK,
live = batch = offline predictions on 100 % of windows. Times in ms after the last sample of the window.

| Rate | Packet (B) | Fills in (ms) | Link transit median / p99 | Feature extraction median / p99 | Queue wait p99 | Prediction RF median | **Total median** | p95 | p99 | max | + 200 ms window: median / p99 | RF / SVM accuracy |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 10 kb/s | 240 | 192 | 6.8 / 10.1 | 7.0 / 19.4 | 50.3 | 0.6 | **137** | 203 | 211 | 214 | 337 / 411 | 83.5 % / 80.0 % |
| 10 kb/s | 20 | 16 | 5.3 / 8.9 | 12.2 / 19.4 | 0.2 | 1.0 | **26** | 34 | 38 | 43 | 226 / 238 | 83.5 % / 80.0 % |
| 32 kb/s | 240 | 60 | 6.1 / 7.8 | 14.5 / 31.0 | 28.0 | 0.6 | **53** | 80 | 89 | 94 | 253 / 289 | 90.5 % / 92.6 % |
| 32 kb/s | 40 | 10 | 5.3 / 8.9 | 14.2 / 31.0 | 0.5 | 0.6 | **24** | 37 | 41 | 42 | 224 / 241 | 90.5 % / 92.6 % |
| 128 kb/s | 240 | 15 | 8.9 / 10.2 | 16.4 / 34.7 | 0.4 | 0.6 | **33** | 51 | 57 | 74 | 233 / 257 | 90.0 % / 92.1 % |
| 128 kb/s | 160 | 10 | 6.0 / 9.7 | 18.4 / 30.5 | 0.2 | 0.7 | **29** | 40 | 44 | 47 | 229 / 244 | 90.0 % / 92.1 % |
| 384 kb/s | 240 | 5 | 5.3 / 8.8 | 18.4 / 41.1 | 0.4 | 0.5 | **26** | 45 | 49 | 54 | 226 / 249 | 90.0 % / 91.3 % |
| 960 kb/s | 240 | 2 | 5.6 / 15.1 | 20.1 / 54.6 | 14.7 | 0.5 | **28** | 63 | 78 | 93 | 228 / 278 | 94.2 % / 92.9 % |

(An earlier 128 kb/s pilot with the same settings, `live_classification_20261105_085900`, gave a median total of 32 ms and p99 of 56 ms.)

![latency breakdown](results/live_classification_20261105_090141/latency_breakdown.png)

### 14.3 Reading the results

1. **The link itself is a small part.** USB plus Bluetooth plus USB takes a median 5 to 9 ms and at most 18 ms (measured at the packet that completed each window); in every run 99 % of these were through in 15 ms or less.
   Of that, the rig's USB hops are under 1 ms, so nearly all of it is Bluetooth waiting for the next connection event (7.5 ms interval) plus the air time.
   The median at one setting varied between runs (128 kb/s, 240 B: 3.7 ms in the pilot, 8.9 ms in the matrix). A likely explanation, not tested: packets
   every 15 ms are exactly two connection intervals, so their position relative to the connection events stays fixed for a whole run and differs between runs.
   Treat link-transit medians as good to about 4 ms.
2. **The firmware's wait for a full packet can be the largest part, and it is a design choice, not a radio limit.** At 10 kb/s a 240-byte packet takes
   192 ms to fill: median total 137 ms (p99 211 ms), against 26 ms (p99 38 ms) with 20-byte packets over the same link, with identical accuracy.
   At 32 kb/s the same change took the median from 53 to 24 ms. Rule of thumb: keep payload / data rate to about 10 ms or less. Large packets also make
   several windows complete at once (50 ms hop against a 192 ms packet), which is where the queue wait at 10 and 32 kb/s with big packets comes from.
   Small packets were tested up to 160 B at 128 kb/s (100 packets/s, no loss). Whether tiny packets at high rates cost capacity was not tested.
3. **With sensibly sized packets the decision comes a median 24 to 33 ms after the last sample, 99 % within 38 to 78 ms.** Most of that is PC compute: feature
   extraction takes 7 to 20 ms (up to 55 ms at the 99th percentile) because the thesis function computes all 176 to 192 features per window in Python while this
   process also reads two USB ports, and the classifier uses only 10 to 35 of them. A version that computes only the selected features would be faster; not measured.
   Prediction is 0.5 to 1 ms. The larger tails at 960 kb/s (p99 78 ms, queue wait p99 15 ms) come with the heaviest USB traffic into the PC (120 kB/s parsed in
   Python) and longer windows (1,500 samples); link transit p99 was 15 ms, so the rest is compute and queueing on the PC, not the radio (inferred from the parts, not tested separately).
4. **The 200 ms window dominates the controller delay.** Window plus processing is a median 224 to 233 ms (p99 238 to 278 ms) for the well-sized packets, so the window is at least 85 % of the median.
   For reference only: Farrell and Weir (2007) found about 100 to 125 ms the optimal controller delay in a box-and-block test with 20 able-bodied subjects and a
   simulated prosthetic hand ([abstract](https://opact.sdu.edu.tr/vufind/SummonRecord/FETCH-LOGICAL-c555t-21c8ddcc2720d3788335d31b1997ad2210628e0ee96d60e36d7b9f054df45e6a3)).
   I did not check whether their figure counts the analysis window, so this is a rough comparison. A 300 ms limit is also often quoted; I could not trace its source.
   If these comparisons hold, the link is not what limits the delay, the window length is. Shorter windows would need their accuracy re-tested (not done).
5. **The link adds no error.** Live, batch and offline predictions agree on 100 % of windows in all 8 runs, so the accuracies are those of the offline pipeline on this slice (they
   match section 13.5 where both exist: 128 kb/s 90.0 % / 92.1 %). The 960 kb/s accuracy (94.2 % / 92.9 %) is new, since that stream was not recorded in section 13.5. It is one
   60 s slice with 12 segments, so differences of a few percent between rates are noise.

### 14.4 What went wrong first

The first attempt (`results/live_classification_20261105_083713`, log `results/live_pilot_attempt1_FAILED_slow_rf_and_usb_stall.log`, not a result) failed twice over.
(1) The Random Forest took 95 to 200 ms per window with sklearn's default parallel setting (and printed about 20,000 warnings), so the classifier fell far behind (37 windows in 8 s).
(2) About 8 s into the stream both boards stopped printing and the USB write to the transmitter timed out after 10 s. **The cause was not established.** The likeliest is that the
overloaded PC process stopped reading the two serial ports, but the logs cannot show it. Changes made: the forest evaluator, an abort with a clear message if either port is silent
for 3 s while streaming or a USB write fails (`HOSTERR ...`, with the classifier backlog), a 3 s write timeout, above-normal process priority and a 1 ms thread switch interval.
It did not recur in the following 9 runs, which is not proof that it is fixed.

### 14.5 What this does not show

- **The window positions come from the offline segmentation, which uses the cue labels.** This tests the data path and the timing, not a deployable controller that must find movements by itself;
  finding them would add its own delay.
- **The sensor is ideal.** No ADC conversion, no microcontroller packing time, no transfer from the sensor to the radio. These add to the figures above.
- **The link was strong and clean** (about 37 dB loss, 0 lost packets, 0 refused sends). Retransmissions on a weaker link cost at least one connection interval (7.5 ms) each; not measured.
- **Compute is Python on a PC.** An embedded or phone receiver would differ in either direction; `compute_complexity_and_latency.py` in the pipeline folder has the embedded estimates.
- **One recording (Vaskov P1), one 60 s slice, each setting once.** Wallace and P2 are not used because their accuracy is not meaningful with this slice design (section 13.5).

### 14.6 To repeat

```
python live_classify.py --tx-port COM5 --rx-port COM6 --rates 10,32,128,384,960 --payloads 240,auto --seconds 60
python live_classify.py --rates 128 --prepare-only          # train and cache the classifiers only, no boards
```
`--payloads` takes bytes (at most 240) or `auto` (a packet that fills in `--max-packet-ms`, default 10). Each run folder has `windows.csv` (every window: true class,
live and offline predictions, each part of its latency) and `summary.json`. The trained classifiers are cached in `results/live_model_cache` (keyed by configuration;
`--retrain` rebuilds them). `--resume <folder>` continues an interrupted run. This section changed `ble_link_test.py` only by two additions to the serial reader
(`Node.packet_hook` and `Node.t_last_data`); nothing else behaves differently.

### 14.7 Weaker link, attempt 1: the lowest power label (5 Nov 2026 09:43, `results/live_classification_20261105_094315`; the latency is NOT valid)

The same five settings (packets that fill in about 10 ms) were repeated at the **-24 dBm label**, the weakest the radio can be set, to see whether a weaker link adds latency.

**What is usable** (counted by the boards or the receiver, not by the PC's clock): no packet was lost at any setting, every stream arrived bit-exact, live, batch and offline predictions
agree on 100 % of windows, so the accuracies are unchanged from section 14.2.

| Rate | Packet (B) | Receiver reads (mean, lowest to highest) | Link loss | Packets lost | Notify attempts refused |
|---|---:|---|---:|---:|---:|
| 10 kb/s | 20 | -63.2 dBm (-64 to -63) | 39.2 dB | 0 | 0.0 % |
| 32 kb/s | 40 | -63.5 dBm (-64 to -63) | 39.5 dB | 0 | 0.0 % |
| 128 kb/s | 160 | -64.3 dBm (-117 to -63) | 40.3 dB | 0 | 0.0 % |
| 384 kb/s | 240 | -64.3 dBm (-103 to -63) | 40.3 dB | 0 | 0.0 % |
| 960 kb/s | 240 | -64.2 dBm (-67 to -60) | 40.2 dB | 0 | **5.9 %** |

(The -117 and -103 dBm entries are single readings in 60; the mean is unaffected beyond about 1 dB.) The only sign of the weaker link is at 960 kb/s: 5.9 % of notify attempts were refused by
the BLE stack against 0.0 % at the 0 dBm label, the first sign of capacity running out. Also noted: today's path loss is 37 to 40 dB, against 34 dB on 4 Nov (section 13.2);
the 24 dB step in the label moved the reading by about 26 dB (the S3 power labels are not calibrated, section 12.7).

**What is not usable: every latency total, feature-extraction time and queue wait in that folder** (see `NOT_A_VALID_LATENCY_MEASUREMENT.md` there; the folder is kept). During the run the PC became 7 to 10 times slower: the script's own
check of feature extraction on an idle process read 10 ms before the first run and 68 to 97 ms before the other four (6 to 9 ms in the section 14.2 matrix). The classifier fell behind the 50 ms hop and the windows queued
behind each other (median queue wait 0.2 s at 10 kb/s up to 2.0 s at 384 kb/s), so the "totals" of 0.26 to 2.2 s measure the PC's backlog and not the link. What was seen afterwards, without proving it is the cause:
the Windows power plan was "Silent", the CPU read about 70 % of its rated performance, both OneDrive processes used about 90 % of a core each (the project folder is inside OneDrive), and a single-threaded
feature-extraction benchmark gave 13 to 17 ms against 7 ms in the morning. Raising the process priority or pinning it to the first 12 logical CPUs made no difference. The host-measured link-transit times in that run
(medians 6.7 to 12.5 ms, p99 15 to 26 ms, against 5 to 9 and at most 15 ms in section 14.2) are only indicative: the busy classifier thread competes with the serial readers for the same interpreter.

**Safeguards added to `live_classify.py` because of this:** every result now has `compute_kept_up` (false if the decoder's queue wait p95 reaches half a hop, or feature extraction p99 reaches a hop), a loud warning in the console, a `pc_kept_up` column in the summary
table, and the PC's power plan and CPU performance at the start (`standalone_timings.pc_state` in `summary.json`; read-only, nothing is changed).

**Why this is not yet the weak-link test.** At -24 dBm the receiver still reads -64 dBm, about 30 dB above the roughly -94 dBm where a link fails, so even a valid run would show little. Retransmissions, rising latency and loss need
about 25 to 30 dB more path loss than the phantom gives, which software cannot add: the receiving antenna must be moved or attenuated (the options of section 13.6). The run should also be repeated on a PC that is not throttled
(power plan Balanced or High performance, OneDrive syncing paused) so the compute part is valid.

## 15. Delay, retransmission and backlog in the unpolarised orientation (11 Nov 2026)

**Setup.** This is the weak-link test asked for at the end of section 14. The receiving antenna was turned 90 degrees on the gel, which puts about −82 dBm at the −24 dBm label (about 11 dB above the 2M sensitivity). The protocol is in `pipeline/docs/three_experiments_declared_protocol.md` (X2, with amendment 1).

**Method.** Script `link_timing.py`: real-time P1 stream, 60 s per run, 240-byte packets, no classifier. Every packet is timed on one PC clock: when it was produced, when it was handed to the transmitter, and when it arrived. `link_timing_analyse.py` computes, for every run:
- link transit;
- the share of packets delayed by whole extra connection intervals (7.5 ms), which is retransmission plus queueing combined: the ESP32 does not count retransmissions;
- the backlog, and when it would overflow a 4–64 kB implant buffer;
- stalls.

The compute jobs on the PC were paused during the runs.

**Runs.**
- `results/link_timing_0dBm_20261111_203841`: reference, 32, 128, 384 and 960 kb/s.
- `results/link_timing_-24dBm_20261111_204333`: stress set, 640, 768, 960 and 1280 kb/s.
- `results/link_timing_0dBm_20261111_204846`: stress set, 640, 768 and 1280 kb/s.

Summary: `results/link_timing_summary_20261111/link_timing_summary.md`.

**Results.**
- **No loss:** 0 of 245,000 packets lost, all bit-exact.
- **Up to 768 kb/s:** 99 % of packets arrived within 13 ms, at both labels.
- **960 kb/s, −24 dBm (at capacity):** 48 % of sends refused, the slowest 1 % of packets 0.5 s late, backlog up to 60 kB.
- **1280 kb/s (over capacity):** the link delivered 995–1023 kb/s, and the backlog grew without limit; a 24 kB buffer would overflow in 0.6–0.7 s.

BLE turns a weak link into delay, not loss. Loss would only begin once the implant's buffer overflowed.
