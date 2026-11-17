#!/usr/bin/env python3
"""power_cal.py -- apply a calibration.json made by power_calibration.py.

The ESP32-C3 sets its transmit power in 3 dB steps (16 levels) and the number you ask for is only a LABEL. The calibration measured,
with two independent receivers, how much signal each label really gives compared with label 0 dBm. This module turns that table into
two helpers used by ble_link_test.py (--cal / --effective-dbm) and rssi_monitor.py:

  cal.effective(label)    -> the power (dBm) the label gives at the receiver, relative to label 0 dBm: label + measured correction
                             (it makes the computed link loss independent of the label; it is not an absolute radiated power)
  cal.label_for(target)   -> the label whose effective power is closest to the power you want

Limits, stated once here and saved with every result that uses it:
  * the staircase is RELATIVE: label 0 dBm is assumed to radiate 0 dBm. The absolute level is not measured (needs a spectrum analyser).
  * the correction is for the CONNECTED link (the mode the data is sent in), measured by the second ESP over a connection. A label is
    only corrected when the two advertising-mode receivers agreed (so the receiver's RSSI scale is trustworthy) and the repetitions
    agreed; otherwise it stays at its nominal value and is reported as "not verified".
  * the table belongs to one board (its MAC is stored); another board needs its own run.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_cal(spec):
    """'latest' -> the newest results/power_cal_*/calibration.json; otherwise a path to a file or to its folder."""
    if spec in (None, ""):
        return None
    if str(spec).lower() == "latest":
        found = sorted((HERE / "results").glob("power_cal_*/calibration.json"))
        if not found:
            raise FileNotFoundError("no results/power_cal_*/calibration.json yet: run power_calibration.py first")
        return found[-1]
    p = Path(spec)
    return p / "calibration.json" if p.is_dir() else p


class PowerCal:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / "calibration.json"
        d = json.load(open(self.path))
        self.mac = d.get("tx_board_mac")
        self.created = d.get("created")
        self.note = d.get("note", "")
        self.anchor = d.get("anchor_label_dbm", 0)
        self.assumption = d.get("anchor_assumption", "")
        self.rows = {int(k): v for k, v in d["table"].items()}

    @property
    def labels(self):
        return sorted(self.rows)

    def verified(self, label):
        return self.rows[label]["verdict"].startswith("both receivers agree")

    def correction(self, label):
        return float(self.rows[label]["correction_db"])

    def effective(self, label):
        return label + self.correction(label)

    def label_for(self, target_dbm):
        """The label whose effective power is nearest the target (a tie goes to the lower power: the safer side)."""
        return min(self.labels, key=lambda l: (abs(self.effective(l) - target_dbm), self.effective(l)))

    def describe(self):
        lines = [f"power calibration {self.path.parent.name} (board {self.mac}, made {self.created}); {self.assumption}",
                 f"  {'label':>6} {'correction':>11} {'effective':>10}  status"]
        for l in self.labels:
            flag = ("  !! above the chip's +20 dBm maximum: the 0 dBm anchor cannot hold, read the effective values as relative"
                    if self.rows[l].get("exceeds_chip_max") else "")
            lines.append(f"  {l:>+6} {self.correction(l):>+10.1f} {self.effective(l):>+9.1f}  "
                         f"{'verified (' + self.rows[l]['verdict'] + ')' if self.verified(l) else 'NOT verified (' + self.rows[l]['verdict'] + '): nominal value used'}{flag}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    p = find_cal(sys.argv[1] if len(sys.argv) > 1 else "latest")
    print(PowerCal(p).describe())
