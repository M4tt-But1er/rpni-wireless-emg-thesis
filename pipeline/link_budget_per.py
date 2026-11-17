"""BLE link budget for the implanted transmitter: theoretical RSSI, BER and PER.

Written 9 Nov 2026 for the thesis simulation chapter (docs/thesis_sim). It
extends link_budget.py (unchanged) from "margin over sensitivity" to the bit
error rate (BER) and packet error rate (PER) that follow from that margin, and
compares the theoretical received power (RSSI) with the gel-phantom
measurements counted by gel_packet_tally.py.

Chain:
  P_rx  = P_tx - L_tissue - L_excess            (antenna gains and mismatch in L_excess)
  N     = -174 dBm/Hz + 10 log10(B) + NF        (receiver noise floor)
  Eb/N0 = P_rx - N  (dB; B = bit rate, so Eb/N0 = SNR)
  BER   = 0.5 exp(-Eb/N0 / 2)                   (non-coherent binary FSK; BLE is GFSK)
  PER   = 1 - (1 - BER)^L                        (independent bit errors, L bits per packet)
NF is not on a datasheet. It is backed out of the sensitivity, which Bluetooth
defines as the input level giving BER = 1e-3: NF = S + 174 - 10 log10(B) - Eb/N0(1e-3).
A coherent-FSK model, BER = Q(sqrt(Eb/N0)), is computed as a check on the
model choice.

Inputs and their status (user = given by the user on 30 Sep 2026):
  P_tx 21 dBm (user: radio's high-power mode); depth 3.5 cm (user);
  tissue 3.8 dB/cm (user; checked here against a plane-wave calculation for muscle);
  sensitivity -94 dBm at BER 1e-3 (user; BLE 2M PHY; not checked against the datasheet);
  B = 2 MHz = bit rate of the 2M PHY (user; occupied bandwidth is not the same thing);
  antenna gains, mismatch, interface and polarisation losses: 0 dB in the theoretical budget,
  then replaced by the excess loss measured in the gel phantom.

Writes cache/link/link_budget_per.json and .md, and the figure
docs/thesis_sim/figures/fig_link.pdf/.png. Nothing else is changed.
"""
import json
import math
import os
import sys

from scipy.special import erfcinv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "link")
FIG = os.path.join(HERE, "docs", "thesis_sim", "figures")

KT_DBM_HZ = -174.0
B_HZ = 2e6
SENS_DBM = -94.0
BER_SENS = 1e-3
P_TX_DESIGN = 21.0
DEPTH_CM = 3.5
ATT_DB_CM = 3.8
L_TISSUE = DEPTH_CM * ATT_DB_CM            # 13.3 dB

# Bits checked by the CRC in one packet: access address 4 B, PDU header 2 B, payload, CRC 3 B.
# A 240-byte notification travels in a 247-byte link-layer payload (4 B L2CAP + 3 B ATT header).
L_BITS_DATA = 8 * (4 + 2 + 247 + 3)        # 2048 bits
L_BITS_REF = 8 * (4 + 2 + 37 + 3)          # 368 bits: the 37-byte test packet of the sensitivity definition

CAPACITY_KBPS = 1020.0                      # measured burst ceiling of the test firmware (README 12.4, 13.2)
SUPERVISION_S = 2.0                         # connection supervision timeout used in all runs
CONN_INTERVAL_S = 7.5e-3
MISALIGN_DB = 21.0                          # largest orientation swing measured in air (README 12.5)


def db(x):
    return 10 * math.log10(x)


def undb(x):
    return 10 ** (x / 10)


# ---------- BER models ----------
def ber_nc(g):                    # non-coherent binary FSK, g = Eb/N0 (linear)
    return 0.5 * math.exp(-g / 2)


def log10_ber_nc(g):
    return math.log10(0.5) - g / (2 * math.log(10))


def g_for_ber_nc(ber):
    return 2 * math.log(0.5 / ber)


def qinv(p):
    return math.sqrt(2) * erfcinv(2 * p)


def g_for_ber_coh(ber):           # coherent binary FSK: BER = Q(sqrt(g))
    return qinv(ber) ** 2


def per_from_ber(ber, bits):
    return -math.expm1(bits * math.log1p(-ber))


def ber_for_per(per, bits):
    return -math.expm1(math.log1p(-per) / bits)


def plane_wave_atten_db_cm(eps_r, sigma, f_hz):
    """Attenuation of a plane wave in a lossy dielectric (dB/cm)."""
    w = 2 * math.pi * f_hz
    eps = eps_r * 8.854187817e-12
    mu = 4e-7 * math.pi
    lt = sigma / (w * eps)
    alpha = w * math.sqrt(mu * eps / 2) * math.sqrt(math.sqrt(1 + lt ** 2) - 1)   # Np/m
    return 8.686 * alpha / 100


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)

    # receiver model calibrated to the sensitivity
    models = {}
    for name, ginv in (("noncoherent", g_for_ber_nc), ("coherent", g_for_ber_coh)):
        g_s = ginv(BER_SENS)
        nf = SENS_DBM - KT_DBM_HZ - db(B_HZ) - db(g_s)
        noise = KT_DBM_HZ + db(B_HZ) + nf
        thr = {}
        for per_t in (1e-2, 1e-3, 1e-6):
            g_t = ginv(ber_for_per(per_t, L_BITS_DATA))
            thr["%g" % per_t] = noise + db(g_t)
        models[name] = {"ebn0_at_sensitivity_db": db(g_s), "implied_nf_db": nf,
                        "noise_floor_dbm": noise, "rssi_for_per_dbm": thr}
    nc = models["noncoherent"]
    noise = nc["noise_floor_dbm"]

    def ber_at(p_rx):
        return ber_nc(undb(p_rx - noise))

    def log10_ber_at(p_rx):
        return log10_ber_nc(undb(p_rx - noise))

    def fmt_p(lg):
        if lg < -100:
            return "<1e-100"
        return "%.1e" % (10 ** lg)

    # the cliff
    cliff = []
    for p in (-95, -94, -93, -92, -91, -90, -89, -88, -87, -86):
        b = ber_at(p)
        cliff.append({"rssi_dbm": p, "margin_db": p - SENS_DBM, "ebn0_db": p - noise, "ber": b,
                      "per_240B": per_from_ber(b, L_BITS_DATA), "per_37B_ref": per_from_ber(b, L_BITS_REF)})
    per_ref_at_sens = per_from_ber(BER_SENS, L_BITS_REF)

    # retransmission: the stream survives while R <= C (1 - PER)
    retx = {}
    for r in (10, 32, 48, 128):
        per_max = 1 - r / CAPACITY_KBPS
        ber_max = ber_for_per(per_max, L_BITS_DATA)
        retx[str(r)] = {"per_max": per_max, "rssi_min_dbm": noise + db(g_for_ber_nc(ber_max))}
    events_in_timeout = int(SUPERVISION_S / CONN_INTERVAL_S)

    # gel phantom measurements
    tally = json.load(open(os.path.join(OUT, "gel_packet_tally.json"), encoding="utf-8"))
    runs = tally["runs"]
    loss = [r["tx_label_dbm"] - r["rssi_rx_mean_dbm"] for r in runs]
    meas = {}
    for lab in (0, -24):
        rs = [r["rssi_rx_mean_dbm"] for r in runs if r["tx_label_dbm"] == lab]
        meas[str(lab)] = (min(rs), max(rs))
    loss_min, loss_max = min(loss), max(loss)
    exc_min, exc_max = loss_min - L_TISSUE, loss_max - L_TISSUE

    # scenarios: theoretical RSSI, BER, PER
    scen = []

    def add(name, p_lo, p_hi, kind):
        lo_b = log10_ber_at(p_lo)
        scen.append({"scenario": name, "kind": kind, "rssi_dbm": [p_lo, p_hi],
                     "margin_db": [p_lo - SENS_DBM, p_hi - SENS_DBM],
                     "margin_over_per1e-6_db": p_lo - nc["rssi_for_per_dbm"]["1e-06"],
                     "log10_ber_worst": lo_b, "ber_worst": fmt_p(lo_b),
                     "per_240B_worst": fmt_p(lo_b + math.log10(L_BITS_DATA)) if lo_b < -10 else "%.1e" % per_from_ber(10 ** lo_b, L_BITS_DATA)})

    add("Budget, 21 dBm, tissue only", P_TX_DESIGN - L_TISSUE, P_TX_DESIGN - L_TISSUE, "theory")
    add("Budget, 0 dBm setting, tissue only", 0 - L_TISSUE, 0 - L_TISSUE, "theory")
    add("Measured in phantom, 0 dBm setting", *meas["0"], "measured")
    add("Budget, -24 dBm setting, tissue only", -24 - L_TISSUE, -24 - L_TISSUE, "theory")
    add("Measured in phantom, -24 dBm setting", *meas["-24"], "measured")
    add("Budget, 21 dBm, with measured excess loss", P_TX_DESIGN - L_TISSUE - exc_max, P_TX_DESIGN - L_TISSUE - exc_min, "theory")
    add("Budget, 21 dBm, excess loss + 21 dB misalignment", P_TX_DESIGN - L_TISSUE - exc_max - MISALIGN_DB,
        P_TX_DESIGN - L_TISSUE - exc_min - MISALIGN_DB, "theory")

    p_tx_min = {
        "tissue_only": nc["rssi_for_per_dbm"]["1e-06"] + L_TISSUE,
        "measured_loss": nc["rssi_for_per_dbm"]["1e-06"] + loss_max,
        "measured_loss_plus_misalignment": nc["rssi_for_per_dbm"]["1e-06"] + loss_max + MISALIGN_DB,
    }

    out = {
        "inputs": {"p_tx_design_dbm": P_TX_DESIGN, "depth_cm": DEPTH_CM, "atten_db_per_cm": ATT_DB_CM,
                   "l_tissue_db": L_TISSUE, "bandwidth_hz": B_HZ, "sensitivity_dbm": SENS_DBM,
                   "ber_at_sensitivity": BER_SENS, "bits_240B_packet": L_BITS_DATA, "bits_37B_ref": L_BITS_REF,
                   "capacity_kbps_measured": CAPACITY_KBPS, "misalignment_allowance_db": MISALIGN_DB},
        "tissue_check": {"eps_r": 52.7, "sigma_S_per_m": 1.74, "f_hz": 2.45e9,
                         "plane_wave_atten_db_cm": plane_wave_atten_db_cm(52.7, 1.74, 2.45e9),
                         "note": "muscle values at 2.45 GHz as commonly tabulated from Gabriel et al. 1996; check before citing"},
        "receiver_models": models,
        "per_37B_at_sensitivity": per_ref_at_sens,
        "cliff": cliff,
        "retransmission": {"per_max_and_rssi_min_by_rate_kbps": retx,
                           "connection_events_in_supervision_timeout": events_in_timeout},
        "phantom": {"runs": tally["summary"]["runs"], "packets_sent": tally["summary"]["packets_sent"],
                    "packets_lost": tally["summary"]["packets_lost"],
                    "all_bit_exact": tally["summary"]["all_integrity_ok"],
                    "loss_95pct_upper_bound": tally["summary"]["loss_95pct_upper_bound"],
                    "rssi_range_dbm": tally["summary"]["rssi_rx_mean_range_dbm"],
                    "measured_link_loss_db": [loss_min, loss_max],
                    "excess_over_tissue_only_db": [exc_min, exc_max],
                    "rssi_by_setting_dbm": meas},
        "scenarios": scen,
        "p_tx_min_for_per_1e-6_dbm": p_tx_min,
    }
    with open(os.path.join(OUT, "link_budget_per.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # markdown summary
    L = []
    L.append("# Link budget: theoretical RSSI, BER, PER (made by link_budget_per.py)\n")
    L.append("Receiver model (non-coherent FSK, calibrated to BER 1e-3 at -94 dBm): Eb/N0 at sensitivity %.2f dB, implied NF %.2f dB, noise floor %.2f dBm. "
             "Coherent-FSK check: %.2f dB, NF %.2f dB.\n" % (nc["ebn0_at_sensitivity_db"], nc["implied_nf_db"], noise,
                                                            models["coherent"]["ebn0_at_sensitivity_db"], models["coherent"]["implied_nf_db"]))
    L.append("RSSI needed for PER (240-byte packet, %d bits): " % L_BITS_DATA + "; ".join(
        "%s: %.2f dBm (coherent %.2f)" % (k, v, models["coherent"]["rssi_for_per_dbm"][k]) for k, v in nc["rssi_for_per_dbm"].items()) + "\n")
    L.append("PER of the 37-byte reference packet at sensitivity: %.3f\n" % per_ref_at_sens)
    L.append("Plane-wave attenuation in muscle (eps_r 52.7, sigma 1.74 S/m, 2.45 GHz): %.2f dB/cm\n" % out["tissue_check"]["plane_wave_atten_db_cm"])
    L.append("\n| RSSI (dBm) | margin (dB) | Eb/N0 (dB) | BER | PER 240 B | PER 37 B |\n|---|---|---|---|---|---|")
    for c in cliff:
        L.append("| %d | %+d | %.1f | %.2e | %.2e | %.2e |" % (c["rssi_dbm"], c["margin_db"], c["ebn0_db"], c["ber"], c["per_240B"], c["per_37B_ref"]))
    L.append("\n| Scenario | RSSI (dBm) | margin over sensitivity (dB) | margin over PER 1e-6 (dB) | BER | PER 240 B |\n|---|---|---|---|---|---|")
    for s in scen:
        r = s["rssi_dbm"]
        rs = "%.1f" % r[0] if abs(r[0] - r[1]) < 0.05 else "%.1f to %.1f" % (r[0], r[1])
        m = s["margin_db"]
        ms = "%.1f" % m[0] if abs(m[0] - m[1]) < 0.05 else "%.1f to %.1f" % (m[0], m[1])
        L.append("| %s | %s | %s | %.1f | %s | %s |" % (s["scenario"], rs, ms, s["margin_over_per1e-6_db"], s["ber_worst"], s["per_240B_worst"]))
    L.append("\nRetransmission: stream carried while R <= C(1 - PER), C = %.0f kb/s: " % CAPACITY_KBPS + "; ".join(
        "%s kb/s: PER up to %.3f, RSSI down to %.2f dBm" % (k, v["per_max"], v["rssi_min_dbm"]) for k, v in retx.items()))
    L.append("\nPhantom: %d runs, %d packets, %d lost, bit-exact %s, RSSI %.1f to %.1f dBm, link loss %.1f to %.1f dB, excess over tissue-only %.1f to %.1f dB, 95%% upper bound on loss %.2e"
             % (out["phantom"]["runs"], out["phantom"]["packets_sent"], out["phantom"]["packets_lost"], out["phantom"]["all_bit_exact"],
                *out["phantom"]["rssi_range_dbm"], loss_min, loss_max, exc_min, exc_max, out["phantom"]["loss_95pct_upper_bound"]))
    L.append("\nMinimum P_tx for PER 1e-6: tissue only %.1f dBm; measured loss %.1f dBm; plus misalignment %.1f dBm" %
             (p_tx_min["tissue_only"], p_tx_min["measured_loss"], p_tx_min["measured_loss_plus_misalignment"]))
    with open(os.path.join(OUT, "link_budget_per.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    make_figure(out, nc, noise)


def make_figure(out, nc, noise):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a8a"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) the cliff
    x = np.linspace(-96, -76, 400)
    g = 10 ** ((x - noise) / 10)
    ber = 0.5 * np.exp(-g / 2)
    per = -np.expm1(L_BITS_DATA * np.log1p(-ber))
    per_ref = -np.expm1(L_BITS_REF * np.log1p(-ber))
    a.semilogy(x, ber, color=BLUE, lw=2, label="BER")
    a.semilogy(x, per, color=ORANGE, lw=2, label="PER (240 B)")
    a.semilogy(x, per_ref, color=AQUA, lw=2, ls="--", label="PER (37 B test)")
    a.axvline(SENS_DBM, color=MUTED, lw=1, ls=":")
    a.text(SENS_DBM + 0.2, 3e-12, "sensitivity\n$-94$ dBm", fontsize=8, color=INK, va="bottom")
    t6 = nc["rssi_for_per_dbm"]["1e-06"]
    a.axvline(t6, color=MUTED, lw=1, ls="--")
    a.text(t6 + 0.3, 0.4, "PER $=10^{-6}$ at $%.1f$ dBm" % t6, fontsize=8, color=INK, va="center")
    a.set_ylim(1e-12, 2)
    a.set_xlim(-96, -76)
    a.set_xlabel("Received power, RSSI (dBm)")
    a.set_ylabel("Error probability")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), frameon=False)
    a.set_title("(a) Error rates near sensitivity", fontsize=10, loc="left")

    # (b) where the link operates
    rows = [s for s in out["scenarios"]]
    labels = ["Budget, 21 dBm\n(tissue only)", "Budget, 0 dBm\n(tissue only)", "Phantom, 0 dBm\n(measured)",
              "Budget, $-24$ dBm\n(tissue only)", "Phantom, $-24$ dBm\n(measured)", "Budget, 21 dBm +\nmeasured excess loss",
              "Budget, 21 dBm + excess\nloss + misalignment"]
    ys = np.arange(len(rows))[::-1]
    for y, s in zip(ys, rows):
        lo, hi = s["rssi_dbm"]
        col = ORANGE if s["kind"] == "measured" else BLUE
        if abs(hi - lo) < 0.05:
            b.plot([lo], [y], "o", color=col, ms=7, mec="white", mew=1.5, zorder=3)
        else:
            b.plot([lo, hi], [y, y], "-", color=col, lw=5, solid_capstyle="round", zorder=3)
    b.axvline(SENS_DBM, color=MUTED, lw=1, ls=":")
    b.axvline(t6, color=MUTED, lw=1, ls="--")
    b.axvspan(-100, nc["rssi_for_per_dbm"]["0.01"], color="#e9e9e9", zorder=0)
    b.set_yticks(ys)
    b.set_yticklabels(labels, fontsize=8)
    b.set_xlim(-100, 12)
    b.set_ylim(-0.7, len(rows) - 0.3)
    b.set_xlabel("Received power, RSSI (dBm)")
    b.plot([], [], "o", color=BLUE, label="theoretical (link budget)")
    b.plot([], [], "-", color=ORANGE, lw=5, label="measured in gel phantom")
    b.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), frameon=False)
    b.set_title("(b) Theoretical and measured RSSI", fontsize=10, loc="left")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_link." + ext), dpi=200 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
