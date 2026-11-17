"""BLE link budget through muscle against implant depth: the user's method, section "BLE Link Budget Through Muscle" of the thesis
draft (Da_new_theisis.tex, 9 Nov 2026), implemented exactly as written there (equations and Table "Input parameters").

Chain: muscle loss tangent -> attenuation and phase constants -> absorption + spreading loss -> received power ->
Eb/N0 from the margin over sensitivity -> BER (non-coherent GFSK) -> PER -> throughput. Evaluated for the 1M and 2M PHYs over
1-15 cm, three spreading models (Friis with the in-tissue wavelength, Friis with the free-space wavelength, none) and an extra
allowance L_other of 0 or 10 dB. Maximum usable depth = depth at which the data-packet PER reaches 1 % (brentq).

Additions for the results chapter (not in the draft method): the model evaluated at the phantom geometry (3.5 cm) and at the
transmit settings used in the gel, compared with the measured received power from gel_packet_tally.py (polarised orientation).

Writes cache/link/ble_depth_model.json and .md, and docs/thesis_sim/figures/fig_ble_depth.pdf/.png.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "link")
FIG = os.path.join(HERE, "docs", "thesis_sim", "figures")

C0 = 299792458.0
EPS0 = 8.8541878128e-12
F = 2.4e9
EPS_R = 52.8
SIGMA = 1.71
P_T = 0.0
G_T = G_R = 0.0
SENS = {"1M": -97.5, "2M": -93.5}          # ESP32-S3 datasheet v1.6, Tables 4-21 and 4-22 (typical, @30.8 % PER)
L_DATA = {"1M": 2088, "2M": 2096}
L_ACK = {"1M": 80, "2M": 88}            # draft gives the 2M value (88 bits); 1M: 1-byte preamble, so 80 bits
RATE = {"1M": 1e6, "2M": 2e6}
T_IFS = 150e-6
APP_BYTES = 244
GAMMA_SENS = 2 * math.log(500)          # Eb/N0 for BER 1e-3, non-coherent GFSK (12.43, 10.94 dB)
DEPTH_PHANTOM_M = 0.035


def tissue():
    w = 2 * math.pi * F
    tan_d = SIGMA / (w * EPS0 * EPS_R)
    k = w / C0
    alpha = k * math.sqrt(EPS_R / 2 * (math.sqrt(1 + tan_d ** 2) - 1))
    beta = k * math.sqrt(EPS_R / 2 * (math.sqrt(1 + tan_d ** 2) + 1))
    return dict(tan_delta=tan_d, alpha_np_m=alpha, alpha_db_cm=8.686 * alpha / 100, beta_rad_m=beta, lambda_m_cm=100 * 2 * math.pi / beta,
                lambda_0_cm=100 * C0 / F)


T = tissue()
LAM = {"tissue": T["lambda_m_cm"] / 100, "free": C0 / F, "none": None}


def losses(d, spread):
    l_abs = 8.686 * T["alpha_np_m"] * d
    l_spr = 0.0 if spread == "none" else 20 * math.log10(4 * math.pi * d / LAM[spread])
    return l_abs, l_spr


def p_r(d, spread, l_other, p_t=P_T):
    l_abs, l_spr = losses(d, spread)
    return p_t + G_T + G_R - l_abs - l_spr - l_other


def ber(pr, phy):
    g_db = 10 * math.log10(GAMMA_SENS) + (pr - SENS[phy])
    g = 10 ** (g_db / 10)
    return 0.5 * math.exp(-g / 2), g_db


def log10_ber(pr, phy):
    g = 10 ** ((10 * math.log10(GAMMA_SENS) + (pr - SENS[phy])) / 10)
    return math.log10(0.5) - g / (2 * math.log(10))


def per(b, bits):
    return -math.expm1(bits * math.log1p(-b))


def throughput(phy, per_data, per_ack):
    t_cycle = L_DATA[phy] / RATE[phy] + T_IFS + L_ACK[phy] / RATE[phy] + T_IFS
    r_max = APP_BYTES * 8 / t_cycle
    return t_cycle, r_max, r_max * (1 - per_data) * (1 - per_ack)


def max_depth(phy, spread, l_other):
    f = lambda d: per(ber(p_r(d, spread, l_other), phy)[0], L_DATA[phy]) - 0.01
    return brentq(f, 0.001, 1.0)


def fmt_p(lg):
    return "<1e-100" if lg < -100 else "%.1e" % (10 ** lg)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    res = {"inputs": dict(f_hz=F, eps_r=EPS_R, sigma=SIGMA, p_t_dbm=P_T, g_dbi=G_T, sensitivity_dbm=SENS, l_data_bits=L_DATA, l_ack_bits=L_ACK),
           "tissue": T}
    # throughput (error-free)
    res["throughput"] = {phy: dict(zip(("t_cycle_us", "r_max_mbps"), (1e6 * throughput(phy, 0, 0)[0], throughput(phy, 0, 0)[1] / 1e6))) for phy in ("1M", "2M")}
    # maximum depth at PER 1 %
    res["max_depth_cm"] = {f"{phy}|{sp}|{lo:g}dB": 100 * max_depth(phy, sp, lo) for phy in ("1M", "2M") for sp in ("tissue", "free", "none") for lo in (0.0, 10.0)}
    # at the phantom depth, P_t = 0 dBm
    at = []
    for sp in ("tissue", "free", "none"):
        for lo in (0.0, 10.0):
            pr = p_r(DEPTH_PHANTOM_M, sp, lo)
            l_abs, l_spr = losses(DEPTH_PHANTOM_M, sp)
            row = dict(spreading=sp, l_other_db=lo, l_abs_db=l_abs, l_spr_db=l_spr, total_loss_db=l_abs + l_spr + lo, p_r_dbm=pr)
            for phy in ("1M", "2M"):
                lb = log10_ber(pr, phy)
                b = 10 ** lb if lb > -300 else 0.0
                pd_, pa = per(b, L_DATA[phy]), per(b, L_ACK[phy])
                row[phy] = dict(margin_db=pr - SENS[phy], ebn0_db=ber(pr, phy)[1], log10_ber=lb, ber=fmt_p(lb),
                                per_data=fmt_p(lb + math.log10(L_DATA[phy])) if lb < -10 else "%.2e" % pd_,
                                goodput_mbps=throughput(phy, pd_, pa)[2] / 1e6)
            at.append(row)
    res["at_phantom_depth_0dBm"] = at
    # comparison with the gel (polarised orientation)
    tally = json.load(open(os.path.join(OUT, "gel_packet_tally.json"), encoding="utf-8"))
    res["gel_polarised"], gel = gel_table(tally)
    up = os.path.join(OUT, "gel_packet_tally_unpolarised.json")
    if os.path.exists(up):
        res["gel_unpolarised"], gel_u = gel_table(json.load(open(up, encoding="utf-8")))
    else:
        gel_u = []
    json.dump(res, open(os.path.join(OUT, "ble_depth_model.json"), "w", encoding="utf-8"), indent=1)
    write_md(res, at, gel, gel_u)
    figure(res)


def gel_table(tally):
    by = tally["summary"]["by_tx_label_dbm"]
    gel = []
    for lab in sorted(by, key=lambda s: -float(s)):
        b = by[lab]
        n = b["packets_sent"]
        gel.append(dict(setting_dbm=float(lab), runs=b["runs"], packets_sent=n, packets_lost=b["packets_lost"],
                        per_measured=b["packets_lost"] / n, per_95_upper=1 - 0.05 ** (1 / n),
                        rssi_min_dbm=b["rssi_min"], rssi_max_dbm=b["rssi_max"],
                        loss_min_db=float(lab) - b["rssi_max"], loss_max_db=float(lab) - b["rssi_min"],
                        model_p_r_tissue_dbm=p_r(DEPTH_PHANTOM_M, "tissue", 0.0, float(lab)),
                        model_p_r_free_dbm=p_r(DEPTH_PHANTOM_M, "free", 0.0, float(lab)),
                        model_p_r_none_dbm=p_r(DEPTH_PHANTOM_M, "none", 0.0, float(lab)),
                        margin_2M_min_db=b["rssi_min"] - SENS["2M"]))
    s = tally["summary"]
    return dict(by_setting=gel, runs=s["runs"], packets_sent=s["packets_sent"], packets_lost=s["packets_lost"],
                per_95_upper=s["loss_95pct_upper_bound"], all_bit_exact=s["all_integrity_ok"],
                rssi_range_dbm=s["rssi_rx_mean_range_dbm"]), gel


def write_md(res, at, gel, gel_u):
    L = ["# BLE link budget through muscle (user's method), made by ble_link_budget_depth.py", "",
         "tan d = %.4f, alpha = %.2f Np/m (%.2f dB/cm), beta = %.1f rad/m, lambda_m = %.2f cm, lambda_0 = %.1f cm" %
         (T["tan_delta"], T["alpha_np_m"], T["alpha_db_cm"], T["beta_rad_m"], T["lambda_m_cm"], T["lambda_0_cm"]),
         "Throughput (error-free): " + "; ".join("%s: T_cycle %.0f us, R_max %.2f Mb/s" % (k, v["t_cycle_us"], v["r_max_mbps"]) for k, v in res["throughput"].items()),
         "", "| Spreading | L_other | L_abs | L_spr | P_r (dBm) | margin 2M | BER 2M | PER 2M | margin 1M |", "|---|---|---|---|---|---|---|---|---|"]
    for r in at:
        L.append("| %s | %g | %.1f | %.1f | %.1f | %.1f | %s | %s | %.1f |" % (r["spreading"], r["l_other_db"], r["l_abs_db"], r["l_spr_db"], r["p_r_dbm"],
                                                                         r["2M"]["margin_db"], r["2M"]["ber"], r["2M"]["per_data"], r["1M"]["margin_db"]))
    L += ["", "Max usable depth (PER 1 %), cm: " + "; ".join("%s %.1f" % (k, v) for k, v in res["max_depth_cm"].items()), "",
          "| Setting (dBm) | runs | packets | lost | PER 95% upper | RSSI (dBm) | loss (dB) | model P_r tissue / free / none |", "|---|---|---|---|---|---|---|---|"]
    for label, rows in (("polarised", gel), ("UNPOLARISED", gel_u)):
        for g in rows:
            L.append("| %s %+g | %d | %d | %d | %.1e | %.1f to %.1f | %.1f to %.1f | %.1f / %.1f / %.1f |" % (
                label, g["setting_dbm"], g["runs"], g["packets_sent"], g["packets_lost"], g["per_95_upper"], g["rssi_min_dbm"], g["rssi_max_dbm"],
                g["loss_min_db"], g["loss_max_db"], g["model_p_r_tissue_dbm"], g["model_p_r_free_dbm"], g["model_p_r_none_dbm"]))
    open(os.path.join(OUT, "ble_depth_model.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a8a"
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    d = np.linspace(0.01, 0.15, 300)
    for sp, col, lab in (("none", AQUA, "absorption only"), ("free", BLUE, r"Friis, free-space $\lambda_0$"), ("tissue", ORANGE, r"Friis, in-tissue $\lambda_m$")):
        ax.plot(100 * d, [p_r(x, sp, 0.0) for x in d], color=col, lw=2, label=lab)
        ax.plot(100 * d, [p_r(x, sp, 10.0) for x in d], color=col, lw=1.2, ls="--")
    for phy, ls, va, dy in (("2M", ":", "bottom", 0.8), ("1M", "-.", "top", -0.8)):
        ax.axhline(SENS[phy], color=MUTED, lw=1, ls=ls)
        ax.text(15.2, SENS[phy] + dy, "%s sensitivity" % phy, fontsize=8, va=va, color=INK)
    g0 = [g for g in res["gel_polarised"]["by_setting"] if g["setting_dbm"] == 0.0][0]
    ax.errorbar([3.5], [(g0["rssi_min_dbm"] + g0["rssi_max_dbm"]) / 2], yerr=[[(g0["rssi_max_dbm"] - g0["rssi_min_dbm"]) / 2]],
                fmt="o", color=INK, ms=6, capsize=3, zorder=5, label="gel, polarised (0 dBm)")
    if "gel_unpolarised" in res:
        u0 = [g for g in res["gel_unpolarised"]["by_setting"] if g["setting_dbm"] == 0.0][0]
        ax.errorbar([3.5], [(u0["rssi_min_dbm"] + u0["rssi_max_dbm"]) / 2], yerr=[[(u0["rssi_max_dbm"] - u0["rssi_min_dbm"]) / 2]],
                    fmt="D", color="#6250d6", mfc="white", mew=1.6, ms=6, capsize=3, zorder=5, label="gel, unpolarised (0 dBm)")
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label=r"dashed: $L_\mathrm{other}=10$ dB")
    ax.set_xlim(1, 15)
    ax.set_ylim(-140, 0)
    ax.set_xlabel("Implant depth $d$ (cm)")
    ax.set_ylabel("Received power $P_r$ (dBm)")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_ble_depth." + ext), dpi=200 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
