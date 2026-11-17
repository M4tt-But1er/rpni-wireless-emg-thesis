"""RF link-budget calculator for the implanted Tx -> external Rx path, to
replace the arbitrary -5..-20dB "attenuation" scale in noise_injection.py
with an SNR grounded in an actual link (see chat discussion). Standalone --
does not touch noise_injection.py / the noise sweep; it just produces a
number that CAN be fed into calibrate_and_inject(..., snr_db=...) once
that function is reworked to target raw/channel SNR instead of post-filter
SNR (separate, not-yet-done step).

Formula:
    P_rx  = P_tx + G_tx - L_match_tx - L_body - L_path - L_match_rx + G_rx
    N_floor = -174 dBm/Hz (kT @ 290K) + 10*log10(B_Hz) + NF
    SNR_rx = P_rx - N_floor

Sign convention matches noise_injection.py's SNR_dB (20*log10(std_signal/
std_noise), i.e. a power-ratio dB, positive = signal above noise) --
dBm subtraction gives the same power-ratio dB directly, no conversion
needed to plug the result into calibrate_and_inject.

Two ways to get a noise-floor reference, since we have sensitivity but not
a datasheet noise figure:
  (a) "margin" mode: SNR = P_rx - sensitivity. Sensitivity already bakes in
      NF + whatever minimum SNR the demodulator needs for its rated BER,
      so this is the standard "link margin above the point where the radio
      stops reliably working" number -- the honest one to use when only
      sensitivity is known, no extra assumption required.
  (b) "thermal" mode: NF is backed out of sensitivity assuming a
      `snr_min_db` demod-margin (default 0dB -- the simplest baseline,
      NOT a measured value), then N_floor is computed from that NF and
      requested bandwidth directly. Included for comparison / sensitivity
      analysis -- with snr_min_db=0 it is numerically identical to (a); a
      higher (more realistic) snr_min_db lowers the implied NF and *raises*
      the resulting SNR estimate, so (a) is the conservative choice.

AGC: not modeled as a separate loss/gain term. Its job is to keep the
digitized signal inside the ADC's usable dynamic range across varying
received power (preventing clipping at high power, quantization-noise
domination at low power) -- it does not change the underlying RF SNR
(same gain applied to signal and its accompanying noise). Since AGC is
present here, we assume no extra digitization penalty on top of SNR_rx.
"""
from dataclasses import dataclass, asdict

BOLTZMANN_NOISE_DBM_HZ = -174.0  # kT at 290K, in dBm/Hz -- physical constant, not a fitted value


def body_loss_db(depth_cm: float, atten_db_per_cm: float) -> float:
    """Linear-in-dB tissue path loss -- the standard first-order biotelemetry
    approximation (attenuation coefficient x path length). Ignores near-
    field/far-field structure of the actual coupling; fine for a first-pass
    budget, not a substitute for a measured or simulated (e.g. FDTD) loss."""
    return atten_db_per_cm * depth_cm


def received_power_dbm(p_tx_dbm: float, l_body_db: float, g_tx_db: float = 0.0,
                        l_match_tx_db: float = 0.0, l_path_db: float = 0.0,
                        l_match_rx_db: float = 0.0, g_rx_db: float = 0.0) -> float:
    """P_rx = P_tx + G_tx - L_match_tx - L_body - L_path - L_match_rx + G_rx.
    Antenna gains/matching losses default to 0dB -- NOT a claim they're
    actually zero, just the placeholder until real antenna numbers exist.
    For an implanted antenna at 2.4GHz (small relative to the ~12.5cm free-
    space wavelength) these are plausibly the biggest unknown in the whole
    budget, easily comparable to or bigger than the tissue loss itself."""
    return (p_tx_dbm + g_tx_db - l_match_tx_db - l_body_db - l_path_db
            - l_match_rx_db + g_rx_db)


def thermal_noise_floor_dbm(bandwidth_hz: float, nf_db: float, temp_k: float = 290.0) -> float:
    import math
    kT_dbm_hz = BOLTZMANN_NOISE_DBM_HZ + 10 * math.log10(temp_k / 290.0)
    return kT_dbm_hz + 10 * math.log10(bandwidth_hz) + nf_db


def implied_noise_figure_db(sensitivity_dbm: float, bandwidth_hz: float,
                             snr_min_db: float = 0.0, temp_k: float = 290.0) -> float:
    """Back out NF from a datasheet sensitivity spec:
        sensitivity = -174 + 10log10(B) + NF + snr_min
    snr_min_db (the demodulator's required SNR/Eb-No margin for its rated
    BER) is NOT given by a bare sensitivity number -- default 0dB is a
    simplifying baseline, not a measured figure. Real GFSK/FSK receivers
    typically need several dB of margin, which would make the TRUE NF lower
    than this function reports at snr_min_db=0 (see mode (b) in module
    docstring) -- this is exactly why "margin" mode is the safer default
    for the actual SNR estimate."""
    import math
    kT_dbm_hz = BOLTZMANN_NOISE_DBM_HZ + 10 * math.log10(temp_k / 290.0)
    return sensitivity_dbm - kT_dbm_hz - 10 * math.log10(bandwidth_hz) - snr_min_db


@dataclass
class LinkBudgetResult:
    p_tx_dbm: float
    l_body_db: float
    p_rx_dbm: float
    bandwidth_hz: float
    sensitivity_dbm: float
    margin_db: float                 # P_rx - sensitivity ("margin" mode SNR estimate
    implied_nf_db_at_snr_min_0: float
    thermal_noise_floor_dbm_at_snr_min_0: float
    snr_thermal_db_at_snr_min_0: float  # identical to margin_db when snr_min_db=0, kept for the cross-check


def link_budget(p_tx_dbm: float, depth_cm: float, atten_db_per_cm: float,
                 bandwidth_hz: float, sensitivity_dbm: float,
                 g_tx_db: float = 0.0, l_match_tx_db: float = 0.0,
                 l_path_db: float = 0.0, l_match_rx_db: float = 0.0,
                 g_rx_db: float = 0.0) -> LinkBudgetResult:
    l_body = body_loss_db(depth_cm, atten_db_per_cm)
    p_rx = received_power_dbm(p_tx_dbm, l_body, g_tx_db, l_match_tx_db,
                               l_path_db, l_match_rx_db, g_rx_db)
    margin = p_rx - sensitivity_dbm

    nf_implied = implied_noise_figure_db(sensitivity_dbm, bandwidth_hz, snr_min_db=0.0)
    n_floor = thermal_noise_floor_dbm(bandwidth_hz, nf_implied)
    snr_thermal = p_rx - n_floor  # == margin at snr_min_db=0, see docstring

    return LinkBudgetResult(
        p_tx_dbm=p_tx_dbm, l_body_db=l_body, p_rx_dbm=p_rx,
        bandwidth_hz=bandwidth_hz, sensitivity_dbm=sensitivity_dbm,
        margin_db=margin, implied_nf_db_at_snr_min_0=nf_implied,
        thermal_noise_floor_dbm_at_snr_min_0=n_floor, snr_thermal_db_at_snr_min_0=snr_thermal,
    )


def print_report(r: LinkBudgetResult):
    print("=== Link budget ===")
    print(f"  P_tx                 = {r.p_tx_dbm:7.2f} dBm")
    print(f"  L_body               = {r.l_body_db:7.2f} dB   (antenna gains/matching losses assumed 0dB -- see caveat)")
    print(f"  P_rx                 = {r.p_rx_dbm:7.2f} dBm")
    print(f"  Bandwidth            = {r.bandwidth_hz/1e6:7.3f} MHz")
    print(f"  Sensitivity          = {r.sensitivity_dbm:7.2f} dBm")
    print(f"  --- SNR estimate (recommended): 'margin' mode ---")
    print(f"  Margin above sensitivity (P_rx - sensitivity) = {r.margin_db:7.2f} dB")
    print(f"  --- cross-check: 'thermal' mode, snr_min assumed 0dB ---")
    print(f"  Implied NF           = {r.implied_nf_db_at_snr_min_0:7.2f} dB  (assumes 0dB demod margin -- likely an overestimate of true NF, see docstring)")
    print(f"  Thermal noise floor  = {r.thermal_noise_floor_dbm_at_snr_min_0:7.2f} dBm")
    print(f"  SNR (thermal mode)   = {r.snr_thermal_db_at_snr_min_0:7.2f} dB  (== margin at snr_min=0, by construction)")


if __name__ == "__main__":
    # User-supplied numbers, 2026-09-30 chat: 2.4GHz, 3.5cm implant depth,
    # 3.8dB/cm body loss (from earlier), B approximated from the radio's max
    # 2Mbps data rate (flagged assumption -- bit rate != occupied bandwidth
    # in general; correct this if the actual channel bandwidth is known),
    # sensitivity -94dBm, high-power-mode Tx = 21dBm, no air gap (L_path=0),
    # antenna gains/matching losses unknown -> 0dB placeholders.
    result = link_budget(
        p_tx_dbm=21.0,
        depth_cm=3.5,
        atten_db_per_cm=3.8,
        bandwidth_hz=2_000_000.0,
        sensitivity_dbm=-94.0,
        l_path_db=0.0,
    )
    print_report(result)
    print()
    print("NOTE: this margin is enormous (see chat) -- at this Tx power and depth, the")
    print("wireless link itself is not plausibly the SNR-limiting factor; antenna/matching")
    print("losses (currently 0dB placeholders) are the most likely missing term big enough")
    print("to matter. Pass g_tx_db/l_match_tx_db/l_match_rx_db/g_rx_db once known.")
