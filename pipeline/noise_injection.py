"""Frequency-dependent Gaussian noise injection -- simulates data loss/link
degradation as added noise whose power ramps up with frequency (models
thermal/quantization-style noise that grows toward the high end of the
band), rather than literal dropped samples.

Injected into the RAW 30kSps signal, but calibrated by what survives the
project's existing 100-499Hz bandpass filter -- because filtering is linear
(filter(x + noise) == filter(x) + filter(noise)), the achieved SNR *within
the band the classifier actually sees* can be computed exactly and hit
precisely, regardless of how much noise energy sits outside the passband
and gets removed anyway.

Severity is expressed as target SNR in dB, measured in-band:
    SNR_dB = 20 * log10(std(filtered_clean) / std(filtered_noise))
"""
import numpy as np

from preprocessing import bandpass_filter

SNR_LEVELS_DB = {
    "mild": 20.0,
    "moderate": 10.0,
    "severe": 0.0,
    "extreme": -10.0,
}

# Round 2: the mild/moderate/severe/extreme sweep above showed almost no
# degradation until noise exceeded signal power -- the informative region
# is entirely past "extreme". Reframed per the user's own request as
# "attenuation" (a positive dB magnitude, bigger = worse) rather than
# signed SNR: attenuation_dB = -SNR_dB. "Up to 20dB" attenuation, with
# 15dB specifically used as the bitrate-sweep's noise condition.
# Kept as a SEPARATE dict (not merged into SNR_LEVELS_DB) so every
# downstream script that iterates SNR_LEVELS_DB keeps producing the
# original 4 results unchanged, and this round is additive, not a replace.
ATTENUATION_LEVELS_DB = {
    "atten5db": -5.0,
    "atten10db": -10.0,
    "atten15db": -15.0,
    "atten20db": -20.0,
}


def generate_shaped_noise(n_samples: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-ish white Gaussian noise shaped so power ramps up linearly with
    frequency (0 at DC, max at Nyquist) -- a "blue/violet" noise profile.
    NOT physical thermal noise (real Johnson-Nyquist noise is flat/white at
    these frequencies) -- kept for backward compatibility with round 1/2's
    existing cached results. See generate_white_noise for the physically-
    correct choice when modeling an actual Tx/Rx thermal noise floor."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    weight = freqs / freqs.max()  # 0 at DC, 1 at Nyquist
    shaped_spectrum = spectrum * weight
    shaped = np.fft.irfft(shaped_spectrum, n=n_samples)
    return shaped.astype(np.float64)


def generate_white_noise(n_samples: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Flat-PSD Gaussian noise -- the physically correct model for receiver
    thermal noise (Johnson-Nyquist), which has no frequency dependence at
    EMG/RF frequencies. `fs` is unused (kept for interface parity with
    generate_shaped_noise so callers can swap the two freely) -- included as
    a parameter rather than dropped, so a future colored-noise variant can
    reuse the same call signature everywhere calibrate_and_inject uses it."""
    return rng.standard_normal(n_samples).astype(np.float64)


NOISE_GENERATORS = {"shaped": generate_shaped_noise, "white": generate_white_noise}


def calibrate_and_inject(clean_raw: np.ndarray, fs: int, snr_db: float,
                          rng: np.random.Generator, noise_color: str = "shaped") -> np.ndarray:
    """Returns clean_raw + calibrated noise, such that filtering both through
    the project's standard bandpass gives the target in-band SNR.

    noise_color: "shaped" (default, unchanged -- every existing round 1/2
    result stays reproducible) or "white" (flat/thermal -- used for the
    Tx/Rx link-realism sweep; see noise_injection.py module docstring)."""
    noise_unit = NOISE_GENERATORS[noise_color](len(clean_raw), fs, rng)

    filtered_clean = bandpass_filter(clean_raw, fs=fs)
    filtered_noise_unit = bandpass_filter(noise_unit, fs=fs)

    target_ratio = 10 ** (snr_db / 20.0)
    std_clean = filtered_clean.std()
    std_noise_unit = filtered_noise_unit.std()
    if std_noise_unit == 0:
        return clean_raw.copy()

    k = std_clean / (target_ratio * std_noise_unit)
    noisy_raw = clean_raw + k * noise_unit

    # sanity: verify the achieved in-band SNR matches the target
    achieved_filtered_noise = bandpass_filter(k * noise_unit, fs=fs)
    achieved_snr_db = 20 * np.log10(std_clean / achieved_filtered_noise.std())
    assert abs(achieved_snr_db - snr_db) < 0.5, \
        f"SNR calibration off: target {snr_db}dB, achieved {achieved_snr_db:.2f}dB"

    return noisy_raw.astype(np.float32)
