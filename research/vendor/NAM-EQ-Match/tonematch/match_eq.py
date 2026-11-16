"""Match-EQ: design an impulse response that corrects the linear (EQ/cab)
difference between the tuned candidate rig and the target recording.

This is the "Tone Match" half: after the NAM capture + input gain reproduce
the nonlinear character, the residual LTAS difference is linear and can be
captured in a FIR filter, exported as a .wav IR loadable in the NAM plugin's
IR slot (or any IR loader).
"""

from __future__ import annotations

import numpy as np

from .audio import EPS, _validate_audio, _validate_sample_rate
from .features import ltas, log_freq_grid, smooth_spectrum_db


def design_match_ir(
    candidate: np.ndarray,
    target: np.ndarray,
    sr: int,
    n_taps: int = 2048,
    max_gain_db: float = 4.5,
    octave_fraction: float = 2.0,
    f_lo_taper: float = 50.0,
    f_hi_taper: float = 11000.0,
    minimum_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design a FIR that makes `candidate` match `target` spectrally.

    Returns (ir, grid_freqs, actual_ir_response_db). Only minimum phase is supported.
    """
    _validate_sample_rate(sr)
    candidate, target = _validate_audio(candidate), _validate_audio(target)
    if sr < 1000:
        raise ValueError("match EQ requires a sample rate of at least 1000 Hz")
    if not isinstance(n_taps, (int, np.integer)) or n_taps < 4:
        raise ValueError("n_taps must be an integer of at least 4")
    if not minimum_phase:
        raise ValueError("only minimum-phase IRs are supported")
    if not np.isfinite(max_gain_db) or not 0 <= max_gain_db <= 60:
        raise ValueError("max_gain_db must be finite and between 0 and 60")
    if not np.isfinite(octave_fraction) or octave_fraction <= 0:
        raise ValueError("octave_fraction must be finite and positive")
    if not 0 < f_lo_taper < f_hi_taper or not np.isfinite(f_hi_taper):
        raise ValueError("taper frequencies must be finite, positive, and increasing")
    grid = log_freq_grid(sr)
    fc, pc = ltas(candidate, sr)
    ft, pt = ltas(target, sr)
    c_db = smooth_spectrum_db(fc, pc, grid, octave_fraction)
    t_db = smooth_spectrum_db(ft, pt, grid, octave_fraction)
    corr_db = t_db - c_db
    corr_db -= np.median(corr_db[(grid > 200) & (grid < 4000)])  # 0 dB midband

    # Taper the correction toward 0 dB at the extremes (don't boost rumble/hiss)
    lo_fade = np.clip(np.log2(grid / (f_lo_taper / 2)) / np.log2(2.0), 0.0, 1.0)
    hi_stop = min(f_hi_taper * 2, sr / 2)
    hi_start = min(f_hi_taper, hi_stop / 2)
    hi_fade = np.clip(np.log2(hi_stop / grid) / np.log2(hi_stop / hi_start), 0.0, 1.0)
    corr_db = corr_db * lo_fade * hi_fade
    corr_db = np.clip(corr_db, -max_gain_db, max_gain_db)

    # Check the quantized, truncated FIR, including the intervals between FFT bins.
    # |dH/dw| <= sum(k*|h[k]|) bounds any peak missed by the sampled response.
    n_check = 1 << int(np.ceil(np.log2(max(65536, 64 * n_taps))))
    lower, upper = 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20)
    for _ in range(40):
        ir = fir_from_log_spectrum(grid, corr_db, sr, n_taps, minimum_phase)
        magnitude = np.abs(np.fft.rfft(ir, n_check))
        margin = np.pi / n_check * np.sum(np.arange(n_taps) * np.abs(ir.astype(float)))
        if magnitude.min() - margin >= lower and magnitude.max() + margin <= upper:
            freqs = np.fft.rfftfreq(n_check, 1 / sr)
            actual_db = np.interp(grid, freqs, 20 * np.log10(np.maximum(magnitude, EPS)))
            return ir, grid, actual_db
        corr_db *= 0.9
    raise ValueError("could not design an IR within the requested gain limit")


def fir_from_log_spectrum(
    grid: np.ndarray,
    gain_db: np.ndarray,
    sr: int,
    n_taps: int = 2048,
    minimum_phase: bool = True,
) -> np.ndarray:
    """Build a FIR from a gain curve defined on a log-frequency grid."""
    _validate_sample_rate(sr)
    grid, gain_db = np.asarray(grid, dtype=float), np.asarray(gain_db, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or gain_db.shape != grid.shape
            or not np.all(np.isfinite(grid)) or not np.all(np.isfinite(gain_db))
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError("gain curve must have finite gains and increasing positive frequencies")
    if not isinstance(n_taps, (int, np.integer)) or n_taps < 4:
        raise ValueError("n_taps must be an integer of at least 4")
    if not minimum_phase:
        raise ValueError("only minimum-phase IRs are supported")
    n_fft = 4 * n_taps
    f_lin = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # interpolate in log-f domain; clamp ends
    lf = np.log10(np.maximum(f_lin, grid[0] / 2))
    # Explicit unity anchors avoid extending a correction to DC or Nyquist.
    curve_f = np.r_[grid[0] / 2, grid[grid < sr / 2], sr / 2]
    curve_db = np.r_[0.0, gain_db[grid < sr / 2], 0.0]
    g_db = np.interp(lf, np.log10(curve_f), curve_db)
    mag = 10.0 ** (g_db / 20.0)

    h = _minimum_phase_fir(mag, n_taps)
    return h.astype(np.float32)


def _minimum_phase_fir(mag: np.ndarray, n_taps: int) -> np.ndarray:
    """Minimum-phase FIR from a magnitude response via the real cepstrum."""
    n_fft = 2 * (len(mag) - 1)
    log_mag = np.log(np.maximum(mag, 1e-8))
    ceps = np.fft.irfft(log_mag)
    w = np.zeros(n_fft)
    w[0] = 1.0
    w[1 : n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0
    min_phase_spec = np.exp(np.fft.rfft(ceps * w))
    h = np.fft.irfft(min_phase_spec)[:n_taps]
    # gentle fade-out on the tail to avoid truncation ripple
    fade = int(0.25 * n_taps)
    h[-fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade)))
    return h


def apply_fir(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve mono audio with an IR, retaining its complete decay tail."""
    from scipy.signal import fftconvolve

    x, ir = _validate_audio(x), _validate_audio(ir)
    if not np.any(ir):
        raise ValueError("IR must contain a nonzero coefficient")
    y = fftconvolve(x, ir, mode="full").astype(np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError("convolution output cannot be represented as float32")
    return y
