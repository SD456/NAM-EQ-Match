"""Frozen stage-3 score aggregation and consensus EQ; no production changes."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/NAM-EQ-Match"))
from tonematch.audio import EPS, _validate_audio, _validate_sample_rate
from tonematch.features import log_freq_grid, ltas, smooth_spectrum_db
from tonematch.match_eq import fir_from_log_spectrum


def aggregate_scores(current_scores, spectrum_scores):
    """Aggregate probes at each fixed (rig, gain), before any minimization.

    Inputs are (probe, rig, gain), with P0 first. Current scores already include
    the unchanged nonlinear score plus 0.25 * production EQ penalty.
    """
    current = np.asarray(current_scores, dtype=float)
    spectrum = np.asarray(spectrum_scores, dtype=float)
    if (current.ndim != 3 or current.shape != spectrum.shape
            or current.shape[0] < 3 or min(current.shape[1:]) < 1
            or not np.isfinite(current).all() or not np.isfinite(spectrum).all()):
        raise ValueError("scores require matching finite (probe>=3, rig>=1, gain>=1) grids")
    return {
        "current_single": current[0].copy(),
        "spectrum_single": spectrum[0].copy(),
        "current_multi": np.median(current, axis=0),
        "spectrum_multi": np.median(spectrum, axis=0),
    }


def _validate_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError("grid requires finite, positive, increasing frequencies")
    return grid


def half_octave_ltas(audio, sr):
    """Compute 1/2-octave LTAS directly, avoiding smoothing a fingerprint twice."""
    _validate_sample_rate(sr)
    audio = _validate_audio(audio)
    if sr < 16000:
        raise ValueError("stage-3 spectral scoring requires the 80-8000 Hz band")
    audio = audio / (np.max(np.abs(audio)) + EPS)
    grid = log_freq_grid(sr)
    freqs, power = ltas(audio, sr)
    curve = smooth_spectrum_db(freqs, power, grid, octave_fraction=2.0)
    band = (grid >= 80) & (grid <= 8000)
    return grid, curve - np.median(curve[band])


def centered_spectral_distance(curves, target, grid):
    """80-8000 Hz log-LTAS MAE after removing the median level difference.

    Leading axes in curves are retained; target is one unpaired recording.
    """
    grid = _validate_grid(grid)
    curves, target = np.asarray(curves, float), np.asarray(target, float)
    band = (grid >= 80) & (grid <= 8000)
    if (curves.ndim < 1 or curves.shape[-1] != len(grid) or target.shape != grid.shape
            or not np.isfinite(curves).all() or not np.isfinite(target).all()
            or np.count_nonzero(band) < 2):
        raise ValueError("spectra must be finite, share the grid and cover the scoring band")
    delta = curves[..., band] - target[band]
    delta = delta - np.median(delta, axis=-1, keepdims=True)
    return np.mean(np.abs(delta), axis=-1)


def make_consensus_ir(curves, grid, sr, n_taps=2048, guard=True,
                      max_gain_db=4.5, min_gain_db=1.0, max_spread_db=3.0):
    """Combine same-rig, same-gain probe EQ responses; return FIR/grid/actual dB.

    Guarded correction requires unanimous sign, <=3 dB spread and >=1 dB median.
    Both variants receive the same final half-octave Gaussian smoothing, so their
    only difference is the guard mask. Unguarded median is the EQ ablation.
    Both enforce the gain cap on
    the actual float32 FIR response over the entire Nyquist interval.
    """
    _validate_sample_rate(sr)
    grid = _validate_grid(grid)
    curves = np.asarray(curves, float)
    if (curves.ndim != 2 or curves.shape[0] < 3 or curves.shape[1] != len(grid)
            or not np.isfinite(curves).all()):
        raise ValueError("consensus requires finite (probe>=3, frequency) curves")
    if not isinstance(n_taps, (int, np.integer)) or n_taps < 4:
        raise ValueError("n_taps must be an integer of at least 4")
    if (not np.isfinite([max_gain_db, min_gain_db, max_spread_db]).all()
            or not 0 <= max_gain_db <= 60 or min_gain_db < 0 or max_spread_db < 0):
        raise ValueError("gain cap, deadzone and spread must be finite and nonnegative")
    steps = np.diff(np.log2(grid))
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError("consensus smoothing requires a uniform log-frequency grid")
    corr_db = np.median(curves, axis=0)
    if guard:
        agree = np.all(np.sign(curves) == np.sign(corr_db), axis=0)
        supported = agree & (np.ptp(curves, axis=0) <= max_spread_db) & (np.abs(corr_db) >= min_gain_db)
        corr_db = np.where(supported, corr_db, 0.0)
    corr_db = gaussian_filter1d(corr_db, sigma=0.5 / steps[0] / 2.355, mode="nearest")
    corr_db = np.clip(corr_db, -max_gain_db, max_gain_db)
    # Match production's bound on float32 quantization, truncation and inter-bin peaks.
    n_check = 1 << int(np.ceil(np.log2(max(65536, 64 * n_taps))))
    lower, upper = 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20)
    for _ in range(40):
        ir = fir_from_log_spectrum(grid, corr_db, sr, n_taps)
        magnitude = np.abs(np.fft.rfft(ir, n_check))
        margin = np.pi / n_check * np.sum(np.arange(n_taps) * np.abs(ir.astype(float)))
        if magnitude.min() - margin >= lower and magnitude.max() + margin <= upper:
            freqs = np.fft.rfftfreq(n_check, 1 / sr)
            actual_db = np.interp(grid, freqs, 20 * np.log10(np.maximum(magnitude, EPS)))
            return ir, grid, actual_db
        corr_db *= 0.9
    raise ValueError("could not design a consensus IR within the requested gain cap")
