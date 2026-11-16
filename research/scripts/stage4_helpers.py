"""CPU-only stage-4 raw-spectrum decisions; old scoring and EQ stay untouched."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/NAM-EQ-Match"))
from tonematch.audio import EPS, _validate_sample_rate
from tonematch.match_eq import fir_from_log_spectrum


def _grid_and_band(grid):
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError("grid requires finite, positive, increasing frequencies")
    band = (grid >= 80) & (grid <= 8000)
    if np.count_nonzero(band) < 2:
        raise ValueError("grid must include at least two 80-8000 Hz bins")
    return grid, band


def _curves(curves, grid, minimum_rows):
    curves = np.asarray(curves, dtype=float)
    if (curves.ndim != 2 or curves.shape[0] < minimum_rows
            or curves.shape[1] != len(grid) or not np.isfinite(curves).all()):
        raise ValueError(f"curves require finite (rows>={minimum_rows}, frequency) values")
    return curves


def raw_correction(probe_spectra, reference, grid):
    """Reference minus median probe, centered in-band before any clipping."""
    grid, band = _grid_and_band(grid)
    probes = _curves(probe_spectra, grid, 3)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != grid.shape or not np.isfinite(reference).all():
        raise ValueError("reference must be finite and share the frequency grid")
    raw = reference - np.median(probes, axis=0)
    return raw - np.median(raw[band])


def fit_null(curves, grid):
    """Pooled empirical maximum absolute E0 difference; no coverage guarantee."""
    grid, band = _grid_and_band(grid)
    curves = _curves(curves, grid, 1)
    return float(np.max(np.abs(curves[:, band])))


def decide_eq(curves, grid, sr, q, *, n_taps=2048, min_gain_db=1.0,
              max_spread_db=3.0, max_gain_db=4.5):
    """Require raw agreement across references beyond the frozen null maximum."""
    _validate_sample_rate(sr)
    if sr < 16000:
        raise ValueError("stage-4 EQ requires the 80-8000 Hz band")
    grid, band = _grid_and_band(grid)
    curves = _curves(curves, grid, 2)
    if not isinstance(n_taps, (int, np.integer)) or n_taps < 4:
        raise ValueError("n_taps must be an integer of at least 4")
    if (not np.isfinite([q, min_gain_db, max_spread_db, max_gain_db]).all()
            or min(q, min_gain_db, max_spread_db, max_gain_db) < 0 or max_gain_db > 60):
        raise ValueError("null, floor, spread and cap must be finite and nonnegative; cap <=60")
    steps = np.diff(np.log2(grid))
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError("smoothing requires a uniform log-frequency grid")

    median = np.median(curves, axis=0)
    supported = (band & np.all(np.sign(curves) == np.sign(median), axis=0)
                 & (np.min(np.abs(curves), axis=0) > max(q, min_gain_db))
                 & (np.ptp(curves, axis=0) <= max_spread_db))
    count = int(np.count_nonzero(supported))
    result = {"apply": bool(count and max_gain_db > 0), "supported_bins": count,
              "q": float(q)}
    if not result["apply"]:
        return {**result, "ir": np.array([1], dtype=np.float32),
                "actual_db": np.zeros_like(grid),
                "reason": "zero_gain_cap" if count else "no_supported_bins"}

    correction = gaussian_filter1d(np.where(supported, median, 0.0),
                                   sigma=0.5 / steps[0] / 2.355, mode="nearest")
    correction = np.clip(correction, -max_gain_db, max_gain_db)
    # Bound inter-bin peaks of the actual quantized, truncated FIR as in stage 3.
    n_check = 1 << int(np.ceil(np.log2(max(65536, 64 * n_taps))))
    lower, upper = 10 ** (-max_gain_db / 20), 10 ** (max_gain_db / 20)
    for _ in range(40):
        ir = fir_from_log_spectrum(grid, correction, sr, n_taps)
        magnitude = np.abs(np.fft.rfft(ir, n_check))
        margin = np.pi / n_check * np.sum(np.arange(n_taps) * np.abs(ir.astype(float)))
        if magnitude.min() - margin >= lower and magnitude.max() + margin <= upper:
            frequencies = np.fft.rfftfreq(n_check, 1 / sr)
            actual = np.interp(grid, frequencies, 20 * np.log10(np.maximum(magnitude, EPS)))
            return {**result, "ir": ir, "actual_db": actual,
                    "reason": "raw_consensus_above_null"}
        correction *= 0.9
    raise ValueError("could not design an EQ FIR within the requested gain cap")


def profile_gain_set(effective_gains, device_gain_grid, level_bounds=(-6, 6)):
    """Compatible device gains under bounded input-level offsets, not confidence."""
    effective = np.asarray(effective_gains, dtype=float)
    device = np.asarray(device_gain_grid, dtype=float)
    bounds = np.asarray(level_bounds, dtype=float)
    if (effective.ndim != 1 or device.ndim != 1 or bounds.shape != (2,)
            or not np.isfinite(effective).all() or not np.isfinite(device).all()
            or not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
        raise ValueError("finite gain vectors and increasing level bounds required")
    offsets = effective[:, None] - device[None, :]
    return device[np.any((offsets >= bounds[0]) & (offsets <= bounds[1]), axis=0)].copy()
