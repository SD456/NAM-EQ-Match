"""Cached same-DI quality comparisons, equivalent to stage2 paired_quality."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import stft

from scripts.stage2_benchmark import SR
from tonematch.audio import EPS, rms
from tonematch.features import log_freq_grid, ltas, smooth_spectrum_db


def quality_table(outputs, teacher_flat_indices, color):
    """Return SC/LTAS arrays [teacher, E0/E1, candidate]; color(x) makes E1.

    Outputs are uncorrected candidates from one DI. Each teacher index points
    into that sequence; the caller supplies the frozen post-EQ including its tail.
    """
    def checked(x):
        x = np.asarray(x, dtype=float)
        if (x.ndim != 1 or not x.size or not np.isfinite(x).all()
                or rms(x) == 0):
            raise ValueError("Quality comparison requires finite, non-silent mono outputs")
        return x

    waves = [checked(x) for x in outputs]
    teachers = list(teacher_flat_indices)
    if not waves or not teachers or any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        or not 0 <= i < len(waves) for i in teachers
    ):
        raise ValueError("Quality table requires outputs and valid teacher indices")
    count = len(waves)
    references = []
    for i in teachers:
        references.append((int(i), len(waves)))
        waves.append(checked(color(waves[i].copy())))
    lengths = [len(x) for x in waves]
    longest = max(lengths)
    padded = [np.pad(x, (0, longest - len(x))) for x in waves]
    shape = (len(teachers), 2, count)
    sc, lt = np.zeros(shape), np.empty(shape)

    @lru_cache(maxsize=None)
    def level(i, n):
        return rms(padded[i][:n])

    # Default STFT framing depends on pair length, even after global zero-padding.
    for fft in (512, 2048, 8192):
        hop = fft // 4
        magnitudes = [np.abs(stft(x, SR, nperseg=fft, noverlap=fft * 3 // 4)[2])
                      for x in padded] if longest >= fft else None

        @lru_cache(maxsize=None)
        def magnitude(i, n):
            if n >= fft:
                value = magnitudes[i][:, :(n + hop - 1) // hop + 1]
            else:
                # Preserve scipy's historical short-input behavior (and validation).
                value = np.abs(stft(padded[i][:n], SR, nperseg=fft,
                                    noverlap=fft * 3 // 4)[2])
            return value, max(np.linalg.norm(value), 1e-12)

        for ti, pair in enumerate(references):
            for eq, ri in enumerate(pair):
                for ci in range(count):
                    n = max(lengths[ci], lengths[ri])
                    c, _ = magnitude(ci, n)
                    r, denominator = magnitude(ri, n)
                    scale = level(ri, n) / level(ci, n)
                    sc[ti, eq, ci] += np.linalg.norm(c * scale - r) / denominator / 3

    grid = log_freq_grid(SR)
    band = (grid >= 80) & (grid <= 8000)

    @lru_cache(maxsize=None)
    def spectrum(i, n):
        x = padded[i][:n]
        peak = np.max(np.abs(x))
        f, p = ltas(x / (peak + EPS), SR)
        return f, p, peak

    @lru_cache(maxsize=None)
    def centered_ltas(i, n, scale=1.0):
        f, p, peak = spectrum(i, n)
        # RMS alignment precedes peak normalization in paired_quality. Retain
        # its EPS-dependent scale rather than assuming perfect level invariance.
        factor = scale * (peak + EPS) / (scale * peak + EPS)
        values = smooth_spectrum_db(f, p * factor**2, grid)
        return values - np.median(values)

    for ti, pair in enumerate(references):
        for eq, ri in enumerate(pair):
            for ci in range(count):
                n = max(lengths[ci], lengths[ri])
                delta = (centered_ltas(ci, n, level(ri, n) / level(ci, n))
                         - centered_ltas(ri, n))[band]
                delta -= np.median(delta)
                lt[ti, eq, ci] = np.mean(np.abs(delta))
    return {"sc": sc, "ltas": lt}
