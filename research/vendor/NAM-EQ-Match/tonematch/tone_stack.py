"""NAM plugin tone stack (Bass / Middle / Treble) emulation and fitting.

Replicates the official NAM plugin's `BasicNamToneStack` exactly
(NeuralAmpModelerPlugin/ToneStack.cpp):

  * Bass:   low shelf,  150 Hz, Q 0.707, gain = 4.0 x (knob - 5)   [+/-20 dB]
  * Middle: peaking,    425 Hz, Q 1.5 when cutting / 0.7 boosting,
                                 gain = 3.0 x (knob - 5)           [+/-15 dB]
  * Treble: high shelf, 1800 Hz, Q 0.707, gain = 2.0 x (knob - 5)  [+/-10 dB]

`fit_tone_stack` finds the knob settings (0..10) that best close the LTAS gap
between the raw NAM output and the target, entirely in the frequency domain
(no reamping per candidate), so suggested values transfer 1:1 to the plugin's
EQ section. This gives a gentler, more "hardware-plausible" correction than a
full match IR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASS_FREQ, BASS_Q = 150.0, 0.707
MID_FREQ = 425.0
TREBLE_FREQ, TREBLE_Q = 1800.0, 0.707


def knob_gains_db(bass: float, middle: float, treble: float) -> tuple[float, float, float]:
    return 4.0 * (bass - 5.0), 3.0 * (middle - 5.0), 2.0 * (treble - 5.0)


# ----------------------------------------------------------------------------
# RBJ biquad coefficients (Audio EQ Cookbook), matching the plugin's filters
# ----------------------------------------------------------------------------


def biquad_coeffs(kind: str, fs: float, f0: float, q: float, gain_db: float):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha
    elif kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    elif kind == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cw
        a2 = 1 - alpha / A
    else:
        raise ValueError(kind)
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def _band_params(band: str, knob: float):
    bass_g, mid_g, treb_g = knob_gains_db(knob, knob, knob)
    if band == "bass":
        return "lowshelf", BASS_FREQ, BASS_Q, bass_g
    if band == "middle":
        q = 1.5 if mid_g < 0.0 else 0.7  # plugin: wider on boost, narrower on cut
        return "peaking", MID_FREQ, q, mid_g
    if band == "treble":
        return "highshelf", TREBLE_FREQ, TREBLE_Q, treb_g
    raise ValueError(band)


def band_response_db(band: str, knob: float, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Magnitude response (dB) of one band at `freqs` for a knob value."""
    kind, f0, q, g = _band_params(band, knob)
    if abs(g) < 1e-9:
        return np.zeros_like(freqs)
    b, a = biquad_coeffs(kind, fs, f0, q, g)
    w = 2.0 * np.pi * freqs / fs
    z = np.exp(-1j * w)
    h = (b[0] + b[1] * z + b[2] * z**2) / (1.0 + a[1] * z + a[2] * z**2)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-9))


def stack_response_db(freqs: np.ndarray, fs: float, bass: float, middle: float, treble: float) -> np.ndarray:
    return (
        band_response_db("bass", bass, freqs, fs)
        + band_response_db("middle", middle, freqs, fs)
        + band_response_db("treble", treble, freqs, fs)
    )


def apply_tone_stack(x: np.ndarray, fs: float, bass: float, middle: float, treble: float) -> np.ndarray:
    """Offline-render audio through the tone stack (bass -> mid -> treble)."""
    from scipy.signal import lfilter

    y = np.asarray(x, dtype=np.float64)
    for band, knob in (("bass", bass), ("middle", middle), ("treble", treble)):
        kind, f0, q, g = _band_params(band, knob)
        if abs(g) < 1e-9:
            continue
        b, a = biquad_coeffs(kind, fs, f0, q, g)
        y = lfilter(b, a, y)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------


@dataclass
class ToneStackFit:
    bass: float
    middle: float
    treble: float
    bass_gain_db: float
    middle_gain_db: float
    treble_gain_db: float
    mse_before: float
    mse_after: float


def fit_tone_stack(
    amped_ltas_db: np.ndarray,
    target_ltas_db: np.ndarray,
    grid: np.ndarray,
    fs: float,
    f_lo: float = 80.0,
    f_hi: float = 10000.0,
    reg: float = 0.01,
) -> ToneStackFit:
    """Find knob settings (0..10) minimizing the LTAS gap.

    Frequency-domain search: precompute each band's response per knob value,
    combine by broadcasting, level-align by median, MSE over f_lo..f_hi with a
    small pull toward noon (5.0) so tiny improvements don't produce extreme
    knob positions. Coarse 0.5-step grid, then 0.1-step refinement.
    """
    sel = (grid >= f_lo) & (grid <= f_hi)
    g = grid[sel]
    a_db = amped_ltas_db[sel]
    t_db = target_ltas_db[sel]
    t_db = t_db - np.median(t_db)

    def search(bass_vals, mid_vals, treb_vals):
        rb = np.stack([band_response_db("bass", v, g, fs) for v in bass_vals])
        rm = np.stack([band_response_db("middle", v, g, fs) for v in mid_vals])
        rt = np.stack([band_response_db("treble", v, g, fs) for v in treb_vals])
        total = (
            rb[:, None, None, :] + rm[None, :, None, :] + rt[None, None, :, :] + a_db
        )
        total = total - np.median(total, axis=-1, keepdims=True)
        mse = np.mean((total - t_db) ** 2, axis=-1)
        pen = reg * (
            (bass_vals[:, None, None] - 5.0) ** 2
            + (mid_vals[None, :, None] - 5.0) ** 2
            + (treb_vals[None, None, :] - 5.0) ** 2
        )
        cost = mse + pen
        i, j, k = np.unravel_index(np.argmin(cost), cost.shape)
        return float(bass_vals[i]), float(mid_vals[j]), float(treb_vals[k]), float(mse[i, j, k])

    coarse = np.arange(0.0, 10.01, 0.5)
    b0, m0, t0, _ = search(coarse, coarse, coarse)

    def around(v):
        return np.clip(np.arange(v - 0.5, v + 0.51, 0.1), 0.0, 10.0)

    b, m, t, mse_after = search(around(b0), around(m0), around(t0))

    a_norm = a_db - np.median(a_db)
    mse_before = float(np.mean((a_norm - t_db) ** 2))
    gains = knob_gains_db(b, m, t)
    return ToneStackFit(
        bass=round(b, 1),
        middle=round(m, 1),
        treble=round(t, 1),
        bass_gain_db=round(gains[0], 2),
        middle_gain_db=round(gains[1], 2),
        treble_gain_db=round(gains[2], 2),
        mse_before=round(mse_before, 3),
        mse_after=round(mse_after, 3),
    )
