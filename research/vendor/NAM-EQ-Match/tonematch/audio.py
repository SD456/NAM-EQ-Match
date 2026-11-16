"""Audio I/O and utility functions."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

EPS = 1e-12


def _validate_sample_rate(sr: int) -> None:
    if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("sample rate must be a positive integer")


def _validate_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("audio must be a nonempty, finite mono array")
    return x


def load_audio(path: str, sample_rate: int | None = None, *, channel: str | None = None) -> tuple[np.ndarray, int]:
    """Load mono float32 audio; multichannel input requires an explicit channel policy."""
    if sample_rate is not None:
        _validate_sample_rate(sample_rate)
    if channel not in (None, "left", "right", "mean"):
        raise ValueError("channel must be left, right, or mean")
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    _validate_sample_rate(sr)
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("audio must be nonempty and finite")
    if x.shape[1] == 1:
        x = x[:, 0]
    elif channel is None:
        raise ValueError("multichannel audio requires channel=left, right, or mean")
    else:
        x = x.mean(axis=1) if channel == "mean" else x[:, 0 if channel == "left" else 1]
    if sample_rate is not None and sr != sample_rate:
        x = resample(x, sr, sample_rate)
        sr = sample_rate
    return x.astype(np.float32), sr


def save_audio(path: str, x: np.ndarray, sample_rate: int) -> None:
    """Write FLOAT WAV, preserving small IR coefficients and samples above unity."""
    _validate_sample_rate(sample_rate)
    x = _validate_audio(x).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("audio cannot be represented as float32")
    sf.write(path, x, sample_rate, format="WAV", subtype="FLOAT")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    _validate_sample_rate(sr_in)
    _validate_sample_rate(sr_out)
    x = _validate_audio(x)
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms(x: np.ndarray) -> float:
    x = _validate_audio(x)
    return float(np.sqrt(np.mean(np.square(x))))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x) + EPS)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin: float) -> float:
    return float(20.0 * np.log10(abs(lin) + EPS))


def match_rms(x: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """Scale x to a target RMS. Returns (scaled, gain_linear)."""
    x = _validate_audio(x)
    if not np.isfinite(target_rms) or target_rms < 0:
        raise ValueError("target RMS must be finite and nonnegative")
    source_rms = rms(x)
    if source_rms == 0 and target_rms > 0:
        raise ValueError("cannot raise silent audio to a nonzero RMS")
    g = target_rms / source_rms if source_rms else 0.0
    return x * g, g


def short_term_rms(x: np.ndarray, sr: int, win_s: float = 0.05, hop_s: float = 0.01) -> np.ndarray:
    """Short-term RMS envelope (linear)."""
    _validate_sample_rate(sr)
    x = _validate_audio(x)
    if not np.isfinite(win_s) or not np.isfinite(hop_s) or win_s <= 0 or hop_s <= 0:
        raise ValueError("RMS window and hop must be finite and positive")
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if len(x) < win:
        return np.array([rms(x)])
    n = 1 + max(0, (len(x) - win)) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(np.square(frames), axis=1))


def active_segment(x: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """Return the most active (loudest sustained) segment of the signal."""
    _validate_sample_rate(sr)
    x = _validate_audio(x)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("segment duration must be finite and positive")
    segment_samples = max(1, int(duration_s * sr))
    if len(x) <= segment_samples:
        return x
    hop = max(1, int(min(0.25, duration_s) * sr))
    starts = np.arange(0, len(x) - segment_samples + 1, hop)
    starts = np.append(starts, len(x) - segment_samples)
    # Cumulative sample energy avoids constructing a large overlapping frame matrix.
    c = np.r_[0.0, np.cumsum(x**2)]
    scores = c[starts + segment_samples] - c[starts]
    start = int(starts[np.argmax(scores)])
    return x[start : start + segment_samples]


def remove_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """Compact quiet frames; do not use before extracting temporal features."""
    env = short_term_rms(x, sr, win_s=0.05, hop_s=0.05)
    env_db = 20.0 * np.log10(env + EPS)
    ref = np.percentile(env_db, 95)
    keep = env_db > (ref + threshold_db)
    if not np.any(keep):
        return x
    hop = int(0.05 * sr)
    mask = np.zeros(len(x), dtype=bool)
    for i, k in enumerate(keep):
        if k:
            mask[i * hop : (i + 1) * hop] = True
    mask[len(env) * hop :] = keep[-1]
    return x[mask[: len(x)]]
