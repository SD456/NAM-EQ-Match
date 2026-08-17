"""Feature extraction and distances for tone matching.

Two kinds of features:

1. LTAS (long-term average spectrum) — linear, fully correctable by a match
   IR/EQ afterwards. Used to *generate* the correction, and only lightly
   penalized during model ranking (extreme corrections indicate a bad base
   match).

2. Nonlinear / dynamic fingerprint — things an EQ can NOT fix: how hard the
   waveform is being squashed (crest, kurtosis, peak density), how much the
   signal is compressed, and how densely the spectrum is filled with
   distortion harmonics. These drive the ranking of candidate NAM captures
   and the input-gain search.

Distortion-measurement notes (learned the hard way):

  * Spectral *flatness* of a high band is a trap: when a clean signal has
    almost no HF content, the band is just noise floor - which is maximally
    flat. That made clean captures look "fizzy" and win against high-gain
    targets. We now measure HF energy on loud frames only, and use spectral
    crest (peak/mean) rather than a log-domain flatness.
  * Every per-feature distance must be *bounded*. An unbounded relative error
    on a near-zero feature (e.g. spectral flatness of tonal material) can
    reach 10+ and outvote every genuine distortion cue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dct
from scipy.signal import welch, stft

from .audio import EPS, remove_silence, short_term_rms

# ----------------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------------


def ltas(x: np.ndarray, sr: int, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average power spectrum via Welch. Returns (freqs, power)."""
    nper = min(n_fft, len(x))
    f, p = welch(x, fs=sr, nperseg=nper, noverlap=nper // 2, window="hann")
    return f, p


def log_freq_grid(sr: int, f_lo: float = 30.0, pts_per_octave: int = 24) -> np.ndarray:
    f_hi = 0.95 * sr / 2
    n_oct = np.log2(f_hi / f_lo)
    n = int(np.ceil(n_oct * pts_per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / pts_per_octave)


def smooth_spectrum_db(
    f: np.ndarray, p: np.ndarray, grid: np.ndarray, octave_fraction: float = 6.0
) -> np.ndarray:
    """Interpolate a power spectrum onto a log grid and smooth by 1/N octave."""
    p_db = 10.0 * np.log10(p + EPS)
    g_db = np.interp(grid, f, p_db)
    pts_per_octave = 1.0 / np.log2(grid[1] / grid[0])
    sigma = pts_per_octave / octave_fraction / 2.355  # FWHM = 1/N octave
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(sigma, 1e-6)) ** 2)
    k /= k.sum()
    padded = np.pad(g_db, radius, mode="edge")
    return np.convolve(padded, k, mode="valid")


# ----------------------------------------------------------------------------
# Mel / MFCC (no librosa dependency)
# ----------------------------------------------------------------------------


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, f_lo: float, f_hi: float) -> np.ndarray:
    m = np.linspace(_hz_to_mel(f_lo), _hz_to_mel(f_hi), n_mels + 2)
    hz = _mel_to_hz(m)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


# ----------------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------------


@dataclass
class ToneFingerprint:
    # --- how hard the waveform is squashed (main distortion cues) ----------
    crest_db: float          # peak/RMS over 400 ms windows; drops as gain rises
    kurtosis: float          # sample-distribution peakiness; clipping flattens it
    peak_density: float      # fraction of samples near the frame peak (square-ness)
    # --- compression --------------------------------------------------------
    dyn_range_db: float
    sustain_ratio: float     # late/early energy of notes; distortion adds sustain
    # --- spectral distortion signature -------------------------------------
    hf_ratio: float          # HF energy share on loud frames (harmonic generation)
    spec_crest_hf: float     # peak/mean in HF band; dense harmonics -> low
    flux: float
    # --- texture ------------------------------------------------------------
    mfcc_std: np.ndarray = field(repr=False, default=None)
    mfcc_dstd: np.ndarray = field(repr=False, default=None)
    # --- linear (correctable by the match IR/EQ) ---------------------------
    ltas_grid: np.ndarray = field(repr=False, default=None)
    ltas_db: np.ndarray = field(repr=False, default=None)


def extract_fingerprint(x: np.ndarray, sr: int) -> ToneFingerprint:
    x = remove_silence(x, sr)
    x = x / (np.max(np.abs(x)) + EPS)  # level-invariant

    # --- Dynamics -----------------------------------------------------------
    env = short_term_rms(x, sr, win_s=0.05, hop_s=0.01)
    env_db = 20.0 * np.log10(env + EPS)
    dyn_range_db = float(np.percentile(env_db, 95) - np.percentile(env_db, 10))

    # Crest factor + waveform-shape stats over 400 ms windows.
    # Distortion squashes peaks: crest falls, the sample distribution flattens
    # (kurtosis down) and more samples sit near the peak (peak_density up).
    win = int(0.4 * sr)
    n = max(1, len(x) // win)
    crests, kurts, dens = [], [], []
    for i in range(n):
        seg = x[i * win : (i + 1) * win]
        if len(seg) < win // 2:
            continue
        r = np.sqrt(np.mean(seg**2) + EPS)
        pk = np.max(np.abs(seg)) + EPS
        if r <= 10 ** (-50 / 20):  # skip near-silence
            continue
        crests.append(20.0 * np.log10(pk / r))
        s = seg / r
        kurts.append(float(np.mean(s**4)))          # kurtosis of a zero-mean frame
        dens.append(float(np.mean(np.abs(seg) > 0.5 * pk)))
    crest_db = float(np.mean(crests)) if crests else 12.0
    kurtosis = float(np.mean(kurts)) if kurts else 6.0
    peak_density = float(np.mean(dens)) if dens else 0.05

    # --- STFT-based ---------------------------------------------------------
    n_fft, hop = 2048, 512
    f, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    S = np.abs(Z) ** 2  # (freq, time)

    # Keep only frames that actually carry signal - measuring "distortion" in
    # the gaps between notes is what made the old fizz feature meaningless.
    frame_e = S.sum(axis=0)
    loud = frame_e > (np.percentile(frame_e, 90) * 10 ** (-20 / 10))
    if np.sum(loud) >= 8:
        S = S[:, loud]
        frame_e = frame_e[loud]

    f_hi = min(8000.0, 0.45 * sr)
    hf = (f >= 2000) & (f <= f_hi)
    band = (f >= 150) & (f <= f_hi)
    hf_e = S[hf].sum(axis=0)
    band_e = S[band].sum(axis=0) + EPS
    # Energy-weighted so loud frames dominate (a quiet decaying tail should not
    # get the same vote as a strummed chord).
    w = band_e / band_e.sum()
    hf_ratio = float(np.sum(w * (hf_e / band_e)))

    hf_band = S[hf] + EPS
    spec_crest = hf_band.max(axis=0) / np.mean(hf_band, axis=0)
    spec_crest_hf = float(np.sum(w * spec_crest))

    # Spectral flux (normalized): how fast the spectrum changes
    Sn = S / (S.sum(axis=0, keepdims=True) + EPS)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(Sn, axis=1) ** 2, axis=0)))) if Sn.shape[1] > 1 else 0.0

    # Sustain: distortion compresses note decay, so energy holds up longer.
    e_env = np.sqrt(short_term_rms(x, sr, win_s=0.05, hop_s=0.05) ** 2)
    if len(e_env) > 8:
        srt = np.sort(e_env)[::-1]
        top = srt[: max(1, len(srt) // 10)].mean()
        mid = np.median(e_env)
        sustain_ratio = float(mid / (top + EPS))
    else:
        sustain_ratio = 0.3

    # --- MFCC texture -------------------------------------------------------
    fb = mel_filterbank(sr, n_fft, n_mels=40, f_lo=60.0, f_hi=min(10000.0, 0.45 * sr))
    mel = np.log(fb @ S + EPS)
    mfcc = dct(mel, type=2, axis=0, norm="ortho")[1:20]  # drop c0 -> EQ-tilt tolerant
    mfcc_std = np.std(mfcc, axis=1)
    dm = np.diff(mfcc, axis=1)
    mfcc_dstd = np.std(dm, axis=1) if dm.shape[1] > 1 else np.zeros(mfcc.shape[0])

    # --- LTAS ---------------------------------------------------------------
    grid = log_freq_grid(sr)
    fw, pw = ltas(x, sr)
    ltas_db = smooth_spectrum_db(fw, pw, grid)
    ltas_db -= np.median(ltas_db)  # level-invariant

    return ToneFingerprint(
        crest_db=crest_db,
        kurtosis=kurtosis,
        peak_density=peak_density,
        dyn_range_db=dyn_range_db,
        sustain_ratio=sustain_ratio,
        hf_ratio=hf_ratio,
        spec_crest_hf=spec_crest_hf,
        flux=flux,
        mfcc_std=mfcc_std,
        mfcc_dstd=mfcc_dstd,
        ltas_grid=grid,
        ltas_db=ltas_db,
    )


# ----------------------------------------------------------------------------
# Distances - every term is bounded in [0, 1] so no single feature can
# dominate the score (an unbounded relative error on a near-zero feature used
# to outvote every distortion cue).
# ----------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    """Symmetric bounded relative difference in [0, 1]."""
    a, b = float(a), float(b)
    return abs(a - b) / (abs(a) + abs(b) + 1e-9)


def _db(a: float, b: float, scale: float = 8.0) -> float:
    """Bounded distance for dB-valued features (`scale` dB -> 1.0)."""
    return min(abs(float(a) - float(b)) / scale, 1.0)


def _log_rel(a: float, b: float, scale: float = 1.0) -> float:
    """Bounded distance on a log ratio - good for positive, wide-range values."""
    a = max(float(a), 1e-9)
    b = max(float(b), 1e-9)
    return min(abs(np.log(a / b)) / scale, 1.0)


def _vec_dist(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
    return min(d, 1.0)


# How much each cue counts. The waveform-squash group (crest/kurtosis/peak
# density) and the harmonic-generation group (hf_ratio/spec_crest) are what
# separate a clean amp from a cranked one, so they carry the weight.
_WEIGHTS = {
    "crest": 1.6,
    "kurtosis": 1.2,
    "peak_density": 1.0,
    "hf_ratio": 1.2,
    "spec_crest_hf": 1.0,
    "dyn_range": 0.5,
    "sustain": 0.5,
    "flux": 0.3,
    "mfcc_texture": 0.5,
    "mfcc_delta": 0.5,
}

# Penalty for a drive-amount mismatch, applied on top of the per-feature terms.
# Asymmetric on purpose: matching a cranked amp with a clean capture is a much
# worse error than the reverse, so under-driving is punished about twice as
# hard as over-driving.
_UNDERDRIVE_PENALTY = 1.5
_OVERDRIVE_PENALTY = 0.7


def distortion_index(fp: ToneFingerprint) -> float:
    """Rough 0..1 'how driven does this sound' summary (higher = more driven).

    Ranges are calibrated so clean..crunch..hi-gain spread across the scale
    instead of all bunching near zero.
    """
    crest = np.clip((22.0 - fp.crest_db) / 12.0, 0.0, 1.0)   # 22 dB clean -> 10 dB squashed
    kurt = np.clip((10.0 - fp.kurtosis) / 7.5, 0.0, 1.0)     # peaky -> flat
    dens = np.clip(fp.peak_density / 0.20, 0.0, 1.0)
    hf = np.clip(fp.hf_ratio / 0.30, 0.0, 1.0)
    return float(0.35 * crest + 0.25 * kurt + 0.2 * dens + 0.2 * hf)


def nonlinear_distance(cand: ToneFingerprint, target: ToneFingerprint) -> dict:
    """EQ-invariant distance between two fingerprints. Lower = better.

    Returns a dict with a 'total' key and per-component breakdown.
    """
    d = {
        "crest": _WEIGHTS["crest"] * _db(cand.crest_db, target.crest_db, scale=8.0),
        "kurtosis": _WEIGHTS["kurtosis"] * _log_rel(cand.kurtosis, target.kurtosis, scale=1.0),
        "peak_density": _WEIGHTS["peak_density"] * _rel(cand.peak_density, target.peak_density),
        "hf_ratio": _WEIGHTS["hf_ratio"] * _rel(cand.hf_ratio, target.hf_ratio),
        "spec_crest_hf": _WEIGHTS["spec_crest_hf"] * _log_rel(cand.spec_crest_hf, target.spec_crest_hf, scale=1.5),
        "dyn_range": _WEIGHTS["dyn_range"] * _db(cand.dyn_range_db, target.dyn_range_db, scale=12.0),
        "sustain": _WEIGHTS["sustain"] * _rel(cand.sustain_ratio, target.sustain_ratio),
        "flux": _WEIGHTS["flux"] * _rel(cand.flux, target.flux),
        "mfcc_texture": _WEIGHTS["mfcc_texture"] * _vec_dist(cand.mfcc_std, target.mfcc_std),
        "mfcc_delta": _WEIGHTS["mfcc_delta"] * _vec_dist(cand.mfcc_dstd, target.mfcc_dstd),
    }
    # Directional guard: penalize any drive mismatch, under-drive worst.
    gap = distortion_index(target) - distortion_index(cand)
    d["drive_gap"] = (
        _UNDERDRIVE_PENALTY * max(gap, 0.0) + _OVERDRIVE_PENALTY * max(-gap, 0.0)
    )
    d["total"] = float(sum(d.values()))
    return d


def eq_correction_db(cand: ToneFingerprint, target: ToneFingerprint) -> np.ndarray:
    """Required EQ correction (dB, on cand.ltas_grid) to map candidate → target."""
    return target.ltas_db - cand.ltas_db


def eq_penalty(cand: ToneFingerprint, target: ToneFingerprint) -> float:
    """Penalty for how much linear correction the candidate would still need."""
    corr = eq_correction_db(cand, target)
    grid = cand.ltas_grid
    sel = (grid >= 80) & (grid <= 8000)
    return float(np.mean(np.abs(corr[sel])) / 18.0)
