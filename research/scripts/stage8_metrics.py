"""Fixed 23-coordinate ERB features and fold-local shared within-chain scores."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.linalg import helmert
from sklearn.covariance import LedoitWolf, OAS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/NAM-EQ-Match"))
from tonematch.audio import _validate_audio, _validate_sample_rate
from tonematch.features import ltas


N_BANDS = 24
N_DIMS = N_BANDS - 1
HELMERT_BASIS = helmert(N_BANDS, full=False).T
HELMERT_BASIS.setflags(write=False)
METHODS = ("euclidean", "diagonal", "ledoit_wolf", "oas")


def erb_feature(audio, sr=48000):
    """Whole-file ERB triangular log energies, projected off constant level.

    The 26 knots are uniform in 21.4*log10(1+0.00437*f), from 80 to
    8000 Hz. Triangles are linear in that ERB coordinate, with unit peaks.
    Welch uses the existing Hann/8192/50%-overlap LTAS definition. Integrated
    band powers use a relative -120 dB floor before 10*log10 and projection.
    Peak normalization is output-feature processing only, never a NAM input.
    """
    _validate_sample_rate(sr)
    if sr < 16000:
        raise ValueError("ERB features require a sample rate of at least 16000 Hz")
    if np.iscomplexobj(audio):
        raise ValueError("audio must be real")
    audio = _validate_audio(audio)
    peak = np.max(np.abs(audio))
    if peak == 0:
        raise ValueError("silent audio has no spectral shape")
    freqs, power = ltas(audio / peak, sr)
    erb = 21.4 * np.log10(1 + 0.00437 * freqs)
    knots = np.linspace(21.4 * np.log10(1 + 0.00437 * 80),
                        21.4 * np.log10(1 + 0.00437 * 8000), N_BANDS + 2)
    rising = (erb[None, :] - knots[:-2, None]) / np.diff(knots)[:-1, None]
    falling = (knots[2:, None] - erb[None, :]) / np.diff(knots)[1:, None]
    weights = np.maximum(0, np.minimum(rising, falling))
    if np.any(weights.sum(axis=1) == 0):
        raise ValueError("audio is too short to resolve every ERB band")
    energies = weights @ power * (freqs[1] - freqs[0])
    maximum = energies.max()
    if maximum <= 0 or not np.isfinite(energies).all():
        raise ValueError("audio has no finite 80-8000 Hz energy")
    bands_db = 10 * np.log10(np.maximum(energies / maximum, 1e-12))
    return bands_db @ HELMERT_BASIS


def _real_array(value, label):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _condition(eigenvalues):
    return float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None


def fit_model(X, method):
    """Fit ONLY caller-supplied training DIs, shaped (chain,gain,DI,23).

    Centers and ML within-chain covariances weight every training DI equally.
    LW/OAS estimate each chain separately before equal-weight chain averaging;
    neither chains nor gains are concatenated into extra sample rows. A fixed
    numeric eigenvalue floor max(1e-10, 1e-10*max_eigenvalue) is applied to the
    shared estimate, and its size is exposed in the diagnostics.
    """
    X = _real_array(X, "training features")
    if (X.ndim != 4 or min(X.shape[:2]) < 1 or X.shape[2] < 2
            or X.shape[3] != N_DIMS):
        raise ValueError("training features require (chain>=1,gain>=1,DI>=2,23)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    centers = X.mean(axis=2)
    residuals = X - centers[:, :, None, :]
    if not np.isfinite(residuals).all():
        raise ValueError("training features exceed numeric range")
    covariances, precisions, logdets, diagnostics = [], [], [], []
    for gi in range(X.shape[1]):
        residual = residuals[:, gi]
        sample = np.einsum("cni,cnj->cij", residual, residual) / X.shape[2]
        if not np.isfinite(sample).all():
            raise ValueError("training covariance exceeds numeric range")
        shared_sample = sample.mean(axis=0)
        shrinkage = None
        if method == "euclidean":
            estimated = np.eye(N_DIMS)
        elif method == "diagonal":
            estimated = np.diag(np.diag(shared_sample))
        else:
            estimator = LedoitWolf if method == "ledoit_wolf" else OAS
            fits = [estimator(assume_centered=True, store_precision=False).fit(row)
                    for row in residual]
            estimated = np.mean([fit.covariance_ for fit in fits], axis=0)
            shrinkage = [float(fit.shrinkage_) for fit in fits]
        eigenvalues, vectors = np.linalg.eigh(estimated)
        floor = max(1e-10, 1e-10 * float(eigenvalues[-1]))
        stabilized = np.maximum(eigenvalues, floor)
        covariance = (vectors * stabilized) @ vectors.T
        precision = (vectors / stabilized) @ vectors.T
        logdet = float(np.log(stabilized).sum())
        sample_eigenvalues = np.linalg.eigvalsh(shared_sample)
        probabilities = stabilized / stabilized.sum()
        diagnostics.append({
            "sample_eigenvalues": sample_eigenvalues.tolist(),
            "estimated_eigenvalues": eigenvalues.tolist(),
            "eigenvalues": stabilized.tolist(),
            "sample_condition": _condition(sample_eigenvalues),
            "condition": _condition(stabilized),
            "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
            "shrinkage_per_chain": shrinkage,
            "eigenvalue_floor": floor,
            "floored_eigenvalues": int(np.count_nonzero(eigenvalues < floor)),
            "logdet": logdet,
        })
        covariances.append(covariance)
        precisions.append(precision)
        logdets.append(logdet)
    return {"method": method, "centers": centers, "covariance": np.asarray(covariances),
            "precision": np.asarray(precisions), "logdet": np.asarray(logdets),
            "diagnostics": diagnostics, "n_train_di": X.shape[2]}


def score_model(query, model):
    """Return (chain,gain) quadratic-form + gain-specific logdet; lower wins."""
    query = _real_array(query, "query")
    centers = _real_array(model["centers"], "centers")
    precision = _real_array(model["precision"], "precision")
    logdet = _real_array(model["logdet"], "logdet")
    if (query.shape != (N_DIMS,) or centers.ndim != 3 or centers.shape[-1] != N_DIMS
            or min(centers.shape[:2]) < 1
            or precision.shape != (centers.shape[1], N_DIMS, N_DIMS)
            or logdet.shape != (centers.shape[1],)):
        raise ValueError("query/model must share 23 coordinates and gain dimensions")
    difference = query - centers
    scores = np.einsum("cgi,gij,cgj->cg", difference, precision, difference) + logdet
    if not np.isfinite(scores).all():
        raise ValueError("scores exceed numeric range")
    return scores
