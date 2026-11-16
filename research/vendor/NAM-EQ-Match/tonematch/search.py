"""Candidate ranking: find the NAM capture + input gain whose nonlinear
character best matches the target recording.

Two-stage search:
  1. Coarse: short DI segment, coarse gain grid, all captures.
  2. Refine: longer segment, finer gain grid, top-N captures only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import active_segment, db_to_lin
from .features import (
    ToneFingerprint,
    eq_penalty,
    extract_fingerprint,
    nonlinear_distance,
)


def _audio(x, label):
    x = np.asarray(x)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"{label} must be nonempty, finite mono audio")
    if not np.any(x):
        raise ValueError(f"{label} is silent")
    return x


@dataclass
class CandidateResult:
    capture: object
    gain_db: float
    score: float
    nl_distance: float
    eq_penalty: float
    breakdown: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return getattr(self.capture, "name", str(self.capture))


def evaluate_gain(
    capture,
    di: np.ndarray,
    sr: int,
    gain_db: float,
    target_fp: ToneFingerprint,
    eq_weight: float = 0.25,
) -> CandidateResult:
    di = _audio(di, "DI")
    if sr <= 0 or getattr(capture, "sample_rate", None) != sr:
        raise ValueError("Capture sample rate must match search sample rate")
    if not np.isfinite(gain_db) or not np.isfinite(eq_weight) or eq_weight < 0:
        raise ValueError("Gain and nonnegative EQ weight must be finite")
    y = capture.process(di * db_to_lin(gain_db))
    y = _audio(y, f"Output of {getattr(capture, 'name', capture)}")
    fp = extract_fingerprint(y, sr)
    nl = nonlinear_distance(fp, target_fp)
    pen = eq_penalty(fp, target_fp)
    if not np.isfinite(list(nl.values())).all() or not np.isfinite(pen):
        raise ValueError(f"Non-finite score for {getattr(capture, 'name', capture)}")
    score = nl["total"] + eq_weight * pen
    if not np.isfinite(score):
        raise ValueError("Non-finite weighted search score")
    return CandidateResult(
        capture=capture,
        gain_db=gain_db,
        score=score,
        nl_distance=nl["total"],
        eq_penalty=pen,
        breakdown=nl,
    )


def best_gain_for_capture(
    capture,
    di: np.ndarray,
    sr: int,
    target_fp: ToneFingerprint,
    gains_db: np.ndarray,
    progress=None,
) -> CandidateResult:
    gains_db = np.asarray(gains_db, dtype=float)
    if gains_db.ndim != 1 or not gains_db.size or not np.isfinite(gains_db).all():
        raise ValueError("Gain grid must be nonempty and finite")
    best: CandidateResult | None = None
    for g in gains_db:
        r = evaluate_gain(capture, di, sr, float(g), target_fp)
        if best is None or r.score < best.score:
            best = r
        if progress:
            progress()
    return best


def rank_captures(
    captures: list,
    di: np.ndarray,
    target: np.ndarray,
    sr: int,
    gain_range_db: tuple[float, float] = (-12.0, 12.0),
    coarse_seg_s: float = 8.0,
    refine_seg_s: float = 16.0,
    refine_top: int = 5,
    progress_cb=None,
) -> list[CandidateResult]:
    """Return only refined top-N results; coarse scores only select finalists."""
    di, target = _audio(di, "DI"), _audio(target, "Target")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("Sample rate must be a positive integer")
    if not isinstance(refine_top, (int, np.integer)) or refine_top < 1:
        raise ValueError("refine_top must be a positive integer")
    if not np.isfinite([coarse_seg_s, refine_seg_s]).all() or not 0 < coarse_seg_s <= refine_seg_s:
        raise ValueError("Segment durations must be finite and 0 < coarse <= refine")
    if int(coarse_seg_s * sr) < 1:
        raise ValueError("Coarse segment must contain at least one sample")
    lo, hi = gain_range_db
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("Gain range must be finite and ordered")
    for capture in captures:
        if getattr(capture, "sample_rate", None) != sr:
            raise ValueError("Capture sample rate must match search sample rate")
    if not captures:
        return []

    coarse_gains = np.unique(np.append(np.arange(lo, hi, 6.0), hi))
    di_refine = active_segment(di, sr, refine_seg_s)
    di_coarse = active_segment(di_refine, sr, coarse_seg_s)
    # DI and reference are independent performances, so share only the reference fingerprint.
    target_fp = extract_fingerprint(target, sr)

    n_coarse = len(captures) * len(coarse_gains)
    n_total = n_coarse + min(refine_top, len(captures)) * 7
    done = [0]

    def tick():
        done[0] += 1
        if progress_cb:
            progress_cb(min(done[0] / n_total, 1.0), "searching")

    # Stage 1: coarse
    coarse = [
        best_gain_for_capture(c, di_coarse, sr, target_fp, coarse_gains, progress=tick)
        for c in captures
    ]
    coarse.sort(key=lambda r: r.score)

    # Stage 2: refine top-N with finer gains on longer audio
    finalists = coarse[:refine_top]
    fine_grids = [np.unique(np.clip(r.gain_db + np.arange(-3., 4.), lo, hi)) for r in finalists]
    n_total = n_coarse + sum(len(grid) for grid in fine_grids)
    results = []
    for r, fine in zip(finalists, fine_grids):
        rr = best_gain_for_capture(r.capture, di_refine, sr, target_fp, fine, progress=tick)
        results.append(rr)
    results.sort(key=lambda r: r.score)
    return results
