"""Audio-only stage-six frontend and independent label-based measurements.

pYIN voiced probability is an observation probability, not pitch correctness.
Normalization here is analysis-only and never changes a NAM input waveform.
"""
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def _analysis_audio(audio, sr, window):
    audio = np.asarray(audio, dtype=float)
    if (isinstance(sr, (bool, np.bool_)) or not isinstance(sr, (int, np.integer))
            or sr < 16000):
        raise ValueError("sample rate must be an integer >= 16000 Hz")
    if window not in (1024, 2048):
        raise ValueError("analysis window must be 1024 or 2048 samples")
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio must be finite, nonempty and mono")
    peak = np.max(np.abs(audio))
    if peak == 0:
        raise ValueError("audio must be nonzero")
    divisor = gcd(int(sr), 16000)
    return resample_poly(audio / peak, 16000 // divisor, int(sr) // divisor)


def estimate_frontend(audio, sr, pitch_window):
    """Estimate full-clip centered pitch with a shared 16 ms reference grid."""
    import librosa

    y = _analysis_audio(audio, sr, pitch_window)
    f0, raw_voiced, probability = librosa.pyin(
        y, sr=16000, fmin=82.4, fmax=1319.0, frame_length=pitch_window,
        hop_length=256, resolution=0.5, center=True, pad_mode="constant",
    )
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256,
                              center=True, pad_mode="constant")[0]
    count = 1 + len(y) // 256
    if any(np.shape(array) != (count,) for array in (f0, raw_voiced, probability, rms)):
        raise ValueError("pitch and RMS must match the centered 16 ms frame grid")
    times = np.arange(count) * 0.016
    duration = len(audio) / sr
    return {"times": times, "f0": f0, "raw_voiced": raw_voiced,
            "probability": probability, "rms_ok": rms > 0.01 * np.percentile(rms, 95),
            "edge_ok": (times >= 0.064) & (times < duration - 0.064),
            "duration_seconds": duration}


def detect_onsets(audio, sr, onset_window=1024):
    """Detect full-clip onsets without labels, cropping or a time-zero onset."""
    import librosa

    y = _analysis_audio(audio, sr, onset_window)
    envelope = librosa.onset.onset_strength(y=y, sr=16000, hop_length=256,
                                           n_fft=onset_window)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=16000,
                                        hop_length=256, backtrack=False, units="frames")
    times = np.unique(np.asarray(frames, dtype=float) * 0.016)
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("onset detector returned invalid times")
    return times


def pitch_metrics(estimate, events, threshold, guard_seconds=0.114):
    """Equal-note pitch metrics over [onset+guard, offset-guard) centers.

    Zero-answer notes retain their denominator. Empty cores are excluded and
    counted explicitly. Confidence bins include raw-voiced eligible core frames
    before thresholding, with the same reciprocal-core-size frame weights.
    """
    if (not np.isfinite(threshold) or not 0 <= threshold <= 1
            or not np.isfinite(guard_seconds) or guard_seconds < 0):
        raise ValueError("invalid threshold or guard")
    names = ("times", "f0", "raw_voiced", "probability", "rms_ok", "edge_ok")
    arrays = {name: np.asarray(estimate[name]) for name in names}
    times = arrays["times"]
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or any(a.shape != times.shape for a in arrays.values())):
        raise ValueError("estimate arrays must share an ordered finite time grid")
    for name in ("raw_voiced", "rms_ok", "edge_ok"):
        if arrays[name].dtype != bool:
            raise ValueError("estimate masks must be boolean")
    probability, f0 = arrays["probability"], arrays["f0"]
    finite_prob = np.isfinite(probability)
    if np.any(finite_prob & ((probability < 0) | (probability > 1))):
        raise ValueError("voiced probability must lie in [0, 1]")
    if not events:
        raise ValueError("at least one monophonic event is required")
    events = sorted(events, key=lambda e: e["onset"])
    previous_offset = 0.0
    for event in events:
        pitch, onset, offset = (event[k] for k in ("pitch", "onset", "offset"))
        if (not np.isfinite([pitch, onset, offset]).all()
                or pitch != int(pitch) or not 40 <= pitch <= 88
                or onset < previous_offset or offset <= onset
                or offset > estimate.get("duration_seconds", np.inf)):
            raise ValueError("events must have valid MIDI 40..88 and nonoverlapping clip bounds")
        previous_offset = offset
    eligible = (arrays["raw_voiced"] & finite_prob & np.isfinite(f0) & (f0 > 0)
                & arrays["rms_ok"] & arrays["edge_ok"])
    accepted = eligible & (probability >= threshold)
    boundaries = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.00001]
    bins = [{"lower": lo, "upper": hi, "n_frames": 0, "weight": 0.0,
             "correct_weight": 0.0} for lo, hi in zip(boundaries, boundaries[1:])]
    notes = []
    for event in events:
        core = ((times >= event["onset"] + guard_seconds)
                & (times < event["offset"] - guard_seconds) & arrays["edge_ok"])
        n = int(core.sum())
        row = {**event, "reference_frames": n, "accepted_frames": 0,
               "correct_frames": 0, "octave_frames": 0}
        if n:
            reference = 440 * 2 ** ((event["pitch"] - 69) / 12)
            cents = np.full(len(times), np.inf)
            cents[eligible] = 1200 * np.log2(f0[eligible] / reference)
            correct = np.abs(cents) <= 50
            octave = np.abs(np.abs(cents) - 1200) <= 50
            for key, mask in (("accepted_frames", accepted),
                              ("correct_frames", accepted & correct),
                              ("octave_frames", accepted & octave)):
                row[key] = int(np.sum(core & mask))
            for bucket in bins:
                mask = (core & eligible & (probability >= bucket["lower"])
                        & (probability < bucket["upper"]))
                bucket["n_frames"] += int(mask.sum())
                bucket["weight"] += float(mask.sum() / n)
                bucket["correct_weight"] += float(np.sum(mask & correct) / n)
        notes.append(row)
    scored = [row for row in notes if row["reference_frames"]]
    count = len(scored)
    a, c, o = (sum(row[key] / row["reference_frames"] for row in scored)
               for key in ("accepted_frames", "correct_frames", "octave_frames"))
    for bucket in bins:
        bucket["conditional_exact"] = (bucket["correct_weight"] / bucket["weight"]
                                        if bucket["weight"] else None)
    return {"n_notes": count, "n_notes_excluded_empty_core": len(notes) - count,
            "total_reference_frames": sum(row["reference_frames"] for row in scored),
            "accepted_weight": a, "correct_weight": c, "octave_weight": o,
            "coverage": a / count if count else None,
            "conditional_exact": c / a if a else None,
            "full_correct": c / count if count else None,
            "conditional_octave": o / a if a else None,
            "n_notes_with_any_accept": sum(row["accepted_frames"] > 0 for row in scored),
            "n_notes_with_correct": sum(row["correct_frames"] > 0 for row in scored),
            "per_note": notes, "confidence_bins": bins}


def match_onsets(predicted, truth, tolerance):
    """Maximum-cardinality ordered one-to-one matching; duplicates remain events."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("onset tolerance must be finite and nonnegative")
    predicted, truth = (np.sort(np.asarray(values, dtype=float)) for values in (predicted, truth))
    if any(values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)
           for values in (predicted, truth)):
        raise ValueError("onset times must be finite nonnegative vectors")
    i = j = 0
    errors = []
    while i < len(predicted) and j < len(truth):
        if predicted[i] < truth[j] - tolerance:
            i += 1
        elif predicted[i] > truth[j] + tolerance:
            j += 1
        else:
            errors.append(float(predicted[i] - truth[j]))
            i += 1
            j += 1
    tp = len(errors)
    fp, fn = len(predicted) - tp, len(truth) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "errors_seconds": errors,
            "precision": tp / len(predicted) if len(predicted) else 0.0,
            "recall": tp / len(truth) if len(truth) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "median_signed_ms": float(np.median(errors) * 1000) if errors else None,
            "median_absolute_ms": float(np.median(np.abs(errors)) * 1000) if errors else None}
