"""Stage-seven audio-only peak merging and assignment-relative label checks."""
import numpy as np

from scripts.frontend_calibration import _analysis_audio


def _times(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("times must be finite nonnegative vectors")
    return values


def onset_peaks(audio, sr, window=1024):
    """Expose the unchanged stage-six detector's frames and raw strengths.

    The caller handles exactly silent audio; the shared normalizer rejects it.
    """
    import librosa

    y = _analysis_audio(audio, sr, window)
    envelope = librosa.onset.onset_strength(y=y, sr=16000, hop_length=256,
                                           n_fft=window)
    frames = np.unique(librosa.onset.onset_detect(
        onset_envelope=envelope, sr=16000, hop_length=256,
        backtrack=False, units="frames"))
    if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
            or np.any(frames < 0) or np.any(frames >= len(envelope))):
        raise ValueError("detector returned invalid frames")
    strengths = np.asarray(envelope)[frames]
    if not np.isfinite(strengths).all() or np.any(strengths < 0):
        raise ValueError("detector returned invalid strengths")
    return {"frames": frames, "times": frames * 0.016, "strengths": strengths}


def merge_peaks(frames, strengths, method, radius_frames):
    """Keep original peaks with anchored-first or strength-ranked suppression."""
    frames, strengths = np.asarray(frames), np.asarray(strengths, dtype=float)
    if (frames.ndim != 1 or strengths.shape != frames.shape
            or (frames.size and not np.issubdtype(frames.dtype, np.integer))
            or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1])
            or not np.isfinite(strengths).all() or np.any(strengths < 0)):
        raise ValueError("frames must be ordered unique nonnegative integers with finite strengths")
    if (isinstance(radius_frames, (bool, np.bool_))
            or not isinstance(radius_frames, (int, np.integer)) or radius_frames <= 0):
        raise ValueError("radius must be a positive integer number of frames")
    if method not in ("first", "strongest"):
        raise ValueError("method must be first or strongest")
    order = (range(len(frames)) if method == "first" else
             sorted(range(len(frames)), key=lambda i: (-strengths[i], int(frames[i]))))
    kept, suppressed = [], {}
    for i in order:
        neighbors = kept[-1:] if method == "first" else kept
        suppressor = next((j for j in neighbors
                           if abs(int(frames[i]) - int(frames[j])) <= radius_frames), None)
        if suppressor is None:
            kept.append(i)
        else:
            suppressed[i] = suppressor
    return {"kept_indices": sorted(kept), "suppressed_by": suppressed}


def match_onset_indices(predicted, truth, tolerance):
    """Ordered greedy matching on the stage-seven integer microsecond grid.

    Stable original indices and unrounded signed errors are retained. The grid
    makes inclusive tolerance boundaries exact for 16 ms peaks / XML labels.
    """
    predicted, truth = _times(predicted), _times(truth)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    pred_us, truth_us = (np.rint(a * 1e6) for a in (predicted, truth))
    tolerance_us = round(tolerance * 1e6)
    pred_order, truth_order = (np.argsort(a, kind="stable") for a in (predicted, truth))
    i = j = 0
    pairs, errors = [], []
    while i < len(predicted) and j < len(truth):
        p, g = int(pred_order[i]), int(truth_order[j])
        if pred_us[p] < truth_us[g] - tolerance_us:
            i += 1
        elif pred_us[p] > truth_us[g] + tolerance_us:
            j += 1
        else:
            pairs.append([p, g])
            errors.append(float(predicted[p] - truth[g]))
            i += 1
            j += 1
    matched_pred, matched_truth = ({pair[k] for pair in pairs} for k in (0, 1))
    tp = len(pairs)
    fp, fn = len(predicted) - tp, len(truth) - tp
    metrics = {"tp": tp, "fp": fp, "fn": fn, "errors_seconds": errors,
               "precision": tp / len(predicted) if len(predicted) else 0.0,
               "recall": tp / len(truth) if len(truth) else 0.0,
               "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "median_signed_ms": float(np.median(errors) * 1000) if errors else None,
               "median_absolute_ms": float(np.median(np.abs(errors)) * 1000) if errors else None}
    return {"pairs": pairs,
            "fp_indices": [k for k in range(len(predicted)) if k not in matched_pred],
            "fn_indices": [k for k in range(len(truth)) if k not in matched_truth],
            "metrics": metrics}


def pair_classes(events, fast_ioi_seconds=0.2):
    """Classify adjacent picked notes on the integer microsecond IOI grid.

    Ringing offsets are irrelevant; inputs are prevalidated ordinary picks.
    """
    if not np.isfinite(fast_ioi_seconds) or fast_ioi_seconds <= 0:
        raise ValueError("fast IOI must be finite and positive")
    onsets = _times([event["onset"] for event in events])
    pitches = np.asarray([event["pitch"] for event in events], dtype=float)
    if (np.any(np.diff(onsets) <= 0) or not np.isfinite(pitches).all()
            or np.any(pitches != np.floor(pitches))):
        raise ValueError("events must have strictly increasing onsets and finite integer MIDI")
    result = {"fast_change": [], "repeat": [], "fast_repeat": []}
    for i in range(1, len(events)):
        pair = [i - 1, i]
        fast = round((onsets[i] - onsets[i - 1]) * 1e6) <= round(fast_ioi_seconds * 1e6)
        if pitches[i] == pitches[i - 1]:
            result["repeat"].append(pair)
            if fast:
                result["fast_repeat"].append(pair)
        elif fast:
            result["fast_change"].append(pair)
    return result


def safety_metrics(raw_matches, merged_matches, events):
    """Compare independent rematches by truth ID, not by removed peak identity.

    These are assignment-relative losses, not physical false-merge guarantees.
    """
    raw, merged = ({g for _, g in matches["pairs"]}
                   for matches in (raw_matches, merged_matches))
    if any(g < 0 or g >= len(events) for g in raw | merged):
        raise ValueError("matched truth indices exceed event bounds")
    result = {"raw_matched_indices": sorted(raw), "merged_matched_indices": sorted(merged)}
    for name, pairs in pair_classes(events).items():
        count = len(pairs)
        row = {"pairs": pairs, "support": count,
               "baseline_recovered_later_lost": sorted({b for _, b in pairs if b in raw - merged}),
               "baseline_both_pair_lost": [pair for pair in pairs
                                            if set(pair) <= raw and not set(pair) <= merged]}
        for prefix, matched in (("raw", raw), ("merged", merged)):
            later = [b for _, b in pairs if b in matched]
            both = [pair for pair in pairs if set(pair) <= matched]
            row.update({f"{prefix}_later_hits": len(later), f"{prefix}_both_pair_hits": len(both),
                        f"{prefix}_later_recall": len(later) / count if count else None,
                        f"{prefix}_both_pair_rate": len(both) / count if count else None,
                        f"{prefix}_later_matched_indices": later,
                        f"{prefix}_both_matched_pairs": both})
        result[name] = row
    return result


def near_label_unmatched_fp_count(matches, predicted, truth, tol=0.05):
    """Count unmatched predictions near matched labels on the microsecond grid."""
    predicted, truth = _times(predicted), _times(truth)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    predicted_us = np.rint(predicted * 1e6)
    matched_truth_us = np.rint(truth[[g for _, g in matches["pairs"]]] * 1e6)
    tolerance_us = round(tol * 1e6)
    return sum(bool(np.any(np.abs(predicted_us[p] - matched_truth_us) <= tolerance_us))
               for p in matches["fp_indices"])
