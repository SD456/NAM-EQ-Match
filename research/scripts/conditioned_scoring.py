"""Experimental pitch/phase-conditioned spectral scoring, with shared support.

Conditions come from dry probes once, and from the wet query alone. Candidate
renders must reuse their dry probe's conditions. pYIN confidence is not proof of
monophony: chords, distorted octave errors and recording shifts remain limits.
No playing-strength labels or out-of-recording generalization are inferred here.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from scripts.multidi_scoring import centered_spectral_distance
from tonematch.features import log_freq_grid, smooth_spectrum_db

PITCH_BINS = {"40-51": (40, 51), "52-63": (52, 63),
              "64-75": (64, 75), "76-88": (76, 88)}


def _audio(audio, sr):
    audio = np.asarray(audio, dtype=float)
    if sr != 48000:
        raise ValueError("conditioned scoring requires 48000 Hz")
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio must be finite, nonempty and mono")
    if not np.any(audio):
        raise ValueError("audio must be nonzero")
    return audio


def audio_conditions(audio48, sr48000=48000, cfg=None):
    """Estimate frozen 16 ms conditions over the first eight seconds.

    cfg permits only ``allow_variable_length`` for short engineering fixtures.
    Edge frames (32 ms) are excluded from voicing; unassigned phase frames stay
    in the voiced coverage denominator. Attack is [0,80 ms), sustain [120,400 ms)
    after a detected onset, each bounded by the next onset. Clip time zero is
    not an assumed onset.
    """
    import librosa

    cfg = {} if cfg is None else cfg
    if set(cfg) - {"allow_variable_length"}:
        raise ValueError("unknown conditioning configuration")
    audio = _audio(audio48, sr48000)
    if not cfg.get("allow_variable_length", False) and len(audio) < 8 * sr48000:
        raise ValueError("conditioning requires at least eight seconds")
    audio = audio[:8 * sr48000]
    n_samples = len(audio)
    audio = audio / np.max(np.abs(audio))
    y = resample_poly(audio, 1, 3)
    frame_count = 1 + n_samples // 768
    times = np.arange(frame_count) * 0.016
    # Explicit centered framing preserves the 48 kHz grid even for partial samples.
    framed_y = np.pad(y, (512, 512))[:1024 + (frame_count - 1) * 256]
    f0, voiced_flag, voiced_prob = librosa.pyin(
        framed_y, sr=16000, fmin=82.4, fmax=1319.0, frame_length=1024,
        hop_length=256, resolution=0.5, center=False,
    )
    if len(f0) != frame_count:
        raise ValueError("pYIN frame grid does not match the spectral grid")
    rms = librosa.feature.rms(y=framed_y, frame_length=1024,
                              hop_length=256, center=False)[0]
    rms_reference = np.percentile(rms, 95)
    voiced = (voiced_flag & (voiced_prob >= 0.8) & np.isfinite(f0)
              & (rms > rms_reference * 0.01)
              & (times >= 0.032) & (times < n_samples / sr48000 - 0.032))
    midi = np.full(frame_count, np.nan)
    midi[voiced] = np.rint(librosa.hz_to_midi(f0[voiced]))
    pitch_bin = np.full(frame_count, "", dtype="U5")
    pitch_masks = {}
    for key, (lo, hi) in PITCH_BINS.items():
        mask = voiced & (midi >= lo) & (midi <= hi)
        pitch_bin[mask] = key
        pitch_masks[key] = mask

    envelope = librosa.onset.onset_strength(y=y, sr=16000, hop_length=256,
                                           n_fft=1024)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=16000, hop_length=256, backtrack=False,
        units="frames",
    )
    onset_times = onset_frames[onset_frames < frame_count] * 0.016
    phase = np.full(frame_count, "unassigned", dtype="U10")
    for i, onset in enumerate(onset_times):
        end = onset_times[i + 1] if i + 1 < len(onset_times) else n_samples / sr48000
        age = times - onset
        phase[(age >= 0) & (age < 0.08) & (times < end)] = "attack"
        phase[(age >= 0.12) & (age < 0.4) & (times < end)] = "sustain"
    masks = {f"{key}:{part}": mask & (phase == part)
             for key, mask in pitch_masks.items() for part in ("attack", "sustain")}
    return {
        "frame_count": frame_count, "n_samples": n_samples, "times": times,
        "voiced": voiced, "midi": midi, "pitch_bin": pitch_bin,
        "phase": phase, "masks": masks, "pitch_masks": pitch_masks,
        "diagnostics": {
            "voiced_frames": int(voiced.sum()), "onset_times": onset_times.tolist(),
            "voiced_probability": voiced_prob,
            "rms_reference": float(rms_reference),
            "analysis_seconds": n_samples / sr48000,
            "warning": "pYIN confidence does not establish monophony or correct octaves",
        },
    }


def conditioned_spectra(audio48, conditions, sr48000=48000):
    """Average intact STFT power frames by dry/query masks, then smooth once.

    A longer render may carry a tail; only the explicit condition sample interval
    is analyzed. A short render or a mismatched mask is an error, never truncation.
    """
    import librosa

    audio = _audio(audio48, sr48000)
    n_samples = conditions["n_samples"]
    if n_samples <= 0 or len(audio) < n_samples:
        raise ValueError("render is shorter than the condition interval")
    audio = audio[:n_samples]
    if not np.any(audio):
        raise ValueError("condition interval must be nonzero")
    audio = audio / np.max(np.abs(audio))
    power = np.abs(librosa.stft(audio, n_fft=2048, hop_length=768,
                               center=True, pad_mode="constant")) ** 2
    if power.shape[1] != conditions["frame_count"]:
        raise ValueError("spectral and condition frame grids differ")
    grid = log_freq_grid(sr48000)
    frequencies = np.fft.rfftfreq(2048, 1 / sr48000)
    band = (grid >= 80) & (grid <= 8000)

    def curves(masks):
        result = {}
        for key, mask in masks.items():
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (power.shape[1],):
                raise ValueError("condition masks must match the full spectral frame grid")
            count = int(mask.sum())
            if count:
                curve = smooth_spectrum_db(frequencies, power[:, mask].mean(axis=1),
                                           grid, octave_fraction=2.0)
                result[key] = {"curve": curve - np.median(curve[band]), "n_frames": count}
        return result

    return {"grid": grid, "pitch_curves": curves(conditions["pitch_masks"]),
            "phase_curves": curves(conditions["masks"])}


def common_strata(probe_conditions, query_conditions, mode="pitch", *,
                  min_pitch_frames=8, min_sustain_frames=8, min_attack_frames=3,
                  min_probes=2, min_coverage=0.6):
    """Freeze support from input conditions, independently of any NAM candidate."""
    if mode not in ("pitch", "phase"):
        raise ValueError("mode must be pitch or phase")
    if (len(probe_conditions) != 3 or min_probes not in (2, 3)
            or min(min_pitch_frames, min_sustain_frames, min_attack_frames) < 1
            or not 0 <= min_coverage <= 1):
        raise ValueError("invalid shared-support thresholds or probe count")
    mask_key = "pitch_masks" if mode == "pitch" else "masks"
    query_voiced = np.asarray(query_conditions["voiced"], dtype=bool)
    matched = np.zeros_like(query_voiced)
    support = {}
    min_frames = {}
    for key, query_mask in query_conditions[mask_key].items():
        threshold = (min_pitch_frames if mode == "pitch" else
                     min_attack_frames if key.endswith(":attack") else min_sustain_frames)
        query_mask = np.asarray(query_mask, dtype=bool)
        if query_mask.shape != query_voiced.shape or np.any(query_mask & ~query_voiced):
            raise ValueError("query masks must be aligned subsets of voiced frames")
        indices = []
        for index, probe in enumerate(probe_conditions):
            probe_mask = np.asarray(probe[mask_key].get(key, np.zeros_like(probe["voiced"])), dtype=bool)
            probe_voiced = np.asarray(probe["voiced"], dtype=bool)
            if probe_mask.shape != probe_voiced.shape or np.any(probe_mask & ~probe_voiced):
                raise ValueError("probe masks must be aligned subsets of voiced frames")
            if probe_mask.sum() >= threshold:
                indices.append(index)
        if query_mask.sum() >= threshold and len(indices) >= min_probes:
            support[key] = indices
            min_frames[key] = threshold
            matched |= query_mask
    voiced_frames = int(query_voiced.sum())
    matched_frames = int(matched.sum())
    coverage = matched_frames / voiced_frames if voiced_frames else 0.0
    has_phases = (mode == "pitch" or
                  all(any(key.endswith(":" + part) for key in support)
                      for part in ("attack", "sustain")))
    reason = ("no_voiced_query" if not voiced_frames else
              "no_common_strata" if not support else
              "missing_attack_or_sustain" if not has_phases else
              "insufficient_coverage" if coverage < min_coverage else "accepted")
    return {"accepted": reason == "accepted", "keys": list(support),
            "support_probes": support, "min_frames": min_frames, "coverage": coverage,
            "voiced_frames": voiced_frames, "matched_frames": matched_frames,
            "reason": reason, "mode": mode}


def score_conditioned(candidate_feature_list, query_feature, decision, mode="pitch"):
    """Median across each stratum's fixed probes, then equal mean over strata."""
    if (mode not in ("pitch", "phase") or decision.get("mode") != mode
            or not decision["accepted"] or not decision["keys"]
            or len(candidate_feature_list) != 3):
        raise ValueError("conditioned scoring requires an accepted shared decision and three probes")
    collection = "pitch_curves" if mode == "pitch" else "phase_curves"
    grid = query_feature["grid"]
    scores = []
    for key in decision["keys"]:
        try:
            target = query_feature[collection][key]
            if target["n_frames"] < decision["min_frames"][key]:
                raise ValueError("query stratum has insufficient frames")
            distances = []
            for index in decision["support_probes"][key]:
                feature = candidate_feature_list[index]
                if not np.array_equal(feature["grid"], grid):
                    raise ValueError("candidate and query frequency grids differ")
                candidate = feature[collection][key]
                if candidate["n_frames"] < decision["min_frames"][key]:
                    raise ValueError("candidate stratum has insufficient frames")
                distances.append(centered_spectral_distance(candidate["curve"], target["curve"], grid))
            if len(distances) < 2:
                raise ValueError("each stratum needs two supporting probes")
            scores.append(float(np.median(distances)))
        except KeyError as exc:
            raise ValueError("candidate/query lacks a fixed shared stratum") from exc
    return float(np.mean(scores))
