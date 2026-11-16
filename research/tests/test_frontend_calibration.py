"""Synthetic implementation checks, not evidence of real-DI performance."""
import numpy as np
import pytest

from scripts.frontend_calibration import (
    detect_onsets, estimate_frontend, match_onsets, pitch_metrics,
)


@pytest.fixture(scope="module")
def synthetic_frontends():
    sr = 44100
    audio = np.zeros(int(2.2 * sr) + 7)
    for onset, frequency in ((0.2, 110), (1.2, 440)):
        t = np.arange(int(0.75 * sr)) / sr
        note = np.minimum(t / 0.005, 1) * np.exp(-t) * (
            np.sin(2 * np.pi * frequency * t) + 0.2 * np.sin(4 * np.pi * frequency * t))
        start = int(onset * sr)
        audio[start:start + len(note)] = note
    before = audio.copy()
    estimates = [estimate_frontend(audio, sr, window) for window in (1024, 2048)]
    np.testing.assert_array_equal(audio, before)
    return audio, sr, estimates


def test_known_notes_and_shared_long_window_grid(synthetic_frontends):
    _, _, estimates = synthetic_frontends
    short, long = estimates
    for name in ("times", "rms_ok", "edge_ok"):
        np.testing.assert_array_equal(short[name], long[name])
    for estimate in estimates:
        np.testing.assert_allclose(np.diff(estimate["times"]), 0.016)
        assert not estimate["edge_ok"][:4].any()
        assert not estimate["edge_ok"][-4:].any()
        for onset, frequency in ((0.2, 110), (1.2, 440)):
            core = ((estimate["times"] >= onset + 0.2)
                    & (estimate["times"] < onset + 0.6))
            voiced = core & estimate["raw_voiced"] & np.isfinite(estimate["f0"])
            assert voiced.sum() >= 0.75 * core.sum()
            assert np.median(np.abs(1200 * np.log2(estimate["f0"][voiced] / frequency))) < 50


@pytest.mark.parametrize("window", [1024, 2048])
def test_onsets_keep_original_time_and_no_assumed_zero(synthetic_frontends, window):
    audio, sr, _ = synthetic_frontends
    onsets = detect_onsets(audio, sr, window)
    assert np.all(np.diff(onsets) > 0)
    assert not np.any(onsets == 0)
    matched = match_onsets(onsets, [0.2, 1.2], 0.08)
    assert matched["tp"] == 2
    assert matched["fn"] == 0


@pytest.mark.parametrize("audio,sr,window", [
    (np.zeros(100), 44100, 1024), (np.ones((100, 2)), 44100, 1024),
    (np.array([1, np.nan]), 44100, 1024), (np.array([1, np.inf]), 44100, 1024),
    ([], 44100, 1024), ([1], 8000, 1024), ([1], 44100.0, 1024),
    ([1], True, 1024), ([1], 44100, 512),
])
def test_invalid_analysis_input(audio, sr, window):
    for frontend in (estimate_frontend, detect_onsets):
        with pytest.raises(ValueError):
            frontend(audio, sr, window)


def handmade_estimate():
    return {"times": np.array([0, .2, .4, .6, .8, 1.2, 1.4, 2]),
            "f0": np.array([880, 440, 880, 440, 440, np.nan, np.nan, 880]),
            "probability": np.array([1, .9, .8, .1, .9, 0, 0, 1]),
            "raw_voiced": np.array([True, True, True, True, False, False, False, True]),
            "rms_ok": np.ones(8, dtype=bool), "edge_ok": np.ones(8, dtype=bool),
            "duration_seconds": 2.2}


def test_equal_note_denominator_thresholds_octaves_and_confidence():
    estimate = handmade_estimate()
    events = [{"pitch": 69, "onset": .1, "offset": .9},
              {"pitch": 57, "onset": 1.1, "offset": 1.5},
              {"pitch": 69, "onset": 2.1, "offset": 2.15}]
    low = pitch_metrics(estimate, events, 0, guard_seconds=0)
    high = pitch_metrics(estimate, events, .9, guard_seconds=0)
    assert low["n_notes"] == 2
    assert low["n_notes_excluded_empty_core"] == 1
    assert low["total_reference_frames"] == 6
    assert low["coverage"] == pytest.approx(.375)
    assert low["conditional_exact"] == pytest.approx(2 / 3)
    assert low["full_correct"] == pytest.approx(.25)
    assert low["conditional_octave"] == pytest.approx(1 / 3)
    assert low["n_notes_with_any_accept"] == low["n_notes_with_correct"] == 1
    assert high["coverage"] == pytest.approx(.125)
    assert high["conditional_exact"] == 1
    assert sum(b["n_frames"] for b in low["confidence_bins"]) == 3
    assert sum(b["weight"] for b in low["confidence_bins"]) == pytest.approx(.75)
    assert low["confidence_bins"] == high["confidence_bins"]
    rejected = pitch_metrics(estimate, events, 1, guard_seconds=0)
    assert rejected["coverage"] == rejected["full_correct"] == 0
    assert rejected["conditional_exact"] is None


def test_event_validation_and_shared_guard():
    estimate = handmade_estimate()
    event = {"pitch": 69, "onset": .1, "offset": .9}
    result = pitch_metrics(estimate, [event], 0, guard_seconds=.15)
    assert result["total_reference_frames"] == 2
    for invalid in ([{**event, "pitch": 89}], [{**event, "onset": -.1}],
                    [{**event, "offset": 3}], [event, event], []):
        with pytest.raises(ValueError):
            pitch_metrics(estimate, invalid, 0)
    with pytest.raises(ValueError):
        pitch_metrics(estimate, [event], np.nan)
    estimate["rms_ok"] = np.ones(7, dtype=bool)
    with pytest.raises(ValueError, match="grid"):
        pitch_metrics(estimate, [event], 0)


def test_onset_matching_cardinality_duplicates_and_inclusive_bounds():
    result = match_onsets([1.05, .95, 2], [1, 2.05], .05)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 0)
    assert result["median_signed_ms"] == pytest.approx(-50)
    duplicate = match_onsets([1, 1], [1], 0)
    assert (duplicate["tp"], duplicate["fp"]) == (1, 1)
    assert match_onsets([.96, 1.02], [1, 1.06], .05)["tp"] == 2
    empty = match_onsets([], [0], .05)
    assert empty["f1"] == 0
    assert empty["median_absolute_ms"] is None
    assert match_onsets([0], [0], 0)["f1"] == 1


@pytest.mark.parametrize("predicted,truth,tolerance", [
    ([np.nan], [1], .05), ([1], [-1], .05), ([[1]], [1], .05),
    ([1], [1], -.05), ([1], [1], np.inf),
])
def test_onset_matching_invalid_input(predicted, truth, tolerance):
    with pytest.raises(ValueError):
        match_onsets(predicted, truth, tolerance)
