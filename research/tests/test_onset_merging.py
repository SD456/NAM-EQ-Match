"""Synthetic contracts only; no real-audio stage-seven evidence."""
import numpy as np
import pytest

from scripts.frontend_calibration import detect_onsets, match_onsets
from scripts.onset_merging import (
    onset_peaks, merge_peaks, match_onset_indices, pair_classes, safety_metrics,
    near_label_unmatched_fp_count,
)


def test_first_is_inclusive_anchored_and_keeps_original_indices():
    frames, strengths = np.array([0, 2, 4, 7]), np.array([1., 4., 3., 2.])
    result = merge_peaks(frames, strengths, "first", 2)
    assert result == {"kept_indices": [0, 2, 3], "suppressed_by": {1: 0}}
    np.testing.assert_array_equal(frames, [0, 2, 4, 7])
    np.testing.assert_array_equal(strengths, [1, 4, 3, 2])


def test_strongest_ranking_ties_and_only_kept_suppressors():
    assert merge_peaks([0, 2, 4], [1, 3, 2], "strongest", 2) == {
        "kept_indices": [1], "suppressed_by": {2: 1, 0: 1}}
    assert merge_peaks([0, 2, 4], [3, 3, 3], "strongest", 2) == {
        "kept_indices": [0, 2], "suppressed_by": {1: 0}}
    result = merge_peaks([0, 2, 4], [3, 1, 4], "strongest", 2)
    assert result == {"kept_indices": [0, 2], "suppressed_by": {1: 2}}


@pytest.mark.parametrize("method", ["first", "strongest"])
def test_empty_merge(method):
    assert merge_peaks([], [], method, 2) == {"kept_indices": [], "suppressed_by": {}}


@pytest.mark.parametrize("frames,strengths,method,radius", [
    ([0, 0], [1, 1], "first", 2), ([2, 1], [1, 1], "first", 2),
    ([1.0], [1], "first", 2), ([-1], [1], "first", 2),
    ([1], [np.nan], "first", 2), ([1], [-1], "first", 2),
    ([1], [], "first", 2), ([1], [1], "bad", 2),
    ([1], [1], "first", True), ([1], [1], "first", 2.0),
    ([1], [1], "first", 0), ([True], [1], "first", 2),
])
def test_merge_rejects_invalid_input(frames, strengths, method, radius):
    with pytest.raises(ValueError):
        merge_peaks(frames, strengths, method, radius)


@pytest.mark.parametrize("window", [1024, 2048])
def test_full_synthetic_audio_exactly_matches_old_detector(window):
    sr = 44100
    audio = np.zeros(sr)
    t = np.arange(sr // 5) / sr
    pulse = np.minimum(t / .002, 1) * np.exp(-20 * t) * np.sin(2 * np.pi * 220 * t)
    for start in (sr // 5, sr * 3 // 5):
        audio[start:start + len(pulse)] = pulse
    before = audio.copy()
    peaks = onset_peaks(audio, sr, window)
    np.testing.assert_array_equal(peaks["times"], detect_onsets(audio, sr, window))
    np.testing.assert_array_equal(peaks["times"], peaks["frames"] * .016)
    assert len(peaks["frames"]) >= 2
    assert np.isfinite(peaks["strengths"]).all()
    np.testing.assert_array_equal(audio, before)


@pytest.mark.parametrize("predicted,truth,tolerance,expected", [
    ([2, 1, 1], [1, 2, 1], 0, [[1, 0], [2, 2], [0, 1]]),
    ([.96, 1.02], [1, 1.06], .05, [[0, 0], [1, 1]]),
    ([1.02, .95], [1], .05, [[1, 0]]),
    ([1.05, .95, 2], [1, 2.05], .05, [[1, 0], [2, 1]]),
    ([], [0], .05, []), ([0], [], 0, []),
])
def test_index_matching_preserves_stable_ids_and_old_metrics(predicted, truth, tolerance, expected):
    result = match_onset_indices(predicted, truth, tolerance)
    assert result["pairs"] == expected
    assert result["metrics"] == match_onsets(predicted, truth, tolerance)
    assert sorted([p for p, _ in expected] + result["fp_indices"]) == list(range(len(predicted)))
    assert sorted([g for _, g in expected] + result["fn_indices"]) == list(range(len(truth)))


@pytest.mark.parametrize("predicted,truth,tolerance", [
    ([np.nan], [0], .05), ([0], [-1], .05), ([[0]], [0], .05),
    ([0], [0], -.1), ([0], [0], np.inf),
])
def test_invalid_matching(predicted, truth, tolerance):
    with pytest.raises(ValueError):
        match_onset_indices(predicted, truth, tolerance)


def test_pair_classes_adjacent_only_and_ignore_ringing_tails():
    events = [{"onset": onset, "pitch": pitch, "offset": 99}
              for onset, pitch in ((0, 60), (.2, 62), (.4, 62), (.7, 62), (.8, 64))]
    assert pair_classes(events) == {"fast_change": [[0, 1], [3, 4]],
                                    "repeat": [[1, 2], [2, 3]], "fast_repeat": [[1, 2]]}
    with pytest.raises(ValueError):
        pair_classes(events[::-1])


def test_microsecond_grid_makes_inclusive_boundaries_exact():
    events = [{"onset": 4.2368, "pitch": 60}, {"onset": 4.4368, "pitch": 62}]
    assert pair_classes(events)["fast_change"] == [[0, 1]]
    assert match_onset_indices([.1], [.15], .05)["pairs"] == [[0, 0]]
    assert match_onset_indices([.1], [.150001], .05)["pairs"] == []
    matches = match_onset_indices([.1, .15], [.1], .05)
    assert near_label_unmatched_fp_count(matches, [.1, .15], [.1]) == 1


def test_rematched_alternate_peak_recovers_truth_and_losses_are_by_truth_id():
    events = [{"onset": onset, "pitch": pitch}
              for onset, pitch in ((0, 60), (.1, 62), (.2, 62))]
    raw = match_onset_indices([0, .1, .12, .2], [0, .1, .2], .05)
    recovered = match_onset_indices([0, .12, .2], [0, .1, .2], .05)
    assert safety_metrics(raw, recovered, events)["fast_change"]["baseline_recovered_later_lost"] == []
    lost = match_onset_indices([0, .2], [0, .1, .2], .05)
    result = safety_metrics(raw, lost, events)
    assert result["fast_change"]["baseline_recovered_later_lost"] == [1]
    assert result["fast_change"]["baseline_both_pair_lost"] == [[0, 1]]
    assert result["repeat"]["baseline_both_pair_lost"] == [[1, 2]]
    assert result["repeat"]["merged_later_recall"] == 1
    assert result["repeat"]["merged_both_pair_rate"] == 0


def test_near_label_proxy_counts_only_unmatched_near_matched_labels_once():
    predicted, truth = [0, .04, .1, .2, .21], [0, .1, .3]
    matches = match_onset_indices(predicted, truth, .05)
    assert near_label_unmatched_fp_count(matches, predicted, truth) == 1
    assert near_label_unmatched_fp_count(match_onset_indices([.1], [], .05), [.1], []) == 0
