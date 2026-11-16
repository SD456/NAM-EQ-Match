"""Gate and abstention checks without NAM rendering or real-audio analysis."""
import json

import numpy as np
import pytest

from scripts.stage5_benchmark import METHODS, PRIMARY, select, summarize
from scripts import stage5_benchmark


def rows_fixture():
    rows = []
    for eq_on in (False, True):
        for index in range(16):
            methods = {}
            for method in METHODS:
                success = index < (13 if method == PRIMARY else 12)
                methods[method] = {"accepted": True, "top1": success, "top3": success,
                                   "gain_db": 0, "forced_truth_gain_db": 0,
                                   "nominal_forced_truth_error_db": 1.0}
            rows.append({"eq_on": eq_on, "methods": methods})
    return rows


def test_gate_passes_only_predeclared_primary():
    rows = rows_fixture()
    summary, gate = summarize(rows)
    assert gate["passed"]
    assert gate["decision"] == "continue_new_take_confirmation"
    assert not gate["production_replacement"]
    assert summary["E0"][PRIMARY]["full_top1"] == 13
    rows[0]["methods"][PRIMARY]["top1"] = False
    assert not summarize(rows)[1]["checks"]["e0_top1"]


def test_abstention_keeps_full_denominator_and_same_subset_controls():
    rows = rows_fixture()
    for row in rows:
        if row is rows[0] or row is rows[16]:
            row["methods"][PRIMARY]["accepted"] = False
            row["methods"][METHODS[0]]["nominal_forced_truth_error_db"] = 100
    summary, gate = summarize(rows)
    primary = summary["E0"][PRIMARY]
    assert primary["total"] == 16 and primary["accepted"] == 15
    assert primary["full_top1"] == 12
    assert primary["full_top1_accuracy"] == 12 / 16
    assert primary["accepted_accuracy"] == 12 / 15
    assert summary["E0"]["same_accepted_subset_controls"][PRIMARY][METHODS[0]]["total"] == 15
    assert not gate["passed"]


def test_zero_acceptance_has_no_median_or_accuracy():
    rows = rows_fixture()
    for row in rows:
        row["methods"][PRIMARY]["accepted"] = False
    summary, gate = summarize(rows)
    assert summary["E0"][PRIMARY]["median_forced_gain_error_db"] is None
    assert summary["E0"][PRIMARY]["accepted_accuracy"] is None
    assert summary["E0"][PRIMARY]["full_top1"] == 0
    assert not gate["passed"]
    assert not gate["checks"]["e0_forced_gain"]


def test_gain_gate_uses_better_control_on_primary_accepted_rows():
    rows = rows_fixture()
    for row in rows[:16]:
        row["methods"][METHODS[1]]["nominal_forced_truth_error_db"] = 0.5
    assert not summarize(rows)[1]["checks"]["e0_forced_gain"]
    for row in rows[:5]:
        row["methods"][PRIMARY]["accepted"] = False
    assert not summarize(rows)[1]["checks"]["coverage"]


def test_rejected_row_does_not_rank_zero_scores():
    decision = {"accepted": False, "coverage": 0.2, "reason": "insufficient_coverage"}
    result = select(np.zeros((12, 25)), list(range(-12, 13)), 0, 0, decision)
    assert result["winner"] is None and result["truth_rank"] is None
    assert result["all_gain_scores"] is None and not result["top1"]


def test_selection_preserves_stable_rig_and_gain_ties():
    scores = np.ones((12, 25))
    scores[3, 18] = 0
    result = select(scores, list(range(-12, 13)), 3, 0)
    assert result["winner"] == 3 and result["gain_db"] == 6
    assert result["truth_rank"] == 1 and result["nominal_forced_truth_error_db"] == 6
    assert result["near_gain_set"]["gains"] == [6.0]
    with pytest.raises(ValueError, match="finite"):
        select(np.full((12, 25), np.nan), list(range(-12, 13)), 3, 0)


def test_summary_requires_each_e_condition_full_denominator():
    with pytest.raises(ValueError, match="16 rows"):
        summarize(rows_fixture()[:-1])


def test_window_control_reuses_identical_stft_with_every_frame(monkeypatch):
    audio = np.ones(384256)

    def spectra(value, conditions, sr):
        assert value is audio and sr == 48000
        assert conditions["n_samples"] == 384000
        assert conditions["frame_count"] == 501
        assert conditions["masks"] == {}
        assert conditions["pitch_masks"]["all"].dtype == bool
        assert conditions["pitch_masks"]["all"].sum() == 501
        return {"pitch_curves": {"all": {"curve": [1, 2, 3]}}}

    monkeypatch.setattr(stage5_benchmark, "conditioned_spectra", spectra)
    assert stage5_benchmark.all_frame_spectrum(audio) == [1, 2, 3]


def test_condition_diagnostics_keeps_counts_without_nan_pitch_values():
    conditions = {"voiced": [True, False, True], "frame_count": 3,
                  "pitch_masks": {"40-51": [True, False, True]},
                  "masks": {"40-51:attack": [True, False, False],
                            "40-51:sustain": [False, False, True]},
                  "midi": np.array([45, np.nan, 45]),
                  "diagnostics": {"onset_times": [0.2]}}
    diagnostic = stage5_benchmark.condition_diagnostics(conditions)
    assert diagnostic == {"voiced_frames": 2, "frame_count": 3,
                          "pitch_frames": {"40-51": 2},
                          "phase_frames": {"40-51:attack": 1, "40-51:sustain": 1},
                          "onset_times": [0.2]}
    json.dumps(diagnostic, allow_nan=False)
