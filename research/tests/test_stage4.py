"""Independent-file separation and the no-EQ decision must also have power."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.stage4_benchmark import gate_oracle, validate_separation
from scripts.stage2_benchmark import read_json, sha

ROOT = Path(__file__).resolve().parents[1]


def test_independent_files_preserve_original_audio_samples():
    meta = read_json(ROOT / "assets/stage4/di-provenance.json")
    validate_separation(meta)
    recordings = {}
    for raw in meta["recordings"]:
        assert sha(ROOT / raw["path"]) == raw["sha256"]
        data, sr = sf.read(ROOT / raw["path"], dtype="float32")
        assert sr == 48000
        np.testing.assert_array_equal(data[:, 0], data[:, 1])
        recordings[raw["id"]] = data[:, 0]
    for item in meta["segments"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
        clip, sr = sf.read(ROOT / item["path"], dtype="float32")
        assert sr == 48000
        np.testing.assert_array_equal(clip, recordings[item["recording_id"]][item["start_frame"]:item["end_frame"]])


@pytest.mark.parametrize("corruption", ["shared_file", "role", "overlap", "short"])
def test_independent_role_corruption_rejected(corruption):
    meta = read_json(ROOT / "assets/stage4/di-provenance.json")
    if corruption == "shared_file":
        for item in meta["segments"]:
            if item["role"] == "holdout":
                item["recording_id"] = "directinput_04"
    elif corruption == "role":
        meta["segments"][-1]["role"] = "probe"
    elif corruption == "overlap":
        meta["segments"][1]["start_frame"] = 0
        meta["segments"][1]["end_frame"] = 384000
    else:
        meta["segments"][-1]["frames"] -= 1
    with pytest.raises(ValueError):
        validate_separation(meta)


def _rows(apply, improved):
    result = []
    for enabled in (False, True):
        for _ in range(8):
            metric = {"harmful": False, "sc_improved": bool(improved and enabled), "ltas_improved": False}
            result.append({"eq_on": enabled, "decision": {"apply": bool(apply and enabled)},
                           "holdouts": {h: {"null_gate": deepcopy(metric)} for h in ("H0", "H1")}})
    return result


def test_always_bypass_is_not_a_successful_eq_detector():
    cfg = read_json(ROOT / "experiments/stage4.json")
    result = gate_oracle(_rows(False, False), cfg)
    assert result["false_apply"] == 0 and result["harmed_cases"] == 0
    assert not result["passed"] and result["e1_applied"] == 0
    assert gate_oracle(_rows(True, True), cfg)["passed"]


def test_benefit_requires_both_unseen_holdouts():
    cfg = read_json(ROOT / "experiments/stage4.json")
    rows = _rows(True, True)
    for row in rows:
        if row["eq_on"]:
            row["holdouts"]["H1"]["null_gate"]["sc_improved"] = False
    assert gate_oracle(rows, cfg)["e1_beneficial_cases"] == 0
    assert not gate_oracle(rows, cfg)["passed"]
