"""Data separation, unchanged audio and diagnostic interpretation guards."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.stage3_benchmark import compare_eq, gain_set, read_json, sha, validate_roles

ROOT = Path(__file__).resolve().parents[1]


def test_new_roles_are_unseen_nonoverlapping_original_samples():
    meta = read_json(ROOT / "assets/stage3/di-provenance.json")
    old = read_json(ROOT / "assets/stage2/di-provenance.json")
    validate_roles(meta, old)
    raw, sr = sf.read(ROOT / meta["recordings"][0]["path"], dtype="float32")
    for item in meta["segments"]:
        data, sample_rate = sf.read(ROOT / item["path"], dtype="float32")
        assert sha(ROOT / item["path"]) == item["sha256"]
        assert sample_rate == sr == 48000
        np.testing.assert_array_equal(data, raw[item["start_frame"]:item["end_frame"], 0])


@pytest.mark.parametrize("kind", ["overlap", "old_overlap", "role", "range"])
def test_role_leakage_is_rejected(kind):
    meta = read_json(ROOT / "assets/stage3/di-provenance.json")
    old = read_json(ROOT / "assets/stage2/di-provenance.json")
    bad = deepcopy(meta)
    if kind == "overlap":
        bad["segments"][-1]["start_frame"] = bad["segments"][0]["start_frame"]
        bad["segments"][-1]["end_frame"] = bad["segments"][0]["end_frame"]
    elif kind == "old_overlap":
        bad["segments"][0]["start_frame"] = old["segments"][0]["start_frame"]
        bad["segments"][0]["end_frame"] = old["segments"][0]["end_frame"]
    elif kind == "role":
        bad["segments"][-1]["role"] = "probe"
    else:
        bad["segments"][-1]["end_frame"] -= 1
    with pytest.raises(ValueError):
        validate_roles(bad, old)


def test_gain_diagnostic_retains_disconnected_sets_and_flat_curves():
    result = gain_set([0, 0.01, 9, 9, 0.01, 0], np.arange(6))
    assert result["gains"] == [0, 1, 4, 5]
    assert result["intervals"] == [[0, 1], [4, 5]]
    assert result["count"] == 4
    flat = gain_set(np.ones(25), np.arange(-12, 13))
    assert flat["count"] == 25 and flat["flat"]


def test_eq_can_be_harmful_even_when_it_still_passes_a_loose_screen():
    gate = read_json(ROOT / "experiments/stage3.json")["gate"]
    bypass = {"mrstft_sc": 0.0, "ltas_mae_db": 0.0}
    result = compare_eq({"mrstft_sc": 0.03, "ltas_mae_db": 0.1}, bypass, gate)
    assert result["passed"] and result["harmful"] and not result["lost_pass"]
    result = compare_eq({"mrstft_sc": 0.01, "ltas_mae_db": 0.3}, bypass, gate)
    assert result["harmful"]


def test_stage2_results_and_runner_remain_unchanged():
    hashes = read_json(ROOT / "artifacts/stage3/baseline-sha256.json")
    assert all(sha(ROOT / path) == digest for path, digest in hashes.items())
