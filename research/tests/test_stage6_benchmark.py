import copy
import json

import numpy as np
import pytest
import soundfile as sf

from scripts import stage6_benchmark as benchmark


def candidate(window,threshold,coverage,precision,full,octave=0):
    return {"pitch_window":window,"threshold":threshold,
            "metrics":{"coverage":coverage,"conditional_exact":precision,
                       "full_correct":full,"conditional_octave":octave}}


def test_selection_requires_both_precision_and_coverage():
    grid = [candidate(1024,0,.99,.89,.8811),candidate(2048,.8,.59,1,.59),
            candidate(2048,.2,.8,.95,.76)]
    assert benchmark.choose_setting(grid,{"min_coverage":.6,"min_conditional_exact":.9}) == {
        "pitch_window":2048,"threshold":.2}


@pytest.mark.parametrize("grid,expected",[
    ([candidate(1024,.1,.8,1,.8,.1),candidate(2048,.1,.8,1,.8,0)],(2048,.1)),
    ([candidate(1024,.1,.8,1,.8),candidate(2048,.8,.8,1,.8)],(1024,.1)),
    ([candidate(1024,.1,.8,1,.8),candidate(1024,.8,.8,1,.8)],(1024,.8))])
def test_selection_uses_frozen_tie_order(grid,expected):
    selected = benchmark.choose_setting(grid,{"min_coverage":.6,"min_conditional_exact":.9})
    assert (selected["pitch_window"],selected["threshold"]) == expected


def test_selection_abstains_when_none_qualifies():
    assert benchmark.choose_setting([candidate(2048,0,0,None,0)],{
        "min_coverage":.6,"min_conditional_exact":.9}) is None


def metric(n_frames,accepted,correct):
    return {"n_notes":1,"n_notes_excluded_empty_core":0,"total_reference_frames":n_frames,
            "accepted_weight":accepted,"correct_weight":correct,"octave_weight":0,
            "n_notes_with_any_accept":int(accepted>0),"n_notes_with_correct":int(correct>0),
            "confidence_bins":[{"n_frames":0,"weight":0,"correct_weight":0} for _ in range(6)]}


def test_combination_keeps_equal_note_weights_across_durations():
    result = benchmark.combine_pitch([metric(100,.5,.5),metric(10,1,.5)])
    assert result["total_reference_frames"] == 110
    assert result["coverage"] == pytest.approx(.75)
    assert result["conditional_exact"] == pytest.approx(2/3)
    assert result["full_correct"] == pytest.approx(.5)


def test_combination_empty_support_has_no_claimed_accuracy():
    result = benchmark.combine_pitch([])
    assert result["n_notes"] == 0 and result["coverage"] is None
    assert result["conditional_exact"] is None
    assert all(x["conditional_exact"] is None for x in result["confidence_bins"])


def test_diagnostic_onset_cannot_rescue_failing_primary():
    heldout = {"grid":[candidate(2048,.2,.8,.95,.76)],
               "onsets":[{"window":1024,"tolerance":.05,"metrics":{"f1":.79}},
                         {"window":2048,"tolerance":.05,"metrics":{"f1":1}},
                         {"window":1024,"tolerance":.1,"metrics":{"f1":1}}]}
    cfg = {"min_coverage":.6,"min_conditional_exact":.9,"min_onset_f1":.8}
    gate = benchmark.final_gate(heldout,{"pitch_window":2048,"threshold":.2},cfg)
    assert not gate["passed"] and gate["checks"]["coverage"]
    assert not gate["checks"]["short_onset_f1"]
    heldout["onsets"][0]["metrics"]["f1"] = .8
    assert benchmark.final_gate(heldout,{"pitch_window":2048,"threshold":.2},cfg)["passed"]


def test_silent_labeled_recording_keeps_all_reference_denominators(tmp_path,monkeypatch):
    cfg = json.loads((benchmark.ROOT/"experiments/stage6.json").read_text(encoding="utf-8"))
    sf.write(tmp_path/"silent.wav",np.zeros(44100),44100,subtype="PCM_16")
    monkeypatch.setattr(benchmark,"ROOT",tmp_path)
    result = benchmark.evaluate([{"id":"silence","split":"fixture","wav":"silent.wav",
        "sample_rate":44100,"frames":44100,"event":{"pitch":57,"onset":.2,"offset":.8}}],cfg)
    assert all(x["metrics"]["n_notes"]==1 and x["metrics"]["coverage"]==0
               and x["metrics"]["conditional_exact"] is None for x in result["grid"])
    assert all(x["metrics"]["fn"]==1 and x["metrics"]["f1"]==0 for x in result["onsets"])


@pytest.mark.parametrize("section,key,value",[("frontend","resolution",.1),
    ("truth","correct_cents",25),("onset","primary_window",2048),
    ("gate","onset_tolerance_seconds",.1)])
def test_config_cannot_claim_settings_the_frozen_code_does_not_use(section,key,value):
    cfg = json.loads((benchmark.ROOT/"experiments/stage6.json").read_text(encoding="utf-8"))
    benchmark.validate_config(cfg)
    cfg[section][key] = value
    with pytest.raises(ValueError,match="update protocol and implementation together"):
        benchmark.validate_config(cfg)
