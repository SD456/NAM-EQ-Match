"""Engineering checks only; these synthetic notes establish no wet-tone accuracy."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.conditioned_scoring import (
    audio_conditions, conditioned_spectra, common_strata, score_conditioned,
)


@pytest.fixture(scope="module")
def plucks():
    sr = 48000
    audio = np.zeros(int(2.4 * sr))
    for start, frequency in zip((0.15, 0.9, 1.65), (155.56349, 220, 440)):
        t = np.arange(int(0.6 * sr)) / sr
        envelope = np.minimum(t / 0.008, 1) * np.exp(-2 * t)
        note = envelope * (np.sin(2 * np.pi * frequency * t)
                           + 0.25 * np.sin(4 * np.pi * frequency * t))
        begin = int(start * sr)
        audio[begin:begin + len(note)] = note
    before = audio.copy()
    conditions = audio_conditions(audio, sr, {"allow_variable_length": True})
    np.testing.assert_array_equal(audio, before)
    return audio, conditions


def test_monophonic_register_pitch_and_onset_masks(plucks):
    _, conditions = plucks
    times = conditions["times"]
    for start, midi, key in zip((0.15, 0.9, 1.65), (51, 57, 69),
                                ("40-51", "52-63", "64-75")):
        stable = (times >= start + 0.12) & (times < start + 0.5)
        voiced = conditions["voiced"] & stable
        assert voiced.sum() >= 0.75 * stable.sum()
        assert np.median(np.abs(conditions["midi"][voiced] - midi)) <= 1
        assert conditions["pitch_masks"][key][stable].sum() >= 0.75 * stable.sum()
        assert conditions["masks"][key + ":attack"].sum() >= 3
        assert conditions["masks"][key + ":sustain"].sum() >= 8
        assert any(abs(onset - start) <= 0.08
                   for onset in conditions["diagnostics"]["onset_times"])
    assert conditions["phase"][0] == "unassigned"
    assert not conditions["voiced"][:2].any()
    attack = conditions["phase"] == "attack"
    sustain = conditions["phase"] == "sustain"
    assert not np.any(attack & sustain)
    for index in np.flatnonzero(sustain):
        prior = [onset for onset in conditions["diagnostics"]["onset_times"]
                 if onset <= times[index]]
        assert prior and 0.12 <= times[index] - prior[-1] < 0.4
    assert all(np.all(mask <= conditions["voiced"])
               for mask in conditions["masks"].values())


@pytest.mark.parametrize("audio", [np.zeros(4800), np.full(4800, np.nan),
                                   np.full(4800, np.inf), np.ones((4800, 2))])
def test_invalid_audio_rejected(audio):
    with pytest.raises(ValueError):
        audio_conditions(audio, 48000, {"allow_variable_length": True})


def test_fixed_window_required():
    with pytest.raises(ValueError, match="eight seconds"):
        audio_conditions(np.ones(4800))


def test_low_note_confidence_gate_is_not_bypassed():
    t = np.arange(24000) / 48000
    audio = np.exp(-2 * t) * (np.sin(2 * np.pi * 110 * t)
                              + 0.25 * np.sin(4 * np.pi * 110 * t))
    conditions = audio_conditions(audio, cfg={"allow_variable_length": True})
    stable = (conditions["times"] > 0.1) & (conditions["times"] < 0.4)
    assert np.median(conditions["diagnostics"]["voiced_probability"][stable]) < 0.8
    assert not conditions["voiced"][stable].any()


def test_spectra_level_invariance_tail_and_alignment(plucks):
    audio, conditions = plucks
    before = audio.copy()
    feature = conditioned_spectra(audio, conditions)
    louder = conditioned_spectra(np.r_[audio * 7, np.ones(1024)], conditions)
    np.testing.assert_array_equal(audio, before)
    for collection in ("pitch_curves", "phase_curves"):
        for key, item in feature[collection].items():
            np.testing.assert_allclose(item["curve"], louder[collection][key]["curve"], atol=1e-8)
    decision = common_strata([conditions] * 3, conditions, mode="phase")
    assert decision["accepted"], decision
    assert score_conditioned([louder] * 3, feature, decision, "phase") < 1e-8
    with pytest.raises(ValueError, match="shorter"):
        conditioned_spectra(audio[:-1], conditions)
    broken = {**conditions, "frame_count": conditions["frame_count"] + 1}
    with pytest.raises(ValueError, match="frame grids"):
        conditioned_spectra(audio, broken)


def condition(pitch_counts, attack=3, sustain=8, extra_voiced=0):
    """Independent manually labelled frames exercise shared-support policy."""
    count = sum(pitch_counts.values()) + extra_voiced
    masks, phases = {}, {}
    offset = 0
    for key, n in pitch_counts.items():
        mask = np.zeros(count, dtype=bool)
        mask[offset:offset + n] = True
        masks[key] = mask
        for part, lo, hi in (("attack", offset, offset + min(attack, n)),
                             ("sustain", offset + attack, min(offset + attack + sustain, offset + n))):
            part_mask = np.zeros(count, dtype=bool)
            part_mask[lo:hi] = True
            phases[key + ":" + part] = part_mask
        offset += n
    return {"voiced": np.ones(count, dtype=bool), "pitch_masks": masks, "masks": phases}


def test_support_uses_two_probes_and_query_coverage_not_candidate_choices():
    query = condition({"40-51": 20, "52-63": 20})
    probes = [condition({"40-51": 10, "52-63": 10}),
              condition({"40-51": 10}), condition({"52-63": 10})]
    decision = common_strata(probes, query)
    assert decision["accepted"] and decision["coverage"] == 1
    assert decision["support_probes"] == {"40-51": [0, 1], "52-63": [0, 2]}
    grid = np.array([80, 200, 1000, 8000.0])

    def features(values):
        return {"grid": grid, "pitch_curves": {
            key: {"curve": np.array([0, value, -value, 0.]), "n_frames": 10}
            for key, value in values.items()}}

    target = features({"40-51": 0, "52-63": 0})
    # Unused probe values are perfect matches: minimizing by candidate would cheat.
    candidates = [features({"40-51": 4, "52-63": 8}),
                  features({"40-51": 12, "52-63": 0}),
                  features({"40-51": 0, "52-63": 16})]
    original = deepcopy(decision)
    assert score_conditioned(candidates, target, decision) == 5
    assert decision == original
    candidates[1]["pitch_curves"]["40-51"]["n_frames"] = 1
    with pytest.raises(ValueError, match="insufficient frames"):
        score_conditioned(candidates, target, decision)
    del candidates[1]["pitch_curves"]["40-51"]
    with pytest.raises(ValueError, match="fixed shared stratum"):
        score_conditioned(candidates, target, decision)


def test_rejection_keeps_voiced_denominator_and_requires_both_phases():
    probes = [condition({"40-51": 20})] * 3
    query = condition({"40-51": 20}, extra_voiced=20)
    decision = common_strata(probes, query)
    assert not decision["accepted"]
    assert decision["coverage"] == 0.5 and decision["voiced_frames"] == 40
    assert decision["reason"] == "insufficient_coverage"
    with pytest.raises(ValueError, match="accepted"):
        score_conditioned([{}] * 3, {}, decision)
    silent = condition({})
    assert common_strata(probes, silent)["reason"] == "no_voiced_query"
    no_attack = condition({"40-51": 20}, attack=0, sustain=20)
    assert common_strata([no_attack] * 3, no_attack, "phase")["reason"] == "missing_attack_or_sustain"
    # Half the voiced frames have no phase; they still reduce phase coverage.
    query = condition({"40-51": 22})
    result = common_strata(probes, query, "phase")
    assert result["coverage"] == 0.5 and not result["accepted"]
