"""Study trust boundaries and paired evaluation, not benchmark success assertions."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.stage2_benchmark import choose_two_stage, paired_quality, sha

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_assets_match_provenance_and_are_unique():
    library = json.loads((ROOT / "assets/stage2/models-ir-provenance.json").read_text(encoding="utf-8"))
    assert len(library["models"]) == 10 and len(library["irs"]) == 2
    assert len({x["sha256"] for x in library["models"]}) == 10
    for item in library["models"] + library["irs"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
        assert item["license"] in ("MIT", "CC0 1.0")
        assert item["sample_rate"] == 48000
    for item in library["irs"]:
        assert sf.info(ROOT / item["path"]).channels == 1


def test_di_clips_preserve_original_samples_and_do_not_overlap():
    meta = json.loads((ROOT / "assets/stage2/di-provenance.json").read_text(encoding="utf-8"))
    raw = meta["recordings"][0]
    assert raw["license"] == "CC BY 4.0"
    assert sha(ROOT / raw["path"]) == raw["sha256"]
    original, sr = sf.read(ROOT / raw["path"], dtype="float32")
    np.testing.assert_array_equal(original[:, 0], original[:, 1])
    previous_end = 0
    for item in meta["segments"]:
        assert item["start_frame"] >= previous_end
        previous_end = item["end_frame"]
        assert sha(ROOT / item["path"]) == item["sha256"]
        clip, clip_sr = sf.read(ROOT / item["path"], dtype="float32")
        assert sr == clip_sr == 48000 and len(clip) == 384000
        np.testing.assert_array_equal(clip, original[item["start_frame"]:item["end_frame"], 0])


def test_exhaustive_and_two_stage_are_distinct_diagnostics():
    gains = np.arange(-12, 13)
    scores = np.full((3, 25), 10.0)
    scores[0, gains == 0] = 1
    scores[1, gains == 6] = 2
    scores[2, [0, 6, 12, 18, 24]] = 3
    scores[2, gains == 5] = 0.1
    coarse, refined, exhaustive = choose_two_stage(scores, gains, refine_top=2)
    assert coarse == [0, 1, 2]
    assert [r[0] for r in refined] == [0, 1]
    assert exhaustive[0][0] == 2


def test_paired_quality_ignores_output_gain_but_detects_wrong_tone():
    t = np.arange(48000) / 48000
    reference = (np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)) * 0.1
    same = paired_quality(reference * 3, reference)
    assert same["mrstft_sc"] < 1e-10 and same["ltas_mae_db"] < 1e-6
    wrong = paired_quality(np.sin(2 * np.pi * 1700 * t) * 0.1, reference)
    assert wrong["mrstft_sc"] > 0.2 and wrong["ltas_mae_db"] > 1.5
