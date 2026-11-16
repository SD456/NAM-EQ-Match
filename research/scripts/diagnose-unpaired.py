"""Record the old cross-performance toy as a diagnostic, not an accuracy gate."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/NAM-EQ-Match"))
from tests.test_synthetic import synth_di
from tonematch.audio import save_audio
from tonematch.nam_backend import MockAmp
from tonematch.pipeline import run_match

output = ROOT / "artifacts/stage1/unpaired_toy"
output.mkdir(parents=True, exist_ok=True)
di, target = output / "di.wav", output / "target.wav"
save_audio(str(di), synth_di(1), 48000)
save_audio(str(target), MockAmp(14, -1.5, 4).process(synth_di(2)), 48000)
captures = [MockAmp(drive_db=drive, name=name) for drive, name in
            [(0, "clean-ish"), (7, "crunch"), (14, "hot"), (24, "fuzz")]]
match = run_match(str(target), str(di), captures, str(output / "match"))
effective = match.best.capture.drive_db + match.renders[0]["input_gain_db"]
result = {
    "purpose": "Cross-performance heuristic limitation diagnostic; synthetic probes, not real guitar/NAM",
    "target_effective_drive_db": 14, "recovered_effective_drive_db": effective,
    "absolute_error_db": abs(effective - 14),
    "within_old_5db_tolerance": bool(abs(effective - 14) <= 5),
    "target_has_extra_tilt_db_per_oct": -1.5, "target_extra_presence_db": 4,
    "candidate_eq": "None of the candidates include target tilt/presence",
    "normalization_db": match.renders[0]["di_normalization_db"],
    "winner": match.best.name, "score": match.best.score,
    "conclusion": "Do not infer cross-performance accuracy from engineering regression passes",
}
path = output / "diagnostic.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
