"""Prepare the fixed stage 3 fallback DI split from the verified stage 2 recording."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "assets/stage2/di/directinput_01.raw.wav"
OUT = ROOT / "assets/stage3/di"
EXPECTED_SHA256 = "b985af2acea79809f572c869a007bd4b5257d4684b0636f26bd09f6482f9d0eb"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha256(RAW) == EXPECTED_SHA256, "Source recording hash mismatch"
    audio, rate = sf.read(RAW, dtype="float32", always_2d=True)
    assert rate == 48000 and audio.shape == (3480000, 2)
    assert np.array_equal(audio[:, 0], audio[:, 1])
    baseline = json.loads((ROOT / "assets/stage2/di-provenance.json").read_text(encoding="utf-8"))
    recording = baseline["recordings"][0]
    assert recording["sha256"] == EXPECTED_SHA256
    OUT.mkdir(parents=True, exist_ok=True)
    segments = []
    # shortcut: same-recording holdout only; replace with separately recorded DI when accessible.
    for name, role, start_seconds in [("P0", "probe", 24), ("P1", "probe", 32),
                                      ("P2", "probe", 40), ("R0", "reference", 48),
                                      ("R1", "reference", 56), ("H0", "holdout", 64)]:
        start, end = start_seconds * rate, (start_seconds + 8) * rate
        samples = audio[start:end, 0]
        assert len(samples) == 384000 and np.max(np.abs(samples)) > 0
        path = OUT / f"{name}.wav"
        sf.write(path, samples, rate, subtype="FLOAT")
        check, check_rate = sf.read(path, dtype="float32")
        assert check_rate == rate and np.array_equal(check, samples)
        segments.append({"id": name, "role": role, "recording_id": recording["id"],
                         "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                         "sample_rate": rate, "channels": 1, "frames": len(samples),
                         "start_frame": start, "end_frame": end,
                         "rms": float(np.sqrt(np.mean(samples ** 2))),
                         "peak": float(np.max(np.abs(samples)))})
    assert all(a["end_frame"] <= b["start_frame"] for a, b in zip(segments, segments[1:]))
    provenance = {"sources": baseline["sources"], "recordings": [recording], "segments": segments,
                  "selection_and_processing": (
                      "Fallback fixed before scoring: P0/P1/P2 24-32/32-40/40-48 s, "
                      "R0/R1 48-56/56-64 s, H0 64-72 s. Half-open sample ranges do not overlap "
                      "each other or stage 2 A/B/C (0-24 s). All segments are from the same "
                      "recording, player and guitar; they are not independent recordings. "
                      "Only one holdout fits in this recording; no H1. Separate 02/03 recordings "
                      "were unavailable because the download host could not be resolved. "
                      "Left channel selected, float32 WAV; no resampling, normalization, gain, "
                      "fades, denoising or effects. Physical recording gain is unspecified.")}
    (ROOT / "assets/stage3/di-provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
