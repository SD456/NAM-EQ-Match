"""Test structural gain/DI-level ambiguity on real NAMs, without claiming picking-force labels."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json
from scripts.stage4_helpers import profile_gain_set
from tonematch.audio import db_to_lin, load_audio
from tonematch.nam_backend import NamCapture


def main():
    folder = ROOT / "artifacts/stage4/gain_witness"
    if (folder / "results.json").exists():
        raise FileExistsError("Witness already exists")
    folder.mkdir(parents=True, exist_ok=True)
    cfg = read_json(ROOT / "experiments/stage4.json")
    definition = cfg["gain_witness"]
    plan = read_json(ROOT / cfg["baseline_plan"])
    old = read_json(ROOT / cfg["baseline_snapshot"])
    assert all(sha(ROOT / path) == digest for path, digest in old.items())
    library = read_json(ROOT / "assets/stage2/models-ir-provenance.json")
    models = {item["id"]: item for item in library["models"]}
    irs = {item["id"]: item for item in library["irs"]}
    identities = {path: sha(ROOT / path) for path in ["experiments/stage4.json", "docs/stage4-protocol.md",
                  "scripts/stage4_gain_witness.py", "scripts/stage4_helpers.py"] + definition["sources"]}
    write_json(folder / "locked-plan.json", {"config": cfg, "source_hashes": identities,
                                              "device": "cuda", "real_force_labels": False})
    torch.set_num_threads(4)
    pairs, observations = [], []
    for index in plan["teachers"]:
        rig = plan["rigs"][index]
        model, cab = models[rig["model_id"]], rig["cab_id"]
        capture = NamCapture(str(ROOT / model["path"]), device="cuda", rig_type=model["rig_type"],
                             cab_ir_path=str(ROOT / irs[cab]["path"]) if cab else None)
        for source in definition["sources"]:
            x, sr = load_audio(str(ROOT / source), 48000)
            assert sr == 48000
            x = x.astype(np.float64) * db_to_lin(cfg["calibration_db"])
            outputs = {}
            for gain in definition["device_gains_db"]:
                for level in definition["digital_levels_db"]:
                    # Two independently set controls multiply in their actual serial positions.
                    scaled_di = x * db_to_lin(level)
                    y = capture.process(scaled_di * db_to_lin(gain))
                    effective = gain + level
                    outputs[gain, level] = y
                    compatible = profile_gain_set([effective], definition["device_grid_db"], definition["level_bounds_db"])
                    assert gain in compatible
                    observations.append({"rig": index, "source": source, "device_gain": gain,
                                         "digital_level": level, "effective_gain": effective,
                                         "compatible_device_gains": compatible,
                                         "known_level_oracle_gain": effective - level})
            for effective, keys in ((0, ((0, 0), (6, -6))), (6, ((0, 6), (6, 0)))):
                a, b = (outputs[key] for key in keys)
                np.testing.assert_allclose(a, b, atol=definition["atol"], rtol=definition["rtol"])
                pairs.append({"rig": index, "source": source, "effective_gain": effective,
                              "distinct_controls": keys, "max_abs_error": float(np.max(np.abs(a - b))),
                              "bitwise_equal": bool(np.array_equal(a, b)), "passed": True})
    assert len(pairs) == 16 and len(observations) == 48
    assert all(sha(ROOT / path) == digest for path, digest in identities.items())
    assert all(sha(ROOT / path) == digest for path, digest in old.items())
    write_json(folder / "results.json", {"source_hashes": identities, "pairs": pairs, "observations": observations,
                "baseline_preserved": True, "max_absolute_wave_error": max(p["max_abs_error"] for p in pairs),
                "conclusion": "Wet-only source-level and device input gain are structurally nonidentifiable; physical picking-force separation has not been tested"})
    print(f"16 equal-effective-gain pairs passed; max waveform error={max(p['max_abs_error'] for p in pairs):g}")


if __name__ == "__main__":
    main()
