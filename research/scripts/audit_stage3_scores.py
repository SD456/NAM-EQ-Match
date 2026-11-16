"""Re-render only reference targets and independently reconstruct every score matrix."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import fingerprint_from_json, read_json, sha, write_json
from scripts.multidi_scoring import half_octave_ltas
from tonematch.audio import db_to_lin, load_audio
from tonematch.features import extract_fingerprint, nonlinear_distance, eq_penalty
from tonematch.nam_backend import NamCapture
from tonematch.tone_stack import biquad_coeffs


def main():
    directory = ROOT / "artifacts/stage3/real"
    result_path = directory / "results.json"
    results = read_json(result_path)
    plan, cfg = results["plan"], results["plan"]["config"]
    assert all(sha(ROOT / path) == digest for path, digest in plan["source_hashes"].items())
    cache_path = directory / "probe-fingerprints.json"
    cache = read_json(cache_path)
    assert cache["identity"] == plan["source_hashes"]
    assert len(cache["items"]) == 900
    meta = plan["di"]
    segments = {item["id"]: item for item in meta["segments"]}
    assert all(key.split("|")[1] in cfg["probes"] for key in cache["items"])
    library = read_json(ROOT / "assets/stage2/models-ir-provenance.json")
    models, irs = ({item["id"]: item for item in library[key]} for key in ("models", "irs"))
    torch.set_num_threads(4)

    @lru_cache(maxsize=4)
    def teacher(index):
        rig = plan["rigs"][index]
        model, cab = models[rig["model_id"]], rig["cab_id"]
        return NamCapture(str(ROOT / model["path"]), device=plan["device"], rig_type=model["rig_type"],
                          cab_ir_path=str(ROOT / irs[cab]["path"]) if cab else None)

    b, a = biquad_coeffs(cfg["post_eq"]["kind"], 48000, cfg["post_eq"]["frequency_hz"],
                        cfg["post_eq"]["q"], cfg["post_eq"]["gain_db"])
    grouped = {}
    for row in results["retrieval"]:
        grouped.setdefault(row["case"], []).append(row)
    assert len(grouped) == 32 and all(len(group) == 4 for group in grouped.values())
    references, comparisons, max_error = {}, 0, 0.0
    for case, rows in grouped.items():
        row = rows[0]
        assert row["reference"] in cfg["references"]
        dry, sr = load_audio(str(ROOT / segments[row["reference"]]["path"]), 48000)
        assert sr == 48000
        calibrated = dry.astype(np.float64) * db_to_lin(cfg["calibration_db"])
        wet = teacher(row["truth"]).process(calibrated * db_to_lin(row["truth_gain_db"]))
        if row["eq_on"]:
            wet = lfilter(b, a, np.pad(wet, (0, 256))).astype(np.float32)
        target = extract_fingerprint(wet, sr)
        grid, target_spectrum = half_octave_ltas(wet, sr)
        references[case] = {"fingerprint": asdict(target), "spectrum": target_spectrum, "grid": grid}
        current = np.zeros((3, 12, 25))
        spectral = np.zeros_like(current)
        band = (grid >= 80) & (grid <= 8000)
        for pi, probe in enumerate(cfg["probes"]):
            for ri in range(12):
                for gi, gain in enumerate(cfg["gains_db"]):
                    item = cache["items"][f"{ri}|{probe}|{gain:g}"]
                    fp = fingerprint_from_json(item["fingerprint"])
                    current[pi, ri, gi] = nonlinear_distance(fp, target)["total"] + 0.25 * eq_penalty(fp, target)
                    delta = np.asarray(item["spectrum"])[band] - target_spectrum[band]
                    spectral[pi, ri, gi] = np.mean(np.abs(delta - np.median(delta)))
        # Deliberately avoid the aggregation helper when auditing its actual results.
        expected = {"current_single": current[0], "spectrum_single": spectral[0],
                    "current_multi": np.median(current, axis=0), "spectrum_multi": np.median(spectral, axis=0)}
        for row in rows:
            actual = np.asarray(row["all_gain_scores"])
            np.testing.assert_allclose(actual, expected[row["method"]], atol=1e-8, rtol=1e-7)
            max_error = max(max_error, float(np.max(np.abs(actual - expected[row["method"]]))))
            comparisons += actual.size
    evidence = {"results_sha256": sha(result_path), "probe_cache_sha256": sha(cache_path),
                "audit_script_sha256": sha(__file__), "references": references,
                "score_matrices_checked": len(grouped) * 4, "score_values_checked": comparisons,
                "max_absolute_score_error": max_error, "passed": True,
                "holdout_rendered": False}
    output = directory / "reference-features.json"
    if output.exists():
        raise FileExistsError("Audit output exists")
    write_json(output, evidence)
    print(f"Audited 32 references, 128 matrices, {comparisons} values; max error {max_error:g}; no H0 rendering")


if __name__ == "__main__":
    main()
