"""Frozen development replay of stage 4; never a new independent validation."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json
from scripts.stage3_benchmark import gain_set
from scripts.stage4_benchmark import validate_separation
from scripts.multidi_scoring import centered_spectral_distance
from scripts.conditioned_scoring import audio_conditions, conditioned_spectra, common_strata, score_conditioned
from tonematch.audio import db_to_lin, load_audio
from tonematch.nam_backend import NamCapture
from tonematch.tone_stack import biquad_coeffs

METHODS = ("frozen_baseline", "windowed_baseline", "pitch_conditioned", "pitch_phase_conditioned")
CONTROLS = METHODS[:2]
PRIMARY = METHODS[-1]
SR = 48000


def metrics(results):
    accepted = [r for r in results if r["accepted"]]
    top1 = sum(r["accepted"] and r["top1"] for r in results)
    top3 = sum(r["accepted"] and r["top3"] for r in results)
    errors = [r["nominal_forced_truth_error_db"] for r in accepted]
    return {"total": len(results), "accepted": len(accepted), "full_top1": top1,
            "full_top3": top3, "full_top1_accuracy": top1 / len(results) if results else None,
            "full_top3_accuracy": top3 / len(results) if results else None,
            "accepted_accuracy": top1 / len(accepted) if accepted else None,
            "median_forced_gain_error_db": float(np.median(errors)) if errors else None,
            "boundary_gain_count": sum(r["gain_db"] in (-12, 12) for r in accepted),
            "forced_truth_boundary_gain_count": sum(r["forced_truth_gain_db"] in (-12, 12) for r in accepted)}


def summarize(rows):
    summary = {}
    for eq_on in (False, True):
        group = [r for r in rows if r["eq_on"] == eq_on]
        if len(group) != 16:
            raise ValueError("Expected 16 rows per E condition")
        summary[f"E{int(eq_on)}"] = {
            method: metrics([r["methods"][method] for r in group]) for method in METHODS}
        summary[f"E{int(eq_on)}"]["same_accepted_subset_controls"] = {
            conditioned: {control: metrics([r["methods"][control] for r in group
                                           if r["methods"][conditioned]["accepted"]])
                          for control in CONTROLS}
            for conditioned in METHODS[2:]}
    e0, e1 = summary["E0"], summary["E1"]
    own_error = e0[PRIMARY]["median_forced_gain_error_db"]
    matched = e0["same_accepted_subset_controls"][PRIMARY]
    checks = {
        "coverage": all(s[PRIMARY]["accepted"] >= 12 for s in (e0, e1)),
        "e0_top1": e0[PRIMARY]["full_top1"] >= max(e0[c]["full_top1"] for c in CONTROLS) + 1,
        "e1_top1": e1[PRIMARY]["full_top1"] >= max(e1[c]["full_top1"] for c in CONTROLS),
        "e0_forced_gain": own_error is not None and all(
            matched[c]["median_forced_gain_error_db"] is not None for c in CONTROLS)
            and own_error <= min(matched[c]["median_forced_gain_error_db"] for c in CONTROLS),
    }
    return summary, {"checks": checks, "passed": all(checks.values()),
                     "decision": "continue_new_take_confirmation" if all(checks.values()) else "retain_baseline",
                     "production_replacement": False}


def select(scores, gains, truth, truth_gain, decision=None):
    result = {"accepted": True, "rejection_reason": None, "coverage": 1.0}
    if decision is not None:
        result.update(accepted=decision["accepted"], rejection_reason=(
            None if decision["accepted"] else decision["reason"]), coverage=decision["coverage"])
    if not result["accepted"]:
        return {**result, "truth_rank": None, "winner": None, "gain_db": None,
                "forced_truth_gain_db": None, "nominal_forced_truth_error_db": None,
                "top1": False, "top3": False, "near_gain_set": None, "all_gain_scores": None}
    scores = np.asarray(scores, float)
    if scores.shape != (12, len(gains)) or not np.isfinite(scores).all():
        raise ValueError("Expected finite 12-rig gain score matrix")
    best = scores.argmin(axis=1)
    ranking = sorted(range(12), key=lambda i: scores[i, best[i]])
    winner = ranking[0]
    forced = gains[int(best[truth])]
    return {**result, "truth_rank": ranking.index(truth) + 1, "winner": winner,
            "gain_db": gains[int(best[winner])], "forced_truth_gain_db": forced,
            "nominal_forced_truth_error_db": abs(forced - truth_gain),
            "top1": winner == truth, "top3": truth in ranking[:3],
            "near_gain_set": gain_set(scores[truth], gains), "all_gain_scores": scores}


def frozen_scores(cache, target, probes, gains, n_rigs=12):
    matrix = np.empty((n_rigs, len(gains)))
    for rig in range(n_rigs):
        for j, gain in enumerate(gains):
            items = [cache["items"][f"{rig}|{p}|{gain}"] for p in probes]
            grid = np.asarray(items[0]["grid"])
            if not all(np.array_equal(grid, item["grid"]) for item in items):
                raise ValueError("Frozen probe frequency grid mismatch")
            matrix[rig, j] = np.median(centered_spectral_distance(
                np.asarray([item["spectrum"] for item in items]), np.asarray(target), grid))
    return matrix


def plan(cfg):
    protected = read_json(ROOT / cfg["baseline_snapshot"])
    for path, digest in protected.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Protected baseline changed: {path}")
    baseline = read_json(ROOT / cfg["baseline_results"])
    meta = read_json(ROOT / cfg["independent_di_manifest"])
    library = read_json(ROOT / cfg["library_manifest"])
    validate_separation(meta)
    assets = {}
    for item in library["models"] + library["irs"] + meta["recordings"] + meta["segments"]:
        path = item["path"]
        assets[path] = sha(ROOT / path)
        if assets[path] != item["sha256"]:
            raise ValueError(f"Changed source asset: {path}")
    old = baseline["plan"]["config"]
    if (cfg.get("sr", cfg.get("sample_rate")) != SR or cfg["probes"] != old["probes"]
            or cfg["test_references"] != old["test_references"]
            or cfg["search_gains_db"] != old["search_gains_db"]
            or cfg["truth_gains_db"] != old["truth_gains_db"]
            or cfg["calibration_db"] != old["calibration_db"] or cfg["post_eq"] != old["post_eq"]):
        raise ValueError("Stage-4 replay identities/calibration must stay frozen")
    if len(baseline["retrieval"]) != 32 or len(baseline["plan"]["rigs"]) != 12 or len(baseline["plan"]["teachers"]) != 4:
        raise ValueError("Expected 32 rows, 12 rigs and four teachers")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen NAM development run requires CUDA")
    sources = ["experiments/stage5.json", "docs/stage5-protocol.md", "docs/stage5-recording-spec.md",
               "docs/stage5-data-search.md", "assets/stage5/recording/session.csv",
               "assets/stage5/recording/events.csv", "assets/stage5/recording/prompt-cards.csv",
               "assets/stage5/discovery-a.json", "assets/stage5/discovery-b.json", "scripts/stage5_benchmark.py",
               "scripts/conditioned_scoring.py", "scripts/multidi_scoring.py", "scripts/stage4_benchmark.py",
               "scripts/stage3_benchmark.py", "scripts/stage2_benchmark.py", cfg["baseline_snapshot"],
               cfg["baseline_results"], cfg["baseline_cache"], cfg["independent_di_manifest"], cfg["library_manifest"],
               "vendor/NAM-EQ-Match/tonematch/nam_backend.py", "vendor/NAM-EQ-Match/tonematch/audio.py",
               "vendor/NAM-EQ-Match/tonematch/features.py", "vendor/NAM-EQ-Match/tonematch/tone_stack.py"]
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "scipy", "librosa", "torch", "neural-amp-modeler", "soundfile")}
    return {"config": cfg, "source_hashes": {path: sha(ROOT / path) for path in sources},
            "assets": assets, "protected_baselines": protected, "di": meta, "library": library,
            "rigs": baseline["plan"]["rigs"], "teachers": baseline["plan"]["teachers"],
            "environment": {"versions": versions, "python": platform.python_version(), "platform": platform.platform(),
                            "device": "cuda", "device_name": torch.cuda.get_device_name(0),
                            "cuda": torch.version.cuda, "precision": "float32_no_tf32", "threads": 4}}


def all_frame_spectrum(audio):
    conditions = {"n_samples": 8 * SR, "frame_count": 501,
                  "pitch_masks": {"all": np.ones(501, dtype=bool)}, "masks": {}}
    return conditioned_spectra(audio, conditions, SR)["pitch_curves"]["all"]["curve"]


def condition_diagnostics(conditions):
    return {"voiced_frames": int(np.sum(conditions["voiced"])),
            "frame_count": conditions["frame_count"],
            "pitch_frames": {key: int(np.sum(mask)) for key, mask in conditions["pitch_masks"].items()},
            "phase_frames": {key: int(np.sum(mask)) for key, mask in conditions["masks"].items()},
            "onset_times": conditions["diagnostics"]["onset_times"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="verify inputs and lock without rendering or target estimation")
    args = parser.parse_args()
    cfg = read_json(ROOT / "experiments/stage5.json")
    output = ROOT / cfg["output"]
    if (output / "results.json").exists():
        raise FileExistsError("Completed development results exist; refusing overwrite")
    torch.set_num_threads(4)
    current = plan(cfg)
    lock_path = output / "locked-plan.json"
    if lock_path.exists() and read_json(lock_path) != current:
        raise ValueError("Frozen plan mismatch; do not adapt this run to observed results")
    if args.freeze:
        output.mkdir(parents=True, exist_ok=True)
        write_json(lock_path, current)
        print("Frozen stage-5 development plan; no NAM render or target estimate performed", flush=True)
        return
    if not lock_path.exists():
        raise FileNotFoundError("Run --freeze separately before development replay")
    identity = sha(lock_path)
    baseline = read_json(ROOT / cfg["baseline_results"])
    old_cache = read_json(ROOT / cfg["baseline_cache"])
    probes, gains = cfg["probes"], cfg["search_gains_db"]
    frozen = []
    for row in baseline["retrieval"]:
        matrix = frozen_scores(old_cache, row["reference_spectrum"], probes, gains)
        np.testing.assert_allclose(matrix, row["all_gain_scores"], atol=1e-10, rtol=0)
        result = select(matrix, gains, row["truth"], row["truth_gain_db"])
        for key in ("winner", "gain_db", "forced_truth_gain_db", "top1", "top3"):
            if result[key] != row[key]:
                raise AssertionError(f"Frozen baseline selection mismatch: {key}")
        frozen.append(result)
    inputs = {item["id"]: load_audio(str(ROOT / item["path"]), SR)[0].astype(np.float64) * db_to_lin(cfg["calibration_db"])
              for item in current["di"]["segments"] if item["id"] in probes + cfg["test_references"]}
    dry_conditions = [audio_conditions(inputs[p], SR) for p in probes]
    models, irs = ({x["id"]: x for x in current["library"][key]} for key in ("models", "irs"))

    def capture(index):
        rig = current["rigs"][index]
        model = models[rig["model_id"]]
        return NamCapture(str(ROOT / model["path"]), device="cuda", rig_type=model["rig_type"],
                          cab_ir_path=str(ROOT / irs[rig["cab_id"]]["path"]) if rig["cab_id"] else None)

    started = time.perf_counter()
    cache_path = output / "probe-features.json"
    cache = read_json(cache_path) if cache_path.exists() else {"identity": identity, "items": {}}
    if cache["identity"] != identity:
        raise ValueError("Candidate feature cache lock mismatch")
    for rig in range(12):
        nam = capture(rig)
        for p, conditions in zip(probes, dry_conditions):
            for gain in gains:
                key = f"{rig}|{p}|{gain}"
                if key not in cache["items"]:
                    audio = nam.process(inputs[p] * db_to_lin(gain))
                    cache["items"][key] = {"conditioned": conditioned_spectra(audio, conditions, SR),
                                           "windowed": all_frame_spectrum(audio)}
        pending = cache_path.with_suffix(".tmp")
        write_json(pending, cache)
        pending.replace(cache_path)
        print(f"development probe features {rig + 1}/12 elapsed={time.perf_counter()-started:.1f}s", flush=True)
        del nam
    if len(cache["items"]) != 900:
        raise ValueError("Expected exactly 900 candidate features")
    eq = cfg["post_eq"]
    b, a = biquad_coeffs(eq["kind"], SR, eq["frequency_hz"], eq["q"], eq["gain_db"])
    rows = []
    # Each of the 32 rows renders its own wet target once; no historical holdout replay.
    for index, old in enumerate(baseline["retrieval"]):
        nam = capture(old["truth"])
        target = nam.process(inputs[old["reference"]] * db_to_lin(old["truth_gain_db"]))
        del nam
        if old["eq_on"]:
            target = lfilter(b, a, np.pad(target, (0, 256))).astype(np.float32)
        conditions = audio_conditions(target, SR)
        query = conditioned_spectra(target, conditions, SR)
        windowed = all_frame_spectrum(target)
        decisions = {method: common_strata(dry_conditions, conditions, mode=mode, **cfg["support"])
                     for method, mode in zip(METHODS[2:], ("pitch", "phase"))}
        matrices = {method: np.zeros((12, len(gains))) for method in METHODS[1:]}
        for rig in range(12):
            for gj, gain in enumerate(gains):
                items = [cache["items"][f"{rig}|{p}|{gain}"] for p in probes]
                matrices[METHODS[1]][rig, gj] = np.median(centered_spectral_distance(
                    np.asarray([item["windowed"] for item in items]), np.asarray(windowed), query["grid"]))
                for method, mode in zip(METHODS[2:], ("pitch", "phase")):
                    if decisions[method]["accepted"]:
                        matrices[method][rig, gj] = score_conditioned(
                            [item["conditioned"] for item in items], query, decisions[method], mode)
        methods = {METHODS[0]: frozen[index]}
        for method in METHODS[1:]:
            methods[method] = select(matrices[method], gains, old["truth"], old["truth_gain_db"], decisions.get(method))
        rows.append({**{key: old[key] for key in ("case", "reference", "truth", "truth_gain_db", "eq_on")},
                     "methods": methods, "query": {**condition_diagnostics(conditions), "decisions": decisions}})
        print(f"development retrieval {index + 1}/32 elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if plan(cfg) != current or sha(lock_path) != identity:
        raise ValueError("Frozen sources changed while running")
    summary, gate = summarize(rows)
    write_json(output / "results.json", {"plan": current, "lock_sha256": identity, "retrieval": rows,
               "dry_probe_conditions": {p: condition_diagnostics(c) for p, c in zip(probes, dry_conditions)},
               "summary": summary, "gate": gate, "baseline_preserved": True,
               "status": "development_replay_not_new_validation", "total_seconds": time.perf_counter() - started})
    print(f"Stage-5 development complete: {gate}", flush=True)


if __name__ == "__main__":
    main()
