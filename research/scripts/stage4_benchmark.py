"""Frozen independent-recording gain baseline and null-calibrated no-EQ decision."""
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import paired_quality, read_json, sha, write_json
from scripts.stage3_benchmark import compare_eq, gain_set
from scripts.multidi_scoring import half_octave_ltas, centered_spectral_distance, make_consensus_ir
from scripts.stage4_helpers import raw_correction, fit_null, decide_eq, profile_gain_set
from tonematch.audio import db_to_lin, load_audio
from tonematch.match_eq import apply_fir, design_match_ir
from tonematch.nam_backend import NamCapture
from tonematch.tone_stack import biquad_coeffs

SR = 48000


def validate_separation(meta):
    expected = {"probe": {"P0", "P1", "P2"}, "calibration_reference": {"C0", "C1"},
                "test_reference": {"T0", "T1"}, "holdout": {"H0", "H1"}}
    if len(meta["segments"]) != 9:
        raise ValueError("Nine frozen clips required")
    seen_recordings = set()
    for role, ids in expected.items():
        group = [x for x in meta["segments"] if x["role"] == role]
        if {x["id"] for x in group} != ids:
            raise ValueError("Missing or incorrect role")
        recordings = {x["recording_id"] for x in group}
        if len(recordings) != 1 or seen_recordings & recordings:
            raise ValueError("Each role must use its own independent recording file")
        seen_recordings |= recordings
        for index, clip in enumerate(group):
            if (clip["frames"] != 8 * SR or clip["sample_rate"] != SR or clip["channels"] != 1
                    or clip["end_frame"] - clip["start_frame"] != clip["frames"]):
                raise ValueError("Require eight-second native 48k mono clips")
            for earlier in group[:index]:
                if max(clip["start_frame"], earlier["start_frame"]) < min(clip["end_frame"], earlier["end_frame"]):
                    raise ValueError("Overlapping clips")
    if "guitar-techs-p3-directinput-01" in seen_recordings or "directinput_01" in seen_recordings:
        raise ValueError("Previously observed recording 01 cannot be an independent validation file")


def gate_oracle(rows, cfg):
    e0, e1 = ([r for r in rows if r["eq_on"] == enabled] for enabled in (False, True))
    if len(e0) != 8 or len(e1) != 8:
        raise ValueError("Expected eight E0 and eight E1 oracle tasks")
    false_apply = sum(r["decision"]["apply"] for r in e0)
    harmed = sum(any(h["null_gate"]["harmful"] for h in r["holdouts"].values()) for r in e0)
    detected = sum(r["decision"]["apply"] for r in e1)
    beneficial = sum(all(not h["null_gate"]["harmful"] and
                         (h["null_gate"]["sc_improved"] or h["null_gate"]["ltas_improved"])
                         for h in r["holdouts"].values()) for r in e1)
    limits = cfg["gate"]
    return {"e0_n": 8, "e1_n": 8, "false_apply": false_apply, "harmed_cases": harmed,
            "e1_applied": detected, "e1_beneficial_cases": beneficial,
            "passed": bool(false_apply <= limits["e0_max_false_apply"] and
                           harmed <= limits["e0_max_harmed_cases"] and
                           detected >= limits["e1_min_apply"] and
                           beneficial >= limits["e1_min_beneficial_cases"])}


def main():
    output = ROOT / "artifacts/stage4/independent"
    if (output / "results.json").exists():
        raise FileExistsError("Completed independent study exists")
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_json(ROOT / "experiments/stage4.json")
    baseline = read_json(ROOT / cfg["baseline_plan"])
    protected = read_json(ROOT / cfg["baseline_snapshot"])
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    meta = read_json(ROOT / cfg["independent_di_manifest"])
    validate_separation(meta)
    library = read_json(ROOT / "assets/stage2/models-ir-provenance.json")
    for item in library["models"] + library["irs"] + meta["recordings"] + meta["segments"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed source asset: {item['path']}")
    inputs = {x["id"]: load_audio(str(ROOT / x["path"]), SR)[0].astype(np.float64) * db_to_lin(cfg["calibration_db"])
              for x in meta["segments"]}
    models, irs = ({x["id"]: x for x in library[key]} for key in ("models", "irs"))
    torch.set_num_threads(4)
    rigs = []
    for rig in baseline["rigs"]:
        model, cab = models[rig["model_id"]], rig["cab_id"]
        capture = NamCapture(str(ROOT / model["path"]), device="cuda", rig_type=model["rig_type"],
                             cab_ir_path=str(ROOT / irs[cab]["path"]) if cab else None)
        rigs.append({**rig, "capture": capture})
    teachers = baseline["teachers"]
    calibration_rigs = [i for i in range(len(rigs)) if i not in teachers]
    sources = ["experiments/stage4.json", "docs/stage4-protocol.md", "scripts/stage4_benchmark.py",
               "scripts/stage4_helpers.py", "scripts/multidi_scoring.py", "scripts/stage3_benchmark.py",
               "scripts/stage2_benchmark.py", cfg["independent_di_manifest"],
               "vendor/NAM-EQ-Match/tonematch/nam_backend.py", "vendor/NAM-EQ-Match/tonematch/match_eq.py"]
    hashes = {path: sha(ROOT / path) for path in sources}
    plan = {"config": cfg, "di": meta, "rigs": baseline["rigs"], "teachers": teachers,
            "calibration_rigs": calibration_rigs, "source_hashes": hashes, "protected_baselines": protected,
            "device": "cuda", "precision": "float32_no_tf32"}
    lock = output / "locked-plan.json"
    if lock.exists() and read_json(lock) != plan:
        raise ValueError("Partial run source mismatch")
    write_json(lock, plan)
    started = time.perf_counter()

    @lru_cache(maxsize=40)
    def render(index, clip, gain):
        return rigs[index]["capture"].process(inputs[clip] * db_to_lin(gain))

    cache_path = output / "probe-spectra.json"
    cache = read_json(cache_path) if cache_path.exists() else {"identity": hashes, "items": {}}
    if cache["identity"] != hashes:
        raise ValueError("Probe cache mismatch")
    spectra = {}
    grid = None
    for index in range(len(rigs)):
        for probe in cfg["probes"]:
            for gain in cfg["search_gains_db"]:
                key = f"{index}|{probe}|{gain}"
                if key not in cache["items"]:
                    frequencies, curve = half_octave_ltas(render(index, probe, gain), SR)
                    cache["items"][key] = {"grid": frequencies, "spectrum": curve}
                item = cache["items"][key]
                spectra[index, probe, gain] = np.asarray(item["spectrum"])
                grid = np.asarray(item["grid"])
        write_json(cache_path, cache)
        print(f"independent probe cache {index+1}/12 elapsed={time.perf_counter()-started:.1f}s", flush=True)
    assert len(cache["items"]) == 900
    render.cache_clear()

    def probe_spectra(index, gain):
        return np.asarray([spectra[index, probe, gain] for probe in cfg["probes"]])

    null_rows = []
    for index in calibration_rigs:
        for gain in cfg["null_calibration"]["gains_db"]:
            for clip in cfg["calibration_references"]:
                _, reference = half_octave_ltas(render(index, clip, gain), SR)
                null_rows.append({"rig": index, "gain": gain, "clip": clip,
                                  "curve": raw_correction(probe_spectra(index, gain), reference, grid)})
    assert len(null_rows) == 32
    q = fit_null(np.asarray([r["curve"] for r in null_rows]), grid)
    write_json(output / "null-calibration.json", {"q": q, "rows": null_rows, "grid": grid,
                                                  "source_hashes": hashes, "test_used": False})
    print(f"locked empirical null Q={q:.6f} dB from 32 curves; test references not used", flush=True)
    b, a = biquad_coeffs(cfg["post_eq"]["kind"], SR, cfg["post_eq"]["frequency_hz"],
                        cfg["post_eq"]["q"], cfg["post_eq"]["gain_db"])

    def color(audio, eq_on):
        return lfilter(b, a, np.pad(audio, (0, 256))).astype(np.float32) if eq_on else audio

    def eq_options(index, gain, targets):
        raw = []
        for target in targets:
            _, reference = half_octave_ltas(target, SR)
            raw.append(raw_correction(probe_spectra(index, gain), reference, grid))
        decision = decide_eq(np.asarray(raw), grid, SR, q, n_taps=cfg["eq_decision"]["n_taps"],
                             min_gain_db=cfg["eq_decision"]["min_gain_db"],
                             max_spread_db=cfg["eq_decision"]["max_spread_db"],
                             max_gain_db=cfg["eq_decision"]["max_gain_db"])
        fits = [design_match_ir(render(index, probe, gain), targets[0], SR,
                                max_gain_db=4.5, octave_fraction=2, n_taps=2048) for probe in cfg["probes"]]
        old_guard, _, _ = make_consensus_ir(np.asarray([x[2] for x in fits]), grid, SR, guard=True)
        return {"bypass": np.array([1], dtype=np.float32), "stage3_guard": old_guard, "null_gate": decision["ir"]}, decision, raw

    def heldout_metrics(index, gain, truth, true_gain, eq_on, options):
        metrics = {}
        for holdout in cfg["holdouts"]:
            reference = color(render(truth, holdout, true_gain), eq_on)
            candidate = render(index, holdout, gain)
            bypass = paired_quality(candidate, reference)
            metrics[holdout] = {variant: compare_eq(bypass if variant == "bypass" else
                                                   paired_quality(apply_fir(candidate, ir), reference), bypass, cfg["gate"])
                                for variant, ir in options.items()}
        return metrics

    oracle, retrieval = [], []
    targets_by_case = {}
    for truth in teachers:
        for gain in cfg["truth_gains_db"]:
            for eq_on in (False, True):
                case = f"{truth}_{gain}_E{int(eq_on)}"
                targets = [color(render(truth, clip, gain), eq_on) for clip in cfg["test_references"]]
                targets_by_case[case] = (truth, gain, eq_on)
                options, decision, raw = eq_options(truth, gain, targets)
                oracle.append({"case": case, "truth": truth, "truth_gain_db": gain, "eq_on": eq_on,
                               "decision": {k: v for k, v in decision.items() if k != "ir"},
                               "raw_test_curves": raw, "irs": options,
                               "holdouts": heldout_metrics(truth, gain, truth, gain, eq_on, options)})
                for clip, target in zip(cfg["test_references"], targets):
                    _, reference = half_octave_ltas(target, SR)
                    all_scores = np.zeros((12, 25))
                    for index in range(12):
                        for gj, search_gain in enumerate(cfg["search_gains_db"]):
                            values = centered_spectral_distance(probe_spectra(index, search_gain), reference, grid)
                            all_scores[index, gj] = np.median(values)
                    best = np.argmin(all_scores, axis=1)
                    ranking = sorted(range(12), key=lambda i: all_scores[i, best[i]])
                    winner = ranking[0]
                    estimated = cfg["search_gains_db"][int(best[winner])]
                    true_estimated = cfg["search_gains_db"][int(best[truth])]
                    near = gain_set(all_scores[truth], cfg["search_gains_db"])
                    retrieval.append({"case": case, "reference": clip, "truth": truth, "truth_gain_db": gain,
                                      "eq_on": eq_on, "winner": winner, "gain_db": estimated,
                                      "forced_truth_gain_db": true_estimated,
                                      "nominal_forced_truth_error_db": abs(true_estimated-gain),
                                      "top1": winner == truth, "top3": truth in ranking[:3],
                                      "near_effective_gains": near,
                                      "compatible_original_gains_under_level_bounds": profile_gain_set(near["gains"], cfg["search_gains_db"], cfg["gain_witness"]["level_bounds_db"]),
                                      "all_gain_scores": all_scores, "reference_spectrum": reference,
                                      "ranking": [{"rig": i, "gain": cfg["search_gains_db"][int(best[i])],
                                                   "score": float(all_scores[i, best[i]])} for i in ranking],
                                      "holdout_bypass": heldout_metrics(winner, estimated, truth, gain, eq_on,
                                                                        {"bypass": np.array([1], np.float32)})})
                print(f"oracle+baseline {case} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    gate = gate_oracle(oracle, cfg)
    deployed = []
    if gate["passed"]:
        for row in retrieval:
            truth, true_gain, eq_on = targets_by_case[row["case"]]
            targets = [color(render(truth, clip, true_gain), eq_on) for clip in cfg["test_references"]]
            options, decision, _ = eq_options(row["winner"], row["gain_db"], targets)
            deployed.append({"case": row["case"], "reference": row["reference"],
                             "decision": {k: v for k, v in decision.items() if k != "ir"},
                             "holdouts": heldout_metrics(row["winner"], row["gain_db"], truth, true_gain, eq_on, options)})
    assert len(oracle) == 16 and len(retrieval) == 32
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    assert all(sha(ROOT / path) == digest for path, digest in hashes.items())
    result = {"plan": plan, "q": q, "null_rows": null_rows, "grid": grid, "oracle": oracle,
              "retrieval": retrieval, "gate": gate, "selected_chain_eq": deployed,
              "selected_chain_eq_status": "evaluated" if gate["passed"] else "not_validated_oracle_gate_failed",
              "baseline_preserved": True, "total_seconds": time.perf_counter()-started,
              "limits": "Different recording files, same player/guitar/interface; hardware level and real picking-force labels unknown. No Free Bird."}
    write_json(output / "results.json", result)
    print(f"completed independent recordings: Q={q:.4f}, oracle_gate={gate}, selected_EQ={len(deployed)}", flush=True)


if __name__ == "__main__":
    main()
