"""Independent-role multi-DI retrieval study, preserving the stage-2 baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import lfilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import (fingerprint_from_json, paired_quality, read_json, sha, write_json)
from scripts.multidi_scoring import (aggregate_scores, centered_spectral_distance,
                                    half_octave_ltas, make_consensus_ir)
from tonematch.audio import db_to_lin, load_audio, rms
from tonematch.features import eq_penalty, extract_fingerprint, nonlinear_distance
from tonematch.match_eq import apply_fir, design_match_ir
from tonematch.nam_backend import NamCapture
from tonematch.search import evaluate_gain
from tonematch.tone_stack import biquad_coeffs

SR = 48000


def validate_roles(meta, old_meta):
    """Reject duplicate or overlapping source samples, including previously used A/B/C."""
    expected = {"P0": "probe", "P1": "probe", "P2": "probe", "R0": "reference",
                "R1": "reference", "H0": "holdout"}
    segments = meta["segments"]
    if len(segments) != len(expected) or {x["id"]: x["role"] for x in segments} != expected:
        raise ValueError("Frozen study requires P0/P1/P2, R0/R1 and H0 with distinct roles")
    for index, item in enumerate(segments):
        if item["frames"] != 8 * SR or item["sample_rate"] != SR or item["channels"] != 1:
            raise ValueError("Each role requires native 48k mono, eight seconds")
        if item["end_frame"] - item["start_frame"] != item["frames"]:
            raise ValueError("Invalid source range")
        for other in segments[:index] + old_meta["segments"]:
            if item["recording_id"] == other["recording_id"] and (
                    max(item["start_frame"], other["start_frame"]) < min(item["end_frame"], other["end_frame"])):
                raise ValueError("Probe/reference/holdout or prior stage shares source samples")


def gain_set(curve, gains, relative_span=0.05, flat_span=1e-9):
    curve, gains = np.asarray(curve, float), np.asarray(gains, float)
    if curve.shape != gains.shape or curve.ndim != 1 or not len(curve) or not np.isfinite(curve).all():
        raise ValueError("Finite one-dimensional score and gain curves required")
    span = float(np.ptp(curve))
    selected = gains if span < flat_span else gains[curve <= np.min(curve) + relative_span * span]
    intervals = []
    for value in selected:
        if intervals and value == intervals[-1][1] + 1:
            intervals[-1][1] = float(value)
        else:
            intervals.append([float(value), float(value)])
    return {"gains": selected.tolist(), "intervals": intervals, "count": len(selected),
            "span_db": float(selected[-1] - selected[0]), "flat": span < flat_span}


def compare_eq(metrics, bypass, gate):
    delta_sc = metrics["mrstft_sc"] - bypass["mrstft_sc"]
    delta_ltas = metrics["ltas_mae_db"] - bypass["ltas_mae_db"]
    passed = metrics["mrstft_sc"] <= gate["sc"] and metrics["ltas_mae_db"] <= gate["ltas_db"]
    passed_before = bypass["mrstft_sc"] <= gate["sc"] and bypass["ltas_mae_db"] <= gate["ltas_db"]
    return {**metrics, "delta_sc": delta_sc, "delta_ltas_db": delta_ltas, "passed": bool(passed),
            "harmful": bool(delta_sc > gate["harm_sc_delta"] or delta_ltas > gate["harm_ltas_delta_db"]),
            "lost_pass": bool(passed_before and not passed),
            "sc_improved": bool(delta_sc < -gate["harm_sc_delta"]),
            "ltas_improved": bool(delta_ltas < -gate["harm_ltas_delta_db"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/stage3/real")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    output = (ROOT / args.out).resolve()
    if (output / "results.json").exists():
        raise FileExistsError("Completed result exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "experiments/stage3.json"
    cfg = read_json(config_path)
    previous_plan = read_json(ROOT / cfg["baseline_plan"])
    baseline_hashes = read_json(ROOT / cfg["baseline_snapshot"])
    if any(sha(ROOT / path) != digest for path, digest in baseline_hashes.items()):
        raise ValueError("Preserved stage-2 baseline has changed")
    meta = read_json(ROOT / cfg["di_manifest"])
    validate_roles(meta, read_json(ROOT / "assets/stage2/di-provenance.json"))
    library_path = ROOT / "assets/stage2/models-ir-provenance.json"
    library = read_json(library_path)
    for item in library["models"] + library["irs"] + meta["recordings"] + meta["segments"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed asset: {item['path']}")
    segments = {x["id"]: load_audio(str(ROOT / x["path"]), SR)[0] for x in meta["segments"]}
    inputs = {key: value.astype(np.float64) * db_to_lin(cfg["calibration_db"]) for key, value in segments.items()}
    models = {x["id"]: x for x in library["models"]}
    irs = {x["id"]: x for x in library["irs"]}
    torch.set_num_threads(4)
    rigs = []
    for rig in previous_plan["rigs"]:
        model, cab = models[rig["model_id"]], rig["cab_id"]
        capture = NamCapture(str(ROOT / model["path"]), device=args.device, rig_type=model["rig_type"],
                             cab_ir_path=str(ROOT / irs[cab]["path"]) if cab else None)
        rigs.append({**rig, "capture": capture})
    identities = {str(path): sha(ROOT / path) for path in (
        "scripts/stage3_benchmark.py", "scripts/multidi_scoring.py", "scripts/stage2_benchmark.py",
        "docs/stage3-protocol.md", "experiments/stage3.json", cfg["di_manifest"],
        "assets/stage2/models-ir-provenance.json", cfg["baseline_plan"],
        "vendor/NAM-EQ-Match/tonematch/audio.py", "vendor/NAM-EQ-Match/tonematch/features.py",
        "vendor/NAM-EQ-Match/tonematch/nam_backend.py", "vendor/NAM-EQ-Match/tonematch/match_eq.py",
        "vendor/NAM-EQ-Match/tonematch/search.py", "vendor/NAM-EQ-Match/tonematch/tone_stack.py")}
    plan = {"config": cfg, "source_hashes": identities, "baseline_hashes": baseline_hashes,
            "rigs": previous_plan["rigs"], "teachers": previous_plan["teacher_indices"],
            "di": meta, "device": args.device, "precision": "float32_no_tf32",
            "input_levels": {key: {"raw_rms": rms(value), "calibrated_peak": float(np.max(np.abs(inputs[key])))}
                             for key, value in segments.items()}}
    lock_path = output / "locked-plan.json"
    if lock_path.exists() and read_json(lock_path) != plan:
        raise ValueError("Partial cache plan mismatch")
    write_json(lock_path, plan)
    gains = np.asarray(cfg["gains_db"], float)
    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    @lru_cache(maxsize=32)
    def render(index, di_id, gain):
        return rigs[index]["capture"].process(inputs[di_id] * db_to_lin(gain))

    cache_path = output / "probe-fingerprints.json"
    cache = read_json(cache_path) if cache_path.exists() else {"identity": identities, "items": {}}
    if cache["identity"] != identities:
        raise ValueError("Probe cache source mismatch")
    fingerprints, spectra = {}, {}
    grid = None
    for index, rig in enumerate(rigs):
        for probe in cfg["probes"]:
            for gain in gains:
                key = f"{index}|{probe}|{gain:g}"
                if key not in cache["items"]:
                    audio = render(index, probe, float(gain))
                    freq, spectrum = half_octave_ltas(audio, SR)
                    cache["items"][key] = {"fingerprint": asdict(extract_fingerprint(audio, SR)),
                                           "grid": freq, "spectrum": spectrum}
                item = cache["items"][key]
                fingerprints[index, probe, float(gain)] = fingerprint_from_json(item["fingerprint"])
                spectra[index, probe, float(gain)] = np.asarray(item["spectrum"])
                freq = np.asarray(item["grid"])
                if grid is not None and not np.array_equal(grid, freq):
                    raise ValueError("Spectral grid changed")
                grid = freq
        write_json(cache_path, cache)
        print(f"probe cache {index + 1}/{len(rigs)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    render.cache_clear()
    cache_seconds = time.perf_counter() - started
    b, a = biquad_coeffs(cfg["post_eq"]["kind"], SR, cfg["post_eq"]["frequency_hz"],
                        cfg["post_eq"]["q"], cfg["post_eq"]["gain_db"])

    def color(audio, eq_on):
        return lfilter(b, a, np.pad(audio, (0, 256))).astype(np.float32) if eq_on else audio

    def eq_options(index, gain, target):
        per_probe = [design_match_ir(render(index, probe, gain), target, SR, **cfg["correction"])
                     for probe in cfg["probes"]]
        curves = np.asarray([fit[2] for fit in per_probe])
        median_ir, _, _ = make_consensus_ir(curves, grid, SR, n_taps=cfg["correction"]["n_taps"], guard=False,
                                           max_gain_db=cfg["correction"]["max_gain_db"])
        guard_ir, _, _ = make_consensus_ir(curves, grid, SR, n_taps=cfg["correction"]["n_taps"], guard=True,
                                          max_gain_db=cfg["correction"]["max_gain_db"],
                                          min_gain_db=cfg["consensus"]["min_gain_db"],
                                          max_spread_db=cfg["consensus"]["max_spread_db"])
        options = {"bypass": np.array([1], dtype=np.float32), "single_fit": per_probe[0][0],
                   "median_fit": median_ir, "guard_fit": guard_ir}
        diagnostics = {}
        for variant, ir in options.items():
            response = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(ir, 131072)), 1e-12))
            peak = float(np.max(np.abs(response)))
            if peak > cfg["correction"]["max_gain_db"] + 1e-3:
                raise ValueError("Actual correction FIR exceeds cap")
            frequencies = np.fft.rfftfreq(131072, 1 / SR)
            band = (frequencies >= 80) & (frequencies <= 8000)
            diagnostics[variant] = {"actual_max_abs_db": peak,
                                    "band_mean_abs_db": float(np.mean(np.abs(response[band]))), "taps": len(ir)}
        return options, diagnostics

    def evaluate_options(index, gain, true_audio, target):
        options, diagnostics = eq_options(index, gain, target)
        dry_output = render(index, "H0", gain)
        bypass = paired_quality(dry_output, true_audio)
        result = {}
        for variant, ir in options.items():
            quality = bypass if variant == "bypass" else paired_quality(apply_fir(dry_output, ir), true_audio)
            result[variant] = {**compare_eq(quality, bypass, cfg["gate"]), **diagnostics[variant]}
        return result

    rows, holdout_rows, oracles, fidelity = [], [], [], []
    for truth in plan["teachers"]:
        for true_gain in cfg["truth_gains_db"]:
            for reference in cfg["references"]:
                for eq_on in (False, True):
                    case = f"{truth}_{true_gain}_{reference}_E{int(eq_on)}"
                    target = color(render(truth, reference, true_gain), eq_on)
                    target_fp = extract_fingerprint(target, SR)
                    _, target_spectrum = half_octave_ltas(target, SR)
                    shape = (len(cfg["probes"]), len(rigs), len(gains))
                    current_scores, spectrum_scores = np.zeros(shape), np.zeros(shape)
                    for pi, probe in enumerate(cfg["probes"]):
                        for ri in range(len(rigs)):
                            for gi, gain in enumerate(gains):
                                fp = fingerprints[ri, probe, float(gain)]
                                current_scores[pi, ri, gi] = nonlinear_distance(fp, target_fp)["total"] + 0.25 * eq_penalty(fp, target_fp)
                                spectrum_scores[pi, ri, gi] = centered_spectral_distance(spectra[ri, probe, float(gain)], target_spectrum, grid)
                    methods = aggregate_scores(current_scores, spectrum_scores)
                    if truth == plan["teachers"][0] and true_gain == 0 and reference == "R0" and not eq_on:
                        for ri in (plan["teachers"][0], plan["teachers"][1]):
                            for gain in (0, 6):
                                direct = evaluate_gain(rigs[ri]["capture"], inputs["P0"], SR, gain, target_fp)
                                gj = int(np.flatnonzero(gains == gain)[0])
                                np.testing.assert_allclose(direct.score, current_scores[0, ri, gj], atol=1e-8)
                                fidelity.append({"rig": ri, "gain": gain, "passed": True})
                    selected = {}
                    for method, scores in methods.items():
                        best_indices = np.argmin(scores, axis=1)
                        order = sorted(range(len(rigs)), key=lambda ri: scores[ri, best_indices[ri]])
                        winner = order[0]
                        win_gi, true_gi = int(best_indices[winner]), int(best_indices[truth])
                        uncertainty = gain_set(scores[truth], gains, relative_span=cfg["gain_set"]["relative_span"],
                                               flat_span=cfg["gain_set"]["flat_span"])
                        row = {"case": case, "method": method, "truth": truth, "truth_gain_db": true_gain,
                               "reference": reference, "eq_on": eq_on, "winner": winner, "gain_db": float(gains[win_gi]),
                               "top1": winner == truth, "top3": truth in order[:3],
                               "forced_truth_gain_db": float(gains[true_gi]),
                               "forced_truth_nominal_gain_error_db": abs(float(gains[true_gi]) - true_gain),
                               "forced_truth_gain_set": uncertainty,
                               "true_gain_in_set": true_gain in uncertainty["gains"],
                               "selected_gain_set": gain_set(scores[winner], gains, relative_span=cfg["gain_set"]["relative_span"], flat_span=cfg["gain_set"]["flat_span"]),
                               "boundary": bool(abs(gains[win_gi]) == 12),
                               "ranking": [{"rig": ri, "gain": float(gains[best_indices[ri]]),
                                            "score": float(scores[ri, best_indices[ri]])} for ri in order],
                               "all_gain_scores": scores}
                        rows.append(row)
                        selected[method] = (winner, float(gains[win_gi]), float(gains[true_gi]))
                    true_h = color(render(truth, "H0", true_gain), eq_on)
                    oracles.append({"case": case, "truth": truth, "eq_on": eq_on,
                                    "variants": evaluate_options(truth, true_gain, true_h, target)})
                    unique_evaluations = {}
                    for method, (winner, gain, forced_gain) in selected.items():
                        key = (winner, gain)
                        if key not in unique_evaluations:
                            unique_evaluations[key] = evaluate_options(winner, gain, true_h, target)
                        holdout_rows.append({"case": case, "method": method, "eq_on": eq_on,
                                             "winner": winner, "gain_db": gain, "reference": reference,
                                             "variants": unique_evaluations[key],
                                             "forced_truth_estimated_gain_bypass": paired_quality(render(truth, "H0", forced_gain), true_h)})
                    print(f"evaluated {case} rows={len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if any(sha(ROOT / path) != digest for path, digest in baseline_hashes.items()):
        raise ValueError("Baseline changed during study")
    if any(sha(ROOT / path) != digest for path, digest in identities.items()):
        raise ValueError("Frozen study sources changed during execution")
    result = {"plan": plan, "retrieval": rows, "holdout": holdout_rows, "oracles": oracles,
              "score_fidelity": fidelity, "baseline_preserved": True, "cache_seconds": cache_seconds,
              "total_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "interpretation": "32 related cases, all unpaired; one same-recording holdout; frozen parameters, no Free Bird. Nominal gain only; uncertainty sets are not confidence intervals."}
    write_json(output / "results.json", result)
    print(f"completed {len(rows)} retrieval rows, {len(holdout_rows)} held-out rows, {len(oracles)} oracle rows", flush=True)


if __name__ == "__main__":
    main()
