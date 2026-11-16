"""Frozen closed-library cross-DI study. No reference-song access or weight tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import lfilter, stft
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/NAM-EQ-Match"
sys.path.insert(0, str(VENDOR))
from tonematch.audio import db_to_lin, load_audio, rms
from tonematch.features import ToneFingerprint, eq_penalty, extract_fingerprint, nonlinear_distance
from tonematch.match_eq import apply_fir, design_match_ir
from tonematch.nam_backend import NamCapture
from tonematch.search import rank_captures
from tonematch.tone_stack import biquad_coeffs

SR = 48000


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                   default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint_from_json(value):
    return ToneFingerprint(**{key: np.asarray(val) if isinstance(val, list) else val for key, val in value.items()})


def score_variants(parts, weighted_eq):
    """Prespecified ablations only. Terms already contain the upstream weights."""
    total = sum(parts.values())
    return {
        "baseline": total + weighted_eq,
        "no_drive_gap": total - parts["drive_gap"] + weighted_eq,
        # The directional drive guard still includes an HF cue in this ablation.
        "no_hf_terms": total - parts["hf_ratio"] - parts["spec_crest_hf"] + weighted_eq,
        "no_dynamic_texture": total - sum(parts[k] for k in ("dyn_range", "sustain", "flux", "mfcc_texture", "mfcc_delta")) + weighted_eq,
        "no_eq_penalty": total,
        "eq_only": weighted_eq,
    }


def choose_two_stage(scores, gains, refine_top=5):
    """Reproduce production selection on equal 8 s inputs; also expose exhaustive ranks."""
    scores, gains = np.asarray(scores), np.asarray(gains)
    if (scores.ndim != 2 or scores.shape[1] != len(gains) or not len(scores)
            or not np.isfinite(scores).all() or refine_top < 1):
        raise ValueError("Invalid score grid")
    coarse_indices = [int(np.flatnonzero(gains == g)[0]) for g in (-12, -6, 0, 6, 12)]
    coarse_best = [coarse_indices[int(np.argmin(row[coarse_indices]))] for row in scores]
    coarse = sorted(range(len(scores)), key=lambda i: scores[i, coarse_best[i]])
    refined = []
    for i in coarse[:refine_top]:
        allowed = np.flatnonzero(np.abs(gains - gains[coarse_best[i]]) <= 3)
        best = int(allowed[np.argmin(scores[i, allowed])])
        refined.append((i, best, float(scores[i, best])))
    refined.sort(key=lambda row: row[2])
    exhaustive = [(i, int(np.argmin(row)), float(np.min(row))) for i, row in enumerate(scores)]
    exhaustive.sort(key=lambda row: row[2])
    return coarse, refined, exhaustive


def paired_quality(candidate, reference):
    """Only for the same held-out DI C, never between different performances."""
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if (candidate.ndim != 1 or reference.ndim != 1 or not len(candidate) or not len(reference)
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()
            or rms(candidate) == 0 or rms(reference) == 0):
        raise ValueError("Quality comparison requires finite, non-silent mono outputs")
    n = max(len(candidate), len(reference))
    candidate = np.pad(candidate, (0, n - len(candidate)))
    reference = np.pad(reference, (0, n - len(reference)))
    candidate *= rms(reference) / rms(candidate)
    errors = []
    for fft in (512, 2048, 8192):
        _, _, c = stft(candidate, SR, nperseg=fft, noverlap=fft * 3 // 4)
        _, _, r = stft(reference, SR, nperseg=fft, noverlap=fft * 3 // 4)
        errors.append(float(np.linalg.norm(np.abs(c) - np.abs(r)) / max(np.linalg.norm(np.abs(r)), 1e-12)))
    fc, fr = extract_fingerprint(candidate, SR), extract_fingerprint(reference, SR)
    band = (fc.ltas_grid >= 80) & (fc.ltas_grid <= 8000)
    delta = fc.ltas_db[band] - fr.ltas_db[band]
    delta -= np.median(delta)
    return {"mrstft_sc": float(np.mean(errors)), "ltas_mae_db": float(np.mean(np.abs(delta)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/stage2/real")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if (out / "results.json").exists():
        raise FileExistsError("Completed study exists; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "experiments/stage2.json"
    cfg = read_json(config_path)
    library_path = ROOT / "assets/stage2/models-ir-provenance.json"
    di_path = ROOT / "assets/stage2/di-provenance.json"
    library, di_meta = read_json(library_path), read_json(di_path)
    for item in library["models"] + library["irs"] + di_meta["recordings"] + di_meta["segments"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed asset: {item['path']}")
    torch.set_num_threads(4)
    segments = {item["id"]: load_audio(str(ROOT / item["path"]), SR)[0] for item in di_meta["segments"]}
    if set(segments) != {"A", "B", "C"} or any(len(x) != 8 * SR for x in segments.values()):
        raise ValueError("Frozen protocol requires three 8-second DIs")
    calibration = db_to_lin(cfg["calibration_rms_dbfs"]) / rms(segments["A"])
    # Production active_segment promotes to float64 before gain multiplication.
    inputs = {key: value.astype(np.float64) * calibration for key, value in segments.items()}
    irs = {item["id"]: item for item in library["irs"]}
    rigs = []
    for model in library["models"]:
        cab_ids = [cfg["primary_cab"]] if model["rig_type"] == "amp-only" else [None]
        if model["id"] in cfg["secondary_cab_models"]:
            cab_ids.append(cfg["secondary_cab"])
        for cab_id in cab_ids:
            rig_id = model["id"] + ("@" + cab_id if cab_id else "")
            cap = NamCapture(str(ROOT / model["path"]), device=args.device,
                             cab_ir_path=str(ROOT / irs[cab_id]["path"]) if cab_id else None,
                             rig_type=model["rig_type"])
            cap.name = rig_id
            rigs.append({"id": rig_id, "model_id": model["id"], "family": model["family"],
                         "cab_id": cab_id, "capture": cap})
    teachers = [next(i for i, r in enumerate(rigs) if r["model_id"] == mid and
                    r["cab_id"] in (None, cfg["primary_cab"])) for mid in cfg["teacher_models"]]
    gains = np.asarray(cfg["gains_db"], float)
    b, a = biquad_coeffs(cfg["post_eq"]["kind"], SR, cfg["post_eq"]["frequency_hz"],
                        cfg["post_eq"]["q"], cfg["post_eq"]["gain_db"])
    asset_identity = {"config_sha256": sha(config_path), "library_sha256": sha(library_path),
                      "di_manifest_sha256": sha(di_path), "features_sha256": sha(VENDOR / "tonematch/features.py"),
                      "backend_sha256": sha(VENDOR / "tonematch/nam_backend.py"),
                      "runner_sha256": sha(__file__), "audio_sha256": sha(VENDOR / "tonematch/audio.py"),
                      "search_sha256": sha(VENDOR / "tonematch/search.py"),
                      "eq_sha256": sha(VENDOR / "tonematch/match_eq.py"),
                      "input_arithmetic": "float64 calibration and gain, backend casts float32"}
    plan = {"config": cfg, "identity": asset_identity,
            "rigs": [{k: v for k, v in r.items() if k != "capture"} for r in rigs],
            "teacher_indices": teachers, "calibration_db": float(20 * np.log10(calibration)),
            "di_levels": {k: {"raw_rms": rms(x), "calibrated_peak": float(np.max(np.abs(inputs[k])))}
                          for k, x in segments.items()}, "device": args.device,
            "precision": "float32_no_tf32", "holdout_use": "C excluded from retrieval, feature cache and EQ fitting"}
    plan_path = out / "locked-plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Existing partial run has a different locked plan")
    write_json(plan_path, plan)

    @lru_cache(maxsize=24)
    def render(index, di_id, gain):
        return rigs[index]["capture"].process(inputs[di_id] * db_to_lin(gain))

    def color(audio, enabled):
        return lfilter(b, a, np.pad(audio, (0, 256))).astype(np.float32) if enabled else audio

    cache_path = out / "fingerprints.json"
    cache = read_json(cache_path) if cache_path.exists() else {"identity": asset_identity, "items": {}}
    if cache["identity"] != asset_identity:
        raise ValueError("Cached features do not match this study")
    fps = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    for index, rig in enumerate(rigs):
        for di_id in cfg["reference_dis"]:
            for gain in gains:
                key = f"{index}|{di_id}|{gain:g}"
                if key not in cache["items"]:
                    cache["items"][key] = asdict(extract_fingerprint(render(index, di_id, float(gain)), SR))
                fps[index, di_id, float(gain)] = fingerprint_from_json(cache["items"][key])
        write_json(cache_path, cache)
        print(f"cache {index + 1}/{len(rigs)} {rig['id']} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    cache_seconds = time.perf_counter() - started
    render.cache_clear()
    records, quality_rows, fidelity = [], [], []
    for truth in teachers:
        for true_gain in cfg["truth_gains_db"]:
            for ref_di in cfg["reference_dis"]:
                for eq_on in (False, True):
                    target = color(render(truth, ref_di, true_gain), eq_on)
                    target_fp = extract_fingerprint(target, SR) if eq_on else fps[truth, ref_di, float(true_gain)]
                    for search_di in cfg["reference_dis"]:
                        same = ref_di == search_di
                        case_id = f"{truth}_{true_gain}_{ref_di}_{search_di}_E{int(eq_on)}"
                        grids = {method: np.zeros((len(rigs), len(gains))) for method in cfg["methods"]}
                        parts_by_key = {}
                        for i in range(len(rigs)):
                            for j, gain in enumerate(gains):
                                fp = fps[i, search_di, float(gain)]
                                parts = nonlinear_distance(fp, target_fp)
                                parts.pop("total")
                                weighted_eq = 0.25 * eq_penalty(fp, target_fp)
                                parts_by_key[i, j] = {**parts, "weighted_eq": weighted_eq}
                                values = score_variants(parts, weighted_eq)
                                for method in grids:
                                    grids[method][i, j] = values[method]
                        selections = {}
                        for method, scores in grids.items():
                            coarse, refined, exhaustive = choose_two_stage(scores, gains, cfg["refine_top"])
                            winner, gj, value = refined[0]
                            selected = [i for i, _, _ in refined]
                            oracle_gj = int(np.argmin(scores[truth]))
                            row = {"case": case_id, "method": method, "truth": truth,
                                   "truth_model": rigs[truth]["model_id"], "truth_gain_db": true_gain,
                                   "reference_di": ref_di, "search_di": search_di, "same_di": same, "eq_on": eq_on,
                                   "winner": winner, "gain_db": float(gains[gj]), "score": value,
                                   "top1": winner == truth, "top3": truth in selected[:3],
                                   "coarse_top5": truth in coarse[:cfg["refine_top"]],
                                   "exhaustive_top1": exhaustive[0][0] == truth,
                                   "exhaustive_top3": truth in [i for i, _, _ in exhaustive[:3]],
                                   "nam_top1": rigs[winner]["model_id"] == rigs[truth]["model_id"],
                                   "family_top1": rigs[winner]["family"] == rigs[truth]["family"],
                                   "nominal_gain_error_if_exact_rig": abs(float(gains[gj]) - true_gain) if winner == truth else None,
                                   "forced_truth_best_gain_db": float(gains[oracle_gj]),
                                   "forced_truth_nominal_gain_error_db": abs(float(gains[oracle_gj]) - true_gain),
                                   "refined_ranking": [{"rig": i, "gain": float(gains[j]), "score": s} for i, j, s in refined],
                                   "exhaustive_ranking": [{"rig": i, "gain": float(gains[j]), "score": s} for i, j, s in exhaustive],
                                   "winner_parts": parts_by_key[winner, gj],
                                   "truth_best_parts": parts_by_key[truth, oracle_gj],
                                   "truth_at_true_gain_parts": parts_by_key[truth, int(np.flatnonzero(gains == true_gain)[0])]}
                            records.append(row)
                            selections[method] = (winner, float(gains[gj]))
                        # Exercise production search itself on four selected controls/cross cases.
                        if truth == teachers[0] and true_gain == 0 and not eq_on:
                            direct = rank_captures([r["capture"] for r in rigs], inputs[search_di], target, SR,
                                                   refine_top=cfg["refine_top"])
                            expected = [row for row in records if row["case"] == case_id and row["method"] == "baseline"][0]
                            expected_ranks = expected["refined_ranking"]
                            assert [r.name for r in direct] == [rigs[x["rig"]]["id"] for x in expected_ranks]
                            np.testing.assert_allclose([r.gain_db for r in direct], [x["gain"] for x in expected_ranks], atol=0)
                            np.testing.assert_allclose([r.score for r in direct], [x["score"] for x in expected_ranks], atol=1e-8)
                            fidelity.append({"case": case_id, "matches_production": True})
                        # C is used only after all selections; fit correction on search DI/reference.
                        winner, gain = selections["baseline"]
                        true_c = color(render(truth, "C", true_gain), eq_on)
                        chosen_train = render(winner, search_di, gain)
                        ir, _, _ = design_match_ir(chosen_train, target, SR, **cfg["correction"])
                        chosen_c = render(winner, "C", gain)
                        truth_train = render(truth, search_di, true_gain)
                        oracle_ir, _, _ = design_match_ir(truth_train, target, SR, **cfg["correction"])
                        metrics = {"selected_before_eq": paired_quality(chosen_c, true_c),
                                   "selected_after_eq": paired_quality(apply_fir(chosen_c, ir), true_c),
                                   "true_rig_true_gain_before_eq": paired_quality(render(truth, "C", true_gain), true_c),
                                   "true_rig_true_gain_fitted_eq": paired_quality(apply_fir(render(truth, "C", true_gain), oracle_ir), true_c)}
                        quality_rows.append({"case": case_id, "truth": truth, "winner": winner, "same_di": same,
                                             "eq_on": eq_on, "metrics": metrics})
                    print(f"cases {rigs[truth]['id']} g={true_gain} ref={ref_di} EQ={eq_on}", flush=True)
    summaries = {}
    for method in cfg["methods"]:
        summaries[method] = {}
        for same in (True, False):
            for eq_on in (False, True):
                group = [r for r in records if r["method"] == method and r["same_di"] == same and r["eq_on"] == eq_on]
                key = ("same" if same else "cross") + ("_eq" if eq_on else "_noeq")
                summary = {"n": len(group), **{metric: sum(r[metric] for r in group) for metric in
                           ("top1", "top3", "coarse_top5", "exhaustive_top1", "exhaustive_top3", "nam_top1", "family_top1")},
                           "median_forced_truth_nominal_gain_error_db": float(np.median([r["forced_truth_nominal_gain_error_db"] for r in group]))}
                if not same:
                    summary["internal_gate_passed"] = all(summary[k] >= len(group) * cfg["gate"][g] for k, g in
                        (("top1", "cross_top1_fraction"), ("top3", "cross_top3_fraction"), ("coarse_top5", "cross_coarse_top5_fraction")))
                summaries[method][key] = summary
    quality_summary = {}
    for same in (True, False):
        for eq_on in (False, True):
            group = [r for r in quality_rows if r["same_di"] == same and r["eq_on"] == eq_on]
            key = ("same" if same else "cross") + ("_eq" if eq_on else "_noeq")
            quality_summary[key] = {variant: {"median_sc": float(np.median([r["metrics"][variant]["mrstft_sc"] for r in group])),
                "median_ltas_db": float(np.median([r["metrics"][variant]["ltas_mae_db"] for r in group])),
                "within_screening_threshold": sum(r["metrics"][variant]["mrstft_sc"] <= cfg["gate"]["holdout_sc"] and
                                                   r["metrics"][variant]["ltas_mae_db"] <= cfg["gate"]["holdout_ltas_db"] for r in group)}
                for variant in group[0]["metrics"]}
    result = {"plan": plan, "cache_seconds": cache_seconds, "total_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "summary": summaries, "holdout_summary": quality_summary, "production_fidelity": fidelity,
              "cases": records, "holdout": quality_rows,
              "interpretation": "Frozen primary baseline. Ablations diagnostic only; no tuning or Free Bird. Nominal gain differences are not physical identification accuracy."}
    write_json(out / "results.json", result)
    print(json.dumps({"baseline": summaries["baseline"], "holdout": quality_summary, "seconds": result["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
