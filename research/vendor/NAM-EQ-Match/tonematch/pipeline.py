"""Local CLI pipeline: retrieve a NAM rig and export one bounded EQ IR.

The DI is normalized once for search. Its complete input gain is exported;
new performances must retain that calibration rather than normalize per file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, freqz

from .audio import active_segment, db_to_lin, lin_to_db, load_audio, match_rms, rms, save_audio
from .match_eq import apply_fir, design_match_ir
from .search import CandidateResult, rank_captures

DEFAULT_SR = 48000


@dataclass
class MatchOutput:
    ranked: list
    best: CandidateResult
    ir_path: str
    render_path: str
    target_ref_path: str
    report_path: str
    plot_path: str | None
    report: dict
    renders: list = field(default_factory=list)


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _finalize(audio, target_rms):
    out, gain = match_rms(audio, target_rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        gain *= 0.99 / peak
        out *= 0.99 / peak
    return out, float(gain)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run_match(
    target_path: str, di_path: str, captures: list, out_dir: str,
    sr: int = DEFAULT_SR, gain_range_db=(-12.0, 12.0), refine_top: int = 5,
    render_top: int = 1, ir_taps: int = 2048, preview_s: float = 30.0, progress_cb=None,
    *, target_channel=None, di_channel=None, target_start_s: float = 0.0,
    target_end_s: float | None = None, di_normalization_db: float | None = None,
    eq_max_gain_db: float = 4.5, eq_octave_fraction: float = 2.0,
) -> MatchOutput:
    if sr != DEFAULT_SR:
        raise ValueError("Stage 1 supports 48000 Hz only")
    if not isinstance(render_top, int) or render_top < 1 or render_top > refine_top:
        raise ValueError("Require 1 <= render_top <= refine_top")
    if not math.isfinite(preview_s) or preview_s < 0:
        raise ValueError("preview_s must be finite and nonnegative")
    if not math.isfinite(target_start_s) or target_start_s < 0:
        raise ValueError("target_start_s must be finite and nonnegative")
    if target_end_s is not None and (not math.isfinite(target_end_s) or target_end_s <= target_start_s):
        raise ValueError("target_end_s must exceed target_start_s")
    if not math.isfinite(eq_max_gain_db) or not 0 <= eq_max_gain_db <= 6:
        raise ValueError("Stage 1 EQ correction limit must be between 0 and 6 dB")
    if not math.isfinite(eq_octave_fraction) or not 0 < eq_octave_fraction <= 2:
        raise ValueError("Stage 1 EQ smoothing must be at least half an octave (0 < fraction <= 2)")
    output = Path(out_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty output directory; existing results are never overwritten")

    def progress(fraction, message):
        if progress_cb:
            progress_cb(fraction, message)

    target, _ = load_audio(target_path, sr, channel=target_channel)
    di_original, _ = load_audio(di_path, sr, channel=di_channel)
    target = target[int(target_start_s * sr):None if target_end_s is None else int(target_end_s * sr)]
    if len(target) < int(0.05 * sr) or rms(target) < 1e-8 or rms(di_original) < 1e-8:
        raise ValueError("Target must contain >=50 ms of non-silent audio; DI must be non-silent")
    if di_normalization_db is None:
        di, normalization_gain = match_rms(di_original, 10 ** (-18 / 20))
        normalization_db = float(lin_to_db(normalization_gain))
    else:
        if not math.isfinite(di_normalization_db):
            raise ValueError("di_normalization_db must be finite")
        normalization_db = float(di_normalization_db)
        normalization_gain = db_to_lin(normalization_db)
        di = di_original * normalization_gain
    progress(0.02, "ranking complete candidate chains")
    ranked = rank_captures(captures, di, target, sr, gain_range_db=gain_range_db,
                           refine_top=refine_top,
                           progress_cb=lambda f, m: progress(0.02 + 0.73 * f, m))
    if not ranked:
        raise ValueError("No candidates could be evaluated")
    output.mkdir(parents=True, exist_ok=True)
    target_ref = target / float(np.max(np.abs(target))) * 0.5
    target_ref_path = output / "target_reference.wav"
    save_audio(str(target_ref_path), target_ref, sr)
    di_preview = di if preview_s == 0 else active_segment(di, sr, preview_s)
    original_preview = di_preview / normalization_gain
    preview_path = output / "preview_di.wav"
    save_audio(str(preview_path), original_preview, sr)
    renders = []
    for index, result in enumerate(ranked[:render_top], start=1):
        progress(0.76 + 0.18 * (index - 1) / render_top, f"exporting rank {index}")
        capture = result.capture
        name = re.sub(r"[^A-Za-z0-9]+", "-", result.name).strip("-")[:40] or "rig"
        prefix = output / f"rank{index:02d}_{name}"
        model_input = di_preview * db_to_lin(result.gain_db)
        amped = capture.process(model_input)
        correction, _, _ = design_match_ir(amped, target, sr, n_taps=ir_taps,
                                           max_gain_db=eq_max_gain_db,
                                           octave_fraction=eq_octave_fraction)
        correction_path = str(prefix) + "_correction_ir.wav"
        save_audio(correction_path, correction, sr)
        correction_disk, _ = load_audio(correction_path, sr)
        _, response = freqz(correction_disk, worN=65536)
        actual_limit = float(np.max(np.abs(20 * np.log10(np.maximum(np.abs(response), 1e-12)))))
        if actual_limit > eq_max_gain_db + 1e-3:
            raise ValueError("Exported correction IR exceeds the requested EQ limit")
        cabinet = getattr(capture, "cab_ir", None)
        combined = correction_disk if cabinet is None else fftconvolve(cabinet, correction_disk).astype(np.float32)
        ir_path = str(prefix) + "_playback_ir.wav"
        save_audio(ir_path, combined, sr)
        ir_disk, _ = load_audio(ir_path, sr)
        raw_process = getattr(capture, "process_raw", capture.process)
        raw_output = apply_fir(raw_process(model_input), ir_disk)
        rendered, output_gain = _finalize(raw_output, rms(target_ref))
        render_path = str(prefix) + "_render.wav"
        save_audio(render_path, rendered, sr)
        source = getattr(capture, "path", None)
        nam_copy = None
        if source and os.path.isfile(source):
            nam_copy = str(prefix) + ".nam"
            shutil.copyfile(source, nam_copy)
        input_db = normalization_db + float(result.gain_db)
        output_db = float(lin_to_db(output_gain))
        input_adjustment = float(getattr(capture, "lv2_input_adjustment_db", 0.0))
        output_adjustment = float(getattr(capture, "lv2_output_adjustment_db", 0.0))
        entry = {
            "schema_version": 1, "rank": index, "name": result.name,
            "file": str(Path(source).resolve()) if source and os.path.isfile(source) else None,
            "nam_copy": str(Path(nam_copy).resolve()) if nam_copy else None,
            "nam_sha256": _sha256(nam_copy) if nam_copy else None,
            "rig_type": getattr(capture, "rig_type", "test-only"),
            "config_adapter": getattr(capture, "config_adapter", []),
            "precision": getattr(capture, "precision", "test-only"),
            "cab_ir_source": getattr(capture, "cab_ir_path", None),
            "cab_ir_sha256": _sha256(capture.cab_ir_path) if getattr(capture, "cab_ir_path", None) else None,
            "di_normalization_db": normalization_db, "search_gain_db": float(result.gain_db),
            "input_gain_db": input_db, "output_gain_db": output_db,
            "lv2_recommended_input_adjustment_db": input_adjustment,
            "lv2_recommended_output_adjustment_db": output_adjustment,
            "lv2_input_control_db": input_db - input_adjustment,
            "lv2_output_control_db": output_db - output_adjustment,
            "lv2_verified": False, "score": float(result.score), "sample_rate": sr,
            "ir": str(Path(ir_path).resolve()), "ir_sha256": _sha256(ir_path),
            "correction_ir": str(Path(correction_path).resolve()),
            "correction_actual_max_abs_db": actual_limit,
            "eq_limit_db": eq_max_gain_db, "eq_octave_fraction": eq_octave_fraction,
            "render": str(Path(render_path).resolve()), "render_frames": len(rendered),
            "preview_di": str(preview_path), "preview_di_frames": len(original_preview),
            "tail_frames": len(ir_disk) - 1,
        }
        settings_path = str(prefix) + "_settings.json"
        _write_json(settings_path, entry)
        entry["settings_json"] = settings_path
        renders.append(entry)
    report = {
        "schema_version": 1, "sample_rate": sr,
        "source_commit": "1074151da12654fcf1125ed6443a1e53ea5b973d + local patches",
        "target": {"path": str(Path(target_path).resolve()), "sha256": _sha256(target_path),
                   "channel": target_channel, "start_s": target_start_s, "end_s": target_end_s,
                   "analyzed_frames": len(target)},
        "di": {"path": str(Path(di_path).resolve()), "sha256": _sha256(di_path),
               "channel": di_channel, "normalization_db": normalization_db},
        "best_model": renders[0], "renders": renders,
        "ranking": [{"name": r.name, "file": getattr(r.capture, "path", None),
                     "search_gain_db": float(r.gain_db), "score": float(r.score),
                     "nonlinear_distance": float(r.nl_distance), "eq_penalty": float(r.eq_penalty)} for r in ranked],
        "ranking_stage": "refined candidates only; heuristic scores are not confidence probabilities",
        "how_to_use": "Original DI -> recorded total input gain -> raw NAM -> playback IR (100% wet, no normalization) -> recorded output gain. Do not add a second cab. Reuse input calibration on new DI. NAM_LV2 controls subtract metadata adjustments; actual plugin verification is pending.",
        "limitations": ["No claim of original equipment recovery or Free Bird similarity",
                        "Mono, native 48 kHz WaveNet only; no time-effect fitting",
                        "Existing NAM is copied, not trained; playback IR remains external"],
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    progress(1.0, "done")
    return MatchOutput(ranked, ranked[0], renders[0]["ir"], renders[0]["render"],
                       str(target_ref_path), str(report_path), None, report, renders)


def render_preset(settings_path: str, di_path: str, output_path: str, *, device="cpu", channel=None):
    """Replay exported files on an unnormalized DI, preserving convolution tail."""
    from .nam_backend import NamCapture

    if Path(output_path).exists():
        raise FileExistsError("Output already exists")
    with open(settings_path, encoding="utf-8") as stream:
        settings = json.load(stream)
    if settings.get("schema_version") != 1 or settings.get("sample_rate") != DEFAULT_SR:
        raise ValueError("Unsupported preset schema or sample rate")
    for path_key, hash_key in (("nam_copy", "nam_sha256"), ("ir", "ir_sha256")):
        if not settings.get(path_key) or _sha256(settings[path_key]) != settings[hash_key]:
            raise ValueError(f"Missing or changed preset asset: {path_key}")
    input_db, output_db = settings["input_gain_db"], settings["output_gain_db"]
    if not all(math.isfinite(value) for value in (input_db, output_db)):
        raise ValueError("Preset gains must be finite")
    capture = NamCapture(settings["nam_copy"], device=device)
    di, _ = load_audio(di_path, DEFAULT_SR, channel=channel)
    ir, _ = load_audio(settings["ir"], DEFAULT_SR)
    rendered = apply_fir(capture.process_raw(di * db_to_lin(input_db)), ir) * db_to_lin(output_db)
    save_audio(output_path, rendered, DEFAULT_SR)
    return rendered
