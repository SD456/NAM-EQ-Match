"""Real pretrained NAM inference and export tests; probes are synthetic, not guitar DI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy.signal import fftconvolve, freqz

from tonematch.audio import db_to_lin, load_audio, save_audio
from tonematch.nam_backend import NamCapture
from tonematch.pipeline import render_preset, run_match
from tonematch.search import rank_captures

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/test_models/Lum_Marshall_2204_1982_AMP_Channel3_A1.nam"
MODEL_HASH = "a9579e9a136b9dd0a360ecd64235bedaa58f4468e7c58dc8e734ca2b19fd042f"
SR = 48000


def probe(seconds=0.4):
    t = np.arange(round(seconds * SR)) / SR
    rng = np.random.default_rng(481)
    return (0.07 * np.sin(2 * np.pi * 110 * t) + 0.02 * np.sin(2 * np.pi * 733 * t)
            + rng.normal(0, 0.01, len(t))).astype(np.float32)


@pytest.fixture(scope="module")
def capture():
    assert MODEL.exists(), "Real pretrained NAM fixture is required; do not silently skip"
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_HASH
    torch.set_num_threads(4)
    return NamCapture(str(MODEL), device="cpu")


def test_real_wavenet_chunked_equals_continuous(capture):
    x = probe(0.65)
    y = capture.process_raw(x, chunk_s=1)
    chunked = capture.process_raw(x, chunk_s=0.17)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.max(np.abs(y)) > 1e-4
    np.testing.assert_allclose(chunked, y, atol=2e-6, rtol=2e-5)


def test_real_nam_search_recovers_known_input_gain(capture):
    x = probe(0.4)
    target = capture.process_raw(x * db_to_lin(6))
    ranked = rank_captures([capture], x, target, SR, gain_range_db=(-12, 12), refine_top=1)
    assert ranked[0].gain_db == 6
    assert ranked[0].score < 1e-5


def test_real_wavenet_cuda_cpu_and_ten_second_boundary(capture):
    assert torch.cuda.is_available(), "Stage 1 on this RTX host requires actual CUDA inference"
    gpu = NamCapture(str(MODEL), device="cuda")
    short = probe(0.4)
    np.testing.assert_allclose(gpu.process_raw(short), capture.process_raw(short), atol=1e-5, rtol=1e-4)
    x = probe(10.25)
    continuous = gpu.process_raw(x, chunk_s=12)
    chunked = gpu.process_raw(x, chunk_s=10)
    np.testing.assert_allclose(chunked, continuous, atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(chunked, gpu.process_raw(x, chunk_s=10))


@pytest.mark.parametrize("normalization_db", [None, 0.0, -3.0])
def test_real_export_replay_keeps_total_gain_float_ir_and_tail(capture, tmp_path, normalization_db):
    x = probe()
    di, target = tmp_path / "di.wav", tmp_path / "target.wav"
    save_audio(str(di), x, SR)
    # Different lengths are deliberate: no paired-DI assumption in the matching API.
    save_audio(str(target), capture.process_raw(probe(0.27)), SR)
    out = run_match(str(target), str(di), [capture], str(tmp_path / "run"),
                    gain_range_db=(0, 0), refine_top=1, preview_s=0,
                    di_normalization_db=normalization_db)
    entry = out.renders[0]
    assert entry["input_gain_db"] == entry["di_normalization_db"] + entry["search_gain_db"]
    assert entry["lv2_verified"] is False
    assert entry["lv2_output_control_db"] == entry["output_gain_db"] - capture.lv2_output_adjustment_db
    assert sf.info(entry["ir"]).subtype == "FLOAT"
    ir, _ = load_audio(entry["ir"], SR)
    _, h = freqz(ir, worN=262144)
    assert np.max(np.abs(20 * np.log10(np.abs(h)))) <= 4.501
    expected = fftconvolve(capture.process_raw(x * db_to_lin(entry["input_gain_db"])), ir).astype(np.float32)
    expected *= db_to_lin(entry["output_gain_db"])
    disk, _ = load_audio(out.render_path, SR)
    assert len(disk) == len(x) + len(ir) - 1
    np.testing.assert_allclose(disk, expected, atol=2e-6, rtol=2e-5)
    replayed = render_preset(entry["settings_json"], str(di), str(tmp_path / "replayed.wav"))
    np.testing.assert_allclose(replayed, disk, atol=2e-6, rtol=2e-5)
    # Reuse the exact same input calibration on a quieter, held-out input.
    quiet = tmp_path / "quiet.wav"
    save_audio(str(quiet), x * 0.25, SR)
    quiet_output = render_preset(entry["settings_json"], str(quiet), str(tmp_path / "quiet_out.wav"))
    assert not np.allclose(quiet_output, replayed)


def test_amp_only_combines_cab_and_correction_once(capture, tmp_path):
    # This short synthetic FIR tests routing/tail correctness; it is not a cabinet capture.
    cabinet = np.array([0.7, -0.15, 0.05, 0.01], dtype=np.float32)
    cab_path = tmp_path / "routing_test_ir.wav"
    save_audio(str(cab_path), cabinet, SR)
    rig = NamCapture(str(MODEL), device="cpu", cab_ir_path=str(cab_path), rig_type="amp-only")
    x = probe()
    np.testing.assert_allclose(rig.process(x), fftconvolve(capture.process_raw(x), cabinet), atol=1e-6)
    di, target = tmp_path / "di.wav", tmp_path / "target.wav"
    save_audio(str(di), x, SR)
    save_audio(str(target), rig.process(x), SR)
    out = run_match(str(target), str(di), [rig], str(tmp_path / "run"),
                    gain_range_db=(0, 0), refine_top=1, preview_s=0, di_normalization_db=0)
    entry = out.renders[0]
    correction, _ = load_audio(entry["correction_ir"], SR)
    combined, _ = load_audio(entry["ir"], SR)
    np.testing.assert_allclose(combined, fftconvolve(cabinet, correction), atol=1e-7)
    replay = render_preset(entry["settings_json"], str(di), str(tmp_path / "replay.wav"))
    disk, _ = load_audio(entry["render"], SR)
    np.testing.assert_allclose(replay, disk, atol=2e-6, rtol=2e-5)


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "original.wav"
    sentinel.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        run_match("missing.wav", "missing.wav", [], str(tmp_path))
    assert sentinel.read_bytes() == b"unchanged"


def test_preset_asset_changes_are_detected(tmp_path):
    settings = tmp_path / "settings.json"
    asset = tmp_path / "wrong.nam"
    asset.write_text("{}")
    settings.write_text(json.dumps({"schema_version": 1, "sample_rate": SR,
                                   "nam_copy": str(asset), "nam_sha256": "wrong"}))
    with pytest.raises(ValueError, match="changed preset"):
        render_preset(str(settings), "unused.wav", str(tmp_path / "new.wav"))


def test_cli_match_and_replay_use_exported_files(capture, tmp_path):
    x = probe(0.2)
    di, target, cab = (tmp_path / name for name in ("di.wav", "target.wav", "test_ir.wav"))
    save_audio(str(di), x, SR)
    save_audio(str(cab), np.array([1.0, 0.1], dtype=np.float32), SR)
    save_audio(str(target), fftconvolve(capture.process_raw(x), [1.0, 0.1]), SR)
    manifest = tmp_path / "candidates.json"
    manifest.write_text(json.dumps([{"path": str(MODEL), "rig_type": "amp-only", "cab_ir_path": str(cab)}]))
    output = tmp_path / "cli_run"
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    command = [sys.executable, str(ROOT / "vendor/NAM-EQ-Match/match.py"),
               "--target", str(target), "--di", str(di), "--manifest", str(manifest),
               "--out", str(output), "--gain-range", "0", "0", "--refine-top", "1",
               "--preview-s", "0", "--di-normalization-db", "0", "--device", "cpu"]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    entry = json.loads((output / "report.json").read_text(encoding="utf-8"))["best_model"]
    replay = tmp_path / "cli_replay.wav"
    command = [sys.executable, str(ROOT / "vendor/NAM-EQ-Match/render.py"),
               "--settings", entry["settings_json"], "--di", str(di), "--out", str(replay)]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    np.testing.assert_allclose(load_audio(str(replay))[0], load_audio(entry["render"])[0], atol=2e-6)


def test_original_wavs_are_intact():
    manifest = json.loads((ROOT / "docs/original-wav-sha256.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = ROOT / "WAV_Data" / item["name"]
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
