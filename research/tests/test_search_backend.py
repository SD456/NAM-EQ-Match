"""Search bookkeeping and invalid-boundary regressions; not a tonal-quality benchmark."""
from types import SimpleNamespace

import numpy as np
import pytest

from tonematch import search


def test_only_refined_candidates_can_win(monkeypatch):
    """A low short-segment score must never compete with long-segment scores."""
    captures = [SimpleNamespace(name=name, sample_rate=48000) for name in ("first", "second", "excluded")]
    calls = []
    monkeypatch.setattr(search, "extract_fingerprint", lambda *_: object())

    def score(capture, di, sr, gain_db, target_fp):
        calls.append((capture.name, len(di), gain_db))
        coarse = len(di) == 48000
        base = {"first": 0, "second": 1, "excluded": 2}[capture.name]
        value = base if coarse else 10 - base
        return search.CandidateResult(capture, gain_db, value, value, 0)

    monkeypatch.setattr(search, "evaluate_gain", score)
    progress = []
    results = search.rank_captures(
        captures, np.ones(96000), np.ones(36000), 48000,
        coarse_seg_s=1, refine_seg_s=2, refine_top=2,
        progress_cb=lambda fraction, _: progress.append(fraction),
    )
    assert [r.name for r in results] == ["second", "first"]
    assert all(r.score >= 9 for r in results)
    assert {g for _, size, g in calls if size == 48000} == {-12, -6, 0, 6, 12}
    assert {g for _, size, g in calls if size == 96000} == {-12, -11, -10, -9}
    assert progress == sorted(progress) and progress[-1] == 1


@pytest.mark.parametrize("options", [
    {"refine_top": 0}, {"refine_top": 1.5},
    {"gain_range_db": (2, -2)}, {"gain_range_db": (0, np.nan)},
    {"coarse_seg_s": 0}, {"refine_seg_s": np.inf},
    {"coarse_seg_s": 4, "refine_seg_s": 2}, {"sr": 0},
])
def test_invalid_search_options_fail(options):
    args = dict(captures=[], di=np.ones(3000), target=np.ones(4000), sr=48000)
    args.update(options)
    with pytest.raises(ValueError):
        search.rank_captures(**args)


@pytest.mark.parametrize("bad", [np.array([]), np.zeros(3000), np.array([np.nan]), np.ones((2, 2))])
def test_invalid_di_fails(bad):
    with pytest.raises(ValueError):
        search.rank_captures([], bad, np.ones(3000), 48000)


def test_capture_sample_rate_must_match():
    with pytest.raises(ValueError, match="sample rate"):
        search.rank_captures([SimpleNamespace(sample_rate=44100)], np.ones(3000), np.ones(4000), 48000)


def test_silent_prefix_does_not_hide_the_active_di():
    sr = 48000
    note = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    di = np.r_[np.zeros(sr * 8), note]
    capture = SimpleNamespace(sample_rate=sr, name="identity", process=lambda x: x)
    result = search.rank_captures([capture], di, note, sr, gain_range_db=(0, 0), refine_top=1)
    assert len(result) == 1 and np.isfinite(result[0].score)


@pytest.mark.parametrize("gains", [[], [np.nan], [[0, 1]]])
def test_invalid_gain_grid_fails(gains):
    with pytest.raises(ValueError, match="Gain grid"):
        search.best_gain_for_capture(None, np.ones(3000), 48000, None, gains)


def test_nonfinite_render_is_not_a_search_result():
    capture = SimpleNamespace(sample_rate=48000, name="invalid", process=lambda x: x * np.nan)
    with pytest.raises(ValueError, match="Output"):
        search.evaluate_gain(capture, np.ones(3000), 48000, 0, None)


def test_nonfinite_score_is_not_a_search_result(monkeypatch):
    capture = SimpleNamespace(sample_rate=48000, name="valid", process=lambda x: x)
    monkeypatch.setattr(search, "extract_fingerprint", lambda *_: object())
    monkeypatch.setattr(search, "nonlinear_distance", lambda *_: {"total": np.nan})
    monkeypatch.setattr(search, "eq_penalty", lambda *_: 0)
    with pytest.raises(ValueError, match="score"):
        search.evaluate_gain(capture, np.ones(3000), 48000, 0, None)


@pytest.mark.parametrize("architecture,sr", [("LSTM", 48000), ("SlimmableContainer", 48000), ("WaveNet", None), ("WaveNet", 44100)])
def test_unsupported_nam_fails_explicitly(tmp_path, architecture, sr):
    import json
    from tonematch.nam_backend import NamCapture
    path = tmp_path / "unsupported.nam"
    path.write_text(json.dumps({"architecture": architecture, "sample_rate": sr}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported|48000"):
        NamCapture(str(path), device="cpu")


@pytest.mark.parametrize("rig_type,cab", [("fullrig", None), ("amp-only", None), ("full-rig", "cab.wav")])
def test_invalid_candidate_chain_is_rejected(tmp_path, rig_type, cab):
    import json
    from tonematch.nam_backend import NamCapture
    path = tmp_path / "invalid_chain.nam"
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000}), encoding="utf-8")
    with pytest.raises(ValueError, match="rig_type|cabinet IR"):
        NamCapture(str(path), device="cpu", rig_type=rig_type, cab_ir_path=cab)


@pytest.fixture(scope="module")
def real_legacy_capture():
    from pathlib import Path
    import torch
    from tonematch.nam_backend import NamCapture
    path = Path(__file__).resolve().parents[1] / "assets/test_models/Lum_Marshall_2204_1982_AMP_Channel3_A1.nam"
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    capture = NamCapture(str(path), device="cpu")
    yield capture
    torch.set_num_threads(threads)


def test_real_legacy_weights_are_loaded_in_original_order(real_legacy_capture):
    import hashlib
    import json
    from pathlib import Path
    capture = real_legacy_capture
    original = Path(capture.path).read_bytes()
    assert hashlib.sha256(original).hexdigest() == "a9579e9a136b9dd0a360ecd64235bedaa58f4468e7c58dc8e734ca2b19fd042f"
    document = json.loads(original)
    # NAM's inverse export serializes every loaded tensor in plugin weight order.
    np.testing.assert_array_equal(capture.model._export_weights(), np.asarray(document["weights"], dtype=np.float32))
    assert capture.config_adapter
    assert capture.sha256 == hashlib.sha256(original).hexdigest()
    assert "head_size" in document["config"]["layers"][0]
    assert capture.metadata == document["metadata"]


def test_real_legacy_chunked_inference_matches_full(real_legacy_capture):
    x = (0.08 * np.sin(2 * np.pi * 220 * np.arange(10007) / 48000)).astype(np.float32)
    full = real_legacy_capture.process_raw(x, chunk_s=1)
    chunked = real_legacy_capture.process_raw(x, chunk_s=0.02)
    assert full.shape == chunked.shape == x.shape
    assert np.isfinite(full).all() and np.max(np.abs(full)) > 1e-6
    np.testing.assert_allclose(chunked, full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mutation,error", [("gated", "Legacy gated"), ("trailing_weight", "weight count")])
def test_legacy_adapter_rejects_unverified_layouts(real_legacy_capture, tmp_path, mutation, error):
    import json
    from pathlib import Path
    from tonematch.nam_backend import NamCapture
    document = json.loads(Path(real_legacy_capture.path).read_bytes())
    if mutation == "gated":
        document["config"]["layers"][0]["gated"] = True
    else:
        document["weights"].append(1.0)
    path = tmp_path / "changed.nam"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        NamCapture(str(path), device="cpu")
