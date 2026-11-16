"""Signal-level regressions; use real WAV I/O and real DSP throughout."""

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import freqz, stft
from scipy.fft import dct

from tonematch.audio import active_segment, load_audio, match_rms, rms, save_audio, short_term_rms
from tonematch.features import EPS, extract_fingerprint, mel_filterbank
from tonematch.match_eq import apply_fir, design_match_ir


@pytest.mark.parametrize("n", [1, 17, 2399, 2400, 2401])
def test_short_rms_uses_available_samples(n):
    x = np.full(n, 0.25)
    assert np.allclose(short_term_rms(x, 48000), 0.25)
    assert np.all(np.isfinite(short_term_rms(np.zeros(n), 48000)))


def test_rms_has_no_artificial_silence_floor():
    assert rms(np.zeros(100)) == 0
    assert np.all(short_term_rms(np.zeros(10000), 48000) == 0)
    assert rms(np.full(100, 1e-10)) == pytest.approx(1e-10, abs=1e-20)
    with pytest.raises(ValueError, match="silent"):
        match_rms(np.zeros(100), 0.1)
    assert np.all(match_rms(np.zeros(100), 0)[0] == 0)
    assert rms(match_rms(np.ones(100), 0.2)[0]) == pytest.approx(0.2)
    for target in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            match_rms(np.ones(100), target)


def test_active_segment_short_duration_and_boundaries():
    x = np.r_[np.ones(2400), np.zeros(48000)]
    assert np.array_equal(active_segment(x, 48000, 0.05), np.ones(2400))
    assert np.array_equal(active_segment(x[:100], 48000, 0.05), x[:100])
    assert len(active_segment(x, 48000, 1e-9)) == 1
    for duration in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            active_segment(x, 48000, duration)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_samples_rejected(bad):
    for operation in (lambda: short_term_rms(bad, 48000),
                      lambda: extract_fingerprint(bad, 48000),
                      lambda: design_match_ir(bad, [1], 48000),
                      lambda: apply_fir([1], bad)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("sr", [0, -1, 48000.5, np.nan, True])
def test_invalid_sample_rates_rejected(sr, tmp_path):
    with pytest.raises(ValueError):
        short_term_rms([1], sr)
    with pytest.raises(ValueError):
        save_audio(tmp_path / "invalid.wav", [1], sr)
    with pytest.raises(ValueError):
        load_audio(tmp_path / "nonexistent.wav", sr)


def test_antiphase_stereo_requires_explicit_policy(tmp_path):
    x = np.sin(np.arange(4096) * 0.13).astype(np.float32)
    path = tmp_path / "antiphase.wav"
    sf.write(path, np.column_stack((x, -x)), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="channel"):
        load_audio(path)
    assert np.array_equal(load_audio(path, channel="left")[0], x)
    assert np.array_equal(load_audio(path, channel="right")[0], -x)
    assert np.array_equal(load_audio(path, channel="mean")[0], np.zeros_like(x))
    resampled, sr = load_audio(path, 24000, channel="left")
    assert sr == 24000 and len(resampled) == len(x) // 2


@pytest.mark.parametrize("x", [np.array([], dtype=float), np.array([np.nan]), np.array([np.inf])])
def test_bad_wav_data_rejected(x, tmp_path):
    path = tmp_path / "bad.wav"
    sf.write(path, x, 48000, subtype="FLOAT")
    with pytest.raises(ValueError):
        load_audio(path)


@pytest.mark.parametrize("n", [1, 100, 2048, 48000])
def test_silent_and_short_fingerprints_finite(n):
    for x in (np.zeros(n), np.sin(np.arange(n) * 0.13)):
        fp = extract_fingerprint(x, 48000)
        for value in vars(fp).values():
            assert np.all(np.isfinite(value))


def test_gaps_do_not_create_false_adjacent_frame_differences():
    sr = 48000
    # Separated steady notes ensure that compacting loud frames creates a false jump.
    t = np.arange(sr) / sr
    x = np.r_[np.sin(2 * np.pi * 330 * t), np.zeros(sr), np.sin(2 * np.pi * 2310 * t)]
    fp = extract_fingerprint(x, sr)
    x /= np.max(np.abs(x)) + EPS
    f, _, z = stft(x, fs=sr, nperseg=2048, noverlap=1536, window="hann")
    power = np.abs(z) ** 2
    e = power.sum(axis=0)
    loud = e > np.percentile(e, 90) * 0.01
    adjacent = loud[:-1] & loud[1:]
    normalized = power / (power.sum(axis=0, keepdims=True) + EPS)
    per_transition = np.linalg.norm(np.diff(normalized, axis=1), axis=0)
    assert fp.flux == pytest.approx(np.mean(per_transition[adjacent]))
    compact_flux = np.mean(np.linalg.norm(np.diff(normalized[:, loud], axis=1), axis=0))
    assert abs(fp.flux - compact_flux) > 1e-4
    fb = mel_filterbank(sr, 2048, 40, 60, 10000)
    mfcc = dct(np.log(fb @ power + EPS), type=2, axis=0, norm="ortho")[1:20]
    expected_delta = np.std(np.diff(mfcc, axis=1)[:, adjacent], axis=1)
    assert np.allclose(fp.mfcc_dstd, expected_delta)


def test_float_ir_roundtrip_preserves_tail_and_over_unity(tmp_path):
    ir = np.array([1.25, -0.4, 1e-7, -1e-9], dtype=np.float32)
    path = tmp_path / "ir.wav"
    save_audio(path, ir, 48000)
    assert sf.info(path).subtype == "FLOAT"
    assert np.array_equal(load_audio(path)[0], ir)


def test_end_impulse_preserves_complete_convolution_tail():
    x = np.r_[np.zeros(99), 1.0]
    ir = np.array([1.0, 0.5, -0.25, 1e-6])
    y = apply_fir(x, ir)
    assert len(y) == len(x) + len(ir) - 1
    assert np.allclose(y[-len(ir):], ir, atol=1e-7)
    with pytest.raises(ValueError):
        apply_fir(x, [0, 0])


@pytest.mark.parametrize("amount", [0, 3, -3, 35, -35])
def test_final_saved_ir_obeys_response_limit(amount, tmp_path):
    sr = 48000
    x = np.random.default_rng(42).normal(0, 0.1, sr * 3)
    # Broadband deterministic tilt, generated directly in the frequency domain.
    frequencies = np.fft.rfftfreq(len(x), 1 / sr)
    tilt_db = amount * np.tanh(np.log2(np.maximum(frequencies, 1) / 1200))
    target = np.fft.irfft(np.fft.rfft(x) * 10 ** (tilt_db / 20), n=len(x))
    ir, grid, corr = design_match_ir(x, target, sr)
    assert len(ir) == 2048
    path = tmp_path / "correction.wav"
    save_audio(path, ir, sr)
    saved, _ = load_audio(path)
    w, response = freqz(saved, worN=131073, include_nyquist=True, fs=sr)
    db = 20 * np.log10(np.abs(response))
    assert np.all(np.isfinite(db))
    assert np.max(np.abs(db)) <= 4.5 + 1e-6
    assert np.allclose(np.interp(grid, w, db), corr, atol=1e-3)
    assert abs(db[0]) < 0.25 and abs(db[-1]) < 0.25
    if amount == 0:
        assert np.max(np.abs(db)) < 1e-5
    else:
        assert np.ptp(db[(w > 100) & (w < 8000)]) > min(abs(amount), 2)
        assert np.sign(np.interp(5000, w, db) - np.interp(200, w, db)) == np.sign(amount)


def test_ir_design_rejects_unsupported_phase_and_bad_limits():
    x = np.ones(8192)
    for options in ({"minimum_phase": False}, {"max_gain_db": np.nan},
                    {"n_taps": 0}, {"octave_fraction": 0}):
        with pytest.raises(ValueError):
            design_match_ir(x, x, 48000, **options)


def test_silent_eq_is_identity_and_zero_limit_is_identity():
    x = np.zeros(1)
    ir, _, correction = design_match_ir(x, x, 48000)
    assert ir[0] == 1 and np.all(ir[1:] == 0)
    assert np.all(correction == 0)
    ir, _, correction = design_match_ir(np.ones(4096), np.zeros(4096), 48000, max_gain_db=0)
    assert ir[0] == 1 and np.all(ir[1:] == 0)
    assert np.all(correction == 0)
