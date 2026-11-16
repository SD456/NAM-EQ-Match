"""Stage-3 invariants: shared gain, level invariance and actual exported EQ cap."""
import numpy as np
import pytest
import soundfile as sf

from scripts.multidi_scoring import (
    aggregate_scores, centered_spectral_distance, half_octave_ltas, make_consensus_ir,
)


def test_aggregation_preserves_shared_gain_and_separates_factors():
    # Independent probe minima would manufacture a score of zero; no shared gain has it.
    current = np.array([[[0, 8, 8], [4, 4, 4]],
                        [[8, 0, 8], [4, 4, 4]],
                        [[8, 8, 0], [4, 4, 4]]], dtype=float)
    scores = aggregate_scores(current, current + 10)
    np.testing.assert_array_equal(scores["current_single"], current[0])
    np.testing.assert_array_equal(scores["spectrum_single"], current[0] + 10)
    np.testing.assert_array_equal(scores["current_multi"], [[8, 8, 8], [4, 4, 4]])
    np.testing.assert_array_equal(scores["spectrum_multi"], scores["current_multi"] + 10)
    assert np.unravel_index(np.argmin(scores["current_multi"]), (2, 3))[0] == 1
    scores["current_single"][0, 0] = -1
    assert current[0, 0, 0] == 0


@pytest.mark.parametrize("current,spectrum", [
    (np.zeros((2, 1, 1)), np.zeros((2, 1, 1))),
    (np.zeros((3, 1, 2)), np.zeros((3, 2, 1))),
    (np.zeros((3, 0, 1)), np.zeros((3, 0, 1))),
    (np.full((3, 1, 1), np.nan), np.zeros((3, 1, 1))),
    (np.zeros((3, 1, 1)), np.full((3, 1, 1), np.inf)),
])
def test_score_grid_rejects_invalid_inputs(current, spectrum):
    with pytest.raises(ValueError):
        aggregate_scores(current, spectrum)


def test_spectral_distance_ignores_level_and_out_of_band_energy():
    grid = np.array([40, 80, 160, 1000, 8000, 12000.0])
    target = np.array([100, -10, -5, 0, -15, 100.0])
    curves = np.stack([target + 20, target + np.array([300, 0, 4, 0, -4, -300])])
    np.testing.assert_allclose(centered_spectral_distance(curves, target, grid), [0, 2])
    assert centered_spectral_distance(target, target, grid) == 0
    with pytest.raises(ValueError):
        centered_spectral_distance(target, target, grid[::-1])
    with pytest.raises(ValueError):
        centered_spectral_distance(target[:-1], target, grid)


def test_half_octave_ltas_is_direct_smoothed_and_level_invariant():
    from tonematch.audio import EPS
    from tonematch.features import ltas, smooth_spectrum_db

    rng = np.random.default_rng(31)
    audio = rng.normal(size=48000) * 0.1
    grid, curve = half_octave_ltas(audio, 48000)
    grid2, louder = half_octave_ltas(audio * 7, 48000)
    np.testing.assert_array_equal(grid, grid2)
    np.testing.assert_allclose(curve, louder, atol=1e-8)
    f, p = ltas(audio / (np.max(np.abs(audio)) + EPS), 48000)
    expected = smooth_spectrum_db(f, p, grid, octave_fraction=2)
    band = (grid >= 80) & (grid <= 8000)
    np.testing.assert_allclose(curve, expected - np.median(expected[band]))


def test_consensus_guard_rejects_weak_or_unshared_corrections():
    from tonematch.features import log_freq_grid

    grid = log_freq_grid(48000)
    weak = np.tile(np.full(len(grid), 0.9), (3, 1))
    ir, _, db = make_consensus_ir(weak, grid, 48000)
    np.testing.assert_allclose(ir, np.r_[1, np.zeros(len(ir) - 1)], atol=1e-7)
    np.testing.assert_allclose(db, 0, atol=1e-7)
    _, _, unguarded = make_consensus_ir(weak, grid, 48000, guard=False)
    assert np.median(unguarded[(grid > 500) & (grid < 3000)]) > 0.8
    for values in ([3, 3, -3], [1, 3, 5]):
        inconsistent = np.array(values)[:, None] * np.ones(len(grid))
        _, _, rejected = make_consensus_ir(inconsistent, grid, 48000)
        np.testing.assert_allclose(rejected, 0, atol=1e-7)
    supported = np.array([2, 3, 4])[:, None] * np.ones(len(grid))
    _, _, accepted = make_consensus_ir(supported, grid, 48000)
    assert np.median(accepted[(grid > 500) & (grid < 3000)]) > 2.8


@pytest.mark.parametrize("guard", [False, True])
def test_consensus_actual_float_export_is_bounded_and_outlier_robust(tmp_path, guard):
    from tonematch.features import log_freq_grid

    grid = log_freq_grid(48000)
    shape = 12 * np.sin(np.log2(grid / 500))
    curves = np.tile(shape, (3, 1))
    ir, actual_grid, response = make_consensus_ir(curves, grid, 48000, guard=guard)
    np.testing.assert_array_equal(actual_grid, grid)
    assert ir.dtype == np.float32 and np.isfinite(ir).all()
    assert np.max(np.abs(response)) <= 4.5
    path = tmp_path / "consensus.wav"
    sf.write(path, ir, 48000, subtype="FLOAT")
    exported, sr = sf.read(path, dtype="float32")
    assert sr == 48000 and sf.info(path).subtype == "FLOAT"
    np.testing.assert_array_equal(exported, ir)
    # Check a denser, independently sampled response after the actual WAV round trip.
    response_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(exported, 2 ** 20)), 1e-12))
    assert np.max(np.abs(response_db)) <= 4.5
    assert np.max(np.abs(response_db)) > 1
    _, _, outlier_response = make_consensus_ir(np.stack([shape, shape, -shape * 10]),
                                              grid, 48000, guard=guard)
    if guard:
        np.testing.assert_allclose(outlier_response, 0, atol=1e-7)
    else:
        np.testing.assert_array_equal(outlier_response, response)


@pytest.mark.parametrize("curves,grid,kwargs", [
    (np.zeros((2, 3)), [100, 200, 400], {}),
    (np.zeros((3, 2)), [100, 200, 400], {}),
    (np.full((3, 3), np.inf), [100, 200, 400], {}),
    (np.zeros((3, 3)), [100, 200, 500], {}),
    (np.zeros((3, 3)), [100, 200, 400], {"max_gain_db": -1}),
    (np.zeros((3, 3)), [100, 200, 400], {"max_spread_db": np.nan}),
    (np.zeros((3, 3)), [100, 200, 400], {"n_taps": 3}),
])
def test_consensus_rejects_invalid_inputs(curves, grid, kwargs):
    with pytest.raises(ValueError):
        make_consensus_ir(curves, grid, 48000, **kwargs)


def test_guard_and_median_differ_only_by_mask_selection():
    from tonematch.features import log_freq_grid

    grid = log_freq_grid(48000)
    # All bins pass the frozen guard, so the identical smoothing must give the same FIR.
    curve = 2 + 0.5 * np.sin(np.log2(grid / 500) * 8)
    probes = np.stack([curve - 0.1, curve, curve + 0.1])
    guarded, _, _ = make_consensus_ir(probes, grid, 48000, guard=True)
    median, _, _ = make_consensus_ir(probes, grid, 48000, guard=False)
    np.testing.assert_array_equal(guarded, median)
