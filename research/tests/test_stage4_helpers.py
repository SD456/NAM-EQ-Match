"""Raw E0 null, nontrivial EQ acceptance and drive-identifiability invariants."""
import numpy as np
import pytest
from scipy.signal import freqz

from scripts.stage4_helpers import decide_eq, fit_null, profile_gain_set, raw_correction
from tonematch.features import log_freq_grid


def test_raw_correction_uses_shared_probe_median_and_removes_output_level():
    grid = np.array([40, 80, 160, 1000, 8000, 12000.0])
    probes = np.array([[0, 2, 4, 6, 8, 10], [2, 0, 6, 4, 10, 8],
                       [1, 1, 5, 5, 9, 9]], dtype=float)
    reference = np.array([100, 8, 11, 15, 20, 200.0])
    actual = raw_correction(probes, reference, grid)
    np.testing.assert_array_equal(actual, [90.5, -1.5, -2.5, 1.5, 2.5, 182.5])
    np.testing.assert_allclose(raw_correction(probes + 19, reference - 31, grid), actual)
    assert abs(actual[-1]) > 100  # The raw curve has not been limited for FIR design.


def test_null_is_empirical_maximum_over_in_band_raw_curves():
    grid = np.array([40, 80, 160, 1000, 8000, 12000.0])
    curves = np.array([[1000, 1, -2, 3, 4, 1000], [999, 2, -7, 1, 0, 999]])
    assert fit_null(curves, grid) == 7


@pytest.mark.parametrize("values,q", [
    ([0, 0], 0), ([1, 1], 0), ([2, 2], 2), ([2, -2], 0),
    ([2, 6], 0), ([10, 30], 0), ([0.9, 3], 0),
])
def test_rejected_raw_curves_return_exact_identity(values, q):
    grid = log_freq_grid(48000)
    curves = np.asarray(values)[:, None] * np.ones(len(grid))
    decision = decide_eq(curves, grid, 48000, q)
    assert decision["apply"] is False
    assert decision["supported_bins"] == 0
    assert decision["reason"] == "no_supported_bins"
    np.testing.assert_array_equal(decision["ir"], np.array([1], dtype=np.float32))
    np.testing.assert_array_equal(decision["actual_db"], np.zeros(len(grid)))


def test_out_of_band_differences_cannot_enable_eq():
    grid = log_freq_grid(48000)
    curves = np.zeros((2, len(grid)))
    curves[:, (grid < 80) | (grid > 8000)] = 100
    assert decide_eq(curves, grid, 48000, 0)["apply"] is False


def test_positive_eq_is_nontrivial_and_actual_float32_fir_respects_cap():
    grid = log_freq_grid(48000)
    shape = 12 * np.tanh(np.log2(grid / 1200))
    curves = np.stack([shape, shape + 0.1])
    decision = decide_eq(curves, grid, 48000, 1.5)
    assert decision["apply"] is True and decision["supported_bins"] > 50
    assert decision["q"] == 1.5
    assert decision["ir"].dtype == np.float32
    assert len(decision["ir"]) == 2048
    _, response = freqz(decision["ir"], worN=2 ** 19, include_nyquist=True)
    actual_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
    assert np.max(np.abs(actual_db)) <= 4.5
    assert np.max(np.abs(actual_db)) > 1
    assert np.max(np.abs(decision["actual_db"])) > 1


def test_gain_profile_keeps_disconnected_sets_and_has_correct_level_sign():
    device = np.arange(-12, 13, dtype=float)
    np.testing.assert_array_equal(profile_gain_set([-8, 8], device, (-1, 1)),
                                  [-9, -8, -7, 7, 8, 9])
    np.testing.assert_array_equal(profile_gain_set([0], device, (2, 4)), [-4, -3, -2])
    np.testing.assert_array_equal(profile_gain_set([0], device), np.arange(-6, 7))
    assert profile_gain_set([], device).size == 0


@pytest.mark.parametrize("call", [
    lambda: raw_correction(np.zeros((2, 3)), np.zeros(3), [100, 200, 400]),
    lambda: raw_correction(np.zeros((3, 3)), [0, np.nan, 0], [100, 200, 400]),
    lambda: fit_null(np.zeros((0, 3)), [100, 200, 400]),
    lambda: fit_null(np.zeros((1, 3)), [100, 50, 400]),
    lambda: fit_null(np.zeros((1, 3)), [10, 20, 40]),
    lambda: decide_eq(np.zeros((1, 3)), [100, 200, 400], 48000, 0),
    lambda: decide_eq(np.zeros((2, 3)), [100, 200, 500], 48000, 0),
    lambda: decide_eq(np.zeros((2, 3)), [100, 200, 400], 48000, np.nan),
    lambda: decide_eq(np.zeros((2, 3)), [100, 200, 400], 48000, -1),
    lambda: decide_eq(np.zeros((2, 3)), [100, 200, 400], 48000, 0, n_taps=3),
    lambda: profile_gain_set([0], [0], (2, -2)),
    lambda: profile_gain_set([np.inf], [0]),
])
def test_invalid_inputs_fail_at_helper_boundaries(call):
    with pytest.raises(ValueError):
        call()
