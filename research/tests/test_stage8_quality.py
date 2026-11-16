"""Compare the optimized table against the historical independent evaluator."""
import numpy as np
import pytest
from scipy.signal import lfilter

from scripts.stage2_benchmark import SR, paired_quality
from scripts.stage8_quality import quality_table
from tonematch.tone_stack import biquad_coeffs


def color(x):
    b, a = biquad_coeffs("highshelf", SR, 1200, 0.7071, 3)
    return lfilter(b, a, np.pad(x, (0, 256))).astype(np.float32)


def test_every_quality_table_entry_matches_historical_paired_quality():
    rng = np.random.default_rng(808)
    t = np.arange(10240) / SR
    clean = (0.14 * np.sin(2 * np.pi * 173 * t)
             + 0.025 * np.sin(2 * np.pi * 1300 * t)).astype(np.float32)
    t = np.arange(12289) / SR
    driven = np.tanh(3 * np.sin(2 * np.pi * 317 * t)) * 0.012
    noise = lfilter([1, -0.85], [1], rng.normal(size=14017)) * 0.7
    # A candidate already equal to an E1 reference exercises the direct path.
    outputs = [[clean, driven, noise][i % 3] * 10 ** ((i - 12) / 20)
               for i in range(24)]
    outputs[2] = color(outputs[0])
    teachers = [0, 1, 12, 13, 6, 7, 20, 21]
    originals = [x.copy() for x in outputs]
    table = quality_table(outputs, teachers, color)
    assert table["sc"].shape == table["ltas"].shape == (8, 2, 24)
    for ti, teacher in enumerate(teachers):
        for eq in (0, 1):
            reference = color(outputs[teacher]) if eq else outputs[teacher]
            for ci, candidate in enumerate(outputs):
                expected = paired_quality(candidate, reference)
                np.testing.assert_allclose(table["sc"][ti, eq, ci], expected["mrstft_sc"], rtol=0, atol=1e-7)
                np.testing.assert_allclose(table["ltas"][ti, eq, ci], expected["ltas_mae_db"], rtol=0, atol=1e-7)
    assert table["sc"][0, 1, 2] == 0
    assert table["ltas"][0, 1, 2] == 0
    for actual, original in zip(outputs, originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("bad", [[], np.zeros(8192), np.array([np.nan]), np.ones((2, 8192))])
def test_quality_table_rejects_invalid_audio(bad):
    with pytest.raises(ValueError, match="finite, non-silent mono"):
        quality_table([bad], [0], color)


@pytest.mark.parametrize("teachers", [[], [-1], [1], [True], [0.5]])
def test_quality_table_rejects_invalid_teacher_indices(teachers):
    with pytest.raises(ValueError, match="valid teacher indices"):
        quality_table([np.ones(8192)], teachers, color)
