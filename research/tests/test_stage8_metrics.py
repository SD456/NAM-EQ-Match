"""Numerical invariants of fold-local ERB/shared-covariance retrieval."""
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf, OAS

from scripts.stage8_metrics import HELMERT_BASIS, erb_feature, fit_model, score_model


def test_fixed_helmert_coordinates_remove_only_constant_level():
    np.testing.assert_allclose(HELMERT_BASIS.T @ HELMERT_BASIS, np.eye(23), atol=1e-14)
    np.testing.assert_allclose(np.ones(24) @ HELMERT_BASIS, 0, atol=1e-14)
    bands = np.arange(24, dtype=float) ** 2
    projected = bands @ HELMERT_BASIS
    np.testing.assert_allclose(projected @ HELMERT_BASIS.T, bands - bands.mean(), atol=1e-12)
    assert np.linalg.norm(projected) > 0


def test_whole_file_erb_features_ignore_output_volume_but_keep_timbre():
    rng = np.random.default_rng(42)
    audio = rng.normal(size=48000).astype(np.float32)
    feature = erb_feature(audio)
    assert feature.shape == (23,) and feature.dtype == np.float64
    np.testing.assert_allclose(erb_feature(audio * 0.001), feature, atol=1e-6)
    np.testing.assert_allclose(erb_feature(audio.astype(float) * -20), feature, atol=1e-11)
    lowpass = np.convolve(audio, np.ones(25) / 25, mode="full")
    assert np.linalg.norm(erb_feature(lowpass) - feature) > 10


@pytest.mark.parametrize("method", ["euclidean", "diagonal", "ledoit_wolf", "oas"])
def test_covariance_excludes_between_chain_and_gain_centers(method):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3, 2, 40, 23))
    offsets = rng.normal(size=(3, 2, 1, 23)) * 100
    original, moved = fit_model(X, method), fit_model(X + offsets, method)
    np.testing.assert_allclose(moved["centers"], original["centers"] + offsets[:, :, 0])
    np.testing.assert_allclose(moved["covariance"], original["covariance"], atol=1e-12)
    query = rng.normal(size=23)
    assert score_model(query, original).shape == (3, 2)
    assert np.isfinite(score_model(query, original)).all()


@pytest.mark.parametrize("method, estimator", [("ledoit_wolf", LedoitWolf), ("oas", OAS)])
def test_shrink_each_chain_before_equal_weight_averaging(method, estimator):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(3, 2, 9, 23))
    X[0, :, :, :3] *= 30
    model = fit_model(X, method)
    for gain in range(2):
        chain_fits = [estimator().fit(chain[gain]) for chain in X]
        expected = np.mean([fit.covariance_ for fit in chain_fits], axis=0)
        np.testing.assert_allclose(model["covariance"][gain], expected, atol=1e-11)
        np.testing.assert_allclose(model["diagnostics"][gain]["shrinkage_per_chain"],
                                   [fit.shrinkage_ for fit in chain_fits])
        pooled = estimator().fit((X[:, gain] - X[:, gain].mean(axis=1, keepdims=True)).reshape(-1, 23))
        assert not np.allclose(expected, pooled.covariance_)


def test_diagonal_is_shared_within_chain_ml_variance_and_euclidean_is_identity():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(2, 2, 35, 23)) * np.arange(1, 24)
    diagonal = fit_model(X, "diagonal")
    np.testing.assert_allclose(diagonal["covariance"],
                               [np.diag(v) for v in X.var(axis=2).mean(axis=0)])
    euclidean = fit_model(X, "euclidean")
    query = rng.normal(size=23)
    np.testing.assert_allclose(score_model(query, euclidean),
                               np.sum((query - X.mean(axis=2)) ** 2, axis=-1))
    np.testing.assert_array_equal(euclidean["logdet"], [0, 0])


def test_cross_gain_scoring_includes_logdet_in_same_23_coordinates():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(50, 23))
    base -= base.mean(axis=0)
    X = np.stack([base, base * 3])[None]
    model = fit_model(X, "diagonal")
    scores = score_model(np.zeros(23), model)[0]
    np.testing.assert_allclose(scores[1] - scores[0], 23 * np.log(9))
    assert scores[0] < scores[1]
    query = np.ones(23)
    for gain in range(2):
        delta = query - model["centers"][0, gain]
        expected = delta @ np.linalg.solve(model["covariance"][gain], delta)
        expected += np.linalg.slogdet(model["covariance"][gain])[1]
        np.testing.assert_allclose(score_model(query, model)[0, gain], expected)


@pytest.mark.parametrize("method", ["euclidean", "diagonal", "ledoit_wolf", "oas"])
def test_degenerate_training_is_positive_definite_and_reports_floor(method):
    model = fit_model(np.full((2, 2, 4, 23), 5, dtype=np.float32), method)
    for covariance, diagnostic in zip(model["covariance"], model["diagnostics"]):
        np.linalg.cholesky(covariance)
        assert diagnostic["condition"] == 1
        assert diagnostic["floored_eigenvalues"] == (0 if method == "euclidean" else 23)
        assert np.isfinite(diagnostic["logdet"])
    assert np.isfinite(score_model(np.ones(23), model)).all()


def test_fit_does_not_mutate_or_retain_caller_training_array():
    X = np.random.default_rng(10).normal(size=(2, 2, 20, 23))
    before = X.copy()
    model = fit_model(X, "ledoit_wolf")
    query = np.ones(23)
    scores = score_model(query, model)
    np.testing.assert_array_equal(X, before)
    X[:] = 1e9
    np.testing.assert_array_equal(score_model(query, model), scores)


@pytest.mark.parametrize("audio,sr", [([], 48000), ([0] * 8192, 48000),
    ([[1, 2]], 48000), ([1, np.nan], 48000), ([1, np.inf], 48000),
    ([1, 2], 48000), (np.ones(8192), 48000), ([1j], 48000),
    (np.ones(8192), 8000), (np.ones(8192), True), (np.ones(8192), 48000.0)])
def test_invalid_audio_rejected(audio, sr):
    with pytest.raises(ValueError):
        erb_feature(audio, sr)


@pytest.mark.parametrize("X,method", [(np.zeros((2, 2, 3, 24)), "diagonal"),
    (np.zeros((2, 2, 1, 23)), "oas"), (np.zeros((0, 2, 3, 23)), "oas"),
    (np.zeros((2, 0, 3, 23)), "oas"), (np.zeros((2, 3, 23)), "oas"),
    (np.full((2, 2, 3, 23), np.nan), "oas"),
    (np.ones((2, 2, 3, 23), dtype=complex), "oas"),
    (np.zeros((2, 2, 3, 23)), "unknown")])
def test_invalid_fit_inputs_rejected(X, method):
    with pytest.raises(ValueError):
        fit_model(X, method)


def test_invalid_query_and_model_shapes_rejected():
    model = fit_model(np.zeros((1, 2, 4, 23)), "euclidean")
    for query in [np.zeros(24), np.zeros((1, 23)), np.full(23, np.inf), np.ones(23) * 1j]:
        with pytest.raises(ValueError):
            score_model(query, model)
    with pytest.raises(ValueError):
        score_model(np.zeros(23), {**model, "logdet": np.zeros(3)})
