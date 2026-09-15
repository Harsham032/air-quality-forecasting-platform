"""Baselines must be exactly what they claim, or the comparison means nothing."""

from __future__ import annotations

import numpy as np
import pytest

from aqf.errors import EvaluationError
from aqf.models.baselines import linear_baseline, persistence, seasonal_naive, window_mean


@pytest.fixture
def windows() -> np.ndarray:
    """Ten windows of 24 hours and 3 features, with distinguishable values."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(10, 24, 3)).astype(np.float32)


def test_persistence_is_the_last_observed_value(windows: np.ndarray) -> None:
    result = persistence(windows, target_index=1)
    np.testing.assert_allclose(result.predictions, windows[:, -1, 1], rtol=1e-6)


def test_seasonal_naive_reaches_back_a_full_period(windows: np.ndarray) -> None:
    result = seasonal_naive(windows, target_index=1, period=24)
    np.testing.assert_allclose(result.predictions, windows[:, 0, 1], rtol=1e-6)

    shorter = seasonal_naive(windows, target_index=1, period=6)
    np.testing.assert_allclose(shorter.predictions, windows[:, 18, 1], rtol=1e-6)


def test_seasonal_naive_says_so_when_the_window_is_too_short(windows: np.ndarray) -> None:
    result = seasonal_naive(windows, target_index=1, period=48)
    np.testing.assert_allclose(result.predictions, windows[:, 0, 1], rtol=1e-6)
    assert "shorter than" in result.description


def test_window_mean_averages_only_the_target(windows: np.ndarray) -> None:
    result = window_mean(windows, target_index=2)
    np.testing.assert_allclose(result.predictions, windows[:, :, 2].mean(axis=1), rtol=1e-5)


def test_ridge_sees_the_same_window_the_networks_do(windows: np.ndarray) -> None:
    y = windows[:, -1, 0] + 0.1
    result = linear_baseline(windows, y, windows)
    assert result.predictions.shape == (len(windows),)
    assert np.isfinite(result.predictions).all()


def test_ridge_recovers_a_linear_relationship() -> None:
    """If the target is a linear function of the window, ridge should find it."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 12, 2))
    y = 2.0 * X[:, -1, 0] - 0.5 * X[:, -2, 1]
    result = linear_baseline(X[:300], y[:300], X[300:], alpha=1e-4)
    assert np.corrcoef(result.predictions, y[300:])[0, 1] > 0.98


def test_a_target_index_outside_the_features_is_refused(windows: np.ndarray) -> None:
    with pytest.raises(EvaluationError, match="outside"):
        persistence(windows, target_index=9)


def test_two_dimensional_input_is_refused() -> None:
    with pytest.raises(EvaluationError, match="windows, lookback, features"):
        persistence(np.zeros((10, 24)), target_index=0)
