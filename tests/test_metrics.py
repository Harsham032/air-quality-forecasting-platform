"""Metric tests, including the guard this project is named for."""

from __future__ import annotations

import numpy as np
import pytest

from aqf.errors import EvaluationError
from aqf.evaluation.metrics import evaluate_forecast, skill_score


def test_a_perfect_forecast_scores_perfectly() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate_forecast(truth, truth, name="oracle")
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.r2 == 1.0
    assert metrics.bias == 0.0
    assert metrics.beats_the_mean


def test_predicting_the_mean_scores_zero_r_squared() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate_forecast(truth, np.full_like(truth, truth.mean()))
    assert metrics.r2 == pytest.approx(0.0)
    assert not metrics.beats_the_mean


def test_negative_r_squared_means_worse_than_the_mean() -> None:
    """The number the original analysis reported, and what it signifies.

    A negative R-squared is not a weak model. It is a model beaten by a constant,
    which on a series with an obvious daily cycle means something upstream is
    broken.
    """
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate_forecast(truth, np.array([40.0, 30.0, 20.0, 10.0]))
    assert metrics.r2 < 0.0
    assert not metrics.beats_the_mean


def test_bias_separates_a_shifted_forecast_from_a_noisy_one() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    shifted = evaluate_forecast(truth, truth + 5.0)
    noisy = evaluate_forecast(truth, truth + np.array([5.0, -5.0, 5.0, -5.0]))
    assert shifted.bias == pytest.approx(5.0)
    assert noisy.bias == pytest.approx(0.0)
    assert shifted.rmse == pytest.approx(noisy.rmse)


def test_a_scaled_target_is_refused() -> None:
    """The guard. A min-max scaled truth cannot produce an interpretable RMSE."""
    truth = np.linspace(0.05, 0.95, 50)
    with pytest.raises(EvaluationError, match="looks min-max scaled"):
        evaluate_forecast(truth, truth + 0.01)


def test_the_guard_can_be_turned_off_deliberately() -> None:
    truth = np.linspace(0.05, 0.95, 50)
    metrics = evaluate_forecast(truth, truth + 0.01, require_physical_units=False)
    assert metrics.rmse == pytest.approx(0.01)


def test_physical_concentrations_pass_the_guard() -> None:
    truth = np.array([45.0, 110.0, 230.0, 88.0])
    metrics = evaluate_forecast(truth, truth * 1.02, unit="ug/m3")
    assert metrics.unit == "ug/m3"
    assert metrics.rmse > 1.0


def test_mape_is_not_reported_when_it_would_be_meaningless() -> None:
    truth = np.array([0.0, 0.0, 100.0, 200.0])
    metrics = evaluate_forecast(truth, truth + 1.0)
    assert np.isnan(metrics.mape)


def test_non_finite_pairs_are_dropped_not_propagated() -> None:
    truth = np.array([10.0, np.nan, 30.0, 40.0])
    prediction = np.array([10.0, 20.0, np.inf, 40.0])
    metrics = evaluate_forecast(truth, prediction)
    assert metrics.n == 2
    assert np.isfinite(metrics.rmse)


def test_scoring_nothing_is_refused() -> None:
    with pytest.raises(EvaluationError, match="nothing to score"):
        evaluate_forecast(np.array([]), np.array([]))


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(EvaluationError, match="prediction has"):
        evaluate_forecast(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_skill_is_zero_against_itself() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate_forecast(truth, truth + 1.0)
    assert skill_score(metrics, metrics) == pytest.approx(0.0)


def test_skill_is_negative_when_the_model_is_worse() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    good = evaluate_forecast(truth, truth + 1.0, name="good")
    bad = evaluate_forecast(truth, truth + 4.0, name="bad")
    assert skill_score(bad, good) < 0.0
    assert skill_score(good, bad) == pytest.approx(0.75)


def test_skill_against_a_flawless_baseline_is_undefined() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    flawless = evaluate_forecast(truth, truth)
    other = evaluate_forecast(truth, truth + 1.0)
    with pytest.raises(EvaluationError, match="skill score is undefined"):
        skill_score(other, flawless)
