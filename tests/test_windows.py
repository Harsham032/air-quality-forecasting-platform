"""Leakage tests.

A time-series split that shuffles, or a scaler fitted on everything, produces
excellent test metrics and a worthless model. These tests assert the absence of
both, which is harder than asserting a number and more important.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqf.data.windows import WindowSpec, make_windows, split_by_time
from aqf.errors import DataQualityError, EvaluationError


def test_a_window_never_contains_the_value_it_predicts() -> None:
    values = np.arange(100, dtype=np.float64).reshape(-1, 1)
    X, y = make_windows(values, values[:, 0], WindowSpec(lookback=5, horizon=1))
    # Window 0 spans rows 0-4 and predicts row 5.
    assert X[0].ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert y[0] == 5.0
    for i in range(len(X)):
        assert y[i] > X[i].max()


def test_a_longer_horizon_pushes_the_target_further_out() -> None:
    values = np.arange(100, dtype=np.float64).reshape(-1, 1)
    _, near = make_windows(values, values[:, 0], WindowSpec(lookback=5, horizon=1))
    _, far = make_windows(values, values[:, 0], WindowSpec(lookback=5, horizon=6))
    assert near[0] == 5.0
    assert far[0] == 10.0


def test_the_split_is_chronological(hourly_series: pd.DataFrame) -> None:
    split = split_by_time(hourly_series, target="NO2(GT)", spec=WindowSpec(lookback=24, horizon=1))
    assert split.timestamps_test.is_monotonic_increasing
    # The test period starts after the training period ends.
    cutoff = hourly_series.index[int(len(hourly_series) * 0.85)]
    assert split.timestamps_test.min() >= cutoff


def test_the_scaler_is_fitted_on_training_data_only(hourly_series: pd.DataFrame) -> None:
    """The statistics must not move when the future changes.

    Doubling every value after the training cut cannot alter the training
    scaler. If it does, the scaler saw the test set.
    """
    spec = WindowSpec(lookback=24, horizon=1)
    baseline = split_by_time(hourly_series, target="NO2(GT)", spec=spec)

    altered = hourly_series.copy()
    cut = int(len(altered) * 0.70)
    altered.iloc[cut:, :] = altered.iloc[cut:, :] * 2.0
    shifted = split_by_time(altered, target="NO2(GT)", spec=spec)

    assert shifted.target_mean == pytest.approx(baseline.target_mean)
    assert shifted.target_scale == pytest.approx(baseline.target_scale)
    # And the training windows themselves are untouched.
    np.testing.assert_allclose(shifted.X_train, baseline.X_train, rtol=1e-6)


def test_no_window_straddles_a_split_boundary(hourly_series: pd.DataFrame) -> None:
    """Windowing before splitting would put training hours inside test windows.

    Counted rather than argued: windows built within each part are strictly
    fewer than windows built across the whole series, by exactly the number the
    two boundaries would have produced.
    """
    spec = WindowSpec(lookback=24, horizon=1)
    split = split_by_time(hourly_series, target="NO2(GT)", spec=spec)

    total = len(split.X_train) + len(split.X_validation) + len(split.X_test)
    across_the_whole_series = len(hourly_series) - spec.lookback - spec.horizon + 1
    assert total == across_the_whole_series - 2 * (spec.lookback + spec.horizon - 1)


def test_inverting_the_scaling_returns_physical_units(hourly_series: pd.DataFrame) -> None:
    split = split_by_time(hourly_series, target="NO2(GT)", spec=WindowSpec())
    restored = split.inverse_target(split.y_test)
    # Back in the range the raw series actually occupies.
    assert restored.min() > hourly_series["NO2(GT)"].min() - 1.0
    assert restored.max() < hourly_series["NO2(GT)"].max() + 1.0
    assert restored.mean() == pytest.approx(hourly_series["NO2(GT)"].mean(), abs=25.0)


def test_scaled_targets_are_not_in_physical_units(hourly_series: pd.DataFrame) -> None:
    """Sanity check on the thing the metric guard is looking for."""
    split = split_by_time(hourly_series, target="NO2(GT)", spec=WindowSpec())
    assert abs(float(split.y_train.mean())) < 1.0
    assert float(split.y_test.mean()) != pytest.approx(float(hourly_series["NO2(GT)"].mean()))


def test_a_series_too_short_to_window_is_refused(hourly_series: pd.DataFrame) -> None:
    with pytest.raises(DataQualityError, match="shorter than a single window"):
        split_by_time(hourly_series.iloc[:100], target="NO2(GT)", spec=WindowSpec(lookback=48))


def test_shares_that_leave_no_test_set_are_refused(hourly_series: pd.DataFrame) -> None:
    with pytest.raises(EvaluationError, match="leave nothing for test"):
        split_by_time(
            hourly_series,
            target="NO2(GT)",
            spec=WindowSpec(),
            train_share=0.8,
            validation_share=0.3,
        )


def test_an_unknown_target_is_refused(hourly_series: pd.DataFrame) -> None:
    with pytest.raises(EvaluationError, match="no target column"):
        split_by_time(hourly_series, target="PM10", spec=WindowSpec())


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(EvaluationError, match="same length"):
        make_windows(np.zeros((50, 2)), np.zeros(40), WindowSpec())
