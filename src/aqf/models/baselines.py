"""Baselines the deep models have to beat.

A forecasting result without a baseline is not a result. Hourly pollutant
concentrations are strongly autocorrelated and strongly diurnal, which means two
models that require no training at all are already good:

*Persistence* predicts the last observed value. On an hourly series this is hard
to beat at a one-hour horizon, because pollution an hour from now mostly is
pollution now.

*Seasonal naive* predicts the value from the same hour yesterday. It captures the
daily traffic and heating cycle that dominates urban air quality, for free.

Published deep-learning forecasts frequently omit both, and a model that loses to
persistence while reporting a respectable RMSE has demonstrated nothing except
that the series is autocorrelated. Every baseline here is evaluated on exactly
the same windows and the same scale as the neural networks, so the comparison is
like for like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import EvaluationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BaselineResult:
    """A baseline's predictions on the test windows."""

    name: str
    predictions: np.ndarray
    description: str


def _check(X: np.ndarray, target_index: int) -> None:
    if X.ndim != 3:
        raise EvaluationError("X must be (windows, lookback, features)")
    if not 0 <= target_index < X.shape[2]:
        raise EvaluationError(f"target_index {target_index} is outside {X.shape[2]} features")


def persistence(X: np.ndarray, target_index: int) -> BaselineResult:
    """Predict the last observed value of the target in each window."""
    _check(X, target_index)
    return BaselineResult(
        name="persistence",
        predictions=X[:, -1, target_index].astype(np.float64),
        description="the target's most recent observed value; no training",
    )


def seasonal_naive(X: np.ndarray, target_index: int, *, period: int = 24) -> BaselineResult:
    """Predict the value from one period earlier, by default the same hour yesterday.

    Falls back to the oldest value in the window when the lookback is shorter
    than the period, rather than failing: a 12-hour lookback cannot see
    yesterday, and saying so is more useful than refusing to run.
    """
    _check(X, target_index)
    lookback = X.shape[1]
    if period < 1:
        raise EvaluationError("period must be at least 1")
    index = lookback - period if lookback >= period else 0
    note = (
        f"the value {period} hours earlier"
        if lookback >= period
        else f"the oldest value in a {lookback}-hour window (shorter than the {period}-hour period)"
    )
    return BaselineResult(
        name=f"seasonal naive ({period}h)",
        predictions=X[:, index, target_index].astype(np.float64),
        description=f"{note}; no training",
    )


def window_mean(X: np.ndarray, target_index: int) -> BaselineResult:
    """Predict the mean of the target across the window."""
    _check(X, target_index)
    return BaselineResult(
        name="window mean",
        predictions=X[:, :, target_index].mean(axis=1).astype(np.float64),
        description="the average of the target over the lookback window; no training",
    )


def linear_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 1.0,
) -> BaselineResult:
    """Ridge regression on the flattened window.

    The honest middle ground between doing nothing and a neural network: it sees
    exactly the same inputs, with no architecture and no training curve. If a
    recurrent model cannot beat a linear map of the same window, the sequence
    structure is not carrying the signal.
    """
    from sklearn.linear_model import Ridge

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise EvaluationError("inputs must be (windows, lookback, features)")

    flat_train = X_train.reshape(len(X_train), -1)
    flat_test = X_test.reshape(len(X_test), -1)
    model = Ridge(alpha=alpha).fit(flat_train, y_train)
    logger.info("linear_baseline_fitted", features=flat_train.shape[1], alpha=alpha)
    return BaselineResult(
        name="ridge regression",
        predictions=np.asarray(model.predict(flat_test), dtype=np.float64),
        description=f"linear map of the flattened window, alpha={alpha}",
    )
