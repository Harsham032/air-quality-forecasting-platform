"""Turning a series into supervised windows, and splitting it in time.

Two decisions here decide whether the reported error means anything.

**The split is chronological, never random.** Shuffling an hourly series before
splitting puts hour 5,000 in training and hour 4,999 in test. The model then
interpolates between neighbours it has already seen, which is not forecasting,
and the reported error can be an order of magnitude too good. This is the most
common way a time-series result becomes unreachable in production.

**The scaler is fitted on training data only.** Fitting it on the whole series
lets the test period's minimum and maximum leak backwards into training - a
subtle leak that inflates results without ever looking wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DataQualityError, EvaluationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WindowSpec:
    """How the series is cut into examples."""

    lookback: int = 24
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise EvaluationError("lookback must be at least 1 hour")
        if self.horizon < 1:
            raise EvaluationError("horizon must be at least 1 hour")


@dataclass
class TemporalSplit:
    """A chronological train/validation/test division, in original units."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target: str
    # The training-set mean and scale of the target, kept so predictions can be
    # returned to micrograms per cubic metre rather than reported on a unitless
    # axis nobody can interpret.
    target_mean: float
    target_scale: float
    timestamps_test: pd.DatetimeIndex

    def describe(self) -> dict[str, float | str]:
        return {
            "target": self.target,
            "features": float(len(self.feature_names)),
            "lookback": float(self.X_train.shape[1]),
            "train_windows": float(len(self.X_train)),
            "validation_windows": float(len(self.X_validation)),
            "test_windows": float(len(self.X_test)),
            "target_mean_train": self.target_mean,
            "target_scale_train": self.target_scale,
        }

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """Return scaled predictions to the target's own units."""
        return np.asarray(scaled, dtype=np.float64) * self.target_scale + self.target_mean


def make_windows(
    values: np.ndarray, target: np.ndarray, spec: WindowSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(n_windows, lookback, n_features)`` inputs and their targets.

    Window ``i`` spans rows ``[i, i + lookback)`` and predicts the target
    ``horizon`` steps past the end of it. Nothing inside a window may come from
    after the value it predicts.
    """
    if values.ndim != 2:
        raise EvaluationError("values must be two-dimensional (time x features)")
    if len(values) != len(target):
        raise EvaluationError("values and target must have the same length")

    n = len(values) - spec.lookback - spec.horizon + 1
    if n <= 0:
        raise DataQualityError(
            f"a lookback of {spec.lookback} and horizon of {spec.horizon} need at least "
            f"{spec.lookback + spec.horizon} rows; {len(values)} are available"
        )

    # Strided view rather than a Python loop: the windows overlap heavily, so
    # materialising each one separately copies the series `lookback` times.
    windows = np.lib.stride_tricks.sliding_window_view(values, spec.lookback, axis=0)
    X = np.ascontiguousarray(windows[:n].transpose(0, 2, 1))
    y = target[spec.lookback + spec.horizon - 1 : spec.lookback + spec.horizon - 1 + n]
    return X.astype(np.float32), np.asarray(y, dtype=np.float32)


def split_by_time(
    frame: pd.DataFrame,
    *,
    target: str,
    spec: WindowSpec,
    train_share: float = 0.7,
    validation_share: float = 0.15,
    features: list[str] | tuple[str, ...] | None = None,
) -> TemporalSplit:
    """Split chronologically, scale on training data, and window each part.

    Windows are built *within* each part rather than across the whole series, so
    no window straddles a boundary. Building them first and splitting afterwards
    would place windows that contain training hours into the test set.
    """
    if target not in frame.columns:
        raise EvaluationError(f"no target column {target!r}")
    if not 0.0 < train_share < 1.0 or not 0.0 < validation_share < 1.0:
        raise EvaluationError("shares must lie strictly between 0 and 1")
    if train_share + validation_share >= 1.0:
        raise EvaluationError("train and validation shares leave nothing for test")

    columns = list(features) if features is not None else list(frame.columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise EvaluationError(f"missing feature columns: {', '.join(missing)}")
    if target not in columns:
        columns = [*columns, target]

    data = frame[columns].astype(np.float64)
    n = len(data)
    train_end = int(n * train_share)
    validation_end = int(n * (train_share + validation_share))
    if (
        min(train_end, validation_end - train_end, n - validation_end)
        <= spec.lookback + spec.horizon
    ):
        raise DataQualityError(
            "one of the three chronological parts is shorter than a single window; "
            "use a shorter lookback or a longer series"
        )

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    # Standardise on training statistics only. Anything else leaks the future.
    mean = train.mean()
    scale = train.std(ddof=0).replace(0.0, 1.0)
    target_index = columns.index(target)

    def prepare(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaled = ((part - mean) / scale).to_numpy(dtype=np.float64)
        return make_windows(scaled, scaled[:, target_index], spec)

    X_train, y_train = prepare(train)
    X_validation, y_validation = prepare(validation)
    X_test, y_test = prepare(test)

    offset = spec.lookback + spec.horizon - 1
    timestamps_test = pd.DatetimeIndex(test.index[offset : offset + len(y_test)])

    split = TemporalSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_names=tuple(columns),
        target=target,
        target_mean=float(mean[target]),
        target_scale=float(scale[target]),
        timestamps_test=timestamps_test,
    )
    logger.info(
        "split_built",
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in split.describe().items()},
    )
    return split
