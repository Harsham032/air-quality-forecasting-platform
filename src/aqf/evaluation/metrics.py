"""Forecast metrics, reported in the target's own units.

**Never on the scaled axis.** An RMSE computed on standardised or min-max scaled
values is a number without a unit, and its size depends entirely on the range the
scaler happened to see. That is how an error of 59 micrograms per cubic metre
gets written down as 0.11 and read as excellent. Every function here takes values
already returned to micrograms per cubic metre, and the evaluation helpers refuse
input that looks like it is still scaled.

R-squared is reported alongside RMSE and MAE because it is the one that exposes a
model that has learned nothing. It compares the model against predicting the
mean of the test period, so a negative value means the model is worse than a
horizontal line. RMSE alone never says that out loud.

The skill score does the same job against a chosen baseline rather than the mean,
which is the comparison that actually decides whether a neural network earned its
place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..errors import EvaluationError


@dataclass(frozen=True)
class ForecastMetrics:
    """How well a forecast did, in the target's units."""

    name: str
    rmse: float
    mae: float
    mape: float
    r2: float
    bias: float
    n: int
    unit: str = ""

    @property
    def beats_the_mean(self) -> bool:
        """Whether the model is better than predicting the test-period average."""
        return self.r2 > 0.0

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "bias": self.bias,
            "n": float(self.n),
            "unit": self.unit,
            "beats_the_mean": self.beats_the_mean,
        }


def _prepare(truth: ArrayLike, prediction: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise EvaluationError(f"truth has {y.shape[0]} values, prediction has {p.shape[0]}")
    if y.size == 0:
        raise EvaluationError("nothing to score")
    finite = np.isfinite(y) & np.isfinite(p)
    if not finite.any():
        raise EvaluationError("no finite pairs to score")
    return y[finite], p[finite]


def evaluate_forecast(
    truth: ArrayLike,
    prediction: ArrayLike,
    *,
    name: str = "model",
    unit: str = "ug/m3",
    require_physical_units: bool = True,
) -> ForecastMetrics:
    """Score a forecast, refusing values that still look scaled.

    ``require_physical_units`` guards the mistake this project exists to correct.
    If the truth lies entirely within a unit interval it is almost certainly
    still min-max scaled, and any RMSE computed from it is uninterpretable. The
    check is a heuristic, so it can be turned off - but it has to be turned off
    deliberately.
    """
    y, p = _prepare(truth, prediction)

    if require_physical_units and y.min() >= -1.0 and y.max() <= 1.0 and np.ptp(y) <= 1.0:
        raise EvaluationError(
            f"the target spans [{y.min():.4f}, {y.max():.4f}], which looks min-max scaled "
            "rather than measured. An RMSE on that axis has no unit and cannot be compared "
            "against anything. Invert the scaling first, or pass "
            "require_physical_units=False if the target genuinely lives in [0, 1]."
        )

    error = p - y
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    # MAPE is undefined where the truth is zero and explodes where it is small;
    # computed only over meaningfully non-zero values, and reported as NaN when
    # too few remain to mean anything.
    scale = np.abs(y)
    usable = scale > max(1e-9, 0.01 * float(np.median(scale)))
    mape = (
        float(np.mean(np.abs(error[usable] / y[usable])) * 100.0)
        if usable.sum() >= 10
        else float("nan")
    )

    variance = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / variance) if variance > 0 else float("nan")

    return ForecastMetrics(
        name=name,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        bias=float(np.mean(error)),
        n=int(y.size),
        unit=unit,
    )


def skill_score(model: ForecastMetrics, baseline: ForecastMetrics) -> float:
    """Fractional reduction in RMSE against a baseline.

    1.0 is a perfect forecast, 0.0 is exactly as good as the baseline, and a
    negative value means the model is worse than the thing it was supposed to
    improve on. This is the number that decides whether a neural network earned
    its training time, and it is the number most air quality papers omit.
    """
    if baseline.rmse <= 0:
        raise EvaluationError("the baseline has zero error; a skill score is undefined")
    return float(1.0 - model.rmse / baseline.rmse)
