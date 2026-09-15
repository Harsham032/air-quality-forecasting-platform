"""Figures.

Matplotlib only, no seaborn, no styling that depends on a theme file. Every
axis is labelled with a unit, because an unlabelled y-axis is how a scaled
error gets mistaken for a physical one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import EvaluationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _figure(figsize: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def _save(fig: Any, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("figure_saved", path=str(out))
    return out


def plot_forecasts(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    columns: list[str] | None = None,
    hours: int = 336,
    unit: str = "ug/m3",
    title: str = "Observed and forecast concentration",
) -> Path:
    """Plot a slice of the test period rather than all of it.

    Ten months of hourly data drawn on one axis is a solid block of ink. Two
    weeks shows whether a forecast tracks the daily cycle or merely lags it by
    an hour, which is the difference that matters here.
    """
    if "observed" not in predictions.columns:
        raise EvaluationError("the prediction frame needs an 'observed' column")
    chosen = columns or [c for c in predictions.columns if c != "observed"]
    missing = [c for c in chosen if c not in predictions.columns]
    if missing:
        raise EvaluationError(f"no such forecast columns: {', '.join(missing)}")

    window = predictions.iloc[:hours]
    fig, ax = _figure((11.0, 4.5))
    ax.plot(window.index, window["observed"], color="black", linewidth=1.6, label="observed")
    for column in chosen:
        ax.plot(window.index, window[column], linewidth=1.1, alpha=0.85, label=column)
    ax.set_xlabel("time")
    ax.set_ylabel(f"concentration ({unit})")
    ax.set_title(f"{title} - first {len(window)} hours of the test period")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_residuals(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    column: str,
    unit: str = "ug/m3",
) -> Path:
    """Residuals against the hour of day, plus their distribution.

    Errors that are flat across the day mean something different from errors
    that spike at the morning peak, and an aggregate RMSE hides which one you
    have.
    """
    if column not in predictions.columns or "observed" not in predictions.columns:
        raise EvaluationError(f"the frame needs 'observed' and {column!r}")

    residual = predictions[column] - predictions["observed"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    hours = pd.DatetimeIndex(predictions.index).hour
    by_hour = residual.groupby(hours)
    means = by_hour.mean()
    spread = by_hour.std(ddof=0)
    left.plot(means.index, means.to_numpy(), color="tab:blue", marker="o", markersize=3)
    left.fill_between(
        means.index,
        (means - spread).to_numpy(),
        (means + spread).to_numpy(),
        alpha=0.2,
        color="tab:blue",
    )
    left.axhline(0.0, color="black", linewidth=0.8)
    left.set_xlabel("hour of day")
    left.set_ylabel(f"forecast - observed ({unit})")
    left.set_title(f"{column}: residual by hour (mean +/- 1 sd)")
    left.grid(alpha=0.25)

    right.hist(residual.to_numpy(), bins=50, color="tab:blue", alpha=0.8)
    right.axvline(0.0, color="black", linewidth=0.8)
    right.set_xlabel(f"forecast - observed ({unit})")
    right.set_ylabel("test hours")
    right.set_title(f"{column}: residual distribution")
    right.grid(alpha=0.25)

    return _save(fig, path)


def plot_training_curves(
    curves: dict[str, dict[str, list[float]]],
    path: str | Path,
    *,
    title: str = "Training and validation loss",
) -> Path:
    """One panel per architecture, both curves on each.

    Plotting only the training loss is how overfitting goes unnoticed; the gap
    between the two lines is the point of the figure.
    """
    if not curves:
        raise EvaluationError("no training curves to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        history = curves[name]
        train = history.get("train_loss", [])
        validation = history.get("validation_loss", [])
        ax.plot(range(1, len(train) + 1), train, label="train", linewidth=1.3)
        ax.plot(range(1, len(validation) + 1), validation, label="validation", linewidth=1.3)
        if validation:
            best = int(np.argmin(validation)) + 1
            ax.axvline(best, color="grey", linestyle="--", linewidth=0.9)
            ax.annotate(
                f"best epoch {best}",
                xy=(best, min(validation)),
                fontsize=7,
                xytext=(4, 6),
                textcoords="offset points",
            )
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (MSE, scaled target)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_missingness(report_frame: pd.DataFrame, path: str | Path) -> Path:
    """How much of each column is the -200 sentinel rather than a measurement."""
    if "column" not in report_frame.columns or "observed_share" not in report_frame.columns:
        raise EvaluationError("the report frame needs 'column' and 'observed_share'")

    ordered = report_frame.sort_values("observed_share")
    fig, ax = _figure((8.0, 0.45 * len(ordered) + 1.6))
    shares = ordered["observed_share"].to_numpy(dtype=float)
    colours = ["tab:red" if share < 0.5 else "tab:blue" for share in shares]
    ax.barh(ordered["column"].astype(str), shares * 100.0, color=colours)
    ax.axvline(50.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("percent of hours actually measured")
    ax.set_xlim(0, 100)
    ax.set_title("Observed share by column (red: dropped, below the 50% threshold)")
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)
