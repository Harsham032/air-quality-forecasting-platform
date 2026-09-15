"""Figures and tables must render from real shapes without a display."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqf.data.loader import missingness_report
from aqf.errors import EvaluationError
from aqf.reporting import plot_forecasts, plot_missingness, plot_residuals, plot_training_curves

pytest.importorskip("matplotlib", reason="matplotlib is needed for the figures")


@pytest.fixture
def predictions() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    index = pd.date_range("2005-01-01", periods=400, freq="h")
    observed = 110.0 + 40.0 * np.sin(2 * np.pi * np.arange(400) / 24.0)
    return pd.DataFrame(
        {
            "observed": observed,
            "gru": observed + rng.normal(0, 8, 400),
            "persistence": np.roll(observed, 1),
        },
        index=index,
    )


def test_a_forecast_plot_is_written(predictions: pd.DataFrame, tmp_path: Path) -> None:
    path = plot_forecasts(predictions, tmp_path / "forecasts.png", hours=168)
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_a_forecast_plot_needs_an_observed_column(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="observed"):
        plot_forecasts(pd.DataFrame({"gru": [1.0, 2.0]}), tmp_path / "x.png")


def test_a_forecast_plot_names_columns_it_cannot_find(
    predictions: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(EvaluationError, match="lstm"):
        plot_forecasts(predictions, tmp_path / "x.png", columns=["lstm"])


def test_a_residual_plot_is_written(predictions: pd.DataFrame, tmp_path: Path) -> None:
    path = plot_residuals(predictions, tmp_path / "residuals.png", column="gru")
    assert path.is_file()


def test_training_curves_render_for_several_architectures(tmp_path: Path) -> None:
    curves = {
        "lstm": {"train_loss": [1.0, 0.6, 0.4], "validation_loss": [1.1, 0.7, 0.8]},
        "gru": {"train_loss": [1.0, 0.5], "validation_loss": [1.0, 0.6]},
    }
    assert plot_training_curves(curves, tmp_path / "curves.png").is_file()


def test_empty_curves_are_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="no training curves"):
        plot_training_curves({}, tmp_path / "curves.png")


def test_the_missingness_plot_renders(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1.0, -200.0, 3.0], "b": [-200.0, -200.0, -200.0]})
    report = missingness_report(frame)
    assert plot_missingness(report.to_frame(), tmp_path / "missingness.png").is_file()


def test_a_malformed_report_frame_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="observed_share"):
        plot_missingness(pd.DataFrame({"column": ["a"]}), tmp_path / "x.png")
