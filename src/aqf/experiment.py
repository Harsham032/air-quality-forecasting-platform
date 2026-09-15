"""Run the comparison end to end.

One function, one config, one set of numbers. The point of putting this in the
package rather than in a script is that it can be tested: a run on three epochs
of fabricated data exercises exactly the code path a real run uses.

Every forecast produced here is inverted back to micrograms per cubic metre
before it is scored. That is the whole correction this project is built around,
so it happens in one place, not per model.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data.loader import DatasetReport, load_air_quality
from .data.windows import TemporalSplit, WindowSpec, split_by_time
from .errors import EvaluationError
from .evaluation.metrics import ForecastMetrics, evaluate_forecast, skill_score
from .logging_utils import get_logger
from .models.baselines import (
    BaselineResult,
    linear_baseline,
    persistence,
    seasonal_naive,
    window_mean,
)
from .models.deep import Architecture, TrainedModel, configure_threads, train_model

logger = get_logger(__name__)


@dataclass
class ModelOutcome:
    """One scored forecast and what it cost to produce."""

    name: str
    kind: str  # "baseline" or "deep"
    metrics: ForecastMetrics
    skill_vs_reference: float
    fit_seconds: float = 0.0
    epochs_run: int = 0
    best_epoch: int = 0
    parameters: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.metrics.to_dict())
        payload.update(
            {
                "name": self.name,
                "kind": self.kind,
                "skill_vs_reference": self.skill_vs_reference,
                "fit_seconds": self.fit_seconds,
                "epochs_run": float(self.epochs_run),
                "best_epoch": float(self.best_epoch),
                "parameters": float(self.parameters),
                "note": self.note,
            }
        )
        return payload


@dataclass
class ExperimentResult:
    """Everything a run produced, in one object worth serialising."""

    config: ExperimentConfig
    dataset: DatasetReport
    split: TemporalSplit
    outcomes: list[ModelOutcome]
    reference_baseline: str
    environment: dict[str, str] = field(default_factory=dict)
    total_seconds: float = 0.0
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    predictions: pd.DataFrame | None = None

    def leaderboard(self) -> pd.DataFrame:
        """Outcomes ordered by test RMSE, best first."""
        frame = pd.DataFrame([outcome.to_dict() for outcome in self.outcomes])
        return frame.sort_values("rmse").reset_index(drop=True)

    def top_names(self, n: int = 3) -> list[str]:
        """The ``n`` lowest-RMSE model names, best first."""
        ordered = sorted(self.outcomes, key=lambda outcome: outcome.metrics.rmse)
        return [outcome.name for outcome in ordered[:n]]

    @property
    def best(self) -> ModelOutcome:
        return min(self.outcomes, key=lambda outcome: outcome.metrics.rmse)

    @property
    def best_deep(self) -> ModelOutcome | None:
        deep = [outcome for outcome in self.outcomes if outcome.kind == "deep"]
        return min(deep, key=lambda outcome: outcome.metrics.rmse) if deep else None

    @property
    def best_baseline(self) -> ModelOutcome | None:
        baselines = [outcome for outcome in self.outcomes if outcome.kind == "baseline"]
        return min(baselines, key=lambda outcome: outcome.metrics.rmse) if baselines else None

    def deep_learning_earned_its_place(self) -> bool:
        """Whether the best network actually beat the best baseline.

        Named as a question rather than a metric because it is the question the
        whole comparison exists to answer, and the answer is allowed to be no.
        """
        deep, baseline = self.best_deep, self.best_baseline
        if deep is None or baseline is None:
            return False
        return deep.metrics.rmse < baseline.metrics.rmse


def describe_environment() -> dict[str, str]:
    """Record what the numbers were produced on.

    A result without its environment is not reproducible, and deep learning
    results move between library versions more than anyone would like.
    """
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        environment["scikit-learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
        pass
    try:
        import tensorflow as tf

        environment["tensorflow"] = tf.__version__
    except ImportError:  # pragma: no cover - exercised only where TF is absent
        environment["tensorflow"] = "not installed"
    return environment


def _score(
    split: TemporalSplit,
    predictions_scaled: np.ndarray,
    *,
    name: str,
    config: ExperimentConfig,
) -> ForecastMetrics:
    """Invert the scaling, then score. Never the other way round."""
    truth = split.inverse_target(split.y_test)
    forecast = split.inverse_target(predictions_scaled)
    return evaluate_forecast(
        truth,
        forecast,
        name=name,
        unit=config.evaluation.unit,
        require_physical_units=config.evaluation.require_physical_units,
    )


def _run_baselines(split: TemporalSplit, config: ExperimentConfig) -> list[BaselineResult]:
    target_index = split.feature_names.index(split.target)
    results = [
        persistence(split.X_test, target_index),
        seasonal_naive(split.X_test, target_index, period=config.baselines.seasonal_period),
        window_mean(split.X_test, target_index),
    ]
    if config.baselines.include_ridge:
        results.append(
            linear_baseline(
                split.X_train,
                split.y_train,
                split.X_test,
                alpha=config.baselines.ridge_alpha,
            )
        )
    return results


def run_experiment(
    config: ExperimentConfig, *, frame: pd.DataFrame | None = None
) -> ExperimentResult:
    """Load, split, fit the baselines and the networks, and score them all.

    ``frame`` lets a caller supply an already-loaded series, which is what the
    tests do; production runs read the file named in the config.
    """
    started = time.perf_counter()
    configure_threads(config.run.n_threads)
    np.random.seed(config.run.seed)

    if frame is None:
        frame, dataset = load_air_quality(
            config.data.path,
            min_observed_share=config.data.min_observed_share,
            interpolate_limit=config.data.interpolate_limit,
            target=config.data.target,
        )
    else:
        from .data.loader import missingness_report

        dataset = missingness_report(frame)

    spec = WindowSpec(lookback=config.split.lookback, horizon=config.split.horizon)
    split = split_by_time(
        frame,
        target=config.data.target,
        spec=spec,
        train_share=config.split.train_share,
        validation_share=config.split.validation_share,
    )

    baseline_results = _run_baselines(split, config)
    baseline_metrics = {
        baseline.name: _score(split, baseline.predictions, name=baseline.name, config=config)
        for baseline in baseline_results
    }

    reference_name = config.evaluation.reference_baseline
    reference = next(
        (metrics for name, metrics in baseline_metrics.items() if name.startswith(reference_name)),
        None,
    )
    if reference is None:
        raise EvaluationError(
            f"reference baseline {reference_name!r} was not among "
            f"{sorted(baseline_metrics)}; skill scores would have nothing to measure against"
        )

    outcomes: list[ModelOutcome] = []
    forecasts: dict[str, np.ndarray] = {}
    for baseline in baseline_results:
        metrics = baseline_metrics[baseline.name]
        outcomes.append(
            ModelOutcome(
                name=baseline.name,
                kind="baseline",
                metrics=metrics,
                skill_vs_reference=skill_score(metrics, reference),
                note=baseline.description,
            )
        )
        forecasts[baseline.name] = split.inverse_target(baseline.predictions)

    curves: dict[str, dict[str, list[float]]] = {}
    for architecture in config.models.architectures:
        logger.info("training", architecture=architecture, windows=len(split.X_train))
        trained: TrainedModel = train_model(
            cast(Architecture, architecture),
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
            epochs=config.models.epochs,
            batch_size=config.models.batch_size,
            patience=config.models.patience,
            units=config.models.units,
            filters=config.models.filters,
            kernel_size=config.models.kernel_size,
            dropout=config.models.dropout,
            learning_rate=config.models.learning_rate,
            seed=config.run.seed,
            n_threads=config.run.n_threads,
        )
        predictions = trained.predict(split.X_test)
        metrics = _score(split, predictions, name=architecture, config=config)
        outcomes.append(
            ModelOutcome(
                name=architecture,
                kind="deep",
                metrics=metrics,
                skill_vs_reference=skill_score(metrics, reference),
                fit_seconds=trained.fit_seconds,
                epochs_run=trained.epochs_run,
                best_epoch=trained.best_epoch,
                parameters=trained.parameters,
                note=(
                    "early stopping restored the best weights"
                    if trained.stopped_early
                    else "ran to the epoch limit without stopping early"
                ),
            )
        )
        forecasts[architecture] = split.inverse_target(predictions)
        curves[architecture] = {
            "train_loss": trained.train_loss,
            "validation_loss": trained.validation_loss,
        }
        logger.info(
            "trained",
            architecture=architecture,
            rmse=round(metrics.rmse, 3),
            r2=round(metrics.r2, 4),
            skill=round(skill_score(metrics, reference), 4),
            seconds=round(trained.fit_seconds, 1),
        )

    predictions_frame = pd.DataFrame(
        {"observed": split.inverse_target(split.y_test), **forecasts},
        index=split.timestamps_test,
    )
    predictions_frame.index.name = "timestamp"

    result = ExperimentResult(
        config=config,
        dataset=dataset,
        split=split,
        outcomes=outcomes,
        reference_baseline=reference.name,
        environment=describe_environment(),
        total_seconds=time.perf_counter() - started,
        curves=curves,
        predictions=predictions_frame,
    )
    logger.info(
        "experiment_finished",
        best=result.best.name,
        best_rmse=round(result.best.metrics.rmse, 3),
        deep_learning_won=result.deep_learning_earned_its_place(),
        seconds=round(result.total_seconds, 1),
    )
    return result


def save_result(result: ExperimentResult, directory: str | Path) -> dict[str, Path]:
    """Write the leaderboard, the predictions and the run's provenance."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    name = result.config.run.name

    leaderboard_path = out / f"{name}-leaderboard.csv"
    result.leaderboard().to_csv(leaderboard_path, index=False)

    predictions_path = out / f"{name}-predictions.csv"
    if result.predictions is not None:
        result.predictions.to_csv(predictions_path)

    quality_path = out / f"{name}-data-quality.csv"
    result.dataset.to_frame().to_csv(quality_path, index=False)

    summary_path = out / f"{name}-summary.json"
    summary = {
        "run": result.config.run.model_dump(),
        "config": result.config.model_dump(mode="json"),
        "environment": result.environment,
        "split": result.split.describe(),
        "reference_baseline": result.reference_baseline,
        "total_seconds": round(result.total_seconds, 2),
        "deep_learning_beat_every_baseline": result.deep_learning_earned_its_place(),
        "outcomes": [outcome.to_dict() for outcome in result.outcomes],
        "training_curves": result.curves,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    written = {
        "leaderboard": leaderboard_path,
        "predictions": predictions_path,
        "data_quality": quality_path,
        "summary": summary_path,
    }
    logger.info("result_saved", **{key: str(value) for key, value in written.items()})
    return written
