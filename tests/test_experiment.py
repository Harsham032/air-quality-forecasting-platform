"""End-to-end tests on synthetic data.

The same code path a real run takes, on a series small enough to finish in
seconds. What is asserted is structure and honesty - that every model is scored
in physical units, that the leaderboard is ordered, that the verdict reflects
what happened - not that any particular model wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aqf.config import ExperimentConfig
from aqf.errors import EvaluationError
from aqf.experiment import describe_environment, run_experiment, save_result
from aqf.reporting import data_quality_table, leaderboard_table, verdict

pytest.importorskip("tensorflow", reason="TensorFlow is needed for the deep models")


@pytest.fixture
def quick_config() -> ExperimentConfig:
    return ExperimentConfig().with_overrides(
        {
            "run.name": "test",
            "models.architectures": "[gru]",
            "models.units": "8",
            "models.epochs": "2",
            "models.patience": "2",
            "models.batch_size": "64",
            "split.lookback": "12",
        }
    )


@pytest.fixture
def result(quick_config: ExperimentConfig, hourly_series: pd.DataFrame):
    return run_experiment(quick_config, frame=hourly_series)


@pytest.mark.slow
def test_every_model_is_scored_in_physical_units(result) -> None:
    """The correction, enforced end to end.

    A concentration RMSE of 0.3 would mean the target was never un-scaled. The
    synthetic series is centred near 110, so a real error is single- or
    double-digit, never a fraction.
    """
    assert result.outcomes
    for outcome in result.outcomes:
        assert outcome.metrics.unit == "ug/m3"
        assert outcome.metrics.rmse > 1.0


@pytest.mark.slow
def test_baselines_and_networks_are_both_present(result) -> None:
    kinds = {outcome.kind for outcome in result.outcomes}
    assert kinds == {"baseline", "deep"}
    assert result.best_baseline is not None
    assert result.best_deep is not None


@pytest.mark.slow
def test_the_leaderboard_is_ordered_by_error(result) -> None:
    rmse = result.leaderboard()["rmse"].tolist()
    assert rmse == sorted(rmse)
    assert result.leaderboard().iloc[0]["name"] == result.best.name


@pytest.mark.slow
def test_the_reference_baseline_scores_zero_skill_against_itself(result) -> None:
    reference = next(o for o in result.outcomes if o.name == result.reference_baseline)
    assert reference.skill_vs_reference == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_the_verdict_matches_the_numbers(result) -> None:
    """The text must not claim a win the table does not show."""
    text = verdict(result)
    if result.deep_learning_earned_its_place():
        assert "did **not** beat" not in text
    else:
        assert "did **not** beat" in text


@pytest.mark.slow
def test_predictions_cover_the_test_hours_and_nothing_else(result) -> None:
    assert result.predictions is not None
    assert len(result.predictions) == len(result.split.y_test)
    assert list(result.predictions.index) == list(result.split.timestamps_test)
    assert "observed" in result.predictions.columns
    for outcome in result.outcomes:
        assert outcome.name in result.predictions.columns


@pytest.mark.slow
def test_the_environment_is_recorded(result) -> None:
    """A deep learning result without its library versions is not reproducible."""
    assert "python" in result.environment
    assert "tensorflow" in result.environment
    assert "numpy" in result.environment


@pytest.mark.slow
def test_saving_writes_everything_needed_to_check_the_claim(result, tmp_path: Path) -> None:
    written = save_result(result, tmp_path)
    for path in written.values():
        assert path.is_file()

    summary = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert summary["outcomes"]
    assert summary["environment"]["python"]
    assert summary["reference_baseline"] == result.reference_baseline
    assert isinstance(summary["deep_learning_beat_every_baseline"], bool)

    leaderboard = pd.read_csv(written["leaderboard"])
    assert len(leaderboard) == len(result.outcomes)
    assert (leaderboard["unit"] == "ug/m3").all()


@pytest.mark.slow
def test_the_tables_render(result) -> None:
    table = leaderboard_table(result)
    assert "| Model |" in table
    assert result.best.name in table
    assert data_quality_table(result.dataset).startswith("| Column |")


@pytest.mark.slow
def test_a_missing_reference_baseline_is_refused(hourly_series: pd.DataFrame) -> None:
    config = ExperimentConfig().with_overrides(
        {
            "models.architectures": "[gru]",
            "models.epochs": "1",
            "split.lookback": "12",
            "evaluation.reference_baseline": "climatology",
        }
    )
    with pytest.raises(EvaluationError, match="skill scores would have nothing"):
        run_experiment(config, frame=hourly_series)


def test_describe_environment_names_the_libraries() -> None:
    environment = describe_environment()
    assert environment["python"]
    assert environment["numpy"]
    assert environment["pandas"]
