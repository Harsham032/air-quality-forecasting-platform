"""Markdown tables for the results document.

Kept separate from the experiment so the numbers are formatted once, in one
place, and the same function produces what goes in ``docs/results.md`` and what
is printed at the end of a run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..data.loader import DatasetReport

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from ..experiment import ExperimentResult


def _number(value: float, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def leaderboard_table(result: ExperimentResult) -> str:
    """Every model on the test set, ordered by RMSE.

    Skill is measured against the reference baseline: 0 means exactly as good as
    that baseline, negative means worse than it.
    """
    unit = result.config.evaluation.unit
    lines = [
        f"| Model | Kind | RMSE ({unit}) | MAE ({unit}) | R2 | Bias ({unit}) | Skill vs {result.reference_baseline} | Fit (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in result.leaderboard().iterrows():
        fit = _number(float(row["fit_seconds"]), 1) if float(row["fit_seconds"]) > 0 else "-"
        lines.append(
            f"| {row['name']} | {row['kind']} | {_number(float(row['rmse']))} | "
            f"{_number(float(row['mae']))} | {_number(float(row['r2']), 4)} | "
            f"{_number(float(row['bias']))} | {_number(float(row['skill_vs_reference']), 4)} | {fit} |"
        )
    return "\n".join(lines)


def data_quality_table(report: DatasetReport, *, limit: int | None = None) -> str:
    """What the sentinel conversion did to each column."""
    lines = [
        "| Column | Observed hours | Sentinel cells | Observed share | Mean with sentinel | Mean observed | Longest gap (h) | Kept |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    columns = report.columns[:limit] if limit else report.columns
    for column in columns:
        lines.append(
            f"| {column.column} | {column.observed:,} | {column.n_sentinel:,} | "
            f"{_number(column.observed_share * 100, 1)}% | {_number(column.mean_with_sentinel, 2)} | "
            f"{_number(column.mean_observed, 2)} | {column.longest_gap:,} | "
            f"{'yes' if column.usable else 'no'} |"
        )
    return "\n".join(lines)


def verdict(result: ExperimentResult) -> str:
    """One paragraph stating what the run actually showed.

    Written to be equally comfortable saying the networks lost, because on this
    dataset at a one-hour horizon that is a plausible outcome and hiding it would
    defeat the purpose of running the comparison.
    """
    best = result.best
    deep = result.best_deep
    baseline = result.best_baseline
    unit = result.config.evaluation.unit

    if deep is None or baseline is None:
        return (
            f"The best model was **{best.name}** at {_number(best.metrics.rmse)} {unit} RMSE. "
            "Only one kind of model was run, so there is no comparison to draw."
        )

    margin = baseline.metrics.rmse - deep.metrics.rmse
    relative = margin / baseline.metrics.rmse if baseline.metrics.rmse else float("nan")

    if margin > 0:
        return (
            f"The best network (**{deep.name}**, {_number(deep.metrics.rmse)} {unit} RMSE) beat the "
            f"best baseline (**{baseline.name}**, {_number(baseline.metrics.rmse)} {unit}) by "
            f"{_number(margin)} {unit}, a {_number(relative * 100, 1)}% reduction in error. "
            f"It cost {_number(deep.fit_seconds, 1)} seconds of training to get that; the baseline "
            "cost nothing."
        )
    return (
        f"The best network (**{deep.name}**, {_number(deep.metrics.rmse)} {unit} RMSE) did **not** beat "
        f"the best baseline (**{baseline.name}**, {_number(baseline.metrics.rmse)} {unit}); it was "
        f"{_number(-margin)} {unit} worse. On a one-step-ahead hourly forecast this is an ordinary "
        "result rather than a failed experiment: the previous hour is a strong predictor of the next "
        "one, and a network has to earn its training time against that."
    )
