#!/usr/bin/env python
"""Run the full comparison and write the results.

    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/fast.yaml --set split.horizon=3

Baselines and networks see identical windows, are scored on identical test
hours, and are reported in micrograms per cubic metre. The script prints the
leaderboard and states plainly whether deep learning beat the baselines, which
is the only conclusion the experiment is entitled to draw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqf.config import ExperimentConfig, load_settings  # noqa: E402
from aqf.errors import AqfError  # noqa: E402
from aqf.experiment import run_experiment, save_result  # noqa: E402
from aqf.logging_utils import configure_logging, get_logger  # noqa: E402
from aqf.reporting import (  # noqa: E402
    data_quality_table,
    leaderboard_table,
    plot_forecasts,
    plot_missingness,
    plot_residuals,
    plot_training_curves,
    verdict,
)

logger = get_logger(__name__)


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects section.key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml", help="experiment configuration")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip plotting")
    return parser.parse_args()


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = parse_args()

    try:
        config = ExperimentConfig.from_yaml(args.config).with_overrides(
            parse_overrides(args.overrides)
        )
    except AqfError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if config.run.n_threads is None and settings.n_threads is not None:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"n_threads": settings.n_threads})}
        )

    try:
        result = run_experiment(config)
    except AqfError as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    written = save_result(result, config.run.output_dir)

    if not args.no_figures:
        figures = Path(config.run.figure_dir)
        name = config.run.name
        plot_missingness(result.dataset.to_frame(), figures / f"{name}-missingness.png")
        if result.predictions is not None:
            plot_forecasts(
                result.predictions,
                figures / f"{name}-forecasts.png",
                columns=result.top_names(3),
                unit=config.evaluation.unit,
                title=f"{config.data.target} forecasts",
            )
            plot_residuals(
                result.predictions,
                figures / f"{name}-residuals.png",
                column=result.best.name,
                unit=config.evaluation.unit,
            )
        if result.curves:
            plot_training_curves(result.curves, figures / f"{name}-training-curves.png")

    unit = config.evaluation.unit
    print()
    print(result.dataset.summary())
    print(
        f"Retained {result.dataset.n_rows_retained:,} hours "
        f"({result.dataset.first_timestamp} to {result.dataset.last_timestamp}) "
        f"across {len(result.dataset.retained_columns)} columns"
    )
    print()
    print(data_quality_table(result.dataset))
    print()
    print(f"Test set: {len(result.split.y_test)} hours, target {config.data.target} in {unit}")
    print(leaderboard_table(result))
    print()
    print(verdict(result))
    print()
    print(
        f"Wrote {written['leaderboard']} and {len(written) - 1} more files to {config.run.output_dir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
