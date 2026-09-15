#!/usr/bin/env python
"""Repeat the comparison across seeds and report the spread.

    python scripts/run_seed_sweep.py --config configs/default.yaml --seeds 5

A single training run reports one draw from a distribution. Two architectures
separated by less than that distribution's width are not distinguishable, and
declaring a winner from one run is how small differences become findings.

Baselines are deterministic, so their numbers repeat exactly; the spread shown
for them is zero by construction, which is itself worth seeing next to the
networks' variation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqf.config import ExperimentConfig  # noqa: E402
from aqf.errors import AqfError  # noqa: E402
from aqf.experiment import run_experiment  # noqa: E402
from aqf.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, default=5, help="how many seeds to run")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        base = ExperimentConfig.from_yaml(args.config)
    except AqfError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for offset in range(args.seeds):
        seed = base.run.seed + offset
        config = base.model_copy(update={"run": base.run.model_copy(update={"seed": seed})})
        logger.info("seed_run_started", seed=seed, index=offset + 1, of=args.seeds)
        result = run_experiment(config)
        for outcome in result.outcomes:
            rows.append(
                {
                    "seed": seed,
                    "name": outcome.name,
                    "kind": outcome.kind,
                    "rmse": outcome.metrics.rmse,
                    "mae": outcome.metrics.mae,
                    "r2": outcome.metrics.r2,
                    "skill_vs_reference": outcome.skill_vs_reference,
                    "fit_seconds": outcome.fit_seconds,
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["name", "kind"])
        .agg(
            seeds=("rmse", "size"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            r2_mean=("r2", "mean"),
            skill_mean=("skill_vs_reference", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
        .sort_values("rmse_mean")
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / f"{base.run.name}-seed-runs.csv"
    summary_path = out / f"{base.run.name}-seed-summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    unit = base.evaluation.unit
    print()
    print(f"{args.seeds} seeds, test RMSE in {unit}")
    print("| Model | Kind | Mean | SD | Min | Max | Mean R2 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in summary.iterrows():
        sd = 0.0 if pd.isna(row["rmse_sd"]) else row["rmse_sd"]
        print(
            f"| {row['name']} | {row['kind']} | {row['rmse_mean']:.3f} | {sd:.3f} | "
            f"{row['rmse_min']:.3f} | {row['rmse_max']:.3f} | {row['r2_mean']:.4f} |"
        )

    deep = summary[summary["kind"] == "deep"]
    baselines = summary[summary["kind"] == "baseline"]
    if not deep.empty and not baselines.empty:
        best_deep = deep.iloc[0]
        best_baseline = baselines.iloc[0]
        gap = float(best_baseline["rmse_mean"] - best_deep["rmse_mean"])

        # Compared against the contender's own spread, not the widest spread in
        # the table. A badly-behaved third model's variance says nothing about
        # whether these two are separable.
        spread = float(best_deep["rmse_sd"]) if pd.notna(best_deep["rmse_sd"]) else 0.0

        # And the question that matters more than any interval: on how many
        # seeds does the ordering actually hold? Means can overlap while one
        # model wins every single run.
        paired = runs.pivot(index="seed", columns="name", values="rmse")
        wins = int((paired[best_baseline["name"]] < paired[best_deep["name"]]).sum())
        seeds = len(paired)

        print()
        print(
            f"Best network {best_deep['name']} averages {best_deep['rmse_mean']:.3f} {unit} "
            f"(sd {spread:.3f}); best baseline {best_baseline['name']} averages "
            f"{best_baseline['rmse_mean']:.3f} {unit} (gap {gap:+.3f})."
        )
        print(f"The baseline is ahead on {wins} of {seeds} seeds.")
        if wins in (0, seeds):
            ahead = best_baseline["name"] if wins == seeds else best_deep["name"]
            print(f"The ordering is consistent across every seed: {ahead} wins each time.")
        elif abs(gap) < spread:
            print(
                "The gap is smaller than the network's own seed-to-seed spread and the "
                "ordering flips between runs, so this sweep does not distinguish them."
            )
        else:
            print(
                "The ordering is not consistent across seeds; more seeds would be needed "
                "to separate them."
            )

    print(f"\nWrote {runs_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
