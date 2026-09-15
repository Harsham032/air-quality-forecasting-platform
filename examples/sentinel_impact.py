#!/usr/bin/env python
"""Run both pipelines side by side and print the difference.

    python examples/sentinel_impact.py

The naive pipeline is not a strawman. It is the ordinary sequence: read the
file, min-max scale it, window it, train a GRU, report RMSE. Every step is
something a competent person would write. The only thing it omits is that -200
means "no reading".

Both pipelines use the same architecture, the same seed, the same number of
epochs and the same chronological split, so the difference in the output is
attributable to the sentinel and nothing else.

This script trains two small networks and takes a couple of minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqf.data.loader import MISSING_SENTINEL, load_air_quality  # noqa: E402
from aqf.data.windows import WindowSpec, make_windows  # noqa: E402
from aqf.errors import DataError, EvaluationError  # noqa: E402
from aqf.evaluation.metrics import evaluate_forecast  # noqa: E402
from aqf.logging_utils import configure_logging  # noqa: E402
from aqf.models.baselines import persistence  # noqa: E402
from aqf.models.deep import train_model  # noqa: E402

TARGET = "NO2(GT)"
LOOKBACK = 24
EPOCHS = 8
SEED = 20260101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", default="data/raw/AirQuality.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def naive_pipeline(path: str, epochs: int) -> dict[str, float]:
    """Read, min-max scale, window, train, report. The sentinel is left in."""
    from sklearn.preprocessing import MinMaxScaler

    frame = pd.read_csv(path, sep=";", decimal=",")
    frame = frame.loc[:, ~frame.columns.str.contains("^Unnamed")].dropna(how="all")
    numeric = frame.select_dtypes(include=[np.number]).dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric.to_numpy(dtype=np.float64))
    target_index = list(numeric.columns).index(TARGET)

    n = len(scaled)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    spec = WindowSpec(lookback=LOOKBACK, horizon=1)
    X_train, y_train = make_windows(scaled[:train_end], scaled[:train_end, target_index], spec)
    X_validation, y_validation = make_windows(
        scaled[train_end:validation_end], scaled[train_end:validation_end, target_index], spec
    )
    X_test, y_test = make_windows(
        scaled[validation_end:], scaled[validation_end:, target_index], spec
    )

    trained = train_model(
        "gru", X_train, y_train, X_validation, y_validation, epochs=epochs, units=32, seed=SEED
    )
    prediction = trained.predict(X_test)

    # Scored on the scaled axis, which is how this is normally reported.
    scaled_metrics = evaluate_forecast(y_test, prediction, require_physical_units=False)

    # And inverted, to see what that error is worth in real units. The naive
    # scaler's range is poisoned by the sentinel, so this inversion is wrong
    # too - but it is the only way to compare the two pipelines at all.
    span = float(numeric[TARGET].max() - numeric[TARGET].min())
    floor = float(numeric[TARGET].min())
    physical = evaluate_forecast(
        y_test * span + floor, prediction * span + floor, require_physical_units=False
    )

    return {
        "target_mean_as_read": float(numeric[TARGET].mean()),
        "scaler_min": floor,
        "scaler_max": float(numeric[TARGET].max()),
        "rmse_on_scaled_axis": scaled_metrics.rmse,
        "r2": scaled_metrics.r2,
        "rmse_in_physical_units": physical.rmse,
        "share_of_scale_used_by_real_data": (float(numeric[TARGET].max()) - 2.0) / span,
    }


def corrected_pipeline(path: str, epochs: int) -> dict[str, float]:
    """The same thing, with the sentinel treated as missing."""
    from aqf.data.windows import split_by_time

    frame, report = load_air_quality(path, target=TARGET)
    split = split_by_time(frame, target=TARGET, spec=WindowSpec(lookback=LOOKBACK, horizon=1))

    trained = train_model(
        "gru",
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
        epochs=epochs,
        units=32,
        seed=SEED,
    )
    truth = split.inverse_target(split.y_test)
    forecast = split.inverse_target(trained.predict(split.X_test))
    metrics = evaluate_forecast(truth, forecast, name="gru")

    target_index = split.feature_names.index(split.target)
    baseline = evaluate_forecast(
        truth, split.inverse_target(persistence(split.X_test, target_index).predictions)
    )

    return {
        "target_mean_observed": float(frame[TARGET].mean()),
        "hours_retained": float(report.n_rows_retained),
        "rmse_in_physical_units": metrics.rmse,
        "r2": metrics.r2,
        "persistence_rmse": baseline.rmse,
        "skill_vs_persistence": 1.0 - metrics.rmse / baseline.rmse,
    }


def main() -> int:
    configure_logging("WARNING")
    args = parse_args()

    if not Path(args.path).is_file():
        print(f"No data at {args.path}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    raw = pd.read_csv(args.path, sep=";", decimal=",")
    sentinel_cells = int(
        (raw.select_dtypes(include=[np.number]) == MISSING_SENTINEL).to_numpy().sum()
    )

    print("=" * 78)
    print(f"The file contains {sentinel_cells:,} cells holding {MISSING_SENTINEL:.0f}.")
    print("=" * 78)

    try:
        naive = naive_pipeline(args.path, args.epochs)
        corrected = corrected_pipeline(args.path, args.epochs)
    except (DataError, EvaluationError) as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    print("\nNAIVE: -200 read as a measurement")
    print(f"  {TARGET} mean as read            {naive['target_mean_as_read']:>10.2f} ug/m3")
    print(
        f"  scaler range                    [{naive['scaler_min']:.1f}, {naive['scaler_max']:.1f}]"
    )
    print(f"  share of the [0,1] axis used    {naive['share_of_scale_used_by_real_data']:>10.1%}")
    print(f"  RMSE on the scaled axis         {naive['rmse_on_scaled_axis']:>10.4f}  <- unitless")
    print(f"  R2                              {naive['r2']:>10.4f}")
    print(f"  RMSE converted back             {naive['rmse_in_physical_units']:>10.2f} ug/m3")

    print("\nCORRECTED: -200 treated as missing")
    print(f"  {TARGET} mean over observed hours{corrected['target_mean_observed']:>10.2f} ug/m3")
    print(f"  hours retained                  {corrected['hours_retained']:>10.0f}")
    print(f"  RMSE                            {corrected['rmse_in_physical_units']:>10.2f} ug/m3")
    print(f"  R2                              {corrected['r2']:>10.4f}")
    print(f"  persistence RMSE                {corrected['persistence_rmse']:>10.2f} ug/m3")
    print(f"  skill vs persistence            {corrected['skill_vs_persistence']:>10.4f}")

    print("\n" + "-" * 78)
    ratio = naive["rmse_in_physical_units"] / corrected["rmse_in_physical_units"]
    print(
        f"Same architecture, same seed, same epochs. The naive pipeline's error is\n"
        f"{ratio:.1f}x larger in physical units, and its scaled RMSE of "
        f"{naive['rmse_on_scaled_axis']:.4f} gives no\nhint of that: it is a small number on an axis "
        f"that means nothing."
    )
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
