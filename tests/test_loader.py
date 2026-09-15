"""The sentinel is the whole point.

Every test here exists because treating -200 as a measurement produces a
plausible-looking pipeline that silently poisons everything downstream. These
tests fail loudly if that ever comes back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqf.data.loader import MISSING_SENTINEL, load_air_quality, missingness_report
from aqf.errors import DataError, DataQualityError

from .conftest import RAW_DATA, requires_raw_data


def test_sentinel_never_survives_loading(raw_like_csv: Path) -> None:
    cleaned, _ = load_air_quality(raw_like_csv, target="NO2(GT)")
    assert not (cleaned == MISSING_SENTINEL).to_numpy().any()
    # Nothing that remains is a physically impossible concentration.
    assert cleaned["NO2(GT)"].min() > 0.0
    assert cleaned["CO(GT)"].min() > 0.0


def test_sentinel_drags_the_mean_below_zero_when_left_alone(raw_like_csv: Path) -> None:
    """The failure mode, demonstrated rather than asserted about.

    Read naively, a concentration column averages to a negative number. No
    reviewer who saw this printed would accept it; the problem is that nobody
    prints it.
    """
    raw = pd.read_csv(raw_like_csv, sep=";", decimal=",")
    naive_mean = raw["NO2(GT)"].mean()

    cleaned, report = load_air_quality(raw_like_csv, target="NO2(GT)")
    honest_mean = cleaned["NO2(GT)"].mean()

    assert naive_mean < honest_mean
    target_report = next(c for c in report.columns if c.column == "NO2(GT)")
    assert target_report.mean_with_sentinel < target_report.mean_observed


def test_a_column_that_barely_exists_is_dropped_not_imputed(raw_like_csv: Path) -> None:
    cleaned, report = load_air_quality(raw_like_csv, target="NO2(GT)")
    assert "NMHC(GT)" in report.dropped
    assert "NMHC(GT)" not in cleaned.columns
    # It is still described, so the reader can see why it went.
    dropped_report = next(c for c in report.columns if c.column == "NMHC(GT)")
    assert dropped_report.observed_share < 0.5
    assert not dropped_report.usable


def test_the_drop_threshold_is_honoured(raw_like_csv: Path) -> None:
    """A stricter threshold must drop more, and `usable` must agree with it."""
    lenient_frame, lenient = load_air_quality(
        raw_like_csv, target="NO2(GT)", min_observed_share=0.05
    )
    strict_frame, strict = load_air_quality(raw_like_csv, target="NO2(GT)", min_observed_share=0.85)

    assert "NMHC(GT)" not in lenient.dropped
    assert "NMHC(GT)" in strict.dropped
    assert set(lenient.dropped) < set(strict.dropped)
    assert len(strict_frame.columns) < len(lenient_frame.columns)

    for column in strict.columns:
        assert column.usable == (column.observed_share >= 0.85)
    for column in lenient.columns:
        assert column.usable == (column.observed_share >= 0.05)


def test_a_threshold_strict_enough_to_drop_the_target_refuses(raw_like_csv: Path) -> None:
    """Silently forecasting a column that was just dropped would be worse.

    The synthetic target is 89% observed, so a 92% threshold removes it. The
    loader must say so rather than continue with a target that is not there.
    """
    with pytest.raises(DataQualityError, match="cannot be forecast"):
        load_air_quality(raw_like_csv, target="NO2(GT)", min_observed_share=0.92)


def test_long_gaps_are_dropped_rather_than_bridged(raw_like_csv: Path) -> None:
    """A 60-hour outage must not become 60 hours of smooth invented data."""
    cleaned, report = load_air_quality(raw_like_csv, target="NO2(GT)", interpolate_limit=6)
    assert report.n_rows_dropped_to_gaps > 0
    assert report.n_rows_retained == len(cleaned)
    assert report.n_rows_retained < report.n_rows
    # The hours inside the long outage are absent, not filled.
    assert len(cleaned) < 900


def test_interpolation_is_counted_not_hidden(raw_like_csv: Path) -> None:
    _, report = load_air_quality(raw_like_csv, target="NO2(GT)")
    assert report.interpolated_cells > 0
    assert report.interpolated_cells < report.n_rows_retained * len(report.retained_columns)
    assert "interpolated" in report.summary()


def test_the_index_is_hourly_and_ordered(raw_like_csv: Path) -> None:
    """A shuffled clock silently destroys a forecasting problem."""
    cleaned, _ = load_air_quality(raw_like_csv, target="NO2(GT)")
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    deltas = cleaned.index.to_series().diff().dropna()
    # Every step is a whole number of hours; gaps are longer, never fractional.
    assert (deltas.dt.total_seconds() % 3600 == 0).all()


def test_a_target_that_is_mostly_missing_is_refused(tmp_path: Path) -> None:
    hours = 300
    index = pd.date_range("2004-03-10 18:00:00", periods=hours, freq="h")
    frame = pd.DataFrame(
        {
            "Date": index.strftime("%d/%m/%Y"),
            "Time": index.strftime("%H.%M.%S"),
            "NO2(GT)": np.where(np.arange(hours) < 30, 100.0, MISSING_SENTINEL),
            "T": np.full(hours, 18.0),
        }
    )
    path = tmp_path / "sparse.csv"
    path.write_text(frame.to_csv(sep=";", index=False, decimal=","), encoding="utf-8")

    with pytest.raises(DataQualityError, match="cannot be forecast"):
        load_air_quality(path, target="NO2(GT)")


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no data file"):
        load_air_quality(tmp_path / "absent.csv")


def test_an_unknown_target_names_what_is_available(raw_like_csv: Path) -> None:
    with pytest.raises(DataError, match="has no column"):
        load_air_quality(raw_like_csv, target="PM2.5")


def test_the_report_counts_sentinels_not_blanks() -> None:
    frame = pd.DataFrame({"a": [1.0, MISSING_SENTINEL, np.nan, 4.0]})
    report = missingness_report(frame)
    column = report.columns[0]
    assert column.n_sentinel == 1
    assert column.n_blank == 1
    assert column.observed == 2
    assert column.observed_share == 0.5


@requires_raw_data()
def test_the_real_file_matches_what_the_documentation_claims() -> None:
    """Guards the numbers quoted in docs/results.md against a silent data change."""
    cleaned, report = load_air_quality(RAW_DATA, target="NO2(GT)")
    assert report.n_rows == 9357
    assert report.total_sentinel_cells == 16701
    assert report.dropped == ["NMHC(GT)"]
    assert report.n_rows_retained == 7620
    assert len(cleaned.columns) == 12
    # The correction, on the real file: a negative mean becomes a plausible one.
    target = next(c for c in report.columns if c.column == "NO2(GT)")
    assert target.mean_with_sentinel == pytest.approx(58.15, abs=0.01)
    assert target.mean_observed == pytest.approx(113.09, abs=0.01)
