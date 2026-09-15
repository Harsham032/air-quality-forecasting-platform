"""Loading the UCI Air Quality series.

**The sentinel is the whole problem.** This dataset does not leave missing
values blank. It writes ``-200``, a value no pollutant concentration, no
temperature in degrees Celsius and no relative humidity can take. To pandas it
is an ordinary number: ``isna()`` returns False for every one of them, and any
pipeline that cleans with ``isna`` followed by ``ffill`` or interpolation runs
to completion, reports no missing data, and trains on 16,701 impossible values.

The damage is not subtle once you look for it. In this file, 8,530 of 9,471 rows
carry at least one sentinel. The target used here, ``NO2(GT)``, has 1,642 of
them - 17.3% - and they drag its mean from 113.1 down to 58.2.

Scaling is where it becomes a reported result. ``MinMaxScaler`` fitted over a
column whose minimum is -200 rather than 2 stretches the range from 338 to 540,
and every sentinel lands on exactly 0.0. A model then predicts a sixth of the
target perfectly, error on the scaled axis collapses, and the number that gets
written down - an RMSE near 0.11 - is an artefact of the encoding rather than a
measure of forecasting skill. Multiplied back into micrograms per cubic metre it
is roughly 59, against a series whose mean is 113.

So the sentinel is converted to a real missing value before anything else
happens, the extent is reported rather than absorbed, and columns too sparse to
model are dropped rather than interpolated into plausible-looking fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataError, DataQualityError
from ..logging_utils import get_logger

logger = get_logger(__name__)

# The value this dataset writes where an instrument reported nothing.
MISSING_SENTINEL = -200.0

# Ground-truth reference analysers, as opposed to the PT08.Sxx metal-oxide
# sensor responses. The (GT) columns are what a forecast should be judged on.
REFERENCE_COLUMNS = ("CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)")
SENSOR_COLUMNS = (
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
)
WEATHER_COLUMNS = ("T", "RH", "AH")


@dataclass(frozen=True)
class ColumnReport:
    """How much of one column actually exists."""

    column: str
    n_rows: int
    n_sentinel: int
    n_blank: int
    observed: int
    observed_share: float
    mean_with_sentinel: float
    mean_observed: float
    longest_gap: int
    # The share this column had to clear to survive. Stored rather than assumed,
    # so `usable` still means something when a caller raises the threshold.
    min_observed_share: float = 0.5

    @property
    def usable(self) -> bool:
        return self.observed_share >= self.min_observed_share

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "column": self.column,
            "n_sentinel": float(self.n_sentinel),
            "n_blank": float(self.n_blank),
            "observed": float(self.observed),
            "observed_share": self.observed_share,
            "mean_with_sentinel": self.mean_with_sentinel,
            "mean_observed": self.mean_observed,
            "longest_gap": float(self.longest_gap),
            "min_observed_share": self.min_observed_share,
            "usable": self.usable,
        }


@dataclass
class DatasetReport:
    """What the file contains, before anybody models it."""

    n_rows: int
    n_columns: int
    columns: list[ColumnReport] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    rows_with_any_sentinel: int = 0
    total_sentinel_cells: int = 0
    # What survived. These describe the frame handed back to the caller; the
    # fields above describe the file as it was read. Conflating the two is how a
    # report ends up claiming 9,357 clean hours from a file that only has 7,620.
    n_rows_retained: int = 0
    n_rows_dropped_to_gaps: int = 0
    interpolated_cells: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""

    @property
    def retained_columns(self) -> list[str]:
        """Columns present in the cleaned frame."""
        dropped = set(self.dropped)
        return [column.column for column in self.columns if column.column not in dropped]

    @property
    def usable_columns(self) -> list[str]:
        """Columns observed often enough to model with."""
        return [column.column for column in self.columns if column.usable]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.columns])

    def summary(self) -> str:
        share = self.rows_with_any_sentinel / self.n_rows if self.n_rows else 0.0
        text = (
            f"{self.n_rows:,} rows x {self.n_columns} columns as read; "
            f"{self.total_sentinel_cells:,} sentinel cells across "
            f"{self.rows_with_any_sentinel:,} rows ({share:.1%})"
        )
        if self.dropped:
            text += f"; dropped {', '.join(self.dropped)}"
        if self.n_rows_retained:
            text += (
                f"; {self.n_rows_retained:,} rows retained "
                f"({self.n_rows_dropped_to_gaps:,} dropped to long gaps, "
                f"{self.interpolated_cells:,} cells interpolated)"
            )
        return text


def _longest_gap(mask: np.ndarray) -> int:
    """Longest run of consecutive missing values.

    A column can be 80% observed and still unusable if the missing fifth is one
    unbroken stretch: interpolating across it invents a week of data.
    """
    longest = current = 0
    for missing in mask:
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return longest


def missingness_report(
    frame: pd.DataFrame,
    *,
    dropped: list[str] | None = None,
    min_observed_share: float = 0.5,
) -> DatasetReport:
    """Measure how much of each column is real, counting the sentinel as missing."""
    numeric = frame.select_dtypes(include=[np.number])
    sentinel_mask = numeric == MISSING_SENTINEL

    report = DatasetReport(
        n_rows=len(frame),
        n_columns=len(frame.columns),
        dropped=list(dropped or []),
        rows_with_any_sentinel=int(sentinel_mask.any(axis=1).sum()),
        total_sentinel_cells=int(sentinel_mask.to_numpy().sum()),
    )

    for column in numeric.columns:
        values = numeric[column]
        is_sentinel = values == MISSING_SENTINEL
        is_blank = values.isna()
        missing = is_sentinel | is_blank
        observed = values[~missing]
        report.columns.append(
            ColumnReport(
                column=str(column),
                n_rows=len(values),
                n_sentinel=int(is_sentinel.sum()),
                n_blank=int(is_blank.sum()),
                observed=int(len(observed)),
                observed_share=float(len(observed) / len(values)) if len(values) else 0.0,
                mean_with_sentinel=float(values.mean()),
                mean_observed=float(observed.mean()) if len(observed) else float("nan"),
                longest_gap=_longest_gap(missing.to_numpy()),
                min_observed_share=min_observed_share,
            )
        )
    return report


def load_air_quality(
    path: str | Path,
    *,
    min_observed_share: float = 0.5,
    interpolate_limit: int = 6,
    target: str = "NO2(GT)",
) -> tuple[pd.DataFrame, DatasetReport]:
    """Read the file, convert the sentinel, and report what survived.

    ``min_observed_share`` drops a column observed less often than this. The
    default of 0.5 removes ``NMHC(GT)``, which is 89.1% sentinel - a column that
    exists in the file and does not exist in any useful sense.

    ``interpolate_limit`` bounds how long a gap may be bridged, in hours. Short
    gaps in an hourly pollutant series are reasonable to interpolate; long ones
    are not, and the rows spanning them are dropped instead. Interpolating a
    two-day outage produces a smooth curve that a model will happily learn and
    that never happened.
    """
    file = Path(path)
    if not file.is_file():
        raise DataError(f"no data file at {file}")

    # European conventions: semicolon separator, comma decimal mark.
    frame = pd.read_csv(file, sep=";", decimal=",")
    frame = frame.loc[:, ~frame.columns.str.contains("^Unnamed")]
    # Trailing blank rows the exporter appends.
    frame = frame.dropna(how="all").reset_index(drop=True)
    if frame.empty:
        raise DataError(f"{file} contains no rows")
    if target not in frame.columns:
        raise DataError(f"{file} has no column {target!r}; found {list(frame.columns)}")

    before = missingness_report(frame, min_observed_share=min_observed_share)
    logger.info(
        "file_read",
        rows=before.n_rows,
        sentinel_cells=before.total_sentinel_cells,
        rows_affected=before.rows_with_any_sentinel,
    )

    timestamps = _parse_timestamps(frame)

    numeric = frame.select_dtypes(include=[np.number]).copy()
    # The single line the original pipeline was missing.
    numeric = numeric.replace(MISSING_SENTINEL, np.nan)

    dropped = [report.column for report in before.columns if not report.usable]
    if target in dropped:
        raise DataQualityError(
            f"the target {target!r} is observed in only "
            f"{next(c.observed_share for c in before.columns if c.column == target):.1%} of rows; "
            "it cannot be forecast from this file"
        )
    numeric = numeric.drop(columns=dropped)

    numeric.index = pd.DatetimeIndex(timestamps)
    numeric = numeric.sort_index()

    # Bridge short outages; drop the rows spanning long ones.
    interpolated = numeric.interpolate(method="time", limit=interpolate_limit, limit_area="inside")
    remaining = interpolated.isna().any(axis=1)
    cleaned = interpolated[~remaining]
    # Cells that hold an interpolated value rather than a measurement, counted
    # on the rows that survive. Reporting it keeps the reader aware of how much
    # of the "clean" series was reconstructed.
    filled = int((numeric.isna() & interpolated.notna())[~remaining].to_numpy().sum())
    if cleaned.empty:
        raise DataQualityError(
            "no complete rows survive after bounded interpolation; the gaps in this "
            "file are longer than the configured limit"
        )

    after = missingness_report(frame, dropped=dropped, min_observed_share=min_observed_share)
    after.n_rows_retained = len(cleaned)
    after.n_rows_dropped_to_gaps = int(remaining.sum())
    after.interpolated_cells = filled
    after.first_timestamp = str(cleaned.index[0])
    after.last_timestamp = str(cleaned.index[-1])
    logger.info(
        "data_loaded",
        rows_in=len(frame),
        rows_out=len(cleaned),
        dropped_columns=dropped,
        rows_dropped_to_long_gaps=int(remaining.sum()),
        interpolated_cells=filled,
        target_mean_observed=round(
            next(c.mean_observed for c in after.columns if c.column == target), 3
        ),
    )
    return cleaned, after


def _parse_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Combine the separate Date and Time columns into one index.

    Times are written ``18.00.00``. Left to infer, pandas reads the dots
    inconsistently across rows, which silently reorders an hourly series - and a
    forecasting model trained on a shuffled clock is measuring nothing.
    """
    if "Date" not in frame.columns or "Time" not in frame.columns:
        raise DataError("the file has no Date and Time columns to index on")
    combined = frame["Date"].astype(str).str.strip() + " " + frame["Time"].astype(str).str.strip()
    timestamps = pd.to_datetime(combined, format="%d/%m/%Y %H.%M.%S", errors="coerce")
    if timestamps.isna().any():
        bad = int(timestamps.isna().sum())
        raise DataError(f"{bad} rows have a Date/Time that does not parse as %d/%m/%Y %H.%M.%S")
    if not timestamps.is_monotonic_increasing:
        logger.warning("timestamps_out_of_order", first=str(timestamps.iloc[0]))
    return timestamps
