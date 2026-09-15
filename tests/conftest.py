"""Shared fixtures.

The synthetic series is built to look like the real file: European separators,
split Date and Time columns, a -200 sentinel scattered through it, and a daily
cycle a forecaster can actually learn. Tests that need the real data are marked
``slow`` and skip when it is absent, so the suite runs on a clean checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAW_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "AirQuality.csv"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260101)


@pytest.fixture
def hourly_series(rng: np.random.Generator) -> pd.DataFrame:
    """A clean hourly frame with a daily cycle, indexed by time."""
    hours = 2000
    index = pd.date_range("2004-03-10 18:00:00", periods=hours, freq="h")
    clock = np.arange(hours)
    daily = 40.0 * np.sin(2 * np.pi * clock / 24.0)
    weekly = 8.0 * np.sin(2 * np.pi * clock / (24.0 * 7))
    target = 110.0 + daily + weekly + rng.normal(0.0, 6.0, hours)
    return pd.DataFrame(
        {
            "NO2(GT)": target,
            "T": 18.0 + 6.0 * np.sin(2 * np.pi * (clock - 6) / 24.0) + rng.normal(0, 1.0, hours),
            "RH": 50.0 + rng.normal(0, 5.0, hours),
        },
        index=index,
    )


@pytest.fixture
def raw_like_csv(tmp_path: Path, rng: np.random.Generator) -> Path:
    """A file shaped like the UCI export, sentinel and all."""
    hours = 900
    index = pd.date_range("2004-03-10 18:00:00", periods=hours, freq="h")
    clock = np.arange(hours)

    frame = pd.DataFrame(
        {
            "Date": index.strftime("%d/%m/%Y"),
            "Time": index.strftime("%H.%M.%S"),
            "CO(GT)": 2.0 + 0.5 * np.sin(2 * np.pi * clock / 24.0) + rng.normal(0, 0.2, hours),
            "NO2(GT)": 110.0 + 40.0 * np.sin(2 * np.pi * clock / 24.0) + rng.normal(0, 6.0, hours),
            "T": 18.0 + 6.0 * np.sin(2 * np.pi * clock / 24.0),
            "RH": 50.0 + rng.normal(0, 5.0, hours),
            # A column that barely exists, to be dropped rather than imputed.
            "NMHC(GT)": np.where(clock < 80, 200.0 + rng.normal(0, 10, hours), -200.0),
        }
    )

    # Scattered short outages, plus one long one that must not be bridged.
    for column in ("CO(GT)", "NO2(GT)"):
        holes = rng.choice(hours, size=40, replace=False)
        frame.loc[holes, column] = -200.0
    frame.loc[400:460, "NO2(GT)"] = -200.0

    path = tmp_path / "AirQuality.csv"
    # The exporter's trailing blank rows and semicolon/comma conventions.
    text = frame.to_csv(sep=";", index=False, decimal=",")
    path.write_text(text + ";;;;;;\n;;;;;;\n", encoding="utf-8")
    return path


def requires_raw_data() -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not RAW_DATA.is_file(),
        reason="the UCI file is not present; run scripts/download_data.py",
    )
