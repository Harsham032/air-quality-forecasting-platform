"""Air quality forecasting on the UCI Air Quality series.

The package exists to answer one question honestly: does a deep sequence model
forecast hourly pollutant concentrations better than persistence does? Getting a
trustworthy answer turns out to depend less on the architecture than on two
things the literature routinely gets wrong - treating the dataset's -200 missing
sentinel as a measurement, and reporting error on a scaled axis instead of in
micrograms per cubic metre.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
