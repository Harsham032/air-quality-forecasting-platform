"""Configuration.

Two kinds of setting, deliberately kept apart.

*Experiment configuration* - what a run does - lives in YAML and is overridable
per run, so a result is reproducible from a file plus a seed rather than from
whatever was typed into a notebook cell that afternoon.

*Deployment settings* - where things live - come from ``AQF_*`` environment
variables. Credentials are never read from YAML, because YAML is a file people
commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError
from .models.deep import ARCHITECTURES


class RunConfig(BaseModel):
    name: str = "default"
    seed: int = 20260101
    output_dir: str = "reports"
    figure_dir: str = "figures"
    n_threads: int | None = None


class DataConfig(BaseModel):
    """Where the series comes from and how much of it survives cleaning."""

    path: str = "data/raw/AirQuality.csv"
    target: str = "NO2(GT)"
    # A column observed less often than this is dropped rather than imputed.
    min_observed_share: float = 0.5
    # How many consecutive hours may be bridged by interpolation.
    interpolate_limit: int = 6

    @field_validator("min_observed_share")
    @classmethod
    def _in_unit_interval(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("min_observed_share must lie in (0, 1]")
        return value

    @field_validator("interpolate_limit")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("interpolate_limit cannot be negative")
        return value


class SplitConfig(BaseModel):
    """The chronological division. Never random - see docs/methodology.md."""

    lookback: int = 24
    horizon: int = 1
    train_share: float = 0.70
    validation_share: float = 0.15

    @field_validator("lookback", "horizon")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1 hour")
        return value

    @model_validator(mode="after")
    def _shares_leave_a_test_set(self) -> SplitConfig:
        if not 0.0 < self.train_share < 1.0 or not 0.0 < self.validation_share < 1.0:
            raise ValueError("shares must lie strictly between 0 and 1")
        if self.train_share + self.validation_share >= 1.0:
            raise ValueError("train and validation shares leave nothing for test")
        return self


class ModelConfig(BaseModel):
    """Capacity and optimisation, shared by every architecture.

    One set of hyperparameters for all four networks is a deliberate choice: a
    comparison where each architecture got a different budget measures the
    budget.
    """

    architectures: tuple[str, ...] = ARCHITECTURES
    units: int = 64
    filters: int = 64
    kernel_size: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    patience: int = 8

    @field_validator("architectures")
    @classmethod
    def _known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one architecture must be listed")
        unknown = [name for name in value if name not in ARCHITECTURES]
        if unknown:
            raise ValueError(
                f"unknown architectures {unknown}; expected any of {list(ARCHITECTURES)}"
            )
        return tuple(value)

    @field_validator("dropout")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        return value

    @field_validator("units", "filters", "kernel_size", "epochs", "batch_size", "patience")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be positive")
        return value


class BaselineConfig(BaseModel):
    """The bar the networks have to clear."""

    seasonal_period: int = 24
    ridge_alpha: float = 1.0
    include_ridge: bool = True

    @field_validator("seasonal_period")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("seasonal_period must be at least 1 hour")
        return value


class EvaluationConfig(BaseModel):
    unit: str = "ug/m3"
    # Guards against reporting an error on a scaled axis. Leave it on.
    require_physical_units: bool = True
    # The baseline every skill score is measured against.
    reference_baseline: str = "persistence"


class ExperimentConfig(BaseModel):
    """Everything a run needs, loaded from YAML."""

    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _kernel_fits_the_window(self) -> ExperimentConfig:
        convolutional = {"cnn", "cnn_lstm"} & set(self.models.architectures)
        if convolutional and self.models.kernel_size > self.split.lookback:
            raise ValueError(
                f"kernel_size {self.models.kernel_size} exceeds the {self.split.lookback}-hour "
                f"lookback, which {sorted(convolutional)} cannot build"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"no configuration file at {file}")
        try:
            payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"{file} is not valid YAML: {exc}") from exc
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ConfigurationError(f"{file} is not a valid configuration: {exc}") from exc

    def with_overrides(self, overrides: dict[str, str]) -> ExperimentConfig:
        """Apply ``section.key=value`` overrides from the command line.

        Values are parsed as YAML so numbers, booleans and lists arrive as the
        types the schema expects rather than as strings.
        """
        if not overrides:
            return self
        payload = self.model_dump()
        for dotted, raw in overrides.items():
            parts = dotted.split(".")
            if len(parts) < 2:
                raise ConfigurationError(f"override '{dotted}' must be section.key")
            cursor: Any = payload
            for part in parts[:-1]:
                if not isinstance(cursor, dict) or part not in cursor:
                    raise ConfigurationError(f"unknown configuration section '{part}'")
                cursor = cursor[part]
            if not isinstance(cursor, dict) or parts[-1] not in cursor:
                raise ConfigurationError(f"unknown configuration key '{dotted}'")
            try:
                cursor[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigurationError(f"override '{dotted}={raw}' is not valid: {exc}") from exc
        try:
            return ExperimentConfig.model_validate(payload)
        except Exception as exc:
            raise ConfigurationError(f"overrides produced an invalid configuration: {exc}") from exc


class Settings(BaseSettings):
    """Deployment settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="AQF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    log_level: str = "INFO"
    data_dir: str = "data"
    dataset_url: str = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    n_threads: int | None = None

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


def load_settings() -> Settings:
    """Read deployment settings from the environment."""
    return Settings()
