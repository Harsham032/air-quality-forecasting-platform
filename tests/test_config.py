"""Configuration must reject what would silently produce a wrong result."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aqf.config import ExperimentConfig, Settings, load_settings
from aqf.errors import ConfigurationError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "fast.yaml", "horizon-6h.yaml"])
def test_every_shipped_config_loads(name: str) -> None:
    config = ExperimentConfig.from_yaml(CONFIG_DIR / name)
    assert config.run.name
    assert config.models.architectures


def test_defaults_are_usable_without_a_file() -> None:
    config = ExperimentConfig()
    assert config.split.lookback == 24
    assert config.evaluation.require_physical_units is True


def test_overrides_are_typed_not_stringly() -> None:
    config = ExperimentConfig().with_overrides({"split.horizon": "6", "models.epochs": "2"})
    assert config.split.horizon == 6
    assert config.models.epochs == 2
    assert isinstance(config.split.horizon, int)


def test_a_list_override_parses(tmp_path: Path) -> None:
    config = ExperimentConfig().with_overrides({"models.architectures": "[gru, cnn]"})
    assert config.models.architectures == ("gru", "cnn")


def test_an_unknown_key_is_rejected_rather_than_ignored() -> None:
    """A typo in an override must fail, not silently run the default."""
    with pytest.raises(ConfigurationError, match="unknown configuration key"):
        ExperimentConfig().with_overrides({"split.lookbak": "48"})


def test_an_unknown_section_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown configuration section"):
        ExperimentConfig().with_overrides({"splitt.lookback": "48"})


def test_a_malformed_override_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="must be section.key"):
        ExperimentConfig().with_overrides({"lookback": "48"})


def test_an_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown architectures"):
        ExperimentConfig().with_overrides({"models.architectures": "[transformer]"})


def test_a_kernel_wider_than_the_window_is_rejected_before_training() -> None:
    """Caught in the config rather than 40 minutes into a run."""
    with pytest.raises(ConfigurationError, match="exceeds"):
        ExperimentConfig().with_overrides({"split.lookback": "2", "models.kernel_size": "5"})


def test_shares_that_leave_no_test_set_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="leave nothing for test"):
        ExperimentConfig().with_overrides(
            {"split.train_share": "0.9", "split.validation_share": "0.2"}
        )


def test_a_missing_config_file_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="no configuration file"):
        ExperimentConfig.from_yaml(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("run: {name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        ExperimentConfig.from_yaml(path)


def test_a_config_that_violates_the_schema_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"models": {"dropout": 1.5}}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a valid configuration"):
        ExperimentConfig.from_yaml(path)


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AQF_ENV", "production")
    monkeypatch.setenv("AQF_LOG_LEVEL", "WARNING")
    settings = Settings(_env_file=None)
    assert settings.is_production
    assert settings.log_level == "WARNING"


def test_settings_default_to_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AQF_ENV", raising=False)
    settings = Settings(_env_file=None)
    assert not settings.is_production


def test_load_settings_returns_settings() -> None:
    assert isinstance(load_settings(), Settings)


def test_the_env_example_matches_the_settings_class() -> None:
    """A documented variable that maps to no field does nothing, silently.

    Pydantic ignores unknown environment variables, so a renamed field leaves
    `.env.example` advertising a setting that has no effect. This catches that
    the moment it happens rather than when someone wonders why their override
    was not applied.
    """
    import re

    example = Path(__file__).resolve().parents[1] / ".env.example"
    documented = set(re.findall(r"^(AQF_\w+)=", example.read_text(encoding="utf-8"), flags=re.M))
    available = {f"AQF_{name.upper()}" for name in Settings.model_fields}

    assert documented - available == set(), "documented but not a setting"
    assert available - documented == set(), "a setting nobody documented"
