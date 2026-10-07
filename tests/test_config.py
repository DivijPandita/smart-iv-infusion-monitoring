"""Tests for the configuration system."""
import pytest

from src.config import ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.bag.initial_volume_ml == 500.0
    assert cfg.ekf.sensor_mode == "fusion"


def test_derived_initial_weight():
    cfg = load_config()
    # 500 mL * 1 g/mL + 25 g bag
    assert cfg.initial_total_weight_g == pytest.approx(525.0)


def test_override_applies():
    cfg = load_config(overrides={"drop_sensor.actual_drop_factor": 14.2})
    assert cfg.drop_sensor.actual_drop_factor == 14.2


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(overrides={"load_cell.noise_std": 0.5})  # typo: missing _g


def test_invalid_values_report_all_errors():
    with pytest.raises(ConfigError) as exc:
        load_config(overrides={"bag.initial_volume_ml": -5, "alerts.critical_min": 99})
    message = str(exc.value)
    assert "initial_volume_ml" in message
    assert "critical_min" in message


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_expected_infusion_time_is_sensible():
    cfg = load_config()
    # mean flow 2.625 mL/min -> about 190 min for 500 mL
    assert 185 < cfg.expected_infusion_time_min() < 195


def test_failure_scenarios_loaded():
    cfg = load_config()
    assert {"load_cell_noisy", "ir_missing_drops", "sensors_disagree"} <= set(cfg.failure_scenarios)
    assert cfg.failure_scenarios["ir_missing_drops"].drop_miss_probability == 0.20