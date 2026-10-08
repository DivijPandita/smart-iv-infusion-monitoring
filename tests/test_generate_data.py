"""Tests for dataset generation, saving and inspection."""
import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.simulation.drop_sensor import simulate_drop_sensor
from src.simulation.generate_data import (
    DROP_COLUMNS, LOAD_CELL_COLUMNS, generate_dataset, inspect_dataset,
    load_dataset, save_dataset,
)
from src.simulation.load_cell import simulate_load_cell


def tmp_cfg(tmp_path):
    """Config whose CSV files go to a temporary folder instead of data/raw."""
    return load_config(overrides={
        "paths.raw_load_cell_file": str(tmp_path / "lc.csv"),
        "paths.raw_drop_file": str(tmp_path / "drops.csv"),
    })


def test_generate_matches_standalone_modules():
    cfg = load_config()
    ds = generate_dataset(cfg)
    pd.testing.assert_frame_equal(ds.load_cell, simulate_load_cell(ds.truth, cfg))
    pd.testing.assert_frame_equal(ds.drops, simulate_drop_sensor(ds.truth, cfg))


def test_unknown_scenario_rejected():
    with pytest.raises(ConfigError, match="Unknown scenario"):
        generate_dataset(load_config(), "does_not_exist")


def test_csv_has_expected_columns(tmp_path):
    cfg = tmp_cfg(tmp_path)
    save_dataset(generate_dataset(cfg), cfg)
    lc, dr = load_dataset(cfg)
    assert list(lc.columns) == LOAD_CELL_COLUMNS
    assert list(dr.columns) == DROP_COLUMNS


def test_save_and_load_roundtrip(tmp_path):
    cfg = tmp_cfg(tmp_path)
    ds = generate_dataset(cfg)
    save_dataset(ds, cfg)
    lc, dr = load_dataset(cfg)
    assert len(lc) == len(ds.load_cell) and len(dr) == len(ds.drops)
    assert np.allclose(lc["measured_weight_g"], ds.load_cell["measured_weight_g"], atol=1e-3)
    assert np.allclose(dr["timestamp"], ds.drops["timestamp"], atol=1e-3)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data"):
        load_dataset(tmp_cfg(tmp_path))


def test_inspection_checks_pass_on_default_data():
    cfg = load_config()
    ds = generate_dataset(cfg)
    checks = inspect_dataset(ds.load_cell, ds.drops, cfg)
    failed = [c for c in checks if not c.passed]
    assert not failed, failed


def test_scenario_changes_only_its_own_sensor_and_window():
    cfg = load_config()
    base = generate_dataset(cfg)
    noisy = generate_dataset(cfg, "load_cell_noisy")     # window: 60-100 min, load cell only

    pd.testing.assert_frame_equal(base.drops, noisy.drops)       # IR sensor untouched
    a, b = base.load_cell, noisy.load_cell
    assert len(a) == len(b)                                      # same dropouts: streams aligned
    t = a["timestamp"].to_numpy()
    wa, wb = a["measured_weight_g"].to_numpy(), b["measured_weight_g"].to_numpy()
    assert np.allclose(wa[t < 3500], wb[t < 3500])               # identical before the window
    assert not np.allclose(wa[(t > 3700) & (t < 5900)], wb[(t > 3700) & (t < 5900)])