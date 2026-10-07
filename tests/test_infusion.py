"""Tests for the ground-truth infusion simulator."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulation.infusion import generate_true_infusion

CONSTANT = {"flow.profile": "constant", "flow.variability_std_ml_min": 0.0}


def test_columns_and_time_grid():
    df = generate_true_infusion(load_config())
    expected = {"timestamp", "true_flow_ml_min", "true_infused_ml", "true_volume_ml",
                "true_weight_g", "true_drop_factor", "true_drop_rate_per_min",
                "true_drops_cum", "true_remaining_time_min"}
    assert expected <= set(df.columns)
    assert df["timestamp"].iloc[0] == 0.0
    assert np.allclose(np.diff(df["timestamp"]), 1.0)          # 1 Hz default


def test_initial_weight_includes_empty_bag():
    df = generate_true_infusion(load_config())
    assert df["true_weight_g"].iloc[0] == pytest.approx(525.0)  # 500 fluid + 25 bag
    assert df["true_volume_ml"].iloc[0] == pytest.approx(500.0)


def test_weight_never_increases_and_flow_positive_until_empty():
    df = generate_true_infusion(load_config())
    assert (np.diff(df["true_weight_g"]) <= 1e-9).all()
    assert (df["true_flow_ml_min"].iloc[:-1] > 0).all()


def test_stops_when_empty():
    df = generate_true_infusion(load_config())
    assert df["true_weight_g"].iloc[-1] == pytest.approx(25.0)   # only the bag is left
    assert df["true_volume_ml"].iloc[-1] == 0.0
    assert 175 < df["timestamp"].iloc[-1] / 60 < 205             # about 190 min


def test_constant_flow_empties_in_exactly_volume_over_flow():
    cfg = load_config(overrides=CONSTANT)                        # 500 mL / 2.5 = 200 min
    df = generate_true_infusion(cfg)
    assert abs(df["timestamp"].iloc[-1] - 200 * 60) <= 1.5


def test_drops_equal_volume_times_drop_factor():
    cfg = load_config()
    df = generate_true_infusion(cfg)
    expected = cfg.bag.initial_volume_ml * cfg.drop_sensor.actual_drop_factor
    assert df["true_drops_cum"].iloc[-1] == pytest.approx(expected, rel=1e-6)
    assert np.allclose(df["true_drop_rate_per_min"],
                       df["true_flow_ml_min"] * df["true_drop_factor"])


def test_piecewise_segments_match_example_profile():
    df = generate_true_infusion(load_config())
    t = df["timestamp"] / 60
    for lo, hi, target in [(1, 4, 2.5), (6, 9, 3.0), (11, 14, 2.2), (16, 19, 2.8)]:
        mean = df.loc[(t >= lo) & (t <= hi), "true_flow_ml_min"].mean()
        assert mean == pytest.approx(target, abs=0.25)


def test_flow_changes_are_smooth():
    df = generate_true_infusion(load_config())
    assert np.abs(np.diff(df["true_flow_ml_min"].iloc[:-1])).max() < 0.1


def test_reproducible_with_same_seed_and_different_with_other_seed():
    a = generate_true_infusion(load_config())
    b = generate_true_infusion(load_config())
    pd.testing.assert_frame_equal(a, b)
    c = generate_true_infusion(load_config(overrides={"simulation.random_seed": 7}))
    n = min(len(a), len(c))
    assert not np.allclose(a["true_flow_ml_min"].iloc[:n], c["true_flow_ml_min"].iloc[:n])


def test_smooth_profile_stays_in_expected_band():
    cfg = load_config(overrides={"flow.profile": "smooth", "flow.variability_std_ml_min": 0.0})
    df = generate_true_infusion(cfg).iloc[:-1]
    assert df["true_flow_ml_min"].min() >= 2.5 - 0.4 - 1e-6
    assert df["true_flow_ml_min"].max() <= 2.5 + 0.4 + 1e-6


def test_run_past_empty_when_stop_disabled():
    cfg = load_config(overrides={**CONSTANT, "simulation.stop_when_empty": False})
    df = generate_true_infusion(cfg)
    assert len(df) == 240 * 60 + 1
    assert df["true_flow_ml_min"].iloc[-1] == 0.0
    assert df["true_weight_g"].iloc[-1] == pytest.approx(25.0)