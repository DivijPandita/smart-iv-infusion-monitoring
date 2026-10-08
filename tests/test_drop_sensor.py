"""Tests for the virtual IR drop counter."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulation.drop_sensor import simulate_drop_sensor
from src.simulation.infusion import generate_true_infusion

PERFECT = {
    "drop_sensor.miss_probability": 0.0,
    "drop_sensor.false_detection_rate_per_min": 0.0,
    "drop_sensor.jitter_std_fraction": 0.0,
}


def make(overrides=None, scenario_name=None):
    cfg = load_config(overrides=overrides)
    truth = generate_true_infusion(cfg)
    scenario = cfg.failure_scenarios[scenario_name] if scenario_name else None
    return cfg, truth, simulate_drop_sensor(truth, cfg, scenario=scenario)


def count_between(ts, lo, hi):
    ts = np.asarray(ts)
    return int(np.searchsorted(ts, hi) - np.searchsorted(ts, lo))


def true_count_between(truth, lo, hi):
    cum = np.interp([lo, hi], truth["timestamp"], truth["true_drops_cum"])
    return cum[1] - cum[0]


def test_columns_sorted_and_flags():
    _, _, df = make()
    assert {"timestamp", "drop_detected", "event_type",
            "true_drop_rate_per_min", "true_drop_factor"} <= set(df.columns)
    assert (np.diff(df["timestamp"]) >= 0).all()
    assert (df["drop_detected"] == 1).all()
    assert (df["timestamp"] >= 0).all()


def test_perfect_sensor_counts_every_drop():
    _, truth, df = make(PERFECT)
    expected = int(np.floor(truth["true_drops_cum"].iloc[-1] + 1e-6))   # 7250
    assert len(df) == expected
    assert (df["event_type"] == "real").all()


def test_perfect_sensor_drops_sit_on_integer_crossings():
    _, truth, df = make(PERFECT)
    cum_at_detection = np.interp(df["timestamp"], truth["timestamp"], truth["true_drops_cum"])
    assert np.allclose(cum_at_detection, np.arange(1, len(df) + 1), atol=1e-6)


def test_miss_probability_removes_expected_fraction():
    _, truth, df = make({**PERFECT, "drop_sensor.miss_probability": 0.2})
    full = int(np.floor(truth["true_drops_cum"].iloc[-1] + 1e-6))
    assert len(df) / full == pytest.approx(0.8, abs=0.03)


def test_false_detections_follow_configured_rate():
    cfg, truth, df = make({**PERFECT, "drop_sensor.false_detection_rate_per_min": 2.0})
    n_false = (df["event_type"] == "false").sum()
    expected = 2.0 * truth["timestamp"].iloc[-1] / 60.0
    assert n_false == pytest.approx(expected, rel=0.2)


def test_jitter_makes_intervals_irregular():
    _, _, perfect = make(PERFECT)
    _, _, jittered = make({**PERFECT, "drop_sensor.jitter_std_fraction": 0.05})

    def interval_std(df):
        ts = df.loc[(df.timestamp > 30) & (df.timestamp < 240), "timestamp"].to_numpy()
        return np.diff(ts).std()

    assert interval_std(jittered) > 3 * interval_std(perfect)


def test_reproducible_and_seed_dependent():
    _, _, a = make()
    _, _, b = make()
    pd.testing.assert_frame_equal(a, b)
    _, _, c = make({"simulation.random_seed": 7})
    assert len(a) != len(c) or not np.allclose(a["timestamp"].to_numpy()[:100],
                                               c["timestamp"].to_numpy()[:100])


def test_missing_drops_scenario_only_affects_its_window():
    overrides = {"drop_sensor.false_detection_rate_per_min": 0.0,
                 "drop_sensor.jitter_std_fraction": 0.0}
    cfg, truth, df = make(overrides, scenario_name="ir_missing_drops")   # window 60-100 min
    inside = count_between(df.timestamp, 3700, 5900) / true_count_between(truth, 3700, 5900)
    outside = count_between(df.timestamp, 600, 3000) / true_count_between(truth, 600, 3000)
    assert inside == pytest.approx(0.80, abs=0.04)
    assert outside == pytest.approx(0.97, abs=0.03)


def test_overcount_scenario_inflates_counts_in_window():
    overrides = {**PERFECT}
    cfg, truth, df = make(overrides, scenario_name="sensors_disagree")   # scale 1.25
    inside = count_between(df.timestamp, 3700, 5900) / true_count_between(truth, 3700, 5900)
    outside = count_between(df.timestamp, 600, 3000) / true_count_between(truth, 600, 3000)
    assert inside == pytest.approx(1.25, abs=0.05)
    assert outside == pytest.approx(1.00, abs=0.02)
    assert (df["event_type"] == "duplicate").sum() > 0


def test_window_counts_are_close_to_truth_with_default_noise():
    _, truth, df = make()
    edges = np.arange(0.0, truth["timestamp"].iloc[-1] + 1, 60.0)
    detected = np.histogram(df["timestamp"], bins=edges)[0]
    true = np.diff(np.interp(edges, truth["timestamp"], truth["true_drops_cum"]))
    err = detected - true
    assert abs(err.mean()) < 1.5
    assert err.std() < 2.5