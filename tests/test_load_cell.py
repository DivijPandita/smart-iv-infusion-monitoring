"""Tests for the virtual load cell."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulation.infusion import generate_true_infusion
from src.simulation.load_cell import simulate_load_cell

# Switch off every impairment except the one under test.
QUIET = {
    "load_cell.noise_std_g": 0.0,
    "load_cell.vibration_amplitude_g": 0.0,
    "load_cell.bias_std_g": 0.0,
    "load_cell.spike_probability": 0.0,
    "load_cell.dropout_probability": 0.0,
    "load_cell.resolution_g": 0.0,
}


def make(overrides=None, scenario_name=None):
    cfg = load_config(overrides=overrides)
    truth = generate_true_infusion(cfg)
    scenario = cfg.failure_scenarios[scenario_name] if scenario_name else None
    return cfg, truth, simulate_load_cell(truth, cfg, scenario=scenario)


def test_columns_and_timestamps():
    _, truth, df = make()
    assert {"timestamp", "true_weight_g", "measured_weight_g", "true_flow_ml_min"} <= set(df.columns)
    assert (np.diff(df["timestamp"]) > 0).all()                 # strictly increasing
    assert set(df["timestamp"]) <= set(truth["timestamp"])      # only real sample times
    assert df["timestamp"].iloc[0] == 0.0


def test_no_impairments_gives_perfect_measurement():
    _, _, df = make(QUIET)
    assert np.allclose(df["measured_weight_g"], df["true_weight_g"])


def test_white_noise_std_matches_config():
    _, _, df = make({**QUIET, "load_cell.noise_std_g": 0.2})
    residual = df["measured_weight_g"] - df["true_weight_g"]
    assert residual.std() == pytest.approx(0.2, rel=0.1)
    assert abs(residual.mean()) < 0.02


def test_quantisation_snaps_to_adc_step():
    _, _, df = make({"load_cell.resolution_g": 0.05})
    ratio = df["measured_weight_g"] / 0.05
    assert np.allclose(ratio, np.round(ratio), atol=1e-6)


def test_spikes_create_large_outliers():
    _, _, df = make({**QUIET, "load_cell.spike_probability": 0.05})
    residual = (df["measured_weight_g"] - df["true_weight_g"]).abs()
    assert residual.max() > 3.0


def test_dropout_removes_samples_but_keeps_first():
    _, truth, df = make({**QUIET, "load_cell.dropout_probability": 0.1})
    assert 0.85 * len(truth) < len(df) < 0.95 * len(truth)
    assert df["timestamp"].iloc[0] == 0.0


def test_reproducible_and_seed_dependent():
    _, _, a = make()
    _, _, b = make()
    pd.testing.assert_frame_equal(a, b)
    _, _, c = make({"simulation.random_seed": 7})
    n = min(len(a), len(c))
    assert not np.allclose(a["measured_weight_g"].iloc[:n], c["measured_weight_g"].iloc[:n])


def test_noise_multiplier_scenario_raises_noise_only_inside_window():
    # first difference removes the slow bias; spikes/vibration disabled for a clean comparison
    cfg, _, df = make({**QUIET, "load_cell.noise_std_g": 0.2}, scenario_name="load_cell_noisy")
    t_min = df["timestamp"] / 60.0
    resid_diff = np.diff((df["measured_weight_g"] - df["true_weight_g"]).to_numpy())
    t_mid = t_min.to_numpy()[1:]
    inside = resid_diff[(t_mid > 65) & (t_mid < 95)].std()     # scenario: 60-100 min
    outside = resid_diff[(t_mid > 10) & (t_mid < 40)].std()
    assert inside / outside > 5.0


def test_default_measurement_is_close_to_truth_and_non_negative():
    _, _, df = make()
    err = (df["measured_weight_g"] - df["true_weight_g"]).abs()
    assert err.median() < 0.6
    assert (df["measured_weight_g"] >= 0).all()