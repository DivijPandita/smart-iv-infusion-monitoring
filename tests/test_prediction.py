"""Tests for remaining volume/time, uncertainty, alerts and the pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.pipeline import evaluate_predictions, run_pipeline
from src.prediction.alerts import (
    classify_remaining_time, classify_remaining_time_array, disagreement_flags, evaluate_alert,
)
from src.prediction.remaining_time import (
    completion_time, format_hms, format_hours_minutes, remaining_time_min,
    remaining_volume_ml, remaining_volume_series,
)
from src.prediction.uncertainty import (
    confidence_interval, effective_flow_std, remaining_time_sigma, sensor_disagreement,
)
from src.simulation.generate_data import generate_dataset


# ---------------- remaining volume ----------------
def test_remaining_volume_from_weight():
    cfg = load_config()
    assert remaining_volume_ml(325.0, cfg) == pytest.approx(300.0)      # (325 - 25 g) / 1


def test_remaining_volume_density_and_never_negative():
    cfg = load_config(overrides={"bag.fluid_density_g_per_ml": 1.25})
    assert remaining_volume_ml(525.0, cfg) == pytest.approx(400.0)
    assert remaining_volume_ml(10.0, cfg) == 0.0


def test_volume_series_fallback_when_weight_missing():
    cfg = load_config()
    w = pd.DataFrame({"t_start_s": [0.0, 60.0, 120.0], "t_end_s": [60.0, 120.0, 180.0],
                      "volume_ml": [497.0, np.nan, np.nan]})
    vol, ok = remaining_volume_series(w, [2.5, 2.5, 2.5], cfg)
    assert vol == pytest.approx([497.0, 494.5, 492.0])
    assert list(ok) == [True, False, False]


# ---------------- remaining time ----------------
def test_remaining_time_basic():
    assert remaining_time_min(300.0, 2.5) == pytest.approx(120.0)
    assert remaining_time_min([300.0, 150.0], [2.5, 3.0]) == pytest.approx([120.0, 50.0])


def test_remaining_time_flow_floor_and_empty():
    assert np.isfinite(remaining_time_min(100.0, 0.0, 0.05))            # no division by zero
    assert remaining_time_min(100.0, 0.0, 0.05) == pytest.approx(2000.0)
    assert remaining_time_min(-5.0, 2.5) == 0.0


def test_time_formatting():
    assert format_hms(113 + 1 / 3) == "01:53:20"
    assert format_hms(float("nan")) == "--:--:--"
    assert format_hours_minutes(113) == "1 h 53 min"
    assert format_hours_minutes(45) == "45 min"
    assert format_hours_minutes(60) == "1 h 00 min"


def test_completion_time():
    now = pd.Timestamp("2025-01-01 12:00:00")
    assert completion_time(now, 90.0) == pd.Timestamp("2025-01-01 13:30:00")


# ---------------- uncertainty ----------------
def test_sigma_t_matches_delta_method():
    # V=300, F=2.5, sigma_F=0.1 -> |dT/dF| = 300/6.25 = 48 -> sigma_T = 4.8
    assert remaining_time_sigma(300, 2.5, 0.0, 0.1) == pytest.approx(4.8)
    # sigma_V = 1 adds (1/2.5)^2 = 0.16 to the variance 23.04
    assert remaining_time_sigma(300, 2.5, 1.0, 0.1) == pytest.approx(np.sqrt(23.2))


def test_sigma_t_matches_monte_carlo():
    rng = np.random.default_rng(0)
    f = rng.normal(2.5, 0.1, 400_000)
    v = rng.normal(300.0, 1.0, 400_000)
    assert (v / f).std() == pytest.approx(remaining_time_sigma(300, 2.5, 1.0, 0.1), rel=0.05)


def test_confidence_interval_symmetric_and_clipped():
    lo, hi = confidence_interval(120.0, 5.0, 1.96)
    assert (lo, hi) == pytest.approx((110.2, 129.8))
    lo, hi = confidence_interval(3.0, 5.0, 1.96)
    assert lo == 0.0 and hi == pytest.approx(12.8)


def test_interval_widens_with_flow_uncertainty_and_volume():
    assert remaining_time_sigma(300, 2.5, 1.0, 0.2) > remaining_time_sigma(300, 2.5, 1.0, 0.1)
    assert remaining_time_sigma(500, 2.5, 1.0, 0.1) > remaining_time_sigma(100, 2.5, 1.0, 0.1)


def test_disagreement_metric():
    diff, rel = sensor_disagreement(2.5, 2.0, 2.5)
    assert diff == pytest.approx(0.5) and rel == pytest.approx(0.2)
    diff, rel = sensor_disagreement(np.nan, 2.0, 2.5)
    assert np.isnan(diff) and np.isnan(rel)


def test_effective_flow_std_grows_with_disagreement():
    cfg = load_config(overrides={"prediction.future_flow_std_ml_min": 0.0,
                                 "prediction.disagreement_inflation": 2.0})
    assert effective_flow_std(0.1, 0.0, cfg) == pytest.approx(0.1)
    assert effective_flow_std(0.1, 0.25, cfg) == pytest.approx(0.15)     # 0.1 * (1 + 2*0.25)
    assert effective_flow_std(0.1, np.nan, cfg) == pytest.approx(0.1)    # unknown -> no inflation
    cfg2 = load_config(overrides={"prediction.future_flow_std_ml_min": 0.25})
    assert effective_flow_std(0.1, 0.0, cfg2) == pytest.approx(np.sqrt(0.01 + 0.0625))


# ---------------- alerts ----------------
def test_alert_thresholds():
    assert classify_remaining_time(45, 30, 10) == "NORMAL"
    assert classify_remaining_time(31, 30, 10) == "NORMAL"
    assert classify_remaining_time(30, 30, 10) == "WARNING"
    assert classify_remaining_time(10.5, 30, 10) == "WARNING"
    assert classify_remaining_time(10, 30, 10) == "CRITICAL"
    assert classify_remaining_time(0, 30, 10) == "CRITICAL"
    assert classify_remaining_time(float("nan"), 30, 10) == "UNKNOWN"
    arr = classify_remaining_time_array([45, 20, 5, np.nan], 30, 10)
    assert list(arr) == ["NORMAL", "WARNING", "CRITICAL", "UNKNOWN"]


def test_disagreement_alert_and_message():
    cfg = load_config()
    state = evaluate_alert(100.0, 90.0, 0.30, cfg)
    assert state.status == "NORMAL" and state.disagreement_alert
    assert "disagree" in state.message.lower()
    assert not evaluate_alert(100.0, 90.0, 0.10, cfg).disagreement_alert
    assert list(disagreement_flags([0.1, 0.3, np.nan], 0.25)) == [False, True, False]


def test_lower_bound_option_alerts_earlier():
    normal = load_config()
    cautious = load_config(overrides={"alerts.use_lower_bound": True})
    assert evaluate_alert(35.0, 25.0, 0.0, normal).status == "NORMAL"
    assert evaluate_alert(35.0, 25.0, 0.0, cautious).status == "WARNING"


# ---------------- pipeline ----------------
@pytest.fixture(scope="module")
def default_run():
    cfg = load_config()
    ds = generate_dataset(cfg)
    return cfg, ds, run_pipeline(cfg, ds.load_cell, ds.drops)


def test_pipeline_outputs_are_consistent(default_run):
    cfg, ds, res = default_run
    need = {"remaining_volume_ml", "remaining_time_min", "ci_low_min", "ci_high_min",
            "completion_time", "alert_status", "disagree_rel", "flow_ekf", "drop_factor_ekf"}
    assert need <= set(res.columns)
    assert (res["ci_low_min"] <= res["remaining_time_min"] + 1e-9).all()
    assert (res["remaining_time_min"] <= res["ci_high_min"] + 1e-9).all()
    assert (res["sigma_T_min"] > 0).all()
    assert set(res["alert_status"]) <= {"NORMAL", "WARNING", "CRITICAL"}
    assert res["timestamp"].is_monotonic_increasing
    assert (res["completion_low"] <= res["completion_time"]).all()
    assert (res["completion_time"] <= res["completion_high"]).all()


def test_pipeline_accuracy_on_simulated_data(default_run):
    cfg, ds, res = default_run
    assert res["alert_status"].iloc[0] == "NORMAL"
    assert res["alert_status"].iloc[-1] == "CRITICAL"
    assert res["remaining_volume_ml"].iloc[-1] < 6.0
    assert res["remaining_time_min"].iloc[-1] < 3.0
    m = evaluate_predictions(res)
    assert m["flow_rmse"] < 0.25
    assert m["time_median_abs_err_min"] < 20.0
    assert 0.0 <= m["ci_coverage"] <= 1.0


def test_disagreement_rises_when_drop_sensor_overcounts():
    cfg = load_config()
    ds = generate_dataset(cfg, "sensors_disagree")                      # 60-100 min, drops x1.25
    res = run_pipeline(cfg, ds.load_cell, ds.drops)
    t = res["t_end_s"] / 60.0
    before = res.loc[(t > 30) & (t <= 60), "disagree_rel"].mean()
    onset = res.loc[(t > 60) & (t <= 70), "disagree_rel"].mean()
    assert onset > 1.5 * before


def test_pipeline_does_not_use_truth_columns(default_run):
    cfg, ds, res = default_run
    bare = run_pipeline(cfg, ds.load_cell[["timestamp", "measured_weight_g"]], ds.drops[["timestamp"]])
    pd.testing.assert_series_equal(res["remaining_time_min"], bare["remaining_time_min"])
    assert not any(c.startswith("ref_") for c in bare.columns)