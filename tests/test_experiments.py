"""Tests for the experiments module."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.experiments import (
    EKF_LABELS, RAW_DROP, RAW_LOAD, compare_one, run_all_modes, run_compare, run_failures,
    score_flow, score_time, window_report,
)
from src.simulation.generate_data import generate_dataset


def test_score_flow_known_values():
    m = score_flow([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], np.array([True, True, True]))
    assert m["flow_mae"] == pytest.approx(1.0)
    assert m["flow_rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["flow_bias"] == pytest.approx(1.0)
    # selection and NaN handling
    m = score_flow([1.0, np.nan, 3.0], [1.0, 1.0, 1.0], np.array([False, True, True]))
    assert m["flow_bias"] == pytest.approx(2.0)
    assert np.isnan(score_flow([np.nan], [1.0], np.array([True]))["flow_rmse"])


def test_score_time_ignores_nan_flow_and_short_remaining():
    volume = np.array([300.0, 300.0, 300.0, 5.0])
    flow = np.array([2.5, np.nan, 3.0, 2.5])
    t_true = np.array([120.0, 120.0, 100.0, 2.0])
    m = score_time(volume, flow, t_true, np.ones(4, dtype=bool))
    # window 1: NaN flow (skipped), window 3: true remaining < 10 min (skipped)
    # window 0: error 0, window 2: |300/3 - 100| = 0
    assert m["time_mae_min"] == pytest.approx(0.0)
    assert np.isnan(score_time(volume, flow, t_true, np.zeros(4, dtype=bool))["time_mae_min"])


@pytest.fixture(scope="module")
def comparison():
    cfg = load_config()
    return cfg, compare_one(cfg)


def test_compare_table_structure(comparison):
    cfg, (table, runs) = comparison
    assert list(table["method"]) == [RAW_LOAD, RAW_DROP, EKF_LABELS["load_cell_only"],
                                     EKF_LABELS["drop_only"], EKF_LABELS["fusion"]]
    assert {"flow_mae", "flow_rmse", "flow_bias", "time_mae_min", "df_final_error",
            "ci_coverage"} <= set(table.columns)
    assert set(runs) == {"fusion", "load_cell_only", "drop_only"}
    assert table["flow_rmse"].notna().all()


def test_only_fusion_learns_the_drop_factor(comparison):
    _, (table, _) = comparison
    t = table.set_index("method")
    start_error = 15.0 - 14.5                                       # initial guess minus truth
    assert t.loc[EKF_LABELS["load_cell_only"], "df_final_error"] == pytest.approx(start_error, abs=1e-6)
    assert t.loc[EKF_LABELS["drop_only"], "df_final_error"] == pytest.approx(start_error, abs=1e-3)
    assert abs(t.loc[EKF_LABELS["fusion"], "df_final_error"]) < start_error


def test_fusion_flow_is_competitive_with_the_raw_drop_counter(comparison):
    _, (table, _) = comparison
    t = table.set_index("method")
    assert t.loc[EKF_LABELS["fusion"], "flow_rmse"] < t.loc[RAW_DROP, "flow_rmse"] * 1.05


# ---------------- failure scenarios ----------------
@pytest.fixture(scope="module")
def failure_reports():
    cfg = load_config()
    base = generate_dataset(cfg)
    base_runs = run_all_modes(cfg, base.load_cell, base.drops)
    out = {}
    for name, sc in cfg.failure_scenarios.items():
        ds = generate_dataset(cfg, name)
        runs = run_all_modes(cfg, ds.load_cell, ds.drops)
        out[name] = (window_report(runs, cfg, sc.start_min, sc.end_min),
                     window_report(base_runs, cfg, sc.start_min, sc.end_min))
    return out


def test_noisy_load_cell_hurts_the_raw_load_cell_flow(failure_reports):
    (est, _), (est0, _) = failure_reports["load_cell_noisy"]
    assert est[RAW_LOAD]["rmse_inside"] > 2.0 * est0[RAW_LOAD]["rmse_inside"]


def test_missing_drops_push_the_raw_drop_flow_down(failure_reports):
    (est, _), (est0, _) = failure_reports["ir_missing_drops"]
    assert est[RAW_DROP]["bias_inside"] < est0[RAW_DROP]["bias_inside"] - 0.1


def test_overcounting_raises_the_disagreement(failure_reports):
    (_, diag), (_, diag0) = failure_reports["sensors_disagree"]
    assert diag["disagree_max_inside_pct"] > diag0["disagree_max_inside_pct"]


# ---------------- end-to-end smoke tests (files are written) ----------------
def test_run_compare_writes_files(tmp_path):
    mean = run_compare(load_config(), n_seeds=1, out_dir=tmp_path, verbose=False)
    assert len(mean) == 5
    for f in ("comparison_mean.csv", "comparison_by_seed.csv", "comparison_plot.html"):
        assert (tmp_path / f).exists()


def test_run_failures_writes_files(tmp_path):
    cfg = load_config()
    table = run_failures(cfg, out_dir=tmp_path, verbose=False)
    assert set(table["scenario"]) == set(cfg.failure_scenarios)
    assert (tmp_path / "failure_metrics.csv").exists()
    for name in cfg.failure_scenarios:
        assert (tmp_path / f"failure_{name}.html").exists()