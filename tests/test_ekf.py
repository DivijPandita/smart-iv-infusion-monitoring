"""Tests for the EKF."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.ekf.ekf import ExtendedKalmanFilter, flow_metrics, run_ekf


def make_ekf(x0=(2.0, 15.0), p=(1.0, 4.0), q=(0.01, 1e-4), r=(0.2, 2.0)):
    return ExtendedKalmanFilter(x0, np.diag(p), q, r)


def synth_windows(n=120, flow=2.5, df=14.5, wn=0.2, dn=1.5, seed=0, weight_ok=True):
    rng = np.random.default_rng(seed)
    t_end = (np.arange(n) + 1) * 60.0
    w = pd.DataFrame({
        "window_index": np.arange(n), "t_start_s": t_end - 60.0, "t_end_s": t_end,
        "flow_weight_ml_min": flow + rng.normal(0, wn, n),
        "drop_rate_per_min": flow * df + rng.normal(0, dn, n),
        "weight_valid": weight_ok,
    })
    if not weight_ok:
        w["flow_weight_ml_min"] = np.nan
    return w


def test_dimensions():
    f = make_ekf()
    assert f.x.shape == (2,) and f.P.shape == (2, 2) and f.R.shape == (2, 2)
    assert f.jacobian(f.x).shape == (2, 2) and f.h(f.x).shape == (2,)


def test_measurement_function_and_jacobian_values():
    x = np.array([2.5, 14.5])
    assert ExtendedKalmanFilter.h(x) == pytest.approx([2.5, 36.25])
    assert ExtendedKalmanFilter.jacobian(x) == pytest.approx(np.array([[1.0, 0.0], [14.5, 2.5]]))


def test_jacobian_matches_numerical_derivative():
    x = np.array([2.7, 14.2])
    eps = 1e-6
    numeric = np.column_stack([
        (ExtendedKalmanFilter.h(x + eps * e) - ExtendedKalmanFilter.h(x - eps * e)) / (2 * eps)
        for e in np.eye(2)])
    assert ExtendedKalmanFilter.jacobian(x) == pytest.approx(numeric, abs=1e-6)


def test_predict_keeps_state_and_grows_covariance_by_q():
    f = make_ekf()
    x_before, p_before = f.x.copy(), f.P.copy()
    f.predict(2.0)
    assert f.x == pytest.approx(x_before)
    assert np.diag(f.P) - np.diag(p_before) == pytest.approx([0.02, 2e-4])


def test_update_moves_state_toward_measurement_and_shrinks_uncertainty():
    f = make_ekf()
    trace_before = np.trace(f.P)
    innov, nis = f.update([3.0, 45.0])
    assert innov == pytest.approx([1.0, 45.0 - 2.0 * 15.0])      # z - h(x)
    assert f.x[0] > 2.0                                          # flow moves toward 3.0
    assert np.trace(f.P) < trace_before
    assert nis > 0


def test_covariance_stays_symmetric_positive_definite():
    f = make_ekf()
    for z in synth_windows(50).itertuples():
        f.predict(1.0)
        f.update([z.flow_weight_ml_min, z.drop_rate_per_min])
        assert np.allclose(f.P, f.P.T)
        assert np.linalg.eigvalsh(f.P).min() > 0


def test_mask_uses_only_selected_measurement():
    f = make_ekf()
    innov, _ = f.update([3.0, 45.0], mask=(False, True))
    assert np.isnan(innov[0]) and np.isfinite(innov[1])
    innov, _ = f.update([np.nan, 40.0])                          # NaN row is skipped automatically
    assert np.isnan(innov[0])


def test_no_usable_measurement_leaves_state_unchanged():
    f = make_ekf()
    x, p = f.x.copy(), f.P.copy()
    innov, nis = f.update([np.nan, 40.0], mask=(True, False))
    assert np.allclose(f.x, x) and np.allclose(f.P, p) and np.isnan(nis)


def test_converges_to_known_flow_and_drop_factor():
    res = run_ekf(synth_windows(150), load_config(), "fusion")
    last = res.iloc[-1]
    assert last.flow_ekf == pytest.approx(2.5, abs=0.15)
    assert last.drop_factor_ekf == pytest.approx(14.5, abs=0.5)
    assert last.drop_factor_std < res.drop_factor_std.iloc[0] / 3     # uncertainty shrank


def test_drop_factor_is_only_learned_with_both_sensors():
    w, cfg = synth_windows(150), load_config()
    assert run_ekf(w, cfg, "load_cell_only").drop_factor_ekf.iloc[-1] == pytest.approx(15.0)
    assert run_ekf(w, cfg, "drop_only").drop_factor_ekf.iloc[-1] == pytest.approx(15.0, abs=1e-3)
    assert abs(run_ekf(w, cfg, "fusion").drop_factor_ekf.iloc[-1] - 15.0) > 0.2


def test_missing_weight_falls_back_to_drop_measurement_only():
    res = run_ekf(synth_windows(30, weight_ok=False), load_config(), "fusion")
    assert not res.used_weight.any() and res.used_drops.all()
    assert res.innov_flow.isna().all()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unknown sensor mode"):
        run_ekf(synth_windows(5), load_config(), "magic")


def test_flow_metrics():
    m = flow_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert m["mae"] == pytest.approx(1.0) and m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["bias"] == pytest.approx(1.0)


def test_ekf_on_simulated_data_beats_raw_and_learns_drop_factor():
    from src.preprocessing.processor import preprocess
    from src.simulation.generate_data import generate_dataset
    cfg = load_config()
    ds = generate_dataset(cfg)
    w = preprocess(ds.load_cell, ds.drops, cfg).windows
    res = run_ekf(w, cfg, "fusion")
    ref = w["ref_flow_ml_min"].to_numpy()
    assert flow_metrics(res.flow_ekf, ref, 10)["rmse"] < 0.25
    assert flow_metrics(res.flow_ekf, ref, 10)["rmse"] < flow_metrics(
        w["flow_drop_nominal_ml_min"], ref, 10)["rmse"] * 1.05
    assert 13.6 < res.drop_factor_ekf.iloc[-1] < 14.9