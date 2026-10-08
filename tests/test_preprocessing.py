"""Tests for preprocessing and windowing."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.preprocessing.processor import (
    MEASUREMENT_COLUMNS, clean_load_cell, preprocess, regularize_load_cell, weight_to_volume,
)
from src.simulation.generate_data import generate_dataset

OFF = {"preprocessing.smoothing_enabled": False, "preprocessing.spike_filter_enabled": False}


def synth_lc(n=601, w0=525.0, slope=-0.05, noise=0.0, seed=0):
    """Straight-line weight: -0.05 g/s = -3 g/min = 3 mL/min."""
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"timestamp": t, "measured_weight_g": w0 + slope * t + rng.normal(0, noise, n)})


def synth_drops(rate_per_min=40.0, duration=600.0):
    return pd.DataFrame({"timestamp": np.arange(0.0, duration, 60.0 / rate_per_min)})


def truth(t):
    return 525.0 - 0.05 * np.asarray(t)


# ---------------- weight -> volume ----------------
def test_weight_to_volume_basic():
    cfg = load_config()
    assert weight_to_volume(525.0, cfg) == pytest.approx(500.0)
    assert weight_to_volume(np.array([525.0, 325.0]), cfg) == pytest.approx([500.0, 300.0])


def test_weight_to_volume_density_and_clipping():
    cfg = load_config(overrides={"bag.fluid_density_g_per_ml": 1.25})
    assert weight_to_volume(525.0, cfg) == pytest.approx(500.0 / 1.25)
    assert weight_to_volume(10.0, cfg) == 0.0          # lighter than the empty bag -> 0, not negative


# ---------------- cleaning ----------------
def test_regularize_places_samples_on_grid_and_marks_missing():
    df = pd.DataFrame({"timestamp": [0.0, 1.0, 2.0, 4.0, 5.0],
                       "measured_weight_g": [10.0, 9.0, 8.0, 6.0, 5.0]})
    t, w = regularize_load_cell(df, 1.0)
    assert len(t) == 6
    assert np.isnan(w[3]) and np.isfinite(np.delete(w, 3)).all()


def test_perfect_signal_passes_through_unchanged():
    cfg = load_config(overrides={"preprocessing.smoothing_enabled": False})
    lc = synth_lc()
    c = clean_load_cell(lc, cfg)
    assert not c["is_spike"].any() and not c["is_missing"].any()
    assert np.allclose(c["weight_clean"], lc["measured_weight_g"])


def test_spike_is_rejected_and_replaced():
    cfg = load_config(overrides={"preprocessing.smoothing_enabled": False})
    lc = synth_lc(noise=0.1)
    lc.loc[300:303, "measured_weight_g"] += [8.0, 4.8, 2.4, 0.8]      # a decaying knock
    c = clean_load_cell(lc, cfg)
    assert abs(c["weight_raw"].iloc[300] - truth(300)) > 7.0           # raw is badly wrong
    assert c["is_spike"].iloc[300:304].all()
    assert c["is_spike"].sum() <= 20                                    # no wave of false alarms
    assert np.abs(c["weight_clean"] - truth(c["timestamp"])).max() < 0.6


def test_short_gap_is_filled_long_gap_is_not():
    cfg = load_config(overrides=OFF)
    lc = synth_lc().drop(index=[100, 101, 102] + list(range(200, 220)))
    c = clean_load_cell(lc, cfg)
    assert np.allclose(c["weight_clean"].iloc[100:103], truth([100, 101, 102]))   # 3 lost: filled
    assert c["weight_clean"].iloc[200:220].isna().all()                            # 20 lost: left NaN
    assert c["weight_clean"].isna().sum() == 20


def test_smoothing_reduces_noise_but_keeps_trend():
    cfg = load_config()
    lc = synth_lc(noise=0.2, seed=1)
    c = clean_load_cell(lc, cfg)
    inner = slice(5, -5)
    raw_err = (c["weight_raw"] - truth(c["timestamp"]))[inner]
    clean_err = (c["weight_clean"] - truth(c["timestamp"]))[inner]
    assert clean_err.std() < 0.6 * raw_err.std()
    assert abs(clean_err.mean()) < 0.05


# ---------------- windows: weight flow ----------------
def test_flow_from_weight_matches_known_slope():
    cfg = load_config(overrides=OFF)
    res = preprocess(synth_lc(), synth_drops(), cfg)
    w = res.windows
    assert np.allclose(w["flow_weight_ml_min"], 3.0)
    assert w["weight_end_g"].iloc[0] == pytest.approx(522.0)
    assert w["volume_ml"].iloc[0] == pytest.approx(497.0)              # (522 - 25 g) / 1 g/mL
    assert w["weight_valid"].all()


def test_flow_respects_density():
    cfg = load_config(overrides={**OFF, "bag.fluid_density_g_per_ml": 2.0})
    w = preprocess(synth_lc(), synth_drops(), cfg).windows
    assert np.allclose(w["flow_weight_ml_min"], 1.5)                   # 3 g/min / 2 g/mL


def test_window_count_and_boundaries():
    cfg = load_config(overrides=OFF)
    w = preprocess(synth_lc(), synth_drops(), cfg).windows
    assert len(w) == 10
    assert w["t_start_s"].iloc[0] == 0.0 and w["t_end_s"].iloc[-1] == 600.0
    assert np.allclose(w["t_end_s"] - w["t_start_s"], 60.0)
    assert np.allclose(np.diff(w["t_end_s"]), 60.0)


def test_overlapping_windows():
    cfg = load_config(overrides={**OFF, "preprocessing.window_step_s": 30.0})
    w = preprocess(synth_lc(), synth_drops(), cfg).windows
    assert len(w) == 19
    assert np.allclose(np.diff(w["t_start_s"]), 30.0)


# ---------------- windows: drop rate ----------------
def test_drop_rate_counts():
    cfg = load_config(overrides=OFF)
    w = preprocess(synth_lc(), synth_drops(40.0), cfg).windows
    assert (w["n_drops"] == 40).all()
    assert np.allclose(w["drop_rate_per_min"], 40.0)
    assert np.allclose(w["flow_drop_nominal_ml_min"], 40.0 / 15.0)


def test_drop_rate_with_30_second_windows():
    cfg = load_config(overrides={**OFF, "preprocessing.window_size_s": 30.0,
                                 "preprocessing.window_step_s": 30.0})
    w = preprocess(synth_lc(), synth_drops(40.0), cfg).windows
    assert len(w) == 20
    assert (w["n_drops"] == 20).all()
    assert np.allclose(w["drop_rate_per_min"], 40.0)                   # still per MINUTE


# ---------------- robustness ----------------
def test_window_with_unfillable_gap_is_invalid():
    cfg = load_config(overrides=OFF)
    lc = synth_lc().drop(index=range(170, 190))                        # 20 s hole around the 180 s edge
    w = preprocess(lc, synth_drops(), cfg).windows
    bad = w[~w["weight_valid"]]
    assert list(bad["window_index"]) == [2, 3]                         # windows touching t = 180 s
    assert bad["flow_weight_ml_min"].isna().all()
    assert w.loc[w["weight_valid"], "flow_weight_ml_min"].round(6).eq(3.0).all()
    assert w.loc[2, "n_missing"] == 10 and w.loc[3, "n_missing"] == 10


def test_invalid_inputs_raise():
    cfg = load_config()
    with pytest.raises(ValueError, match="at least 2"):
        preprocess(pd.DataFrame({"timestamp": [], "measured_weight_g": []}), synth_drops(), cfg)
    with pytest.raises(ValueError, match="columns"):
        preprocess(pd.DataFrame({"timestamp": [0.0, 1.0]}), synth_drops(), cfg)
    with pytest.raises(ValueError, match="shorter"):
        preprocess(synth_lc(n=30), synth_drops(), cfg)


# ---------------- on simulated data ----------------
def test_end_to_end_accuracy_on_simulated_data():
    cfg = load_config()
    ds = generate_dataset(cfg)
    res = preprocess(ds.load_cell, ds.drops, cfg)
    w = res.windows
    assert 180 < len(w) < 200
    assert w["weight_valid"].all()
    assert res.stats["n_unfilled"] <= 30

    flow_err = w["flow_weight_ml_min"] - w["ref_flow_ml_min"]
    assert flow_err.std() < 0.30 and abs(flow_err.mean()) < 0.05

    drop_err = w["drop_rate_per_min"] - w["ref_drop_rate_per_min"]
    assert -1.5 < drop_err.mean() < 0.2 and drop_err.std() < 2.5


def test_measurements_do_not_depend_on_truth_columns():
    cfg = load_config()
    ds = generate_dataset(cfg)
    full = preprocess(ds.load_cell, ds.drops, cfg).windows
    bare = preprocess(ds.load_cell[["timestamp", "measured_weight_g"]],
                      ds.drops[["timestamp"]], cfg).windows
    pd.testing.assert_frame_equal(full[MEASUREMENT_COLUMNS], bare[MEASUREMENT_COLUMNS])
    assert not any(c.startswith("ref_") for c in bare.columns)         # no truth available -> none added
    assert "ref_flow_ml_min" in full.columns