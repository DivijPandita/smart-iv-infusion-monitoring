"""Tests for the file boundary: raw CSV -> pipeline -> results CSV."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.pipeline import run_pipeline
from src.simulation.generate_data import generate_dataset, save_dataset

DATE_COLS = ["timestamp", "completion_time", "completion_low", "completion_high"]


def tmp_cfg(tmp_path):
    return load_config(overrides={
        "paths.raw_load_cell_file": str(tmp_path / "lc.csv"),
        "paths.raw_drop_file": str(tmp_path / "drops.csv"),
    })


def test_pipeline_reads_the_csv_files(tmp_path):
    cfg = tmp_cfg(tmp_path)
    ds = generate_dataset(cfg)
    save_dataset(ds, cfg)
    from_files = run_pipeline(cfg)                                    # reads the CSVs
    in_memory = run_pipeline(cfg, ds.load_cell, ds.drops)
    assert len(from_files) == len(in_memory)
    assert np.allclose(from_files["remaining_time_min"], in_memory["remaining_time_min"],
                       rtol=0.02, atol=1.0)                           # CSV rounding only


def test_results_csv_roundtrip_keeps_what_the_dashboard_needs(tmp_path):
    cfg = load_config()
    ds = generate_dataset(cfg)
    res = run_pipeline(cfg, ds.load_cell, ds.drops)
    path = tmp_path / "results.csv"
    res.round(5).to_csv(path, index=False)
    back = pd.read_csv(path, parse_dates=DATE_COLS)
    assert len(back) == len(res)
    assert pd.api.types.is_datetime64_any_dtype(back["completion_time"])
    assert back["alert_status"].iloc[-1] == res["alert_status"].iloc[-1]
    assert back["remaining_time_min"].iloc[10] == pytest.approx(
        res["remaining_time_min"].iloc[10], abs=1e-4)