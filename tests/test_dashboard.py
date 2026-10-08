"""Tests for the dashboard pieces (no browser needed)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import charts, components
from src.config import load_config
from src.pipeline import run_pipeline
from src.simulation.generate_data import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_all_figures_build_from_pipeline_output():
    cfg = load_config()
    ds = generate_dataset(cfg)
    res = run_pipeline(cfg, ds.load_cell, ds.drops)
    lc = ds.load_cell[["timestamp", "measured_weight_g", "true_weight_g"]]
    vis, total = res.iloc[:80], res["t_end_s"].iloc[-1] / 60.0
    figs = [
        charts.weight_figure(lc[lc.timestamp <= vis.t_end_s.iloc[-1]], total),
        charts.flow_figure(vis, total),
        charts.drop_factor_figure(vis, res, cfg.drop_sensor.nominal_drop_factor, total),
        charts.remaining_time_figure(vis, res, cfg, total),
        charts.disagreement_figure(vis, res, cfg, total),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) >= 1 for f in figs)
    assert all(len(f.data) >= 2 for f in figs[:4])     # chart 5 has one trace until an alert fires
    hidden = charts.flow_figure(vis, total, show_truth=False)       # without the ground truth
    assert len(hidden.data) < len(figs[1].data)


def test_fmt_num():
    assert components.fmt_num(float("nan")) == "n/a"
    assert components.fmt_num(None) == "n/a"
    assert components.fmt_num(2.5, "{:.2f}", "mL/min") == "2.50 mL/min"


def test_card_html_escapes_text():
    html = components.card_html("A<b>", "1 & 2", "sub")
    assert "A&lt;b&gt;" in html and "1 &amp; 2" in html and "<b>" not in html


def test_alert_log_lists_status_changes_and_disagreement_onset():
    vis = pd.DataFrame({
        "t_end_s": [60.0, 120.0, 180.0, 240.0],
        "alert_status": ["NORMAL", "NORMAL", "WARNING", "CRITICAL"],
        "remaining_time_min": [100.0, 50.0, 20.0, 5.0],
        "disagreement_alert": [False, True, True, False],
    })
    log = components.alert_log(vis)
    assert list(log.columns) == ["Time (min)", "Event", "Remaining time"]
    assert list(log["Event"]) == ["Status: NORMAL", "Sensor disagreement started",
                                  "Status: WARNING", "Status: CRITICAL"]


def test_app_runs_headless_and_renders_without_errors():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["sim_t"] = 1e9            # start at the end of the replay
    at.run()
    assert not at.exception, at.exception
    assert not at.error