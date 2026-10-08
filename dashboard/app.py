"""Smart IV Infusion Monitor: Streamlit dashboard (SOFTWARE SIMULATION, not a medical device).

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # make `import src` work under `streamlit run`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import charts, components  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.prediction.alerts import evaluate_alert  # noqa: E402
from src.simulation.generate_data import generate_dataset, load_dataset  # noqa: E402

cfg0 = load_config()
st.set_page_config(page_title="Smart IV Infusion Monitor (Simulation)", page_icon="💧",
                   layout="wide")

LIVE = "Live simulation (interactive)"
SAVED = "Saved results file"
DATE_COLS = ["timestamp", "completion_time", "completion_low", "completion_high"]
REQUIRED = {"t_end_s", "weight_g", "remaining_volume_ml", "volume_from_weight", "flow_weight_ml_min",
            "flow_drop_ml_min", "flow_ekf", "flow_std", "drop_factor_ekf", "drop_factor_std",
            "drop_rate_per_min", "disagree_rel", "disagreement_alert", "remaining_time_min",
            "sigma_T_min", "ci_low_min", "ci_high_min", "completion_time", "alert_status"}
LC_COLS = ["timestamp", "measured_weight_g", "true_weight_g"]

SS = st.session_state
SS.setdefault("seed", int(cfg0.simulation.random_seed))
SS.setdefault("playing", False)
SS.setdefault("sim_t", 0.0)
SS.setdefault("at_end", False)
SS.setdefault("last_tick", time.time())
SS.setdefault("signature", None)


# ---------------------------------------------------------------- playback callbacks
def _reset_replay() -> None:
    SS.playing, SS.sim_t, SS.at_end, SS.last_tick = False, 0.0, False, time.time()


def _start() -> None:
    if SS.get("at_end"):
        SS.sim_t, SS.at_end = 0.0, False
    SS.playing, SS.last_tick = True, time.time()


def _pause() -> None:
    SS.playing = False


def _jump_end() -> None:
    SS.playing, SS.sim_t = False, float("inf")


def _new_dataset() -> None:
    SS.seed = random.randint(1, 1_000_000)
    _reset_replay()


# ---------------------------------------------------------------- cached data
@st.cache_data(show_spinner="Simulating sensors and running the EKF...")
def run_live(overrides: tuple, scenario: str | None):
    cfg = load_config(overrides=dict(overrides))
    ds = generate_dataset(cfg, scenario)
    res = run_pipeline(cfg, ds.load_cell, ds.drops)
    return res, ds.load_cell[LC_COLS].reset_index(drop=True)


@st.cache_data
def load_saved(mtime: float):                       # mtime only invalidates the cache
    cfg = load_config()
    path = cfg.project_path(cfg.paths.processed_results_file)
    res = pd.read_csv(path, parse_dates=DATE_COLS)
    lc, _ = load_dataset(cfg)
    return res, lc[LC_COLS].reset_index(drop=True)


def _slider(label, lo, hi, default, step, fmt="%.2f", help=None):
    default = min(max(float(default), float(lo)), float(hi))
    return st.slider(label, float(lo), float(hi), default, float(step), format=fmt, help=help)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("💧 Controls")
source = st.sidebar.radio("Data source", [LIVE, SAVED],
                          help="Saved mode replays data/processed/ekf_results.csv "
                               "(from `python -m src.pipeline`). Settings below are ignored.")

b1, b2 = st.sidebar.columns(2)
b1.button("▶ Start", on_click=_start)
b2.button("⏸ Pause", on_click=_pause)
b3, b4 = st.sidebar.columns(2)
b3.button("↺ Reset", on_click=_reset_replay)
b4.button("⏭ Jump to end", on_click=_jump_end)
st.sidebar.button("🎲 Generate New Dataset", on_click=_new_dataset,
                  help="New random seed with the current settings")
st.sidebar.caption(f"Dataset seed: {SS.seed}")

speed = st.sidebar.slider("Simulation speed (simulated s per real s)", 10, 1500,
                          int(cfg0.dashboard.replay_speed), 10)
show_truth = st.sidebar.checkbox("Show simulation ground truth", True,
                                 help="True values exist only because this is a simulation.")

with st.sidebar.form("params"):
    st.caption("Edit the values, then press **Apply settings**. Applying restarts the replay.")
    with st.expander("IV bag", expanded=True):
        vol = _slider("Initial bag volume (mL)", 100, 1000, cfg0.bag.initial_volume_ml, 50, "%.0f")
        empty = _slider("Empty bag weight (g)", 0, 100, cfg0.bag.empty_bag_weight_g, 5, "%.0f")
    with st.expander("Flow", expanded=True):
        base_flow = _slider("Initial / base flow rate (mL/min)", 1.0, 6.0,
                            cfg0.flow.base_flow_ml_min, 0.1, "%.1f")
        variability = _slider("Flow-rate variability (mL/min, std)", 0.0, 0.5,
                              cfg0.flow.variability_std_ml_min, 0.01)
    with st.expander("Tubing drop factor", expanded=True):
        nominal = _slider("Nominal drop factor (drops/mL)", 10.0, 20.0,
                          cfg0.drop_sensor.nominal_drop_factor, 0.5, "%.1f")
        actual = _slider("Actual drop factor (drops/mL)", 12.0, 18.0,
                         cfg0.drop_sensor.actual_drop_factor, 0.1, "%.1f")
    with st.expander("Sensor noise", expanded=True):
        lc_noise = _slider("Load-cell noise (g, std)", 0.0, 2.0, cfg0.load_cell.noise_std_g, 0.05)
        miss_pct = _slider("Drop counter: missed drops (%)", 0.0, 30.0,
                           100 * cfg0.drop_sensor.miss_probability, 1.0, "%.0f")
        false_rate = _slider("Drop counter: false detections (per min)", 0.0, 5.0,
                             cfg0.drop_sensor.false_detection_rate_per_min, 0.1, "%.1f")
        scenario = st.selectbox("Sensor failure scenario",
                                ["none"] + sorted(cfg0.failure_scenarios))
    with st.expander("Processing & alerts"):
        win_options = sorted({30.0, 45.0, 60.0, 90.0, 120.0, float(cfg0.preprocessing.window_size_s)})
        window = st.select_slider("Window size (s)", options=win_options,
                                  value=float(cfg0.preprocessing.window_size_s))
        warn = _slider("Warning threshold (min)", 5, 120, cfg0.alerts.warning_min, 5, "%.0f")
        crit = _slider("Critical threshold (min)", 1, 60, cfg0.alerts.critical_min, 1, "%.0f")
    with st.expander("EKF tuning (advanced)"):
        r_w = _slider("R: weight-flow noise (mL/min)", 0.02, 1.0,
                      cfg0.ekf.r_weight_flow_std_ml_min, 0.01)
        r_d = _slider("R: drop-rate noise (drops/min)", 0.2, 10.0,
                      cfg0.ekf.r_drop_rate_std_drops_min, 0.1, "%.1f")
        q_f = _slider("Q: flow process noise (var per min)", 0.0005, 0.1,
                      cfg0.ekf.q_flow_var_per_min, 0.0005, "%.4f")
    st.form_submit_button("Apply settings", type="primary")

# ---------------------------------------------------------------- data
if source == LIVE:
    max_dur = max(float(cfg0.simulation.max_duration_min),
                  math.ceil((1.1 * vol / base_flow + 10.0) / 10.0) * 10.0)
    data_ov = {
        "bag.initial_volume_ml": float(vol),
        "bag.empty_bag_weight_g": float(empty),
        "flow.base_flow_ml_min": float(base_flow),
        "flow.variability_std_ml_min": float(variability),
        "drop_sensor.nominal_drop_factor": float(nominal),
        "drop_sensor.actual_drop_factor": float(actual),
        "drop_sensor.miss_probability": float(miss_pct) / 100.0,
        "drop_sensor.false_detection_rate_per_min": float(false_rate),
        "load_cell.noise_std_g": float(lc_noise),
        "preprocessing.window_size_s": float(window),
        "preprocessing.window_step_s": float(window),
        "ekf.initial_drop_factor": float(nominal),             # EKF starts from the printed value
        "ekf.initial_flow_ml_min": round(0.8 * float(base_flow), 3),   # deliberately imperfect
        "ekf.r_weight_flow_std_ml_min": float(r_w),
        "ekf.r_drop_rate_std_drops_min": float(r_d),
        "ekf.q_flow_var_per_min": float(q_f),
        "simulation.max_duration_min": float(max_dur),
        "simulation.random_seed": int(SS.seed),
    }
    live_ov = {"alerts.warning_min": float(warn), "alerts.critical_min": float(crit)}
    scenario_arg = None if scenario == "none" else scenario
    sig = (source, tuple(sorted(data_ov.items())), scenario_arg)
    try:
        res, lc = run_live(tuple(sorted({**data_ov, **live_ov}.items())), scenario_arg)
        cfg = load_config(overrides={**data_ov, **live_ov})
    except Exception as exc:                                    # bad settings -> show, don't crash
        st.error(f"Could not run the simulation with these settings: {exc}")
        st.stop()
else:
    cfg, scenario_arg, sig = cfg0, None, (source, None, None)
    path = cfg.project_path(cfg.paths.processed_results_file)
    if not path.exists():
        st.error(f"{path.name} not found. Run `python -m src.simulation.generate_data` and "
                 f"`python -m src.pipeline` first.")
        st.stop()
    try:
        res, lc = load_saved(path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read the saved results: {exc}")
        st.stop()
    if not REQUIRED <= set(res.columns):
        st.error("The saved results file is missing columns. Re-run `python -m src.pipeline`.")
        st.stop()

if SS.signature is None:
    SS.signature = sig
elif SS.signature != sig:                                       # data changed -> restart replay
    SS.signature = sig
    _reset_replay()

# ---------------------------------------------------------------- replay clock
t_total = float(res["t_end_s"].iloc[-1])
now = time.time()
if SS.playing:
    SS.sim_t += (now - SS.last_tick) * speed
SS.last_tick = now
if SS.sim_t >= t_total:
    SS.sim_t, SS.playing, SS.at_end = t_total, False, True
else:
    SS.at_end = False
k = int((res["t_end_s"] <= SS.sim_t + 1e-9).sum())              # windows revealed so far
total_min = t_total / 60.0

# ---------------------------------------------------------------- page
components.inject_css()
st.title("💧 " + cfg.dashboard.title)
components.disclaimer()

if scenario_arg:
    sc = cfg.failure_scenarios[scenario_arg]
    st.warning(f"Failure scenario active: {sc.description} "
               f"(from {sc.start_min:.0f} to {sc.end_min:.0f} min)")

clock = pd.Timestamp(cfg.simulation.start_datetime) + pd.to_timedelta(SS.sim_t, unit="s")
st.progress(min(SS.sim_t / t_total, 1.0),
            text=f"Simulated clock {clock:%H:%M:%S}  |  {SS.sim_t / 60:.0f} of {total_min:.0f} min  "
                 f"|  {k} of {len(res)} windows processed")

if k == 0:
    st.info("Press **▶ Start** in the sidebar to begin the replay, or **⏭ Jump to end** to see "
            "the complete run. The first result appears after one full window of data.")
else:
    vis = res.iloc[:k]
    row = vis.iloc[-1]
    lc_vis = lc[lc["timestamp"] <= row["t_end_s"]]

    state = evaluate_alert(float(row["remaining_time_min"]), float(row["ci_low_min"]),
                           float(row["disagree_rel"]), cfg)
    components.status_banner(state)
    components.metric_grid(row, cfg)

    left, right = st.columns(2)
    with left:
        components.show_plot(charts.weight_figure(lc_vis, total_min, show_truth))
    with right:
        components.show_plot(charts.flow_figure(vis, total_min, show_truth))
    left, right = st.columns(2)
    with left:
        components.show_plot(charts.drop_factor_figure(
            vis, res, cfg.drop_sensor.nominal_drop_factor, total_min, show_truth))
    with right:
        components.show_plot(charts.remaining_time_figure(vis, res, cfg, total_min, show_truth))
    left, right = st.columns(2)
    with left:
        components.show_plot(charts.disagreement_figure(vis, res, cfg, total_min))
    with right:
        st.markdown("**Alert log**")
        st.dataframe(components.alert_log(vis), hide_index=True)

    with st.expander("EKF internals (last 10 windows)"):
        cols = ["t_end_s", "flow_weight_ml_min", "drop_rate_per_min", "flow_ekf", "flow_std",
                "drop_factor_ekf", "drop_factor_std", "nis"]
        st.dataframe(vis[cols].tail(10).round(3), hide_index=True)
        st.caption("nis = normalised innovation squared; it averages about 2 when the filter's "
                   "noise settings (R, Q) are consistent with the data.")

# ---------------------------------------------------------------- animation loop
if SS.playing:
    time.sleep(cfg.dashboard.update_interval_ms / 1000.0)
    st.rerun()