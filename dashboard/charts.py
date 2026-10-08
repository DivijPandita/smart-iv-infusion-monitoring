"""Plotly figures for the dashboard. Every function takes already-sliced data
(`vis` = windows revealed so far, `full` = the whole run, used only to fix axis ranges)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

C_WEIGHT, C_DROP, C_EKF, C_TRUE, C_BAD = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f", "#d62728"


def _layout(fig: go.Figure, title: str, ytitle: str, total_min: float, height: int = 340):
    fig.update_layout(title=dict(text=title, x=0.01), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.update_xaxes(title_text="time (min)", range=[0, total_min])
    fig.update_yaxes(title_text=ytitle)
    return fig


def _band(fig, x, lo, hi, rgba: str, name: str) -> None:
    fig.add_scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=rgba, name=name, hoverinfo="skip")


def weight_figure(lc_vis: pd.DataFrame, total_min: float, show_truth: bool = True) -> go.Figure:
    step = max(1, len(lc_vis) // 1500)                 # keep the plot light
    d = lc_vis.iloc[::step]
    fig = go.Figure()
    fig.add_scatter(x=d["timestamp"] / 60.0, y=d["measured_weight_g"], mode="lines",
                    name="measured (load cell)", line=dict(color=C_WEIGHT, width=1))
    if show_truth and "true_weight_g" in d.columns:
        fig.add_scatter(x=d["timestamp"] / 60.0, y=d["true_weight_g"], mode="lines",
                        name="true (simulation reference)", line=dict(color=C_TRUE, dash="dash"))
    return _layout(fig, "1. IV bag weight", "weight (g)", total_min)


def flow_figure(vis: pd.DataFrame, total_min: float, show_truth: bool = True) -> go.Figure:
    x = vis["t_end_s"] / 60.0
    fig = go.Figure()
    _band(fig, x, vis["flow_ekf"] - vis["flow_std"], vis["flow_ekf"] + vis["flow_std"],
          "rgba(44,160,44,0.18)", "EKF +/- 1 sigma")
    fig.add_scatter(x=x, y=vis["flow_weight_ml_min"], mode="lines", name="weight-derived",
                    line=dict(color=C_WEIGHT, width=1))
    fig.add_scatter(x=x, y=vis["flow_drop_ml_min"], mode="lines", name="drop-derived (EKF DF)",
                    line=dict(color=C_DROP, width=1))
    fig.add_scatter(x=x, y=vis["flow_ekf"], mode="lines", name="EKF fused",
                    line=dict(color=C_EKF, width=3))
    if show_truth and "ref_flow_ml_min" in vis.columns:
        fig.add_scatter(x=x, y=vis["ref_flow_ml_min"], mode="lines",
                        name="true (simulation reference)", line=dict(color="black", dash="dash"))
    return _layout(fig, "2. Flow rate", "mL/min", total_min)


def drop_factor_figure(vis: pd.DataFrame, full: pd.DataFrame, nominal_df: float,
                       total_min: float, show_truth: bool = True) -> go.Figure:
    x = vis["t_end_s"] / 60.0
    fig = go.Figure()
    _band(fig, x, vis["drop_factor_ekf"] - vis["drop_factor_std"],
          vis["drop_factor_ekf"] + vis["drop_factor_std"], "rgba(44,160,44,0.18)",
          "EKF +/- 1 sigma")
    fig.add_scatter(x=x, y=vis["drop_factor_ekf"], mode="lines", name="EKF estimate",
                    line=dict(color=C_EKF, width=3))
    fig.add_scatter(x=[0, total_min], y=[nominal_df] * 2, mode="lines",
                    name=f"initial nominal ({nominal_df:g})", line=dict(color=C_DROP, dash="dot"))
    values = [nominal_df] + list(full["drop_factor_ekf"].dropna())
    if show_truth and "ref_drop_factor" in vis.columns:
        true_df = float(vis["ref_drop_factor"].iloc[-1])
        fig.add_scatter(x=[0, total_min], y=[true_df] * 2, mode="lines",
                        name=f"true ({true_df:g}, simulation reference)",
                        line=dict(color="black", dash="dash"))
        values.append(true_df)
    fig = _layout(fig, "3. Drop factor (self-calibration)", "drops/mL", total_min)
    fig.update_yaxes(range=[min(values) - 0.6, max(values) + 0.6])
    return fig


def remaining_time_figure(vis: pd.DataFrame, full: pd.DataFrame, cfg, total_min: float,
                          show_truth: bool = True) -> go.Figure:
    x = vis["t_end_s"] / 60.0
    fig = go.Figure()
    _band(fig, x, vis["ci_low_min"], vis["ci_high_min"], "rgba(44,160,44,0.18)",
          "95% interval (lower / upper bound)")
    fig.add_scatter(x=x, y=vis["remaining_time_min"], mode="lines", name="predicted",
                    line=dict(color=C_EKF, width=3))
    if show_truth and "ref_remaining_time_min" in vis.columns:
        fig.add_scatter(x=x, y=vis["ref_remaining_time_min"], mode="lines",
                        name="true (simulation reference)", line=dict(color="black", dash="dash"))
    fig.add_hline(y=cfg.alerts.warning_min, line_dash="dot", line_color=STATUS_WARN,
                  annotation_text="warning", annotation_position="top right")
    fig.add_hline(y=cfg.alerts.critical_min, line_dash="dot", line_color=C_BAD,
                  annotation_text="critical", annotation_position="top right")
    fig = _layout(fig, "4. Remaining time with confidence interval", "minutes", total_min)
    ymax = np.nanmax(full["ci_high_min"].iloc[3:]) if len(full) > 3 else np.nan
    if np.isfinite(ymax):
        fig.update_yaxes(range=[0, ymax * 1.05])
    return fig


STATUS_WARN = "#ed6c02"


def disagreement_figure(vis: pd.DataFrame, full: pd.DataFrame, cfg, total_min: float) -> go.Figure:
    x = vis["t_end_s"] / 60.0
    pct = vis["disagree_rel"] * 100.0
    thr = cfg.alerts.disagreement_relative_threshold * 100.0
    fig = go.Figure()
    fig.add_scatter(x=x, y=pct, mode="lines", name="|weight flow - drop flow| / EKF flow",
                    line=dict(color=C_WEIGHT, width=2))
    flagged = vis["disagreement_alert"].astype(bool)
    if flagged.any():
        fig.add_scatter(x=x[flagged], y=pct[flagged], mode="markers", name="alert",
                        marker=dict(color=C_BAD, size=8))
    fig.add_hline(y=thr, line_dash="dot", line_color=C_BAD,
                  annotation_text="alert threshold", annotation_position="top right")
    fig = _layout(fig, "5. Sensor disagreement", "% of flow", total_min)
    top = np.nanmax(full["disagree_rel"] * 100.0) if len(full) else np.nan
    fig.update_yaxes(range=[0, max(thr * 1.5, (top if np.isfinite(top) else 0) * 1.1)])
    return fig