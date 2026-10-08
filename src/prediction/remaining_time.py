"""Remaining volume, remaining time and predicted completion time.

    remaining_volume = (measured_weight - empty_bag_weight) / density
    remaining_time   = remaining_volume / flow
    completion_time  = now + remaining_time
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.preprocessing.processor import weight_to_volume


def remaining_volume_ml(weight_g, cfg: AppConfig):
    """Total measured weight [g] -> fluid still in the bag [mL] (never negative)."""
    return weight_to_volume(weight_g, cfg)


def remaining_time_min(volume_ml, flow_ml_min, min_flow_ml_min: float = 0.05):
    """T = V / F in minutes. The flow is floored so a stalled estimate cannot give infinity."""
    v = np.maximum(np.asarray(volume_ml, dtype=float), 0.0)
    f = np.maximum(np.asarray(flow_ml_min, dtype=float), min_flow_ml_min)
    return v / f


def remaining_volume_series(windows: pd.DataFrame, flow_ml_min,
                            cfg: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    """Remaining volume per window, with a fallback when the weight is missing.

    Normally the volume comes from the window-end weight. If that weight is NaN,
    we integrate instead:  V = V_previous - flow * dt.
    Returns (volume_ml, from_weight) where from_weight is False for fallback windows.
    """
    volume = windows["volume_ml"].to_numpy(dtype=float).copy()
    flow = np.asarray(flow_ml_min, dtype=float)
    t_end = windows["t_end_s"].to_numpy(dtype=float)
    from_weight = np.isfinite(volume)

    prev_v = cfg.bag.initial_volume_ml
    prev_t = float(windows["t_start_s"].iloc[0])
    for i in range(len(volume)):
        if not from_weight[i]:
            volume[i] = max(prev_v - flow[i] * (t_end[i] - prev_t) / 60.0, 0.0)
        prev_v, prev_t = volume[i], t_end[i]
    return volume, from_weight


def completion_time(now, remaining_min):
    """Clock time at which the bag is predicted to be empty (works for scalars and arrays)."""
    return now + pd.to_timedelta(remaining_min, unit="m")


def format_hms(minutes: float) -> str:
    """113.333 min -> '01:53:20'."""
    if minutes is None or not np.isfinite(minutes):
        return "--:--:--"
    total = int(round(minutes * 60.0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_hours_minutes(minutes: float) -> str:
    """113 min -> '1 h 53 min';  45 min -> '45 min'."""
    if minutes is None or not np.isfinite(minutes):
        return "n/a"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"