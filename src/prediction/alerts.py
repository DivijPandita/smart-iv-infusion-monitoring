"""Predictive alert logic.

    NORMAL   remaining time >  warning_min
    WARNING  remaining time <= warning_min
    CRITICAL remaining time <= critical_min
plus an independent SENSOR DISAGREEMENT alert when |weight flow - drop flow| / flow > threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import AppConfig
from src.prediction.remaining_time import format_hours_minutes

SEVERITY = {"UNKNOWN": -1, "NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class AlertState:
    status: str                 # NORMAL | WARNING | CRITICAL | UNKNOWN
    disagreement_alert: bool
    message: str


def classify_remaining_time(minutes, warning_min: float, critical_min: float) -> str:
    if minutes is None or not np.isfinite(minutes):
        return "UNKNOWN"
    if minutes <= critical_min:
        return "CRITICAL"
    if minutes <= warning_min:
        return "WARNING"
    return "NORMAL"


def classify_remaining_time_array(minutes, warning_min: float, critical_min: float) -> np.ndarray:
    """Vectorised version. NaN gives 'UNKNOWN'."""
    m = np.asarray(minutes, dtype=float)
    return np.select([m <= critical_min, m <= warning_min, m > warning_min],
                     ["CRITICAL", "WARNING", "NORMAL"], default="UNKNOWN")


def disagreement_flags(rel_disagreement, threshold: float) -> np.ndarray:
    """True where the relative disagreement exceeds the threshold (NaN counts as False)."""
    return np.nan_to_num(np.asarray(rel_disagreement, dtype=float), nan=0.0) > threshold


def evaluate_alert(remaining_min: float, ci_low_min: float, rel_disagreement: float,
                   cfg: AppConfig) -> AlertState:
    """Single-moment alert evaluation (used by the dashboard)."""
    a = cfg.alerts
    use_low = a.use_lower_bound and np.isfinite(ci_low_min)
    basis = ci_low_min if use_low else remaining_min
    status = classify_remaining_time(basis, a.warning_min, a.critical_min)
    dis = bool(np.isfinite(rel_disagreement) and rel_disagreement > a.disagreement_relative_threshold)

    if status == "CRITICAL":
        message = f"Infusion nearly complete: about {format_hours_minutes(remaining_min)} left"
    elif status == "WARNING":
        message = f"Infusion ending soon: about {format_hours_minutes(remaining_min)} left"
    elif status == "NORMAL":
        message = "Infusion progressing normally"
    else:
        message = "Remaining time unavailable"
    if dis:
        message += f" | Sensors disagree by {100 * rel_disagreement:.0f}%: estimate less reliable"
    return AlertState(status, dis, message)