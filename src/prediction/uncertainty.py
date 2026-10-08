"""Uncertainty of the remaining-time prediction (first-order error propagation).

    T = V / F
    dT/dV = 1/F          dT/dF = -V / F^2
    sigma_T^2 = (sigma_V / F)^2 + (V * sigma_F / F^2)^2      (V and F independent)
    95% interval:  T +/- z * sigma_T,  lower end clipped at 0
"""
from __future__ import annotations

import numpy as np

from src.config import AppConfig


def remaining_time_sigma(volume_ml, flow_ml_min, sigma_volume_ml, sigma_flow_ml_min,
                         min_flow_ml_min: float = 0.05):
    """Standard deviation of T [min] by the delta method."""
    v = np.maximum(np.asarray(volume_ml, dtype=float), 0.0)
    f = np.maximum(np.asarray(flow_ml_min, dtype=float), min_flow_ml_min)
    term_v = np.asarray(sigma_volume_ml, dtype=float) / f
    term_f = v * np.asarray(sigma_flow_ml_min, dtype=float) / f ** 2
    return np.sqrt(term_v ** 2 + term_f ** 2)


def sensor_disagreement(flow_weight, flow_drop, flow_fused, min_flow_ml_min: float = 0.05):
    """(absolute difference [mL/min], relative difference = abs / fused flow).

    NaN where the weight flow is missing (the sensors cannot be compared).
    """
    fw = np.asarray(flow_weight, dtype=float)
    fd = np.asarray(flow_drop, dtype=float)
    diff = np.abs(fw - fd)
    rel = diff / np.maximum(np.asarray(flow_fused, dtype=float), min_flow_ml_min)
    return diff, rel


def effective_flow_std(ekf_flow_std, rel_disagreement, cfg: AppConfig):
    """Flow uncertainty used for the time prediction.

        sigma_eff = sqrt(sigma_EKF^2 + sigma_future^2) * (1 + g * disagreement)

    - sigma_EKF   : sqrt(P[0,0]) from the EKF (grows when sensors are noisy)
    - sigma_future: prediction.future_flow_std_ml_min (the flow may change before the bag empties)
    - g           : prediction.disagreement_inflation (a safety margin, not a derived quantity)
    """
    pc = cfg.prediction
    rel = np.nan_to_num(np.asarray(rel_disagreement, dtype=float), nan=0.0)
    base = np.sqrt(np.asarray(ekf_flow_std, dtype=float) ** 2 + pc.future_flow_std_ml_min ** 2)
    return base * (1.0 + pc.disagreement_inflation * rel)


def confidence_interval(remaining_min, sigma_min, z: float = 1.96):
    """(lower, upper) of T +/- z*sigma. The lower end never goes below 0."""
    t = np.asarray(remaining_min, dtype=float)
    s = np.asarray(sigma_min, dtype=float)
    return np.maximum(t - z * s, 0.0), t + z * s