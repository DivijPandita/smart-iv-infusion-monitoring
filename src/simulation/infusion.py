"""Ground-truth IV infusion simulator.

Produces the hidden "true" state of the infusion on a regular time grid.
The virtual sensors (load cell, IR drop counter) observe this truth and add
their own noise; the EKF never sees these columns.

Physical relationships used:
    volume_infused  = integral of flow dt
    remaining       = initial_volume - volume_infused
    weight          = empty_bag_weight + remaining * density
    drop_rate       = flow * drop_factor            [drops/min]
    drops_so_far    = volume_infused * drop_factor

Run a quick demo:   python -m src.simulation.infusion
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, FlowConfig, load_config

# The true flow never goes below this (gravity drip doesn't stop randomly).
MIN_TRUE_FLOW_ML_MIN = 0.1


# ----------------------------------------------------------------------
# Random number streams
# ----------------------------------------------------------------------
def spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Create independent, reproducible random streams from one seed."""
    names = ("flow", "load_cell", "drop_sensor")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


# ----------------------------------------------------------------------
# Flow profile building blocks
# ----------------------------------------------------------------------
def _moving_average(x: np.ndarray, window: float) -> np.ndarray:
    """Centered moving average with edge padding (same length as input)."""
    w = int(round(window))
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def _piecewise_flow(t_s: np.ndarray, cfg: FlowConfig, dt_s: float) -> np.ndarray:
    """Step profile (segments x base flow), softened by a moving average."""
    durations = np.array([s["duration_min"] for s in cfg.segments], dtype=float) * 60.0
    scales = np.array([s["scale"] for s in cfg.segments], dtype=float)
    edges = np.cumsum(durations)                       # end time of each segment

    t_local = np.mod(t_s, edges[-1]) if cfg.repeat_segments else t_s
    idx = np.minimum(np.searchsorted(edges, t_local, side="right"), len(scales) - 1)

    step = cfg.base_flow_ml_min * scales[idx]
    return _moving_average(step, cfg.transition_s / dt_s)


def _ou_process(n: int, dt_s: float, std: float, tau_s: float,
                rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck process: slow, mean-reverting random wiggle.

    Exact discretisation:  x[k+1] = phi * x[k] + sqrt(1 - phi^2) * std * N(0,1)
    with phi = exp(-dt/tau). The long-run standard deviation equals `std`.
    """
    if std <= 0:
        return np.zeros(n)
    phi = np.exp(-dt_s / tau_s)
    shocks = rng.normal(0.0, std * np.sqrt(1.0 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, std)          # start from the stationary distribution
    for k in range(1, n):
        x[k] = phi * x[k - 1] + shocks[k]
    return x


def build_flow_profile(t_s: np.ndarray, cfg: AppConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """True flow rate [mL/min] at every time in `t_s` (before the bag empties)."""
    f = cfg.flow
    dt_s = float(t_s[1] - t_s[0])

    if f.profile == "piecewise":
        flow = _piecewise_flow(t_s, f, dt_s)
    elif f.profile == "smooth":
        flow = f.base_flow_ml_min + f.smooth_amplitude_ml_min * np.sin(
            2.0 * np.pi * t_s / (f.smooth_period_min * 60.0))
    else:  # constant
        flow = np.full_like(t_s, f.base_flow_ml_min)

    wiggle = _ou_process(len(t_s), dt_s, f.variability_std_ml_min,
                         f.variability_tau_min * 60.0, rng)
    return np.maximum(flow + wiggle, MIN_TRUE_FLOW_ML_MIN)


# ----------------------------------------------------------------------
# Main generator
# ----------------------------------------------------------------------
def generate_true_infusion(cfg: AppConfig | None = None,
                           rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Simulate the true infusion. One row per sample time.

    Columns
    -------
    timestamp              seconds since the start of the infusion
    true_flow_ml_min       true flow rate
    true_infused_ml        volume that has left the bag so far
    true_volume_ml         fluid still in the bag
    true_weight_g          what a perfect scale would read (fluid + empty bag)
    true_drop_factor       actual tubing drop factor [drops/mL]
    true_drop_rate_per_min true drop rate = flow * drop_factor
    true_drops_cum         true cumulative (fractional) drop count
    true_remaining_time_min  minutes until the bag is empty (NaN if it never empties)
    """
    cfg = cfg or load_config()
    if rng is None:
        rng = spawn_rngs(cfg.simulation.random_seed)["flow"]

    fs = cfg.simulation.sample_rate_hz
    n = int(round(cfg.simulation.max_duration_min * 60.0 * fs)) + 1
    t_s = np.arange(n) / fs

    flow = build_flow_profile(t_s, cfg, rng)             # mL/min

    # Volume infused = integral of flow (trapezoid rule). Flow is per minute, dt in minutes.
    dt_min = 1.0 / (fs * 60.0)
    infused_raw = np.concatenate([[0.0], np.cumsum(0.5 * (flow[1:] + flow[:-1]) * dt_min)])

    v0 = cfg.bag.initial_volume_ml
    remaining = np.maximum(v0 - infused_raw, 0.0)
    remaining[remaining < 1e-6] = 0.0                    # kill float dust
    empty = remaining <= 0.0

    # Once empty, nothing flows.
    flow = np.where(empty, 0.0, flow)
    infused = v0 - remaining

    # Optionally stop the recording at the moment the bag empties.
    t_empty_s = float(t_s[np.argmax(empty)]) if empty.any() else np.nan
    if cfg.simulation.stop_when_empty and empty.any():
        last = int(np.argmax(empty))
        keep = slice(0, last + 1)
        t_s, flow, infused, remaining = t_s[keep], flow[keep], infused[keep], remaining[keep]

    df_factor = cfg.drop_sensor.actual_drop_factor
    density = cfg.bag.fluid_density_g_per_ml

    return pd.DataFrame({
        "timestamp": t_s,
        "true_flow_ml_min": flow,
        "true_infused_ml": infused,
        "true_volume_ml": remaining,
        "true_weight_g": cfg.bag.empty_bag_weight_g + remaining * density,
        "true_drop_factor": df_factor,
        "true_drop_rate_per_min": flow * df_factor,
        "true_drops_cum": infused * df_factor,
        "true_remaining_time_min": (t_empty_s - t_s) / 60.0,
    })


# ----------------------------------------------------------------------
# Demo:  python -m src.simulation.infusion
# ----------------------------------------------------------------------
def _hhmm(minutes: float) -> str:
    total = int(round(minutes))
    return f"{total // 60:02d}:{total % 60:02d}"


def main() -> None:
    cfg = load_config()
    df = generate_true_infusion(cfg)
    t_min = df["timestamp"] / 60.0

    print("=== True infusion simulation ===")
    print(f"Rows (samples)        : {len(df)}  (every {1 / cfg.simulation.sample_rate_hz:.0f} s)")
    print(f"Infusion duration     : {t_min.iloc[-1]:.1f} min ({_hhmm(t_min.iloc[-1])})")
    print(f"Weight start -> end   : {df['true_weight_g'].iloc[0]:.1f} g -> "
          f"{df['true_weight_g'].iloc[-1]:.1f} g")
    print(f"Volume start -> end   : {df['true_volume_ml'].iloc[0]:.1f} mL -> "
          f"{df['true_volume_ml'].iloc[-1]:.1f} mL")
    print(f"Flow min / mean / max : {df['true_flow_ml_min'][df['true_flow_ml_min'] > 0].min():.2f} / "
          f"{df['true_flow_ml_min'].mean():.2f} / {df['true_flow_ml_min'].max():.2f} mL/min")
    print(f"Total drops           : {df['true_drops_cum'].iloc[-1]:.0f} "
          f"({cfg.bag.initial_volume_ml:.0f} mL x {cfg.drop_sensor.actual_drop_factor} drops/mL)")

    print("\nMean true flow inside each 5-minute segment (first cycle, edges excluded):")
    for label, lo, hi in [("0-5 min", 1, 4), ("5-10 min", 6, 9),
                          ("10-15 min", 11, 14), ("15-20 min", 16, 19)]:
        m = df.loc[(t_min >= lo) & (t_min <= hi), "true_flow_ml_min"].mean()
        print(f"  {label:<10} -> {m:.2f} mL/min")


if __name__ == "__main__":
    main()