"""Preprocessing and windowing of the raw sensor logs.

    raw load cell  -> regular grid -> spike rejection -> gap filling -> light smoothing
    raw drop times -> counted per window
    both           -> one row per window: [flow_weight_ml_min, drop_rate_per_min]

The EKF only ever receives the MEASUREMENT columns of the window table.
Columns that start with `ref_` are simulation references (the hidden truth),
used only for plots and scoring. They are added only if the raw data contains
truth columns, so a real sensor log would simply not have them.

Run a demo:   python -m src.preprocessing.processor
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config

HAMPEL_SCALE = 1.4826          # converts median(|x|) into a standard deviation (Gaussian data)
SIGMA_WINDOW_SAMPLES = 41      # window for the local noise estimate
SIGMA_MIN_PERIODS = 11
MIN_SIGMA_G = 0.02             # never treat the noise as smaller than this (avoids 0 threshold)
SPIKE_TAIL_SAMPLES = 3         # a knock decays over ~4 samples; reject the tail too
SPIKE_GAP_FILL_S = 15.0        # a rejected knock (+ tail) may be bridged up to this long

MEASUREMENT_COLUMNS = [
    "window_index", "t_start_s", "t_end_s",
    "weight_start_g", "weight_end_g", "fluid_weight_g", "volume_ml",
    "flow_weight_ml_min", "n_drops", "drop_rate_per_min",
    "flow_drop_nominal_ml_min", "weight_valid", "n_missing", "n_spikes",
]


@dataclass
class PreprocessResult:
    clean: pd.DataFrame      # 1 Hz cleaned load-cell series (for plots / inspection)
    windows: pd.DataFrame    # one row per window (this is what the EKF consumes)
    stats: dict[str, int]    # counts of what preprocessing did


# ----------------------------------------------------------------------
# Weight -> volume
# ----------------------------------------------------------------------
def weight_to_volume(weight_g, cfg: AppConfig) -> np.ndarray:
    """Total measured weight [g] -> fluid volume [mL].

        fluid_mass = measured_weight - empty_bag_weight      (never negative)
        volume     = fluid_mass / density
    """
    fluid_mass = np.asarray(weight_g, dtype=float) - cfg.bag.empty_bag_weight_g
    return np.maximum(fluid_mass, 0.0) / cfg.bag.fluid_density_g_per_ml


# ----------------------------------------------------------------------
# Load-cell cleaning
# ----------------------------------------------------------------------
def regularize_load_cell(load_cell: pd.DataFrame, sample_rate_hz: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Place samples on a regular time grid. Missing samples become NaN."""
    needed = {"timestamp", "measured_weight_g"}
    if not needed <= set(load_cell.columns):
        raise ValueError(f"Load-cell data needs columns {sorted(needed)}")

    df = (load_cell[["timestamp", "measured_weight_g"]]
          .replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp"))
    if len(df) < 2:
        raise ValueError("Need at least 2 valid load-cell samples")

    dt = 1.0 / sample_rate_hz
    t0 = float(df["timestamp"].iloc[0])
    slot = np.rint((df["timestamp"].to_numpy() - t0) / dt).astype(int)
    n = int(slot.max()) + 1

    grid_t = t0 + np.arange(n) * dt
    weight = np.full(n, np.nan)
    _, first = np.unique(slot, return_index=True)      # if two samples share a slot, keep the first
    weight[slot[first]] = df["measured_weight_g"].to_numpy()[first]
    return grid_t, weight

def _hampel_pass(median_source: np.ndarray, evaluate: np.ndarray, window: int,
                 threshold: float, exclude: np.ndarray | None = None) -> np.ndarray:
    """One Hampel pass: flag samples of `evaluate` far from the rolling median of `median_source`.

    `exclude` marks samples whose residuals must NOT be used to estimate the noise level
    (samples already known to be outliers).
    """
    median = pd.Series(median_source).rolling(window, center=True, min_periods=3).median()
    resid = (pd.Series(evaluate) - median).to_numpy()

    resid_for_scale = np.abs(resid)
    if exclude is not None:
        resid_for_scale = np.where(exclude, np.nan, resid_for_scale)
    scale = (pd.Series(resid_for_scale)
             .rolling(SIGMA_WINDOW_SAMPLES, center=True, min_periods=SIGMA_MIN_PERIODS)
             .median().to_numpy() * HAMPEL_SCALE)

    finite = np.isfinite(scale)
    typical = float(np.median(scale[finite])) if finite.any() else 0.0
    floor = max(0.75 * typical, MIN_SIGMA_G)            # guards against a tiny local estimate
    sigma = np.where(finite, np.maximum(scale, floor), floor)

    with np.errstate(invalid="ignore"):                # NaN residual -> comparison is False
        return np.abs(resid) > threshold * sigma


def hampel_spike_flags(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """True where a sample is an outlier relative to its local median.

    Pass 1 finds the obvious outliers. Pass 2 removes them, recomputes the median and noise
    scale from clean samples only, and re-judges every sample. This stops a knock from
    shifting the median and making its healthy neighbours look like outliers.
    """
    flags = _hampel_pass(x, x, window, threshold)
    if flags.any():
        clean_source = np.where(flags, np.nan, x)
        flags = _hampel_pass(clean_source, x, window, threshold, exclude=flags)
    return flags


def extend_flags_forward(flags: np.ndarray, n: int) -> np.ndarray:
    """Also flag the n samples after every flagged sample (the decaying tail of a knock)."""
    out = flags.copy()
    for k in range(1, n + 1):
        out[k:] |= flags[:-k]
    return out


def fill_short_gaps(x: np.ndarray, dt_s: float, max_gap_s: float,
                    spike_mask: np.ndarray | None = None,
                    spike_max_gap_s: float | None = None) -> np.ndarray:
    """Linearly interpolate NaN runs whose surrounding valid samples are close enough.

    A run that contains a rejected spike may be bridged up to `spike_max_gap_s`;
    a run of merely lost samples only up to `max_gap_s`.
    """
    out = x.copy()
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) < 2:
        return out
    a, b = valid[:-1], valid[1:]
    for i0, i1 in zip(a[(b - a) > 1], b[(b - a) > 1]):
        limit = max_gap_s
        if spike_mask is not None and spike_mask[i0 + 1:i1].any():
            limit = max(max_gap_s, spike_max_gap_s or max_gap_s)
        if (i1 - i0) * dt_s <= limit + 1e-9:
            out[i0 + 1:i1] = np.interp(np.arange(i0 + 1, i1), [i0, i1], [x[i0], x[i1]])
    return out


def clean_load_cell(load_cell: pd.DataFrame, cfg: AppConfig | None = None) -> pd.DataFrame:
    """Regularise, reject spikes, fill short gaps, lightly smooth.

    Returns columns: timestamp, weight_raw, is_missing, is_spike,
    weight_filled (after spike removal + gap filling), weight_clean (after smoothing).
    NaN in weight_clean means "no trustworthy value here".
    """
    cfg = cfg or load_config()
    pp = cfg.preprocessing
    dt = 1.0 / cfg.simulation.sample_rate_hz

    t, raw = regularize_load_cell(load_cell, cfg.simulation.sample_rate_hz)
    missing = ~np.isfinite(raw)

    if pp.spike_filter_enabled:
        flags = hampel_spike_flags(raw, pp.spike_filter_window_samples, pp.spike_mad_threshold)
        spike = extend_flags_forward(flags, SPIKE_TAIL_SAMPLES) & ~missing
    else:
        spike = np.zeros(len(raw), dtype=bool)

    candidate = raw.copy()
    candidate[spike] = np.nan
    filled = fill_short_gaps(candidate, dt, pp.max_gap_s,
                             spike_mask=spike, spike_max_gap_s=SPIKE_GAP_FILL_S)

    clean = filled.copy()
    if pp.smoothing_enabled and pp.smoothing_samples > 1:
        clean = (pd.Series(filled).rolling(pp.smoothing_samples, center=True, min_periods=1)
                 .mean().to_numpy())
        clean[~np.isfinite(filled)] = np.nan           # smoothing must not invent data in holes

    return pd.DataFrame({
        "timestamp": t, "weight_raw": raw, "is_missing": missing, "is_spike": spike,
        "weight_filled": filled, "weight_clean": clean,
    })


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------
def _value_at(t_valid: np.ndarray, w_valid: np.ndarray, t_query: np.ndarray,
              max_dist_s: float) -> np.ndarray:
    """Interpolate the weight at the query times; NaN if no valid sample is within max_dist_s."""
    t_query = np.asarray(t_query, dtype=float)
    if len(t_valid) == 0:
        return np.full(t_query.shape, np.nan)
    value = np.interp(t_query, t_valid, w_valid)
    j = np.searchsorted(t_valid, t_query)
    last = len(t_valid) - 1
    left = np.abs(t_query - t_valid[np.clip(j - 1, 0, last)])
    right = np.abs(t_valid[np.clip(j, 0, last)] - t_query)
    value[np.minimum(left, right) > max_dist_s + 1e-9] = np.nan
    return value


def _per_window_sum(flag: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(flag.astype(int))])
    return cs[i1] - cs[i0]


def build_windows(clean: pd.DataFrame, drops: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Aggregate the cleaned weight and the drop timestamps into windows."""
    pp = cfg.preprocessing
    size, step = pp.window_size_s, pp.window_step_s
    t = clean["timestamp"].to_numpy()
    duration = t[-1] - t[0]
    if duration < size:
        raise ValueError(f"Data ({duration:.0f} s) is shorter than one window ({size:.0f} s)")

    n_windows = int(np.floor((duration - size) / step + 1e-9)) + 1
    starts = t[0] + np.arange(n_windows) * step
    ends = starts + size
    minutes = size / 60.0

    # ---- weight side ----
    w = clean["weight_clean"].to_numpy()
    ok = np.isfinite(w)
    w_start = _value_at(t[ok], w[ok], starts, pp.max_gap_s)
    w_end = _value_at(t[ok], w[ok], ends, pp.max_gap_s)
    flow_weight = (w_start - w_end) / cfg.bag.fluid_density_g_per_ml / minutes

    # ---- drop side ----
    ts = np.sort(drops["timestamp"].to_numpy(dtype=float))
    ts = ts[np.isfinite(ts)]
    n_drops = np.searchsorted(ts, ends, side="left") - np.searchsorted(ts, starts, side="left")
    drop_rate = n_drops / minutes

    # ---- diagnostics ----
    i0 = np.searchsorted(t, starts, side="left")
    i1 = np.searchsorted(t, ends, side="left")

    return pd.DataFrame({
        "window_index": np.arange(n_windows),
        "t_start_s": starts,
        "t_end_s": ends,
        "weight_start_g": w_start,
        "weight_end_g": w_end,
        "fluid_weight_g": w_end - cfg.bag.empty_bag_weight_g,
        "volume_ml": weight_to_volume(w_end, cfg),
        "flow_weight_ml_min": flow_weight,
        "n_drops": n_drops,
        "drop_rate_per_min": drop_rate,
        "flow_drop_nominal_ml_min": drop_rate / cfg.drop_sensor.nominal_drop_factor,
        "weight_valid": np.isfinite(flow_weight),
        "n_missing": _per_window_sum(clean["is_missing"].to_numpy(), i0, i1),
        "n_spikes": _per_window_sum(clean["is_spike"].to_numpy(), i0, i1),
    })


def attach_reference(windows: pd.DataFrame, load_cell: pd.DataFrame,
                     drops: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Add `ref_*` columns (hidden truth) when the raw data carries truth columns."""
    out = windows.copy()
    if "true_weight_g" not in load_cell.columns:
        return out
    minutes = cfg.preprocessing.window_size_s / 60.0
    tt, tw = load_cell["timestamp"].to_numpy(), load_cell["true_weight_g"].to_numpy()
    w_s = np.interp(out["t_start_s"], tt, tw)
    w_e = np.interp(out["t_end_s"], tt, tw)
    ref_flow = (w_s - w_e) / cfg.bag.fluid_density_g_per_ml / minutes
    out["ref_flow_ml_min"] = ref_flow
    out["ref_weight_end_g"] = w_e
    if "true_drop_factor" in drops.columns and len(drops):
        factor = float(drops["true_drop_factor"].iloc[0])
        out["ref_drop_factor"] = factor
        out["ref_drop_rate_per_min"] = ref_flow * factor
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def preprocess(load_cell: pd.DataFrame, drops: pd.DataFrame,
               cfg: AppConfig | None = None) -> PreprocessResult:
    """Raw sensor logs in -> cleaned series + window table out."""
    cfg = cfg or load_config()
    clean = clean_load_cell(load_cell, cfg)
    windows = attach_reference(build_windows(clean, drops, cfg), load_cell, drops, cfg)

    kept_raw = (~clean["is_missing"] & ~clean["is_spike"]).sum()
    stats = {
        "n_grid_samples": len(clean),
        "n_missing": int(clean["is_missing"].sum()),
        "n_spike_samples": int(clean["is_spike"].sum()),
        "n_filled": int(clean["weight_filled"].notna().sum() - kept_raw),
        "n_unfilled": int(clean["weight_filled"].isna().sum()),
        "n_windows": len(windows),
        "n_windows_weight_invalid": int((~windows["weight_valid"]).sum()),
    }
    return PreprocessResult(clean, windows, stats)


# ----------------------------------------------------------------------
# Demo:  python -m src.preprocessing.processor
# ----------------------------------------------------------------------
def _err_summary(err) -> str:
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    return f"mean {e.mean():+.3f}, std {e.std():.3f}, max|err| {np.abs(e).max():.3f}"


def main() -> None:
    from src.simulation.generate_data import load_dataset

    cfg = load_config()
    lc, drops = load_dataset(cfg)                      # read the CSV files, as the pipeline will
    res = preprocess(lc, drops, cfg)
    w, s, pp = res.windows, res.stats, cfg.preprocessing

    print("=== Preprocessing & windowing ===")
    print(f"Load-cell grid samples   : {s['n_grid_samples']}")
    print(f"  lost samples           : {s['n_missing']}")
    print(f"  spike samples rejected : {s['n_spike_samples']}")
    print(f"  gap samples filled     : {s['n_filled']}   left missing: {s['n_unfilled']}")
    print(f"Windows                  : {s['n_windows']}  "
          f"({pp.window_size_s:.0f} s windows, step {pp.window_step_s:.0f} s)")
    print(f"Windows with valid weight: {s['n_windows'] - s['n_windows_weight_invalid']} / {s['n_windows']}")

    view = pd.DataFrame({
        "t_end_min": w["t_end_s"] / 60.0,
        "weight_end_g": w["weight_end_g"],
        "volume_ml": w["volume_ml"],
        "flow_weight": w["flow_weight_ml_min"],
        "n_drops": w["n_drops"],
        "drop_rate": w["drop_rate_per_min"],
        "flow_drop(15)": w["flow_drop_nominal_ml_min"],
        "ref_flow": w["ref_flow_ml_min"],
    })
    print("\nFirst 5 windows:")
    print(view.head(5).round(2).to_string(index=False))

    # naive estimate: raw weight at the window edges, no cleaning at all
    c = res.clean
    ok = c["weight_raw"].notna().to_numpy()
    t, raw = c["timestamp"].to_numpy(), c["weight_raw"].to_numpy()
    dt = 1.0 / cfg.simulation.sample_rate_hz
    n_s = _value_at(t[ok], raw[ok], w["t_start_s"].to_numpy(), dt)
    n_e = _value_at(t[ok], raw[ok], w["t_end_s"].to_numpy(), dt)
    naive = (n_s - n_e) / cfg.bag.fluid_density_g_per_ml / (pp.window_size_s / 60.0)

    print("\n--- Weight-derived flow vs simulation reference (mL/min) ---")
    print(f"naive (raw edges)   : {_err_summary(naive - w['ref_flow_ml_min'])}")
    print(f"preprocessed        : {_err_summary(w['flow_weight_ml_min'] - w['ref_flow_ml_min'])}")
    print("\n--- Drop rate vs simulation reference (drops/min) ---")
    print(f"drop counter        : {_err_summary(w['drop_rate_per_min'] - w['ref_drop_rate_per_min'])}")
    print("\n--- Flow from drops using the PRINTED drop factor ---")
    err = w["flow_drop_nominal_ml_min"] - w["ref_flow_ml_min"]
    rel = 100 * err.mean() / w["ref_flow_ml_min"].mean()
    print(f"flow_drop(15) error : mean {err.mean():+.3f} mL/min ({rel:+.1f}%)  <- the EKF must fix this")


if __name__ == "__main__":
    main()