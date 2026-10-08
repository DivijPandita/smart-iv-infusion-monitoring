"""End-to-end pipeline:

    raw CSV -> preprocessing/windowing -> EKF -> remaining volume/time
            -> confidence interval -> alerts -> data/processed/ekf_results.csv

    python -m src.pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.ekf.ekf import run_ekf
from src.prediction.alerts import (classify_remaining_time_array, disagreement_flags)
from src.prediction.remaining_time import (
    completion_time, format_hms, format_hours_minutes, remaining_time_min,
    remaining_volume_series,
)
from src.prediction.uncertainty import (
    confidence_interval, effective_flow_std, remaining_time_sigma, sensor_disagreement,
)
from src.preprocessing.processor import preprocess
from src.simulation.generate_data import load_dataset


def reference_empty_time_s(load_cell: pd.DataFrame, cfg: AppConfig) -> float:
    """Time at which the TRUE bag empties (simulation reference). NaN if no truth is available."""
    if "true_weight_g" not in load_cell.columns:
        return float("nan")
    t = load_cell["timestamp"].to_numpy(dtype=float)
    tw = load_cell["true_weight_g"].to_numpy(dtype=float)
    empty = cfg.bag.empty_bag_weight_g
    hit = np.flatnonzero(tw <= empty + 1e-3)
    if hit.size:
        return float(t[hit[0]])
    n = min(60, len(t) - 1)                       # last samples were lost: extrapolate
    slope = (tw[-1 - n] - tw[-1]) / (t[-1] - t[-1 - n])
    return float(t[-1] + (tw[-1] - empty) / slope) if slope > 0 else float("nan")


def run_pipeline(cfg: AppConfig | None = None, load_cell: pd.DataFrame | None = None,
                 drops: pd.DataFrame | None = None, mode: str | None = None) -> pd.DataFrame:
    """Run everything and return one row per window. Only MEASUREMENT columns feed the EKF."""
    cfg = cfg or load_config()
    if (load_cell is None) != (drops is None):
        raise ValueError("Pass both load_cell and drops, or neither (then the CSV files are read)")
    if load_cell is None:
        load_cell, drops = load_dataset(cfg)

    w = preprocess(load_cell, drops, cfg).windows
    ekf = run_ekf(w, cfg, mode)
    pc, ac = cfg.prediction, cfg.alerts

    flow = ekf["flow_ekf"].to_numpy()
    df_est = ekf["drop_factor_ekf"].to_numpy()
    flow_weight = w["flow_weight_ml_min"].to_numpy(dtype=float)
    flow_drop = w["drop_rate_per_min"].to_numpy(dtype=float) / df_est

    volume, from_weight = remaining_volume_series(w, flow, cfg)
    t_rem = remaining_time_min(volume, flow, pc.min_flow_ml_min)

    dis_abs, dis_rel = sensor_disagreement(flow_weight, flow_drop, flow, pc.min_flow_ml_min)
    sigma_flow = effective_flow_std(ekf["flow_std"].to_numpy(), dis_rel, cfg)
    sigma_t = remaining_time_sigma(volume, flow, pc.volume_std_ml, sigma_flow, pc.min_flow_ml_min)
    ci_low, ci_high = confidence_interval(t_rem, sigma_t, pc.confidence_z)

    basis = ci_low if ac.use_lower_bound else t_rem
    status = classify_remaining_time_array(basis, ac.warning_min, ac.critical_min)

    now = pd.Timestamp(cfg.simulation.start_datetime) + pd.to_timedelta(
        w["t_end_s"].to_numpy(dtype=float), unit="s")

    out = pd.DataFrame({
        "window_index": w["window_index"].to_numpy(),
        "t_end_s": w["t_end_s"].to_numpy(),
        "timestamp": now,
        "weight_g": w["weight_end_g"].to_numpy(),
        "weight_valid": w["weight_valid"].to_numpy(),
        "remaining_volume_ml": volume,
        "volume_from_weight": from_weight,
        "flow_weight_ml_min": flow_weight,
        "drop_rate_per_min": w["drop_rate_per_min"].to_numpy(),
        "flow_drop_ml_min": flow_drop,
        "flow_ekf": flow,
        "flow_std": ekf["flow_std"].to_numpy(),
        "drop_factor_ekf": df_est,
        "drop_factor_std": ekf["drop_factor_std"].to_numpy(),
        "nis": ekf["nis"].to_numpy(),
        "disagree_abs_ml_min": dis_abs,
        "disagree_rel": dis_rel,
        "sigma_flow_eff": sigma_flow,
        "remaining_time_min": t_rem,
        "sigma_T_min": sigma_t,
        "ci_low_min": ci_low,
        "ci_high_min": ci_high,
        "completion_time": completion_time(now, t_rem),
        "completion_low": completion_time(now, ci_low),
        "completion_high": completion_time(now, ci_high),
        "alert_status": status,
        "disagreement_alert": disagreement_flags(dis_rel, ac.disagreement_relative_threshold),
    })

    # ---- simulation references (hidden truth; used only for plots and scoring) ----
    for src, dst in (("ref_flow_ml_min", "ref_flow_ml_min"), ("ref_drop_factor", "ref_drop_factor"),
                     ("ref_weight_end_g", "ref_weight_g")):
        if src in w.columns:
            out[dst] = w[src].to_numpy()
    t_empty = reference_empty_time_s(load_cell, cfg)
    if np.isfinite(t_empty):
        out["ref_remaining_time_min"] = np.maximum((t_empty - out["t_end_s"]) / 60.0, 0.0)
    return out


def evaluate_predictions(res: pd.DataFrame, skip: int = 10,
                         min_true_remaining_min: float = 10.0) -> dict[str, float]:
    """Score the pipeline against the simulation reference (needs ref_* columns)."""
    need = {"ref_flow_ml_min", "ref_drop_factor", "ref_remaining_time_min"}
    if not need <= set(res.columns):
        raise ValueError("Reference columns are missing, so the results cannot be scored")
    flow_err = (res["flow_ekf"] - res["ref_flow_ml_min"]).iloc[skip:]
    r = res.iloc[skip:]
    r = r[r["ref_remaining_time_min"] > min_true_remaining_min]
    err = r["remaining_time_min"] - r["ref_remaining_time_min"]
    inside = (r["ref_remaining_time_min"] >= r["ci_low_min"]) & \
             (r["ref_remaining_time_min"] <= r["ci_high_min"])
    return {
        "flow_mae": float(flow_err.abs().mean()),
        "flow_rmse": float(np.sqrt((flow_err ** 2).mean())),
        "df_error_final": float(res["drop_factor_ekf"].iloc[-1] - res["ref_drop_factor"].iloc[0]),
        "time_mae_min": float(err.abs().mean()),
        "time_median_abs_err_min": float(err.abs().median()),
        "time_mape_pct": float((err.abs() / r["ref_remaining_time_min"]).mean() * 100),
        "ci_coverage": float(inside.mean()),
        "mean_ci_width_min": float((r["ci_high_min"] - r["ci_low_min"]).mean()),
        "n_scored": float(len(r)),
    }


def snapshot_text(row: pd.Series) -> str:
    """Text version of the dashboard panel for one window."""
    bar = "=" * 52
    return "\n".join([
        bar, "        SMART IV INFUSION MONITORING", "          SOFTWARE PROTOTYPE (simulation)", bar,
        f"Bag weight        : {row.weight_g:7.1f} g      Remaining volume : {row.remaining_volume_ml:6.1f} mL",
        f"EKF flow rate     : {row.flow_ekf:7.2f} mL/min Drop rate        : {row.drop_rate_per_min:6.1f} drops/min",
        f"EKF drop factor   : {row.drop_factor_ekf:7.2f} drops/mL",
        f"Remaining time    : {format_hms(row.remaining_time_min)}",
        f"Completion        : {row.completion_time:%H:%M:%S}",
        f"95% interval      : {format_hms(row.ci_low_min)} - {format_hms(row.ci_high_min)}",
        f"Sensor disagreement: {100 * row.disagree_rel:5.1f} %" if np.isfinite(row.disagree_rel)
        else "Sensor disagreement: n/a",
        f"System status     : {row.alert_status}" + ("   [SENSOR DISAGREEMENT]" if row.disagreement_alert else ""),
        bar,
    ])


def main() -> None:
    cfg = load_config()
    res = run_pipeline(cfg)
    out = cfg.project_path(cfg.paths.processed_results_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(out, index=False)

    print("=== Pipeline complete ===")
    print(f"Windows: {len(res)}   saved: {out.relative_to(cfg.project_path('.'))}")
    print(f"EKF sensor mode: {cfg.ekf.sensor_mode}")

    print("\nRemaining-time prediction over the run (times as HH:MM:SS):")
    print(" t_min  vol_mL  flow  DF_ekf  predicted   95% interval          true     status")
    ref_ok = "ref_remaining_time_min" in res.columns
    for i in sorted({0, 4, 9, 19, 39, 59, 89, 119, 149, len(res) - 1}):
        if i >= len(res):
            continue
        r = res.iloc[i]
        true_txt = format_hms(r.ref_remaining_time_min) if ref_ok else "n/a"
        print(f" {r.t_end_s / 60:5.0f} {r.remaining_volume_ml:7.1f} {r.flow_ekf:5.2f} {r.drop_factor_ekf:7.2f}"
              f"  {format_hms(r.remaining_time_min)}  {format_hms(r.ci_low_min)}-{format_hms(r.ci_high_min)}"
              f"  {true_txt}  {r.alert_status}")

    if ref_ok:
        m = evaluate_predictions(res)
        print("\n--- Accuracy vs simulation reference (after 10 window warm-up) ---")
        print(f"Flow           : MAE {m['flow_mae']:.3f}  RMSE {m['flow_rmse']:.3f} mL/min")
        print(f"Drop factor    : final error {m['df_error_final']:+.2f} drops/mL")
        print(f"Remaining time : MAE {m['time_mae_min']:.1f} min, median {m['time_median_abs_err_min']:.1f} min, "
              f"MAPE {m['time_mape_pct']:.1f}%  ({int(m['n_scored'])} windows with >10 min left)")
        print(f"95% interval   : contains the true time in {100 * m['ci_coverage']:.0f}% of windows, "
              f"mean width {m['mean_ci_width_min']:.0f} min")

    print("\n--- Alerts ---")
    for level, limit in (("WARNING", cfg.alerts.warning_min), ("CRITICAL", cfg.alerts.critical_min)):
        hit = res[res["alert_status"] == level]
        if len(hit):
            r = hit.iloc[0]
            extra = f", true remaining then: {r.ref_remaining_time_min:.0f} min" if ref_ok else ""
            print(f"First {level:<8} at t = {r.t_end_s / 60:5.1f} min (threshold {limit:.0f} min{extra})")
    print(f"Disagreement alerts: {int(res['disagreement_alert'].sum())} of {len(res)} windows")

    k = int(np.argmin(np.abs(res["t_end_s"] - 3600.0)))
    print(f"\nSnapshot at t = {res.t_end_s.iloc[k] / 60:.0f} min:")
    print(snapshot_text(res.iloc[k]))
    print(f"(about {format_hours_minutes(res.remaining_time_min.iloc[k])} remaining)")


if __name__ == "__main__":
    main()