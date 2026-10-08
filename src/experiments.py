"""Experiments: sensor comparison (Steps 16) and sensor failures (Step 17).

    python -m src.experiments                    # both
    python -m src.experiments compare --seeds 5  # load cell only vs drop only vs fusion
    python -m src.experiments failures           # the 3 failure scenarios from config.yaml

Outputs go to results/ (CSV tables and interactive HTML plots).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.pipeline import evaluate_predictions, run_pipeline
from src.prediction.remaining_time import remaining_time_min
from src.simulation.generate_data import generate_dataset

SKIP_WINDOWS = 10                 # ignore the EKF warm-up when scoring
MIN_TRUE_REMAINING_MIN = 10.0     # remaining-time errors are only scored with > 10 min left
WARMUP_END_MIN = 15.0             # "before the failure" starts here
MODES = ("fusion", "load_cell_only", "drop_only")
EKF_LABELS = {"fusion": "EKF fusion", "load_cell_only": "EKF load cell only",
              "drop_only": "EKF drop only"}
RAW_LOAD = "Load cell raw"
RAW_DROP = "Drop counter raw (printed DF)"
NAN = float("nan")


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def run_all_modes(cfg: AppConfig, load_cell: pd.DataFrame, drops: pd.DataFrame
                  ) -> dict[str, pd.DataFrame]:
    """Run the full pipeline once per sensor mode on the same raw data."""
    return {m: run_pipeline(cfg, load_cell, drops, m) for m in MODES}


def flow_estimators(runs: dict[str, pd.DataFrame], cfg: AppConfig) -> dict[str, np.ndarray]:
    """Every flow estimate we compare, one value per window."""
    f = runs["fusion"]
    return {
        RAW_LOAD: f["flow_weight_ml_min"].to_numpy(dtype=float),
        RAW_DROP: f["drop_rate_per_min"].to_numpy(dtype=float) / cfg.drop_sensor.nominal_drop_factor,
        EKF_LABELS["load_cell_only"]: runs["load_cell_only"]["flow_ekf"].to_numpy(dtype=float),
        EKF_LABELS["drop_only"]: runs["drop_only"]["flow_ekf"].to_numpy(dtype=float),
        EKF_LABELS["fusion"]: f["flow_ekf"].to_numpy(dtype=float),
    }


def score_flow(flow, ref, select) -> dict[str, float]:
    """MAE / RMSE / bias of a flow estimate against the reference, on the selected windows."""
    err = (np.asarray(flow, dtype=float) - np.asarray(ref, dtype=float))[select]
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"flow_mae": NAN, "flow_rmse": NAN, "flow_bias": NAN}
    return {"flow_mae": float(np.abs(err).mean()),
            "flow_rmse": float(np.sqrt((err ** 2).mean())),
            "flow_bias": float(err.mean())}


def score_time(volume, flow, t_true, select, min_flow: float = 0.05) -> dict[str, float]:
    """Remaining-time error when T = V / flow is computed from the given flow estimate."""
    t_pred = remaining_time_min(volume, flow, min_flow)
    t_true = np.asarray(t_true, dtype=float)
    sel = np.asarray(select) & (t_true > MIN_TRUE_REMAINING_MIN) & np.isfinite(t_pred)
    if not sel.any():
        return {"time_mae_min": NAN, "time_median_abs_min": NAN, "time_mape_pct": NAN}
    err = t_pred[sel] - t_true[sel]
    return {"time_mae_min": float(np.abs(err).mean()),
            "time_median_abs_min": float(np.median(np.abs(err))),
            "time_mape_pct": float((np.abs(err) / t_true[sel]).mean() * 100.0)}


def _masked(fn, values, mask) -> float:
    """fn(values[mask]) ignoring NaN; NaN if nothing is selected."""
    v = np.asarray(values, dtype=float)[mask]
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else NAN


def _fmt(x, fmt="{:.1f}") -> str:
    return "n/a" if x is None or not np.isfinite(x) else fmt.format(x)


# ----------------------------------------------------------------------
# Step 16: sensor comparison
# ----------------------------------------------------------------------
def compare_one(cfg: AppConfig, load_cell: pd.DataFrame | None = None,
                drops: pd.DataFrame | None = None):
    """Compare all five estimators on ONE dataset. Returns (table, runs)."""
    if load_cell is None:
        ds = generate_dataset(cfg)
        load_cell, drops = ds.load_cell, ds.drops
    runs = run_all_modes(cfg, load_cell, drops)
    base = runs["fusion"]

    select = np.arange(len(base)) >= SKIP_WINDOWS
    ref = base["ref_flow_ml_min"].to_numpy()
    t_true = base["ref_remaining_time_min"].to_numpy()
    volume = base["remaining_volume_ml"].to_numpy()

    mode_of = {label: mode for mode, label in EKF_LABELS.items()}
    rows = []
    for name, flow in flow_estimators(runs, cfg).items():
        row = {"method": name,
               **score_flow(flow, ref, select),
               **score_time(volume, flow, t_true, select, cfg.prediction.min_flow_ml_min)}
        if name in mode_of:
            ev = evaluate_predictions(runs[mode_of[name]], SKIP_WINDOWS, MIN_TRUE_REMAINING_MIN)
            row.update(df_final_error=ev["df_error_final"], ci_coverage=ev["ci_coverage"],
                       mean_ci_width_min=ev["mean_ci_width_min"])
        else:
            row.update(df_final_error=NAN, ci_coverage=NAN, mean_ci_width_min=NAN)
        rows.append(row)
    return pd.DataFrame(rows), runs


def save_comparison_plot(runs: dict[str, pd.DataFrame], cfg: AppConfig, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    base = runs["fusion"]
    x = base["t_end_s"] / 60.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Flow-rate estimates (click legend to show/hide)",
                                        "Drop-factor estimates"))
    hidden = {EKF_LABELS["load_cell_only"], EKF_LABELS["drop_only"]}
    for name, flow in flow_estimators(runs, cfg).items():
        fig.add_scatter(x=x, y=flow, mode="lines", name=name, row=1, col=1,
                        visible="legendonly" if name in hidden else True,
                        line=dict(width=3 if name == EKF_LABELS["fusion"] else 1))
    fig.add_scatter(x=x, y=base["ref_flow_ml_min"], mode="lines", name="true flow (simulation)",
                    line=dict(color="black", dash="dash"), row=1, col=1)
    for mode, label in EKF_LABELS.items():
        fig.add_scatter(x=x, y=runs[mode]["drop_factor_ekf"], mode="lines", name=f"DF: {label}",
                        row=2, col=1)
    fig.add_scatter(x=x, y=base["ref_drop_factor"], mode="lines", name="true DF (simulation)",
                    line=dict(color="black", dash="dash"), row=2, col=1)
    fig.update_xaxes(title_text="time (min)", row=2, col=1)
    fig.update_yaxes(title_text="mL/min", row=1, col=1)
    fig.update_yaxes(title_text="drops/mL", row=2, col=1)
    fig.update_layout(title="Sensor comparison (software simulation)", height=750,
                      hovermode="x unified")
    fig.write_html(path)


def run_compare(cfg: AppConfig | None = None, n_seeds: int = 5, out_dir: Path | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Experiments 1 to 3. Returns the table averaged over seeds."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir) if out_dir else cfg.project_path(cfg.paths.experiments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, first = [], None
    for i in range(n_seeds):
        c = copy.deepcopy(cfg)
        c.simulation.random_seed = cfg.simulation.random_seed + i
        table, runs = compare_one(c)
        table.insert(0, "seed", c.simulation.random_seed)
        tables.append(table)
        if first is None:
            first = (runs, c)

    by_seed = pd.concat(tables, ignore_index=True)
    numeric = by_seed.drop(columns="seed")
    mean = numeric.groupby("method", sort=False).mean(numeric_only=True)
    std = numeric.groupby("method", sort=False).std(numeric_only=True)

    by_seed.round(5).to_csv(out_dir / "comparison_by_seed.csv", index=False)
    mean.round(5).to_csv(out_dir / "comparison_mean.csv")
    save_comparison_plot(first[0], first[1], out_dir / "comparison_plot.html")

    if verbose:
        print(f"=== Experiments 1-3: sensor comparison (mean of {n_seeds} seeds, "
              f"first {SKIP_WINDOWS} windows excluded) ===")
        shown = mean[["flow_mae", "flow_rmse", "flow_bias", "time_mae_min",
                      "time_median_abs_min", "df_final_error", "ci_coverage"]]
        print(shown.round(3).to_string(na_rep="-"))
        print("\nSpread between seeds (std):")
        print(std[["flow_rmse", "time_mae_min"]].round(3).to_string(na_rep="-"))
        print("\nColumns: flow_* in mL/min; time_* = error of the remaining-time prediction in min;")
        print("df_final_error = final drop-factor estimate minus true value (single-sensor modes")
        print("cannot learn it, so this is just their starting error); ci_coverage = fraction of")
        print("windows where the true remaining time fell inside the 95% interval.")
        print(f"\nSaved to {out_dir}: comparison_mean.csv, comparison_by_seed.csv, comparison_plot.html")
    return mean


# ----------------------------------------------------------------------
# Step 17: sensor-failure experiments
# ----------------------------------------------------------------------
def window_report(runs: dict[str, pd.DataFrame], cfg: AppConfig, start_min: float, end_min: float):
    """Score every estimator inside and before a time window, plus fusion diagnostics."""
    base = runs["fusion"]
    t = base["t_end_s"].to_numpy(dtype=float) / 60.0
    inside = (t > start_min) & (t <= end_min)
    before = (t > WARMUP_END_MIN) & (t <= start_min)
    ref = base["ref_flow_ml_min"].to_numpy()

    est = {}
    for name, flow in flow_estimators(runs, cfg).items():
        a, b = score_flow(flow, ref, inside), score_flow(flow, ref, before)
        est[name] = {"rmse_inside": a["flow_rmse"], "bias_inside": a["flow_bias"],
                     "rmse_before": b["flow_rmse"]}

    dis = base["disagree_rel"].to_numpy(dtype=float) * 100.0
    alert = base["disagreement_alert"].to_numpy(dtype=bool)
    width = (base["ci_high_min"] - base["ci_low_min"]).to_numpy(dtype=float)
    hits = np.flatnonzero(inside & alert)
    diag = {
        "disagree_before_pct": _masked(np.mean, dis, before),
        "disagree_inside_pct": _masked(np.mean, dis, inside),
        "disagree_max_inside_pct": _masked(np.max, dis, inside),
        "alert_frac_before": _masked(np.mean, alert.astype(float), before),
        "alert_frac_inside": _masked(np.mean, alert.astype(float), inside),
        "first_alert_after_onset_min": float(t[hits[0]] - start_min) if hits.size else NAN,
        "ci_width_before_min": _masked(np.mean, width, before),
        "ci_width_inside_min": _masked(np.mean, width, inside),
    }
    return est, diag


def save_failure_plot(runs, cfg: AppConfig, name: str, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sc = cfg.failure_scenarios[name]
    base = runs["fusion"]
    x = base["t_end_s"] / 60.0
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Flow rate", "Sensor disagreement (% of flow)",
                                        "Remaining time with 95% interval"))
    est = flow_estimators(runs, cfg)
    fig.add_scatter(x=x, y=est[RAW_LOAD], name=RAW_LOAD, line=dict(width=1), row=1, col=1)
    fig.add_scatter(x=x, y=est[RAW_DROP], name=RAW_DROP, line=dict(width=1), row=1, col=1)
    fig.add_scatter(x=x, y=est[EKF_LABELS["fusion"]], name="EKF fusion", line=dict(width=3),
                    row=1, col=1)
    fig.add_scatter(x=x, y=base["ref_flow_ml_min"], name="true flow",
                    line=dict(color="black", dash="dash"), row=1, col=1)

    fig.add_scatter(x=x, y=base["disagree_rel"] * 100.0, name="disagreement", row=2, col=1)
    fig.add_hline(y=cfg.alerts.disagreement_relative_threshold * 100.0, line_dash="dot",
                  line_color="red", row=2, col=1)

    fig.add_scatter(x=x, y=base["ci_high_min"], line=dict(width=0), showlegend=False,
                    hoverinfo="skip", row=3, col=1)
    fig.add_scatter(x=x, y=base["ci_low_min"], line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(44,160,44,0.2)", name="95% interval", row=3, col=1)
    fig.add_scatter(x=x, y=base["remaining_time_min"], name="predicted",
                    line=dict(width=3), row=3, col=1)
    fig.add_scatter(x=x, y=base["ref_remaining_time_min"], name="true remaining",
                    line=dict(color="black", dash="dash"), row=3, col=1)

    for r in (1, 2, 3):
        fig.add_vrect(x0=sc.start_min, x1=sc.end_min, fillcolor="red", opacity=0.08,
                      line_width=0, row=r, col=1)
    fig.update_xaxes(title_text="time (min)", row=3, col=1)
    fig.update_layout(title=f"Failure scenario: {name}. {sc.description} (shaded)",
                      height=900, hovermode="x unified")
    fig.write_html(path)


def _print_failure(name, sc, est, est0, diag, diag0) -> None:
    print(f"\n=== Scenario: {name}  ({sc.start_min:.0f} to {sc.end_min:.0f} min) ===")
    print(sc.description)
    tab = pd.DataFrame({
        "RMSE in window": {k: v["rmse_inside"] for k, v in est.items()},
        "RMSE, no failure": {k: v["rmse_inside"] for k, v in est0.items()},
        "bias in window": {k: v["bias_inside"] for k, v in est.items()},
        "RMSE before": {k: v["rmse_before"] for k, v in est.items()},
    })
    print("Flow error vs true flow (mL/min):")
    print(tab.round(3).to_string(na_rep="-"))
    print(f"Disagreement (fusion), mean: before {_fmt(diag['disagree_before_pct'])}%, "
          f"in window {_fmt(diag['disagree_inside_pct'])}% "
          f"(no failure: {_fmt(diag0['disagree_inside_pct'])}%), "
          f"max in window {_fmt(diag['disagree_max_inside_pct'])}%")
    print(f"Disagreement alerts: before {_fmt(100 * diag['alert_frac_before'], '{:.0f}')}% of windows, "
          f"in window {_fmt(100 * diag['alert_frac_inside'], '{:.0f}')}% "
          f"(no failure: {_fmt(100 * diag0['alert_frac_inside'], '{:.0f}')}%); "
          f"first alert {_fmt(diag['first_alert_after_onset_min'])} min after onset")
    print(f"95% interval width: before {_fmt(diag['ci_width_before_min'])} min, "
          f"in window {_fmt(diag['ci_width_inside_min'])} min "
          f"(no failure: {_fmt(diag0['ci_width_inside_min'])} min)")


def run_failures(cfg: AppConfig | None = None, out_dir: Path | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Step 17: run each failure scenario and compare with a no-failure run."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir) if out_dir else cfg.project_path(cfg.paths.experiments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ds = generate_dataset(cfg)
    base_runs = run_all_modes(cfg, base_ds.load_cell, base_ds.drops)

    rows = []
    for name, sc in cfg.failure_scenarios.items():
        ds = generate_dataset(cfg, name)
        runs = run_all_modes(cfg, ds.load_cell, ds.drops)
        est, diag = window_report(runs, cfg, sc.start_min, sc.end_min)
        est0, diag0 = window_report(base_runs, cfg, sc.start_min, sc.end_min)

        for estimator, v in est.items():
            rows.append({"scenario": name, "estimator": estimator,
                         "rmse_in_window": v["rmse_inside"], "bias_in_window": v["bias_inside"],
                         "rmse_in_window_no_failure": est0[estimator]["rmse_inside"],
                         "rmse_before": v["rmse_before"]})
        save_failure_plot(runs, cfg, name, out_dir / f"failure_{name}.html")
        if verbose:
            _print_failure(name, sc, est, est0, diag, diag0)

    table = pd.DataFrame(rows)
    table.round(5).to_csv(out_dir / "failure_metrics.csv", index=False)
    if verbose:
        print(f"\nSaved to {out_dir}: failure_metrics.csv and one failure_<scenario>.html per scenario")
    return table


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sensor comparison and failure experiments.")
    p.add_argument("experiment", nargs="?", default="all", choices=["all", "compare", "failures"])
    p.add_argument("--seeds", type=int, default=5, help="number of random datasets for 'compare'")
    args = p.parse_args(argv)
    cfg = load_config()
    if args.experiment in ("all", "compare"):
        run_compare(cfg, args.seeds)
    if args.experiment in ("all", "failures"):
        run_failures(cfg)


if __name__ == "__main__":
    main()