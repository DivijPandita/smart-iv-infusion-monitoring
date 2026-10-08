"""Generate, save, load and inspect the simulated raw sensor data.

    python -m src.simulation.generate_data                      # normal run
    python -m src.simulation.generate_data --scenario ir_missing_drops
    python -m src.simulation.generate_data --seed 7 --plot

Outputs (paths come from config/config.yaml):
    data/raw/load_cell_data.csv     one row per load-cell sample
    data/raw/drop_counter_data.csv  one row per detected drop
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import AppConfig, ConfigError, load_config
from src.simulation.drop_sensor import simulate_drop_sensor
from src.simulation.infusion import generate_true_infusion, spawn_rngs
from src.simulation.load_cell import simulate_load_cell

LOAD_CELL_COLUMNS = ["timestamp", "true_weight_g", "measured_weight_g", "true_flow_ml_min"]
DROP_COLUMNS = ["timestamp", "drop_detected", "event_type",
                "true_drop_rate_per_min", "true_drop_factor"]


@dataclass
class SimulatedDataset:
    """Everything one simulation run produces."""
    truth: pd.DataFrame        # hidden ground truth (1 Hz, never fed to the EKF)
    load_cell: pd.DataFrame    # what the load cell reported
    drops: pd.DataFrame        # what the IR sensor reported
    scenario: str | None = None


# ----------------------------------------------------------------------
# Generate / save / load
# ----------------------------------------------------------------------
def generate_dataset(cfg: AppConfig | None = None,
                     scenario: str | None = None) -> SimulatedDataset:
    """Run truth + both sensors in memory (no files). Optionally inject a failure."""
    cfg = cfg or load_config()

    sc = None
    if scenario:
        if scenario not in cfg.failure_scenarios:
            raise ConfigError(
                f"Unknown scenario '{scenario}'. Available: {sorted(cfg.failure_scenarios)}")
        sc = cfg.failure_scenarios[scenario]

    rngs = spawn_rngs(cfg.simulation.random_seed)      # independent, reproducible streams
    truth = generate_true_infusion(cfg, rngs["flow"])
    load_cell = simulate_load_cell(truth, cfg, rngs["load_cell"], sc)
    drops = simulate_drop_sensor(truth, cfg, rngs["drop_sensor"], sc)
    return SimulatedDataset(truth, load_cell, drops, scenario)


def save_dataset(ds: SimulatedDataset, cfg: AppConfig) -> tuple[Path, Path]:
    """Write both sensor logs to CSV and return the two paths."""
    lc_path = cfg.project_path(cfg.paths.raw_load_cell_file)
    dr_path = cfg.project_path(cfg.paths.raw_drop_file)
    lc_path.parent.mkdir(parents=True, exist_ok=True)
    dr_path.parent.mkdir(parents=True, exist_ok=True)

    ds.load_cell[LOAD_CELL_COLUMNS].round(
        {"timestamp": 3, "true_weight_g": 4, "measured_weight_g": 3, "true_flow_ml_min": 4}
    ).to_csv(lc_path, index=False)
    ds.drops[DROP_COLUMNS].round(
        {"timestamp": 3, "true_drop_rate_per_min": 3, "true_drop_factor": 3}
    ).to_csv(dr_path, index=False)
    return lc_path, dr_path


def load_dataset(cfg: AppConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two raw CSV files back (this is what the pipeline will call)."""
    cfg = cfg or load_config()
    lc_path = cfg.project_path(cfg.paths.raw_load_cell_file)
    dr_path = cfg.project_path(cfg.paths.raw_drop_file)
    for p in (lc_path, dr_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run:  python -m src.simulation.generate_data")
    return pd.read_csv(lc_path), pd.read_csv(dr_path)


# ----------------------------------------------------------------------
# Inspection
# ----------------------------------------------------------------------
class Check(NamedTuple):
    name: str
    passed: bool
    detail: str


def inspect_dataset(load_cell: pd.DataFrame, drops: pd.DataFrame,
                    cfg: AppConfig) -> list[Check]:
    """Basic sanity checks on the raw data."""
    checks: list[Check] = []
    t = load_cell["timestamp"].to_numpy()
    w = load_cell["measured_weight_g"].to_numpy()
    dt = 1.0 / cfg.simulation.sample_rate_hz

    checks.append(Check("Load-cell timestamps strictly increasing",
                        bool((np.diff(t) > 0).all()), f"{len(t)} rows"))
    checks.append(Check("Typical sample spacing matches sample_rate_hz",
                        bool(np.isclose(np.median(np.diff(t)), dt, atol=1e-3)),
                        f"median spacing {np.median(np.diff(t)):.3f} s"))
    checks.append(Check("No NaN and no negative weights",
                        bool(np.isfinite(w).all() and (w >= 0).all()),
                        f"min {w.min():.2f} g, max {w.max():.2f} g"))

    start_est = float(np.median(w[:10]))
    checks.append(Check("Starting weight is near fluid + empty bag",
                        abs(start_est - cfg.initial_total_weight_g) < 2.0,
                        f"{start_est:.1f} g vs expected {cfg.initial_total_weight_g:.1f} g"))

    end_true = float(load_cell["true_weight_g"].iloc[-1])
    checks.append(Check("Bag ends (nearly) empty: only the bag is left",
                        abs(end_true - cfg.bag.empty_bag_weight_g) < 0.5,
                        f"final true weight {end_true:.2f} g"))

    ts = drops["timestamp"].to_numpy()
    checks.append(Check("Drop timestamps sorted and non-negative",
                        bool((np.diff(ts) >= 0).all() and (ts >= 0).all()),
                        f"{len(ts)} detected events"))

    expected = cfg.bag.initial_volume_ml * cfg.drop_sensor.actual_drop_factor
    ratio = len(ts) / expected
    checks.append(Check("Detected drop count is within 15% of the true count",
                        0.85 < ratio < 1.15,
                        f"{len(ts)} detected vs {expected:.0f} true ({100 * (ratio - 1):+.1f}%)"))
    return checks


def block_flow_table(load_cell: pd.DataFrame, drops: pd.DataFrame, cfg: AppConfig,
                     n_blocks: int = 4, block_min: float = 5.0) -> pd.DataFrame:
    """Compare what each sensor says about the flow in consecutive time blocks.

    This is a preview of Step 7 (windowing), computed the crudest possible way.
    """
    t = load_cell["timestamp"].to_numpy()
    w = load_cell["measured_weight_g"].to_numpy()
    ts = drops["timestamp"].to_numpy()
    rows = []
    for i in range(n_blocks):
        t0, t1 = i * block_min * 60.0, (i + 1) * block_min * 60.0
        w0, w1 = np.interp([t0, t1], t, w)
        n_drops = int(np.searchsorted(ts, t1) - np.searchsorted(ts, t0))
        mask = (t >= t0) & (t < t1)
        rows.append({
            "block": f"{i * block_min:.0f}-{(i + 1) * block_min:.0f} min",
            "true_flow": load_cell.loc[mask, "true_flow_ml_min"].mean(),
            "load_cell_flow": (w0 - w1) / cfg.bag.fluid_density_g_per_ml / block_min,
            "drops_per_min": n_drops / block_min,
            "drops/nominal_DF": n_drops / block_min / cfg.drop_sensor.nominal_drop_factor,
            "drops/actual_DF": n_drops / block_min / cfg.drop_sensor.actual_drop_factor,
        })
    return pd.DataFrame(rows)


def save_preview_plot(ds: SimulatedDataset, cfg: AppConfig, minutes: float = 10.0) -> Path:
    """Interactive HTML preview of the first few minutes of both sensors."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lc = ds.load_cell[ds.load_cell["timestamp"] <= minutes * 60.0]
    dr = ds.drops[ds.drops["timestamp"] <= minutes * 60.0]
    bins = np.arange(0.0, minutes * 60.0 + 10.0, 10.0)
    counts = np.histogram(dr["timestamp"], bins=bins)[0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Load cell: bag weight", "IR sensor: drops per 10 s"))
    fig.add_scatter(x=lc["timestamp"], y=lc["measured_weight_g"], name="measured", row=1, col=1)
    fig.add_scatter(x=lc["timestamp"], y=lc["true_weight_g"], name="true", row=1, col=1)
    fig.add_bar(x=bins[:-1], y=counts, name="drops / 10 s", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="g", row=1, col=1)
    fig.update_layout(title="Raw simulated data preview (software simulation)", height=650)

    out = cfg.project_path(cfg.paths.experiments_dir) / "raw_data_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    return out


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate simulated IV sensor data.")
    parser.add_argument("--scenario", help="failure scenario name from config.yaml")
    parser.add_argument("--seed", type=int, help="override simulation.random_seed")
    parser.add_argument("--plot", action="store_true", help="also write an HTML preview")
    args = parser.parse_args(argv)

    cfg = load_config(overrides={"simulation.random_seed": args.seed}
                      if args.seed is not None else None)
    ds = generate_dataset(cfg, args.scenario)
    lc_path, dr_path = save_dataset(ds, cfg)

    root = cfg.project_path(".").resolve()
    print("=== Raw data generated ===")
    print(f"Scenario        : {args.scenario or 'none (normal operation)'}")
    print(f"Random seed     : {cfg.simulation.random_seed}")
    for label, path, df in (("Load-cell file", lc_path, ds.load_cell),
                            ("Drop file     ", dr_path, ds.drops)):
        print(f"{label} : {path.resolve().relative_to(root)}  "
              f"({len(df)} rows, {path.stat().st_size / 1024:.0f} KB)")

    lc_back, dr_back = load_dataset(cfg)          # prove the files can be read back
    print(f"Reloaded OK     : {len(lc_back)} load-cell rows, {len(dr_back)} drop rows")

    print("\n--- Sanity checks ---")
    checks = inspect_dataset(lc_back, dr_back, cfg)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")

    print("\n--- What each sensor says about the flow (mL/min) ---")
    print(block_flow_table(lc_back, dr_back, cfg).round(2).to_string(index=False))
    print("(drops/nominal_DF uses the printed 15; drops/actual_DF uses the hidden true value.)")

    print("\n--- First rows of each file ---")
    print(lc_back.head(3).to_string(index=False))
    print(dr_back.head(3).to_string(index=False))

    if args.plot:
        print(f"\nPreview plot    : {save_preview_plot(ds, cfg).resolve().relative_to(root)}")

    if not all(c.passed for c in checks):
        raise SystemExit("Some sanity checks failed.")


if __name__ == "__main__":
    main()