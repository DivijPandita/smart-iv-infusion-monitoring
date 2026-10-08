"""Extended Kalman Filter for IV flow rate and tubing drop factor.

State      x = [flow (mL/min), drop_factor (drops/mL)]
Process    x_k = x_(k-1) + w,       w ~ N(0, Q)         (random walk)
Measure    z = [flow_from_weight (mL/min), drop_rate (drops/min)]
           z = h(x) + v,  h(x) = [flow, flow * drop_factor],  v ~ N(0, R)
Jacobian   H = [[1, 0], [drop_factor, flow]]

Run a demo:   python -m src.ekf.ekf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config

MIN_FLOW = 0.01          # state clamps keep the filter physical
MIN_DROP_FACTOR = 1.0
SENSOR_MASKS = {         # which measurement rows each mode may use: (weight, drops)
    "fusion": (True, True),
    "load_cell_only": (True, False),
    "drop_only": (False, True),
}


class ExtendedKalmanFilter:
    """Minimal, explicit 2-state EKF. Every equation is written out."""

    def __init__(self, x0, P0, q_var_per_min, r_std):
        self.x = np.asarray(x0, dtype=float).reshape(2)
        self.P = np.asarray(P0, dtype=float).reshape(2, 2)
        self.q = np.asarray(q_var_per_min, dtype=float).reshape(2)   # variance per minute
        self.R = np.diag(np.square(np.asarray(r_std, dtype=float)))  # 2x2 diagonal

    # ---- models ----
    @staticmethod
    def h(x: np.ndarray) -> np.ndarray:
        """Measurement function: what the sensors should read if the state is x."""
        return np.array([x[0], x[0] * x[1]])

    @staticmethod
    def jacobian(x: np.ndarray) -> np.ndarray:
        """H = dh/dx evaluated at the CURRENT state."""
        return np.array([[1.0, 0.0],
                         [x[1], x[0]]])

    # ---- step 1: predict ----
    def predict(self, dt_min: float) -> None:
        """x_k|k-1 = f(x_k-1) = x_k-1 ;  P_k|k-1 = F P F^T + Q  with F = I."""
        F = np.eye(2)
        Q = np.diag(self.q * dt_min)             # more time -> more allowed wander
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ---- step 2: update ----
    def update(self, z, mask=(True, True)) -> tuple[np.ndarray, float]:
        """Correct the state with a measurement.

        mask selects which measurement rows are used (a NaN in z is also skipped).
        Returns (innovation[2] with NaN for unused rows, NIS).
        NIS = y^T S^-1 y is a consistency score: its average should be about the
        number of rows used (1 or 2) if R and Q are well tuned.
        """
        z = np.asarray(z, dtype=float)
        rows = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(z))
        innovation = np.full(2, np.nan)
        if rows.size == 0:
            return innovation, float("nan")

        H = self.jacobian(self.x)[rows]                       # (k, 2)
        R = self.R[np.ix_(rows, rows)]                        # (k, k)
        y = z[rows] - self.h(self.x)[rows]                    # innovation
        S = H @ self.P @ H.T + R                              # innovation covariance
        K = np.linalg.solve(S, H @ self.P).T                  # Kalman gain (2, k), no explicit inverse

        self.x = self.x + K @ y
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T         # Joseph form
        self.P = 0.5 * (self.P + self.P.T)                    # enforce symmetry
        self.x[0] = max(self.x[0], MIN_FLOW)
        self.x[1] = max(self.x[1], MIN_DROP_FACTOR)

        innovation[rows] = y
        return innovation, float(y @ np.linalg.solve(S, y))


# ----------------------------------------------------------------------
# Run the filter over the window table
# ----------------------------------------------------------------------
def run_ekf(windows: pd.DataFrame, cfg: AppConfig | None = None,
            mode: str | None = None) -> pd.DataFrame:
    """Run the EKF window by window. Only MEASUREMENT columns are read (never ref_*)."""
    cfg = cfg or load_config()
    e = cfg.ekf
    mode = mode or e.sensor_mode
    if mode not in SENSOR_MASKS:
        raise ValueError(f"Unknown sensor mode '{mode}'. Use one of {sorted(SENSOR_MASKS)}")
    use_weight, use_drops = SENSOR_MASKS[mode]

    df_std, q_df = e.initial_drop_factor_std, e.q_drop_factor_var_per_min
    if mode == "drop_only":                 # one sensor cannot learn DF: keep the printed value
        df_std, q_df = 1e-6, 0.0

    ekf = ExtendedKalmanFilter(
        x0=[e.initial_flow_ml_min, e.initial_drop_factor],
        P0=np.diag([e.initial_flow_std ** 2, df_std ** 2]),
        q_var_per_min=[e.q_flow_var_per_min, q_df],
        r_std=[e.r_weight_flow_std_ml_min, e.r_drop_rate_std_drops_min],
    )

    prev_t = float(windows["t_start_s"].iloc[0])
    rows = []
    for w in windows.itertuples(index=False):
        dt_min = (w.t_end_s - prev_t) / 60.0
        prev_t = w.t_end_s

        ekf.predict(dt_min)
        z = np.array([w.flow_weight_ml_min if w.weight_valid else np.nan,
                      w.drop_rate_per_min])
        innov, nis = ekf.update(z, (use_weight, use_drops))

        rows.append({
            "window_index": w.window_index, "t_end_s": w.t_end_s,
            "flow_ekf": ekf.x[0], "drop_factor_ekf": ekf.x[1],
            "p_flow": ekf.P[0, 0], "p_flow_df": ekf.P[0, 1], "p_df": ekf.P[1, 1],
            "flow_std": np.sqrt(ekf.P[0, 0]), "drop_factor_std": np.sqrt(ekf.P[1, 1]),
            "innov_flow": innov[0], "innov_drop": innov[1], "nis": nis,
            "used_weight": bool(use_weight and np.isfinite(z[0])),
            "used_drops": bool(use_drops),
        })
    return pd.DataFrame(rows)


def flow_metrics(est: np.ndarray, ref: np.ndarray, skip: int = 0) -> dict[str, float]:
    """MAE and RMSE of a flow estimate against the reference, ignoring the first `skip` windows."""
    err = np.asarray(est, float)[skip:] - np.asarray(ref, float)[skip:]
    err = err[np.isfinite(err)]
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean())}


# ----------------------------------------------------------------------
# Demo (Step 9):  python -m src.ekf.ekf
# ----------------------------------------------------------------------
def main() -> None:
    from src.preprocessing.processor import preprocess
    from src.simulation.generate_data import load_dataset

    cfg = load_config()
    lc, drops = load_dataset(cfg)
    windows = preprocess(lc, drops, cfg).windows
    res = run_ekf(windows, cfg, "fusion")
    ref = windows["ref_flow_ml_min"].to_numpy()
    true_df = float(windows["ref_drop_factor"].iloc[0])
    minutes = cfg.preprocessing.window_size_s / 60.0
    eff_df = windows["n_drops"].sum() / (ref * minutes).sum()   # drops actually counted per mL

    print("=== EKF run (fusion) ===")
    print(f"Windows: {len(res)}   start x = [{cfg.ekf.initial_flow_ml_min}, "
          f"{cfg.ekf.initial_drop_factor}]")
    print(f"True drop factor: {true_df:.2f}   effective (as counted by the IR sensor): {eff_df:.2f}")

    print("\nConvergence of the drop factor (self-calibration):")
    print(" window  t_min  flow_est  flow_true  DF_est   DF_std")
    for i in [0, 2, 5, 10, 20, 40, 80, len(res) - 1]:
        r = res.iloc[min(i, len(res) - 1)]
        print(f" {int(r.window_index):>5}  {r.t_end_s / 60:>5.0f}  {r.flow_ekf:>8.2f}  "
              f"{ref[int(r.window_index)]:>9.2f}  {r.drop_factor_ekf:>6.2f}  {r.drop_factor_std:>6.3f}")

    print("\nFlow accuracy vs simulation reference, after a 10 window warm-up (mL/min):")
    modes = {
        "load cell raw (weight window flow)": windows["flow_weight_ml_min"].to_numpy(),
        "drop counter raw (printed DF 15)": windows["flow_drop_nominal_ml_min"].to_numpy(),
        "EKF load cell only": run_ekf(windows, cfg, "load_cell_only")["flow_ekf"].to_numpy(),
        "EKF drop only": run_ekf(windows, cfg, "drop_only")["flow_ekf"].to_numpy(),
        "EKF fusion": res["flow_ekf"].to_numpy(),
    }
    for name, est in modes.items():
        m = flow_metrics(est, ref, skip=10)
        print(f"  {name:<36} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  bias {m['bias']:+.3f}")

    print(f"\nMean NIS (consistency, ideal about 2 for fusion): {res['nis'].mean():.2f}")

    out = cfg.project_path(cfg.paths.processed_results_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["window_index", "t_start_s", "t_end_s", "volume_ml", "flow_weight_ml_min",
            "drop_rate_per_min", "flow_drop_nominal_ml_min", "weight_valid",
            "ref_flow_ml_min", "ref_drop_factor", "ref_weight_end_g"]
    merged = windows[keep].merge(res.drop(columns=["t_end_s"]), on="window_index")
    merged.round(5).to_csv(out, index=False)
    print(f"Saved: {out.relative_to(cfg.project_path('.'))}  ({len(merged)} rows)")


if __name__ == "__main__":
    main()