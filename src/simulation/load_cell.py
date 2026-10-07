"""Virtual load cell + HX711 equivalent.

Turns the hidden true bag weight into a realistic noisy measurement:

    measured = true_weight
             + white Gaussian noise          (electrical noise)
             + vibration                     (sinusoidal ripple)
             + slowly varying bias           (thermal drift, Ornstein-Uhlenbeck)
             + spikes                        (sudden knocks that decay over a few samples)
    then quantised to the ADC step and clipped at 0 g.
    Finally some samples are dropped (lost readings).

Run a demo:   python -m src.simulation.load_cell
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, FailureScenario, load_config
from src.simulation.infusion import _ou_process, generate_true_infusion, spawn_rngs

# A knock on the bag decays over a few samples instead of lasting one sample only.
SPIKE_KERNEL = np.array([1.0, 0.6, 0.3, 0.1])


def simulate_load_cell(
    true_df: pd.DataFrame,
    cfg: AppConfig | None = None,
    rng: np.random.Generator | None = None,
    scenario: FailureScenario | None = None,
) -> pd.DataFrame:
    """Simulate the load cell reading of the true weight.

    Parameters
    ----------
    true_df  : output of generate_true_infusion()
    scenario : optional failure scenario; inside [start_min, end_min) the white
               noise is multiplied by scenario.load_cell_noise_multiplier

    Returns
    -------
    DataFrame with columns: timestamp [s], true_weight_g, measured_weight_g,
    true_flow_ml_min. Rows for dropped samples are missing (timestamps skip).
    """
    cfg = cfg or load_config()
    if rng is None:
        rng = spawn_rngs(cfg.simulation.random_seed)["load_cell"]
    lc = cfg.load_cell

    t = true_df["timestamp"].to_numpy(dtype=float)
    true_w = true_df["true_weight_g"].to_numpy(dtype=float)
    n = len(t)
    dt_s = float(t[1] - t[0])

    # ---- all random draws, in a fixed order (reproducibility) ----
    white = rng.standard_normal(n)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    bias = _ou_process(n, dt_s, lc.bias_std_g, lc.bias_tau_min * 60.0, rng)
    spike_hit = rng.random(n) < lc.spike_probability
    spike_sign = rng.choice([-1.0, 1.0], n)
    spike_size = rng.uniform(0.5, 1.5, n)
    dropout_draw = rng.random(n)

    # ---- noise multiplier (failure scenario) ----
    multiplier = np.ones(n)
    if scenario is not None:
        window = (t >= scenario.start_min * 60.0) & (t < scenario.end_min * 60.0)
        multiplier[window] = scenario.load_cell_noise_multiplier

    # ---- components ----
    noise = white * lc.noise_std_g * multiplier
    vibration = lc.vibration_amplitude_g * np.sin(2.0 * np.pi * lc.vibration_freq_hz * t + phase)
    impulses = spike_hit * spike_sign * spike_size * lc.spike_magnitude_g
    spikes = np.convolve(impulses, SPIKE_KERNEL)[:n]

    measured = true_w + noise + vibration + bias + spikes

    # ---- ADC behaviour ----
    if lc.resolution_g > 0:
        measured = np.round(measured / lc.resolution_g) * lc.resolution_g
    measured = np.maximum(measured, 0.0)

    out = pd.DataFrame({
        "timestamp": t,
        "true_weight_g": true_w,
        "measured_weight_g": measured,
        "true_flow_ml_min": true_df["true_flow_ml_min"].to_numpy(),
    })

    # ---- lost samples (never lose the first one) ----
    keep = dropout_draw >= lc.dropout_probability
    keep[0] = True
    return out[keep].reset_index(drop=True)


# ----------------------------------------------------------------------
# Demo:  python -m src.simulation.load_cell
# ----------------------------------------------------------------------
def _robust_std(x: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation (ignores outliers)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def main() -> None:
    cfg = load_config()
    truth = generate_true_infusion(cfg)
    lc_df = simulate_load_cell(truth, cfg)

    residual = (lc_df["measured_weight_g"] - lc_df["true_weight_g"]).to_numpy()

    # 60 s weight-difference flow estimate vs the true 60 s flow (what Step 7 will do)
    s = lc_df.set_index("timestamp")
    later = s["measured_weight_g"].reindex(s.index + 60.0).to_numpy()
    later_true = s["true_weight_g"].reindex(s.index + 60.0).to_numpy()
    est = (s["measured_weight_g"].to_numpy() - later) / cfg.bag.fluid_density_g_per_ml
    ref = (s["true_weight_g"].to_numpy() - later_true) / cfg.bag.fluid_density_g_per_ml
    ok = ~np.isnan(est) & ~np.isnan(ref)
    flow_err = (est - ref)[ok]

    print("=== Virtual load cell ===")
    print(f"Samples kept / possible : {len(lc_df)} / {len(truth)} "
          f"({len(truth) - len(lc_df)} lost)")
    print(f"Weight error (meas-true): mean {residual.mean():+.3f} g, std {residual.std():.3f} g, "
          f"robust std {_robust_std(residual):.3f} g")
    print(f"Largest error           : {np.abs(residual).max():.2f} g")
    print(f"Samples with |error|>2 g: {(np.abs(residual) > 2).sum()}  (spikes)")
    print(f"ADC step                : {cfg.load_cell.resolution_g} g")
    print("--- flow from 60 s weight difference ---")
    print(f"Flow error std          : {flow_err.std():.3f} mL/min  (includes spikes)")
    print(f"Flow error robust std   : {_robust_std(flow_err):.3f} mL/min  (spikes ignored)")
    print("\nFirst 5 rows:")
    print(lc_df.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()