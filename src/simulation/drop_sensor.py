"""Virtual IR drop counter.

Converts the true cumulative drop count into discrete detection events:

    true drop k happens when  true_drops_cum  crosses the integer k
    detected time = true time + jitter
    some drops are missed, some phantom drops are added,
    and (failure scenario) some drops can be double counted.

Output is event based: ONE ROW PER DETECTED DROP.

Run a demo:   python -m src.simulation.drop_sensor
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig, FailureScenario, load_config
from src.simulation.infusion import generate_true_infusion, spawn_rngs

MAX_JITTER_FRACTION = 0.45   # jitter never exceeds 45% of the drop interval


def true_drop_times(true_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Times [s] at which each real drop falls, plus the local drop rate there.

    Drop k falls when the cumulative drop count crosses k. np.interp inverts
    the monotonic (time -> cumulative drops) curve to find that time.
    """
    t = true_df["timestamp"].to_numpy(dtype=float)
    cum = true_df["true_drops_cum"].to_numpy(dtype=float)
    rate = true_df["true_drop_rate_per_min"].to_numpy(dtype=float).copy()

    last = int(np.argmax(cum >= cum.max() - 1e-9))       # first index where the bag is empty
    t, cum, rate = t[: last + 1], cum[: last + 1], rate[: last + 1]
    if last > 0:
        rate[last] = rate[last - 1]                      # flow is zeroed in the last row; avoid /0

    n_drops = int(np.floor(cum[-1] + 1e-6))
    k = np.arange(1, n_drops + 1)
    times = np.interp(k, cum, t)
    local_rate = np.interp(times, t, rate)
    return times, local_rate


def simulate_drop_sensor(
    true_df: pd.DataFrame,
    cfg: AppConfig | None = None,
    rng: np.random.Generator | None = None,
    scenario: FailureScenario | None = None,
) -> pd.DataFrame:
    """Simulate the IR drop counter.

    Returns one row per detected drop:
        timestamp               detection time [s]
        drop_detected           always 1 (kept so the table reads like a sensor log)
        event_type              'real' | 'false' | 'duplicate'  (simulation reference only!)
        true_drop_rate_per_min  true drop rate at that moment (simulation reference)
        true_drop_factor        true tubing drop factor (simulation reference)
    The EKF pipeline must use only `timestamp`.
    """
    cfg = cfg or load_config()
    if rng is None:
        rng = spawn_rngs(cfg.simulation.random_seed)["drop_sensor"]
    ds = cfg.drop_sensor

    times, local_rate = true_drop_times(true_df)
    n = len(times)
    interval_s = 60.0 / local_rate
    t_end = float(times[-1]) if n else 0.0

    # ---- all random draws up front, in a fixed order (reproducibility) ----
    jitter_z = rng.standard_normal(n)
    miss_u = rng.random(n)
    dup_u = rng.random(n)
    dup_offset = rng.uniform(0.2, 0.6, n)
    n_false = rng.poisson(ds.false_detection_rate_per_min * t_end / 60.0)
    false_times = rng.uniform(0.0, t_end, n_false)

    # ---- per-drop miss / duplicate probabilities (scenario aware) ----
    miss_p = np.full(n, ds.miss_probability)
    dup_p = np.zeros(n)
    if scenario is not None:
        in_window = (times >= scenario.start_min * 60.0) & (times < scenario.end_min * 60.0)
        if scenario.drop_miss_probability is not None:
            miss_p[in_window] = scenario.drop_miss_probability
        scale = scenario.drop_rate_scale
        if scale < 1.0:        # under-counting: extra misses
            miss_p[in_window] = 1.0 - (1.0 - miss_p[in_window]) * scale
        elif scale > 1.0:      # over-counting: duplicates
            dup_p[in_window] = min(scale - 1.0, 1.0)

    detected = miss_u >= miss_p
    duplicated = detected & (dup_u < dup_p)

    # ---- jitter ----
    jitter = np.clip(jitter_z * ds.jitter_std_fraction,
                     -MAX_JITTER_FRACTION, MAX_JITTER_FRACTION) * interval_s
    detected_times = np.maximum(times + jitter, 0.0)
    duplicate_times = detected_times + dup_offset * interval_s

    events = pd.concat([
        pd.DataFrame({"timestamp": detected_times[detected], "event_type": "real"}),
        pd.DataFrame({"timestamp": duplicate_times[duplicated], "event_type": "duplicate"}),
        pd.DataFrame({"timestamp": false_times, "event_type": "false"}),
    ], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)

    # simulation-reference columns
    t_full = true_df["timestamp"].to_numpy(dtype=float)
    events["drop_detected"] = 1
    events["true_drop_rate_per_min"] = np.interp(
        events["timestamp"], t_full, true_df["true_drop_rate_per_min"].to_numpy())
    events["true_drop_factor"] = cfg.drop_sensor.actual_drop_factor
    return events[["timestamp", "drop_detected", "event_type",
                   "true_drop_rate_per_min", "true_drop_factor"]]


# ----------------------------------------------------------------------
# Demo:  python -m src.simulation.drop_sensor
# ----------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    truth = generate_true_infusion(cfg)
    drops = simulate_drop_sensor(truth, cfg)

    true_total = int(np.floor(truth["true_drops_cum"].iloc[-1] + 1e-6))
    counts = drops["event_type"].value_counts()
    real, false = int(counts.get("real", 0)), int(counts.get("false", 0))

    # 60 s window counts vs the true number of drops in each window
    t_end = truth["timestamp"].iloc[-1]
    edges = np.arange(0.0, t_end + 1, 60.0)
    detected_counts = np.histogram(drops["timestamp"], bins=edges)[0]
    true_counts = np.diff(np.interp(edges, truth["timestamp"], truth["true_drops_cum"]))
    err = detected_counts - true_counts

    mean_true_rate = true_counts.mean()
    flow_nominal = mean_true_rate / cfg.drop_sensor.nominal_drop_factor
    flow_actual = mean_true_rate / cfg.drop_sensor.actual_drop_factor

    print("=== Virtual IR drop counter ===")
    print(f"True drops                  : {true_total}")
    print(f"Detected events (total)     : {len(drops)}")
    print(f"  real drops detected       : {real}  ({true_total - real} missed, "
          f"{100 * (true_total - real) / true_total:.1f}%)")
    print(f"  false detections          : {false}")
    print(f"Mean drop rate (true)       : {mean_true_rate:.1f} drops/min")
    print("--- 60 s window drop counts ---")
    print(f"Count error (detected-true) : mean {err.mean():+.2f}, std {err.std():.2f} drops/min")
    print("--- why the drop factor matters ---")
    print(f"Flow from drops / actual DF : {flow_actual:.3f} mL/min (correct)")
    print(f"Flow from drops / nominal DF: {flow_nominal:.3f} mL/min "
          f"({100 * (flow_nominal / flow_actual - 1):+.1f}% error if printed DF is trusted)")
    print("\nFirst 5 rows:")
    print(drops.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()