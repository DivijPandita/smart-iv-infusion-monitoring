"""Configuration loading, overriding and validation.

Usage:
    from src.config import load_config
    cfg = load_config()                                    # defaults from config/config.yaml
    cfg = load_config(overrides={"load_cell.noise_std_g": 0.5})
    print(cfg.bag.initial_volume_ml)
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


# ----------------------------------------------------------------------
# Section definitions (defaults mirror config.yaml; YAML values win)
# ----------------------------------------------------------------------
@dataclass
class PathsConfig:
    raw_load_cell_file: str = "data/raw/load_cell_data.csv"
    raw_drop_file: str = "data/raw/drop_counter_data.csv"
    processed_results_file: str = "data/processed/ekf_results.csv"
    experiments_dir: str = "results"


@dataclass
class BagConfig:
    initial_volume_ml: float = 500.0
    empty_bag_weight_g: float = 25.0
    fluid_density_g_per_ml: float = 1.0


@dataclass
class SimulationConfig:
    random_seed: int = 42
    start_datetime: str = "2025-01-01 12:00:00"
    max_duration_min: float = 240.0
    sample_rate_hz: float = 1.0
    stop_when_empty: bool = True


@dataclass
class FlowConfig:
    profile: str = "piecewise"
    base_flow_ml_min: float = 2.5
    segments: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"duration_min": 5.0, "scale": 1.00},
            {"duration_min": 5.0, "scale": 1.20},
            {"duration_min": 5.0, "scale": 0.88},
            {"duration_min": 5.0, "scale": 1.12},
        ]
    )
    repeat_segments: bool = True
    transition_s: float = 30.0
    variability_std_ml_min: float = 0.08
    variability_tau_min: float = 5.0
    smooth_amplitude_ml_min: float = 0.4
    smooth_period_min: float = 40.0


@dataclass
class LoadCellConfig:
    noise_std_g: float = 0.20
    vibration_amplitude_g: float = 0.15
    vibration_freq_hz: float = 0.35
    bias_std_g: float = 0.30
    bias_tau_min: float = 30.0
    spike_probability: float = 0.002
    spike_magnitude_g: float = 6.0
    resolution_g: float = 0.01


@dataclass
class DropSensorConfig:
    nominal_drop_factor: float = 15.0
    actual_drop_factor: float = 14.5
    miss_probability: float = 0.03
    false_detection_rate_per_min: float = 0.5
    jitter_std_fraction: float = 0.05


@dataclass
class PreprocessingConfig:
    window_size_s: float = 60.0
    window_step_s: float = 60.0
    max_gap_s: float = 5.0
    spike_filter_enabled: bool = True
    spike_filter_window_samples: int = 9
    spike_mad_threshold: float = 6.0
    smoothing_enabled: bool = True
    smoothing_samples: int = 5


@dataclass
class EKFConfig:
    sensor_mode: str = "fusion"
    initial_flow_ml_min: float = 2.0
    initial_drop_factor: float = 15.0
    initial_flow_std: float = 1.0
    initial_drop_factor_std: float = 2.0
    q_flow_var_per_min: float = 0.01
    q_drop_factor_var_per_min: float = 0.0001
    r_weight_flow_std_ml_min: float = 0.30
    r_drop_rate_std_drops_min: float = 2.0


@dataclass
class PredictionConfig:
    confidence_z: float = 1.96
    min_flow_ml_min: float = 0.05


@dataclass
class AlertsConfig:
    warning_min: float = 30.0
    critical_min: float = 10.0
    disagreement_relative_threshold: float = 0.25


@dataclass
class DashboardConfig:
    title: str = "Smart IV Infusion Monitoring: Software Prototype"
    replay_speed: float = 120.0
    update_interval_ms: int = 400


@dataclass
class FailureScenario:
    description: str = ""
    start_min: float = 0.0
    end_min: float = 0.0
    load_cell_noise_multiplier: float = 1.0
    drop_miss_probability: float | None = None   # None -> keep the normal value
    drop_rate_scale: float = 1.0


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    bag: BagConfig = field(default_factory=BagConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    load_cell: LoadCellConfig = field(default_factory=LoadCellConfig)
    drop_sensor: DropSensorConfig = field(default_factory=DropSensorConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    failure_scenarios: dict[str, FailureScenario] = field(default_factory=dict)

    # ---- derived quantities (computed, never stored in YAML) ----
    @property
    def initial_fluid_weight_g(self) -> float:
        return self.bag.initial_volume_ml * self.bag.fluid_density_g_per_ml

    @property
    def initial_total_weight_g(self) -> float:
        """What the load cell reads at t = 0: fluid + empty bag."""
        return self.initial_fluid_weight_g + self.bag.empty_bag_weight_g

    def mean_flow_ml_min(self) -> float:
        """Average true flow over the profile (rough, ignores the random wiggle)."""
        if self.flow.profile == "piecewise":
            total = sum(s["duration_min"] for s in self.flow.segments)
            weighted = sum(s["duration_min"] * s["scale"] for s in self.flow.segments)
            return self.flow.base_flow_ml_min * weighted / total
        return self.flow.base_flow_ml_min

    def expected_infusion_time_min(self) -> float:
        return self.bag.initial_volume_ml / self.mean_flow_ml_min()

    def project_path(self, relative: str) -> Path:
        """Turn a path from the config into an absolute path under the project root."""
        return PROJECT_ROOT / relative

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
_SECTION_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "bag": BagConfig,
    "simulation": SimulationConfig,
    "flow": FlowConfig,
    "load_cell": LoadCellConfig,
    "drop_sensor": DropSensorConfig,
    "preprocessing": PreprocessingConfig,
    "ekf": EKFConfig,
    "prediction": PredictionConfig,
    "alerts": AlertsConfig,
    "dashboard": DashboardConfig,
}


def _build(cls: type, data: dict[str, Any] | None, section: str):
    """Create a dataclass from a dict, rejecting unknown keys (catches typos)."""
    data = dict(data or {})
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}': {sorted(unknown)}. Valid keys: {sorted(valid)}"
        )
    return cls(**data)


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply {'section.key': value} overrides in place."""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot override '{dotted_key}': '{part}' is not a section")
        node[parts[-1]] = value


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load, override and validate the configuration."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top level of the config file must be a mapping (key: value)")

    raw = copy.deepcopy(raw)
    if overrides:
        _apply_overrides(raw, overrides)

    known_top = set(_SECTION_CLASSES) | {"failure_scenarios"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"Unknown top-level section(s): {sorted(unknown_top)}")

    sections = {name: _build(cls, raw.get(name), name) for name, cls in _SECTION_CLASSES.items()}
    scenarios = {
        name: _build(FailureScenario, body, f"failure_scenarios.{name}")
        for name, body in (raw.get("failure_scenarios") or {}).items()
    }

    cfg = AppConfig(failure_scenarios=scenarios, **sections)
    validate(cfg)
    return cfg


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate(cfg: AppConfig) -> None:
    """Check physical and logical sanity. Reports ALL problems at once."""
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    # bag
    check(cfg.bag.initial_volume_ml > 0, "bag.initial_volume_ml must be > 0")
    check(cfg.bag.empty_bag_weight_g >= 0, "bag.empty_bag_weight_g must be >= 0")
    check(cfg.bag.fluid_density_g_per_ml > 0, "bag.fluid_density_g_per_ml must be > 0")

    # simulation
    check(cfg.simulation.max_duration_min > 0, "simulation.max_duration_min must be > 0")
    check(cfg.simulation.sample_rate_hz > 0, "simulation.sample_rate_hz must be > 0")

    # flow
    check(cfg.flow.profile in {"piecewise", "smooth", "constant"},
          "flow.profile must be piecewise, smooth or constant")
    check(cfg.flow.base_flow_ml_min > 0, "flow.base_flow_ml_min must be > 0")
    check(cfg.flow.variability_std_ml_min >= 0, "flow.variability_std_ml_min must be >= 0")
    check(cfg.flow.variability_tau_min > 0, "flow.variability_tau_min must be > 0")
    if cfg.flow.profile == "piecewise":
        check(len(cfg.flow.segments) > 0, "flow.segments must not be empty")
        for i, seg in enumerate(cfg.flow.segments):
            check({"duration_min", "scale"} <= set(seg),
                  f"flow.segments[{i}] needs 'duration_min' and 'scale'")
            if {"duration_min", "scale"} <= set(seg):
                check(seg["duration_min"] > 0, f"flow.segments[{i}].duration_min must be > 0")
                check(seg["scale"] > 0, f"flow.segments[{i}].scale must be > 0")

    # sensors
    lc, ds = cfg.load_cell, cfg.drop_sensor
    check(lc.noise_std_g >= 0, "load_cell.noise_std_g must be >= 0")
    check(lc.bias_std_g >= 0, "load_cell.bias_std_g must be >= 0")
    check(lc.bias_tau_min > 0, "load_cell.bias_tau_min must be > 0")
    check(0 <= lc.spike_probability <= 1, "load_cell.spike_probability must be in [0, 1]")
    check(lc.resolution_g >= 0, "load_cell.resolution_g must be >= 0")
    check(ds.nominal_drop_factor > 0, "drop_sensor.nominal_drop_factor must be > 0")
    check(ds.actual_drop_factor > 0, "drop_sensor.actual_drop_factor must be > 0")
    check(0 <= ds.miss_probability < 1, "drop_sensor.miss_probability must be in [0, 1)")
    check(ds.false_detection_rate_per_min >= 0,
          "drop_sensor.false_detection_rate_per_min must be >= 0")
    check(ds.jitter_std_fraction >= 0, "drop_sensor.jitter_std_fraction must be >= 0")

    # preprocessing
    pp = cfg.preprocessing
    check(pp.window_size_s > 0, "preprocessing.window_size_s must be > 0")
    check(0 < pp.window_step_s <= pp.window_size_s,
          "preprocessing.window_step_s must be > 0 and <= window_size_s")
    check(pp.max_gap_s > 0, "preprocessing.max_gap_s must be > 0")
    check(pp.spike_filter_window_samples >= 3, "preprocessing.spike_filter_window_samples must be >= 3")
    check(pp.smoothing_samples >= 1, "preprocessing.smoothing_samples must be >= 1")

    # EKF
    e = cfg.ekf
    check(e.sensor_mode in {"fusion", "load_cell_only", "drop_only"},
          "ekf.sensor_mode must be fusion, load_cell_only or drop_only")
    check(e.initial_flow_ml_min > 0, "ekf.initial_flow_ml_min must be > 0")
    check(e.initial_drop_factor > 0, "ekf.initial_drop_factor must be > 0")
    for name in ("initial_flow_std", "initial_drop_factor_std",
                 "r_weight_flow_std_ml_min", "r_drop_rate_std_drops_min"):
        check(getattr(e, name) > 0, f"ekf.{name} must be > 0")
    for name in ("q_flow_var_per_min", "q_drop_factor_var_per_min"):
        check(getattr(e, name) >= 0, f"ekf.{name} must be >= 0")

    # prediction / alerts
    check(cfg.prediction.confidence_z > 0, "prediction.confidence_z must be > 0")
    check(cfg.prediction.min_flow_ml_min > 0, "prediction.min_flow_ml_min must be > 0")
    check(cfg.alerts.critical_min > 0, "alerts.critical_min must be > 0")
    check(cfg.alerts.critical_min < cfg.alerts.warning_min,
          "alerts.critical_min must be smaller than alerts.warning_min")
    check(cfg.alerts.disagreement_relative_threshold > 0,
          "alerts.disagreement_relative_threshold must be > 0")

    # dashboard
    check(cfg.dashboard.replay_speed > 0, "dashboard.replay_speed must be > 0")
    check(cfg.dashboard.update_interval_ms > 0, "dashboard.update_interval_ms must be > 0")

    # failure scenarios
    for name, sc in cfg.failure_scenarios.items():
        check(sc.end_min > sc.start_min >= 0,
              f"failure_scenarios.{name}: need 0 <= start_min < end_min")
        check(sc.load_cell_noise_multiplier > 0,
              f"failure_scenarios.{name}.load_cell_noise_multiplier must be > 0")
        check(sc.drop_rate_scale > 0, f"failure_scenarios.{name}.drop_rate_scale must be > 0")
        if sc.drop_miss_probability is not None:
            check(0 <= sc.drop_miss_probability < 1,
                  f"failure_scenarios.{name}.drop_miss_probability must be in [0, 1)")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


# ----------------------------------------------------------------------
# Quick self-check:  python -m src.config
# ----------------------------------------------------------------------
def _hhmm(minutes: float) -> str:
    return f"{int(minutes) // 60:02d}:{int(round(minutes % 60)):02d}"


def main() -> None:
    cfg = load_config()
    t = cfg.expected_infusion_time_min()
    print("=== Configuration loaded successfully ===")
    print(f"Config file        : {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Bag                : {cfg.bag.initial_volume_ml} mL fluid + "
          f"{cfg.bag.empty_bag_weight_g} g empty bag = {cfg.initial_total_weight_g} g total")
    print(f"Flow profile       : {cfg.flow.profile}, base {cfg.flow.base_flow_ml_min} mL/min")
    print(f"Expected duration  : ~{t:.1f} min ({_hhmm(t)})")
    print(f"Drop factor        : nominal {cfg.drop_sensor.nominal_drop_factor}, "
          f"actual {cfg.drop_sensor.actual_drop_factor} drops/mL")
    print(f"Window             : {cfg.preprocessing.window_size_s:.0f} s "
          f"(step {cfg.preprocessing.window_step_s:.0f} s)")
    print(f"EKF sensor mode    : {cfg.ekf.sensor_mode}")
    print(f"EKF initial state  : flow {cfg.ekf.initial_flow_ml_min} mL/min, "
          f"drop factor {cfg.ekf.initial_drop_factor} drops/mL")
    print(f"Alert thresholds   : warning {cfg.alerts.warning_min:.0f} min, "
          f"critical {cfg.alerts.critical_min:.0f} min")
    print(f"Failure scenarios  : {', '.join(cfg.failure_scenarios)}")
    print(f"Random seed        : {cfg.simulation.random_seed}")


if __name__ == "__main__":
    main()