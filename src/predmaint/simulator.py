"""Synthetic vibration and temperature data with injected degradation patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START_TIMESTAMP = pd.Timestamp("2026-01-01", tz="UTC")
COLUMNS = ["timestamp", "machine_id", "machine_type", "vibration_rms", "temperature_c"]
MACHINE_TYPES = ("pump", "motor")
DEGRADATIONS = ("none", "linear_wear", "exponential_wear", "step_fault", "overheating")

# Temperature noise (deg C) per unit of vibration noise_level (mm/s).
TEMP_NOISE_FACTOR = 5.0
# Amplitude (deg C) of the daily temperature cycle.
DAILY_TEMP_AMPLITUDE = 2.0
# Share of the vibration wear term that also shows up as extra temperature (deg C per mm/s).
WEAR_TEMP_FRACTION = 0.3
# Scale (mm/s) of the exponential wear term: scale * (exp(rate * dt) - 1).
EXP_WEAR_SCALE = 0.5
# Sudden vibration jump of a step fault, as a multiple of the baseline vibration.
STEP_FAULT_JUMP_FACTOR = 1.0


@dataclass(frozen=True)
class MachineSpec:
    """Parameters for one simulated machine.

    ``noise_level`` is the standard deviation of vibration noise in mm/s;
    temperature noise is ``noise_level * TEMP_NOISE_FACTOR`` deg C.
    ``degradation_start`` is the hour index at which degradation begins and
    ``degradation_rate`` its per-hour rate (meaning depends on the pattern).
    """

    machine_id: str
    machine_type: str
    baseline_vibration_mm_s: float
    baseline_temp_c: float
    noise_level: float
    degradation: str = "none"
    degradation_start: int = 0
    degradation_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.machine_type not in MACHINE_TYPES:
            raise ValueError(f"machine_type must be one of {MACHINE_TYPES}, got {self.machine_type!r}")
        if self.degradation not in DEGRADATIONS:
            raise ValueError(f"degradation must be one of {DEGRADATIONS}, got {self.degradation!r}")
        if self.degradation_start < 0:
            raise ValueError("degradation_start must be non-negative")


def _degradation_terms(spec: MachineSpec, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return additive (vibration, temperature) offsets for the spec's pattern."""
    active = t >= spec.degradation_start
    dt = np.where(active, t - spec.degradation_start, 0).astype(float)
    vib = np.zeros_like(dt)
    temp = np.zeros_like(dt)
    rate = spec.degradation_rate

    if spec.degradation == "linear_wear":
        vib = rate * dt
        temp = WEAR_TEMP_FRACTION * vib
    elif spec.degradation == "exponential_wear":
        vib = EXP_WEAR_SCALE * (np.exp(rate * dt) - 1.0)
        temp = WEAR_TEMP_FRACTION * vib
    elif spec.degradation == "step_fault":
        jump = STEP_FAULT_JUMP_FACTOR * spec.baseline_vibration_mm_s
        vib = np.where(active, jump, 0.0) + rate * dt
        temp = WEAR_TEMP_FRACTION * vib
    elif spec.degradation == "overheating":
        temp = rate * dt
    return vib, temp


def simulate_machine(spec: MachineSpec, hours: int, seed: int) -> pd.DataFrame:
    """Simulate ``hours`` hourly readings for one machine."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(hours)

    vibration = spec.baseline_vibration_mm_s + rng.normal(0.0, spec.noise_level, hours)
    temperature = (
        spec.baseline_temp_c
        + DAILY_TEMP_AMPLITUDE * np.sin(2 * np.pi * t / 24.0)
        + rng.normal(0.0, spec.noise_level * TEMP_NOISE_FACTOR, hours)
    )
    vib_offset, temp_offset = _degradation_terms(spec, t)
    vibration = np.clip(vibration + vib_offset, 0.0, None)
    temperature = temperature + temp_offset

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(START_TIMESTAMP, periods=hours, freq="h").as_unit("ns"),
            "machine_id": spec.machine_id,
            "machine_type": spec.machine_type,
            "vibration_rms": vibration,
            "temperature_c": temperature,
        },
        columns=COLUMNS,
    )


def _random_spec(rng: np.random.Generator, index: int, degradation: str, hours: int) -> MachineSpec:
    machine_type = MACHINE_TYPES[int(rng.integers(len(MACHINE_TYPES)))]
    if machine_type == "pump":
        base_vib = rng.uniform(2.0, 3.5)
        base_temp = rng.uniform(45.0, 60.0)
    else:
        base_vib = rng.uniform(1.0, 2.5)
        base_temp = rng.uniform(55.0, 70.0)
    noise = base_vib * rng.uniform(0.05, 0.10)

    start, rate = 0, 0.0
    if degradation != "none":
        start = int(hours * rng.uniform(0.3, 0.6))
        span = max(hours - start, 1)
        # Total vibration growth (mm/s) reached by the last hour.
        growth = base_vib * rng.uniform(1.0, 2.5)
        if degradation == "linear_wear":
            rate = growth / span
        elif degradation == "exponential_wear":
            rate = float(np.log1p(growth / EXP_WEAR_SCALE)) / span
        elif degradation == "step_fault":
            rate = 0.5 * growth / span
        elif degradation == "overheating":
            rate = rng.uniform(15.0, 30.0) / span

    return MachineSpec(
        machine_id=f"M{index + 1:03d}",
        machine_type=machine_type,
        baseline_vibration_mm_s=float(base_vib),
        baseline_temp_c=float(base_temp),
        noise_level=float(noise),
        degradation=degradation,
        degradation_start=start,
        degradation_rate=float(rate),
    )


def simulate_fleet(n_machines: int, hours: int, seed: int) -> tuple[pd.DataFrame, list[MachineSpec]]:
    """Build a deterministic mixed fleet and simulate every machine.

    Roughly a third of the machines are healthy; the rest cycle through the
    degradation patterns. With two or more machines there is always at least
    one healthy and one degrading machine.
    """
    if n_machines <= 0:
        raise ValueError("n_machines must be positive")
    rng = np.random.default_rng(seed)

    n_healthy = max(1, round(n_machines / 3))
    if n_machines >= 2:
        n_healthy = min(n_healthy, n_machines - 1)
    faulty = DEGRADATIONS[1:]
    degradations = ["none"] * n_healthy + [faulty[i % len(faulty)] for i in range(n_machines - n_healthy)]
    degradations = [degradations[i] for i in rng.permutation(n_machines)]

    specs = [_random_spec(rng, i, d, hours) for i, d in enumerate(degradations)]
    machine_seeds = rng.integers(0, 2**32, size=n_machines)
    frames = [simulate_machine(s, hours, int(ms)) for s, ms in zip(specs, machine_seeds)]
    return pd.concat(frames, ignore_index=True), specs
