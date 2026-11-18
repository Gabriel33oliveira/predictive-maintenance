"""Rolling-statistics anomaly detection on per-machine sensor streams."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = {"vibration_rms": "vibration_z", "temperature_c": "temperature_z"}
# Length (hours/rows) of the recent window used for the baseline-drift check.
RECENT_WINDOW = 24
# Signal used for the baseline-drift ratio.
BASELINE_SIGNAL = "vibration_rms"

SUMMARY_COLUMNS = [
    "machine_id",
    "anomaly_count",
    "alarm_count",
    "first_alarm_timestamp",
    "max_vibration_z",
    "max_temperature_z",
    "latest_baseline_deviation",
]


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Z-score of each point against the previous ``window`` points.

    The rolling mean and std are shifted by one so the current point never
    contributes to its own baseline. Points with fewer than ``min_periods``
    prior observations are NaN. Where the prior std is zero (a perfectly flat
    history) the z-score is 0 instead of a division error or infinity.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if min_periods is None:
        min_periods = window
    min_periods = max(2, min(min_periods, window))

    values = pd.Series(series, dtype=float)
    prior = values.shift(1).rolling(window, min_periods=min_periods)
    mean = prior.mean()
    std = prior.std()
    flat = std.notna() & (std <= 0)
    z = (values - mean) / std.where(~flat)
    z = z.where(~flat, 0.0)
    return z.rename(series.name)


def _persistent(flags: pd.Series, persistence: int) -> pd.Series:
    """True where ``flags`` has been True for at least ``persistence`` consecutive rows."""
    run_id = (flags != flags.shift()).cumsum()
    run_length = flags.groupby(run_id).cumcount() + 1
    return flags & (run_length >= persistence)


def _detect_one(group: pd.DataFrame, window: int, z_threshold: float, persistence: int) -> pd.DataFrame:
    min_periods = max(2, window // 2)
    out = pd.DataFrame(index=group.index)
    for signal, z_col in SIGNALS.items():
        out[z_col] = rolling_zscore(group[signal], window, min_periods)
    is_anomaly = (out["vibration_z"].abs() > z_threshold) | (out["temperature_z"].abs() > z_threshold)
    out["is_anomaly"] = is_anomaly
    out["alarm"] = _persistent(is_anomaly, persistence)

    baseline = group[BASELINE_SIGNAL].iloc[:window].mean()
    recent = group[BASELINE_SIGNAL].rolling(RECENT_WINDOW, min_periods=1).mean()
    out["baseline_deviation"] = recent / baseline if baseline > 0 else np.nan
    return out


def detect_anomalies(
    df: pd.DataFrame, window: int = 48, z_threshold: float = 3.0, persistence: int = 3
) -> pd.DataFrame:
    """Flag anomalous readings per machine.

    Returns a copy of ``df`` with added columns:

    - ``vibration_z`` / ``temperature_z``: rolling z-scores (see :func:`rolling_zscore`).
    - ``is_anomaly``: either ``|z|`` exceeds ``z_threshold``.
    - ``alarm``: the anomaly has persisted for at least ``persistence``
      consecutive readings, which suppresses single-point noise.
    - ``baseline_deviation``: rolling mean of the last 24 vibration readings
      divided by the mean of the machine's first ``window`` readings
      (1.0 means no drift).

    Statistics never cross machines. Rows are processed in timestamp order
    within each machine; the output keeps the input's row order and index.
    """
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    ordered = df.sort_values(["machine_id", "timestamp"], kind="stable") if "timestamp" in df else df
    parts = [
        _detect_one(group, window, z_threshold, persistence)
        for _, group in ordered.groupby("machine_id", sort=False)
    ]
    cols = [*SIGNALS.values(), "is_anomaly", "alarm", "baseline_deviation"]
    added = pd.concat(parts) if parts else pd.DataFrame(columns=cols)
    result = df.copy()
    for col in cols:
        result[col] = added[col].reindex(df.index)
    result["is_anomaly"] = result["is_anomaly"].astype(bool)
    result["alarm"] = result["alarm"].astype(bool)
    return result


def summarize_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per machine summarizing the output of :func:`detect_anomalies`."""
    rows = []
    for machine_id, group in df.groupby("machine_id", sort=True):
        if "timestamp" in group:
            group = group.sort_values("timestamp", kind="stable")
            alarm_times = group.loc[group["alarm"], "timestamp"]
            first_alarm = alarm_times.iloc[0] if len(alarm_times) else pd.NaT
        else:
            first_alarm = pd.NaT
        rows.append(
            {
                "machine_id": machine_id,
                "anomaly_count": int(group["is_anomaly"].sum()),
                "alarm_count": int(group["alarm"].sum()),
                "first_alarm_timestamp": first_alarm,
                "max_vibration_z": group["vibration_z"].abs().max(),
                "max_temperature_z": group["temperature_z"].abs().max(),
                "latest_baseline_deviation": group["baseline_deviation"].iloc[-1] if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
