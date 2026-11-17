"""CSV helpers for sensor data frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` as CSV (ISO-8601 timestamps), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by :func:`save_csv`, parsing ``timestamp`` as UTC."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns")
    return df
