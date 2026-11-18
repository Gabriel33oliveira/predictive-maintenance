import numpy as np
import pandas as pd
import pytest

from predmaint.anomaly import SUMMARY_COLUMNS, detect_anomalies, rolling_zscore, summarize_anomalies
from predmaint.simulator import START_TIMESTAMP, simulate_fleet


def make_machine(machine_id, vibration, temperature=None, seed=0):
    n = len(vibration)
    if temperature is None:
        temperature = 50.0 + np.random.default_rng(seed).normal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(START_TIMESTAMP, periods=n, freq="h"),
            "machine_id": machine_id,
            "machine_type": "pump",
            "vibration_rms": vibration,
            "temperature_c": temperature,
        }
    )


def flat_noise(n=200, seed=1, level=2.0, sigma=0.05):
    return level + np.random.default_rng(seed).normal(0.0, sigma, n)


def test_rolling_zscore_constant_series_is_zero_and_finite():
    z = rolling_zscore(pd.Series(np.full(100, 3.0)), window=10, min_periods=5)
    valid = z.dropna()
    assert len(valid) == 100 - 5
    assert np.isfinite(valid).all()
    assert (valid == 0.0).all()


def test_rolling_zscore_excludes_current_point():
    s = pd.Series([1.0, 2.0, 1.0, 2.0, 10.0])
    z = rolling_zscore(s, window=4, min_periods=4)
    prior = s.iloc[:4]
    assert z.iloc[4] == pytest.approx((10.0 - prior.mean()) / prior.std())


def test_single_spike_is_anomaly_but_not_alarm():
    vib = flat_noise()
    vib[120] += 2.0
    out = detect_anomalies(make_machine("M001", vib), window=48, z_threshold=3.0, persistence=3)
    assert out.loc[120, "is_anomaly"]
    assert not out["alarm"].any()


def test_sustained_shift_raises_alarm():
    vib = flat_noise()
    vib[120:] += 1.0
    out = detect_anomalies(make_machine("M001", vib), persistence=3)
    assert out["alarm"].any()
    assert not out.loc[:119, "alarm"].any()
    assert out.loc[199, "baseline_deviation"] == pytest.approx(1.5, rel=0.05)


def test_grouping_does_not_leak_between_machines():
    quiet = make_machine("A", flat_noise(seed=2))
    loud = make_machine("B", flat_noise(seed=3, level=20.0, sigma=5.0))
    alone = detect_anomalies(quiet)
    combined = detect_anomalies(pd.concat([loud, quiet], ignore_index=True))
    from_combined = combined[combined["machine_id"] == "A"].reset_index(drop=True)
    cols = ["vibration_z", "temperature_z", "is_anomaly", "alarm", "baseline_deviation"]
    pd.testing.assert_frame_equal(from_combined[cols], alone[cols])


def test_input_is_not_mutated():
    df = make_machine("M001", flat_noise())
    before = df.copy()
    out = detect_anomalies(df)
    pd.testing.assert_frame_equal(df, before)
    assert out is not df
    summarize_anomalies(out)
    assert list(df.columns) == list(before.columns)


def test_summary_has_one_row_per_machine():
    spiky = flat_noise(seed=4)
    spiky[100:] += 1.0
    df = pd.concat([make_machine("A", flat_noise(seed=5)), make_machine("B", spiky)], ignore_index=True)
    summary = summarize_anomalies(detect_anomalies(df))
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary["machine_id"]) == ["A", "B"]
    b = summary.set_index("machine_id").loc["B"]
    assert b["alarm_count"] > 0
    assert b["first_alarm_timestamp"] >= START_TIMESTAMP + pd.Timedelta(hours=100)


def test_summary_first_alarm_is_nat_without_alarms():
    summary = summarize_anomalies(detect_anomalies(make_machine("A", np.full(100, 2.0), np.full(100, 50.0))))
    assert summary.loc[0, "alarm_count"] == 0
    assert pd.isna(summary.loc[0, "first_alarm_timestamp"])


def test_fleet_healthy_machines_alarm_less_than_step_faults():
    df, specs = simulate_fleet(n_machines=12, hours=500, seed=7)
    summary = summarize_anomalies(detect_anomalies(df)).set_index("machine_id")
    healthy = [s.machine_id for s in specs if s.degradation == "none"]
    faulty = [s.machine_id for s in specs if s.degradation == "step_fault"]
    assert healthy and faulty
    assert summary.loc[healthy, "alarm_count"].max() < summary.loc[faulty, "alarm_count"].min()
