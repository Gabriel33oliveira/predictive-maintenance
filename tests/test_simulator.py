import pandas as pd
import pandas.testing as pdt
import pytest

from predmaint.simulator import COLUMNS, MachineSpec, simulate_fleet, simulate_machine

HOURS = 400
START = 200


def make_spec(degradation="none", rate=0.0, start=START):
    return MachineSpec(
        machine_id="T001",
        machine_type="pump",
        baseline_vibration_mm_s=2.5,
        baseline_temp_c=50.0,
        noise_level=0.15,
        degradation=degradation,
        degradation_start=start,
        degradation_rate=rate,
    )


def test_fleet_columns_and_row_count():
    df, specs = simulate_fleet(6, 72, seed=1)
    assert list(df.columns) == COLUMNS
    assert len(df) == 6 * 72
    assert len(specs) == 6
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert df.groupby("machine_id").size().eq(72).all()


def test_same_seed_identical_different_seed_differs():
    a, specs_a = simulate_fleet(5, 100, seed=42)
    b, specs_b = simulate_fleet(5, 100, seed=42)
    c, _ = simulate_fleet(5, 100, seed=43)
    pdt.assert_frame_equal(a, b)
    assert specs_a == specs_b
    assert not a["vibration_rms"].equals(c["vibration_rms"])


def test_simulate_machine_deterministic():
    spec = make_spec("linear_wear", 0.01)
    pdt.assert_frame_equal(simulate_machine(spec, HOURS, 7), simulate_machine(spec, HOURS, 7))


def test_healthy_machine_is_stable():
    df = simulate_machine(make_spec(), HOURS, seed=0)
    q = HOURS // 4
    first = df["vibration_rms"].iloc[:q].mean()
    last = df["vibration_rms"].iloc[-q:].mean()
    assert abs(first - last) < 0.1


@pytest.mark.parametrize("degradation,rate", [("linear_wear", 0.02), ("exponential_wear", 0.012)])
def test_wear_increases_vibration(degradation, rate):
    df = simulate_machine(make_spec(degradation, rate), HOURS, seed=0)
    before = df["vibration_rms"].iloc[:START].mean()
    end = df["vibration_rms"].iloc[-50:].mean()
    assert end > before + 1.0


def test_overheating_raises_late_temperature():
    df = simulate_machine(make_spec("overheating", 0.1), HOURS, seed=0)
    before = df["temperature_c"].iloc[:START].mean()
    end = df["temperature_c"].iloc[-50:].mean()
    assert end > before + 10.0


def test_step_fault_jumps_at_start():
    df = simulate_machine(make_spec("step_fault", 0.001), HOURS, seed=0)
    vib = df["vibration_rms"]
    before = vib.iloc[START - 24 : START].mean()
    after = vib.iloc[START : START + 24].mean()
    # Jump equals the baseline vibration (2.5 mm/s).
    assert after - before > 2.0


def test_fleet_has_healthy_and_degrading_machines():
    _, specs = simulate_fleet(6, 100, seed=5)
    kinds = [s.degradation for s in specs]
    assert "none" in kinds
    assert any(k != "none" for k in kinds)
    assert {s.machine_type for s in specs} <= {"pump", "motor"}


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):
        make_spec("bogus")
