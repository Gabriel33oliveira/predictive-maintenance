import pandas as pd
import pandas.testing as pdt

from predmaint.io import load_csv, save_csv
from predmaint.simulator import COLUMNS, simulate_fleet


def test_csv_round_trip_preserves_schema(tmp_path):
    df, _ = simulate_fleet(4, 48, seed=3)
    path = save_csv(df, tmp_path / "out" / "readings.csv")
    loaded = load_csv(path)

    assert list(loaded.columns) == COLUMNS
    assert isinstance(loaded["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    for col in ("vibration_rms", "temperature_c"):
        assert loaded[col].dtype == df[col].dtype == "float64"
    pdt.assert_series_equal(loaded["timestamp"], df["timestamp"])
    pdt.assert_frame_equal(loaded, df, check_exact=False)
