import pandas as pd
import numpy as np
from src.market import make_features, FEATURES

def test_feature_builder_creates_expected_columns():
    idx = pd.date_range("2022-01-01", periods=100, freq="B")
    data = pd.DataFrame({
        "close": np.linspace(100, 150, 100),
        "volume": np.linspace(1000, 2000, 100)
    }, index=idx)
    result = make_features(data, horizon=5)
    assert set(FEATURES).issubset(result.columns)
    assert "target_return" in result.columns
    assert len(result) == 100

def test_target_is_future_return():
    idx = pd.date_range("2022-01-01", periods=20, freq="B")
    close = np.arange(100, 120, dtype=float)
    data = pd.DataFrame({"close": close, "volume": 1000}, index=idx)
    result = make_features(data, horizon=2)
    assert np.isclose(result["target_return"].iloc[0], close[2] / close[0] - 1)
