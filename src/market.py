from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV data. Raises a readable error if no data is returned."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please enter a stock ticker, for example AAPL or MSFT.")
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No market data found for {ticker}. Check the ticker or internet connection.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]
    required = {"close", "volume"}
    if not required.issubset(data.columns):
        raise RuntimeError("The downloaded data is missing required Close or Volume columns.")
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    return data.dropna(subset=["close"])

def make_features(data: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create lag/technical features and a future-return target for model training."""
    df = data.copy()
    close = df["close"].astype(float)
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["sma_5_ratio"] = close / close.rolling(5).mean() - 1
    df["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma_50_ratio"] = close / close.rolling(50).mean() - 1
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["target_return"] = close.shift(-horizon) / close - 1
    return df.replace([np.inf, -np.inf], np.nan)

FEATURES = [
    "return_1d", "return_5d", "return_10d", "volatility_10d",
    "sma_5_ratio", "sma_20_ratio", "sma_50_ratio", "volume_change"
]
