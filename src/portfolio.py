from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def optimize_portfolio(tickers, period="2y"):
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if len(tickers) < 2:
        raise ValueError("Enter at least two comma-separated ticker symbols.")
    prices = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    returns = prices.pct_change().dropna(how="any")
    returns = returns.dropna(axis=1)
    if returns.shape[1] < 2 or len(returns) < 60:
        raise ValueError("Not enough overlapping price history for these assets.")
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(mu)
    def objective(w):
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -float(w @ mu) / vol
    result = minimize(objective, np.ones(n) / n, method="SLSQP",
                      bounds=[(0, 1)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                      options={"maxiter": 500})
    weights = result.x if result.success else np.ones(n) / n
    return pd.DataFrame({"Ticker": returns.columns, "Weight": weights}).sort_values("Weight", ascending=False)
