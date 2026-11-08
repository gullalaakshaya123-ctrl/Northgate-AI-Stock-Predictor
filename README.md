# Northgate AI Stock Predictor

An educational AI-powered stock-market analysis prototype built with Python, machine learning, Streamlit, and public market/news feeds.

## Features
- Historical OHLCV market-data download through Yahoo Finance (`yfinance`)
- Exploratory price visualization and technical/lag feature engineering
- Five-day (configurable 1/5/10 trading-day) future-return prediction
- Random Forest and HistGradientBoosting comparison using a chronological holdout
- MAE, RMSE, and R² evaluation
- Financial headline sentiment scoring using VADER and Google News RSS
- Illustrative long-only portfolio allocation optimization using historical returns
- Interactive Streamlit dashboard

## Important limitation
This is a student learning project, not a professional trading system. Stock prices are noisy and difficult to predict. Model estimates and portfolio weights are experimental, based on historical data, and are **not financial advice** or guaranteed returns. News sentiment is a simple headline-level proxy, not a full financial-language model.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run dashboard
```powershell
streamlit run app.py
```
Enter a ticker such as `AAPL`, select the history period and forecast horizon, then click **Analyze stock**. News headlines and portfolio optimization can be run from their respective sections.

## Run tests
```powershell
pytest -q
```

## Project structure
```text
Northgate_AI_Stock_Predictor/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── market.py
│   ├── model.py
│   ├── news.py
│   └── portfolio.py
└── tests/
    └── test_market.py
```

## Data and reproducibility
Market prices and news headlines are fetched at runtime. No large downloaded datasets are committed. Results can vary as markets and news change. An internet connection is required for live market/news features.

## Author
Akshaya Gullala
