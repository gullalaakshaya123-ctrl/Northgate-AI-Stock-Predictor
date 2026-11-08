import streamlit as st
import pandas as pd
import plotly.express as px

from src.market import fetch_history
from src.model import train_and_evaluate, get_feature_importance
from src.news import fetch_news_sentiment
from src.portfolio import optimize_portfolio

st.set_page_config(page_title="Northgate AI Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Northgate AI Stock Predictor")
st.caption("Educational market-analysis prototype | Not investment advice")

with st.sidebar:
    st.header("Market settings")
    ticker = st.text_input("Stock ticker", "AAPL").strip().upper()
    period = st.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=2)
    horizon = st.selectbox("Forecast horizon", [1, 5, 10], index=1)
    run = st.button("Analyze stock", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner(f"Downloading and analyzing {ticker}..."):
            history = fetch_history(ticker, period)
            metrics, best_name, predicted_return, predicted_price, n_train, n_test = train_and_evaluate(history, horizon)
            st.session_state["analysis"] = {
                "ticker": ticker, "history": history, "metrics": metrics,
                "best_name": best_name, "predicted_return": predicted_return,
                "predicted_price": predicted_price, "horizon": horizon,
                "train": n_train, "test": n_test
            }
    except Exception as exc:
        st.error(str(exc))

if "analysis" in st.session_state:
    a = st.session_state["analysis"]
    hist = a["history"]
    latest = float(hist["close"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest adjusted close", f"${latest:,.2f}")
    c2.metric(f"Estimated price in {a['horizon']} trading day(s)", f"${a['predicted_price']:,.2f}",
              f"{a['predicted_return']*100:+.2f}% model estimate")
    c3.metric("Selected model", a["best_name"])
    c4.metric("Training / test rows", f"{a['train']} / {a['test']}")
    st.subheader(f"{a['ticker']} historical price")
    st.plotly_chart(px.line(hist.reset_index(), x=hist.index.name or "Date", y="close",
                            labels={"close": "Adjusted close (USD)"}, title="Historical adjusted closing price"),
                    use_container_width=True)
    st.subheader("Model evaluation (chronological holdout)")
    st.dataframe(a["metrics"].style.format({"MAE (return)": "{:.4f}", "RMSE (return)": "{:.4f}", "R²": "{:.3f}"}),
                 use_container_width=True)
    st.caption("A lower MAE/RMSE is better. Historical test performance does not guarantee future accuracy.")
    st.subheader("Feature importance")
    st.info("Feature importance is available for the Random Forest model. The selected model is evaluated above.")
    try:
        from src.model import FEATURES
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        # Display explanatory feature definitions, avoiding claims about a different fitted model.
        st.write(pd.DataFrame({"Feature": FEATURES, "Meaning": [
            "Previous trading-day return", "Previous 5-day return", "Previous 10-day return",
            "10-day return volatility", "Price relative to 5-day average",
            "Price relative to 20-day average", "Price relative to 50-day average",
            "Daily volume percentage change"
        ]}))
    except Exception:
        pass
    st.subheader("Financial news sentiment")
    if st.button("Load latest news headlines"):
        try:
            news = fetch_news_sentiment(a["ticker"])
            if news:
                news_df = pd.DataFrame(news)
                st.metric("Average headline sentiment", f"{news_df['Sentiment'].mean():+.3f}")
                st.dataframe(news_df, use_container_width=True)
            else:
                st.info("No headlines were returned. Try again later.")
        except Exception as exc:
            st.warning(f"News could not be loaded: {exc}")

st.divider()
st.subheader("Portfolio allocation optimizer")
st.write("Explore a long-only, fully-invested allocation using historical return/risk estimates.")
assets = st.text_input("Tickers separated by commas", "AAPL, MSFT, GOOGL")
if st.button("Optimize illustrative allocation"):
    try:
        allocation = optimize_portfolio(assets.split(","))
        st.plotly_chart(px.pie(allocation, names="Ticker", values="Weight", title="Illustrative portfolio weights"),
                        use_container_width=True)
        st.dataframe(allocation.style.format({"Weight": "{:.1%}"}), use_container_width=True)
        st.caption("Optimization uses historical sample data and assumptions; it is not personalized financial advice.")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.caption("Model estimates are experimental. Markets are volatile; do not use this tool as the sole basis for investment decisions.")
