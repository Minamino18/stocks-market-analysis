# ====================================================
# ADVANCED REAL-TIME STOCK MARKET ANALYSIS DASHBOARD
# ====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    layout="wide"
)

st.title("📊 Advanced Real-Time Stock Market Analysis")
st.caption("End-to-End Data Analytics & Financial Modeling Project")

# -------------------------------
# AUTO REFRESH (60 seconds)
# -------------------------------
refresh_rate = 60
st.write(f"🔄 Auto-refresh every {refresh_rate} seconds")
time.sleep(1)

# -------------------------------
# USER INPUT
# -------------------------------
stocks = st.multiselect(
    "Select Stocks",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    default=["AAPL", "MSFT"]
)

start_date = st.date_input(
    "Start Date",
    datetime.today() - timedelta(days=365)
)

end_date = st.date_input(
    "End Date",
    datetime.today()
)

market_index = "^GSPC"  # S&P 500 for Beta

# -------------------------------
# DATA ACQUISITION
# -------------------------------
@st.cache_data
def load_data(symbols, start, end):
    return yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker"
    )

raw_data = load_data(stocks + [market_index], start_date, end_date)

# -------------------------------
# PROCESS DATA
# -------------------------------
def prepare_data(symbol):
    df = raw_data[symbol].copy()
    df.dropna(inplace=True)

    df["Daily Return"] = df["Close"].pct_change()
    df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    return df

data = {s: prepare_data(s) for s in stocks}
market_df = prepare_data(market_index)

# -------------------------------
# PRICE & MOVING AVERAGES
# -------------------------------
st.header("📈 Price & Moving Averages")

for s in stocks:
    df = data[s]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA 50", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_200"], name="MA 200", line=dict(dash="dot")))

    fig.update_layout(title=f"{s} Price Trend", height=450)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RISK, RETURN, SHARPE, BETA
# -------------------------------
st.header("⚖️ Risk Metrics & Performance")

metrics = []

risk_free_rate = 0.02  # 2%

for s in stocks:
    df = data[s]
    returns = df["Daily Return"].dropna()
    market_returns = market_df["Daily Return"].dropna()

    annual_return = returns.mean() * 252
    annual_risk = returns.std() * np.sqrt(252)

    sharpe = (annual_return - risk_free_rate) / annual_risk

    covariance = np.cov(returns.align(market_returns, join="inner")[0],
                        market_returns.align(returns, join="inner")[0])[0][1]
    beta = covariance / market_returns.var()

    metrics.append({
        "Stock": s,
        "Annual Return": annual_return,
        "Annual Risk": annual_risk,
        "Sharpe Ratio": sharpe,
        "Beta (vs S&P 500)": beta
    })

metrics_df = pd.DataFrame(metrics)
st.dataframe(metrics_df, use_container_width=True)

# -------------------------------
# RISK VS RETURN PLOT
# -------------------------------
fig_rr = px.scatter(
    metrics_df,
    x="Annual Risk",
    y="Annual Return",
    text="Stock",
    title="Risk vs Return Analysis"
)
fig_rr.update_traces(textposition="top center")
st.plotly_chart(fig_rr, use_container_width=True)

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
st.header("🔗 Correlation Heatmap")

returns_df = pd.DataFrame({s: data[s]["Daily Return"] for s in stocks})
corr = returns_df.corr()

fig_corr = px.imshow(corr, text_auto=True, title="Stock Return Correlation")
st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------
# MACHINE LEARNING TREND (LINEAR REGRESSION)
# -------------------------------
st.header("🤖 Machine Learning Price Trend")

for s in stocks:
    df = data[s].dropna()

    df["Time"] = np.arange(len(df))
    X = df[["Time"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    df["Trend"] = model.predict(X)

    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual"))
    fig_ml.add_trace(go.Scatter(x=df.index, y=df["Trend"], name="ML Trend"))

    fig_ml.update_layout(title=f"{s} Linear Regression Trend")
    st.plotly_chart(fig_ml, use_container_width=True)

# -------------------------------
# REAL-TIME PRICES
# -------------------------------
st.header("⏱️ Latest Market Prices")

latest = []
for s in stocks:
    ticker = yf.Ticker(s)
    latest.append({
        "Stock": s,
        "Price": ticker.fast_info["last_price"]
    })

latest_df = pd.DataFrame(latest)
st.table(latest_df)

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
st.header("⬇️ Download Analysis")

csv = metrics_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Financial Metrics (CSV)",
    data=csv,
    file_name="stock_analysis_metrics.csv",
    mime="text/csv"
)

st.success("✅ Full real-time analytics pipeline executed successfully")

time.sleep(refresh_rate)
st.experimental_rerun()
