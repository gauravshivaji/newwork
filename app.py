import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import date, timedelta

# ------------- PAGE CONFIG -------------
st.set_page_config(
    page_title="Nifty 50 SMA & RSI Dashboard",
    layout="wide"
)

st.title("📊 Nifty 50 Stock Dashboard — SMA(20/50/200) + RSI")
st.write(
    "Nifty 50 dashboard showing candlestick price with SMA 20/50/200 and RSI "
    "for three fixed timeframes:\n"
    "- Weekly: last 10 years\n"
    "- Daily: last 3 years\n"
    "- Hourly: last 60 days"
)

# ------------- NIFTY 50 TICKERS -------------
NIFTY50_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "SBI": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "HUL": "HINDUNILVR.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M_M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Grasim": "GRASIM.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Divi's Lab": "DIVISLAB.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Bajaj Auto": "BAJAJ_AUTO.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Nestle India": "NESTLEIND.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "UPL": "UPL.NS",
    "Shree Cement": "SHREECEM.NS",
    "Hindalco": "HINDALCO.NS",
    "JSW Energy": "JSWENERGY.NS"  # can replace if needed
}

# ------------- SIDEBAR CONTROLS -------------
st.sidebar.header("⚙️ Settings")

stock_name = st.sidebar.selectbox(
    "Select Nifty 50 Stock",
    options=list(NIFTY50_TICKERS.keys()),
    index=0
)

ticker = NIFTY50_TICKERS[stock_name]

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ using Streamlit & yfinance")

# ------------- DATE RANGES (FIXED) -------------
today = date.today()
daily_start = today - timedelta(days=3 * 365)   # ~3 years
weekly_start = today - timedelta(days=10 * 365) # ~10 years
hourly_start = today - timedelta(days=60)       # 60 days

# ------------- DATA DOWNLOAD FUNCTION -------------
@st.cache_data(show_spinner=True)
def load_data(ticker_symbol, start, end, interval="1d"):
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(
        ticker_symbol,
        start=start,
        end=end + timedelta(days=1),  # include end date
        interval=interval,
        auto_adjust=True
    )
    df = df.copy()
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# ------------- INDICATOR CALCULATION -------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- SMA ---
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # --- RSI (ensure 1D close series) ---
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    df["RSI"] = rsi_indicator.rsi()

    return df

# ------------- CHART FUNCTIONS -------------
def plot_price_and_sma(df: pd.DataFrame, title_suffix: str):
    fig_price = go.Figure()

    # Candlestick price
    fig_price.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    # SMAs
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df["SMA20"],
        mode="lines",
        name="SMA20"
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df["SMA50"],
        mode="lines",
        name="SMA50"
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df["SMA200"],
        mode="lines",
        name="SMA200"
    ))

    fig_price.update_layout(
        title=f"{title_suffix} — Candlestick with SMA 20/50/200",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)


def plot_rsi(df: pd.DataFrame, title_suffix: str):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode="lines",
        name="RSI"
    ))

    fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold (30)")

    fig_rsi.update_layout(
        title=f"{title_suffix} — RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=350,
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ------------- MAIN -------------
st.subheader(f"🧾 {stock_name} ({ticker}) — Fixed Timeframe View")

tab_daily, tab_hourly, tab_weekly = st.tabs(["📅 Daily (3Y)", "⏱ Hourly (60D)", "📆 Weekly (10Y)"])

# ---- DAILY TAB (3 years) ----
with tab_daily:
    st.markdown("**Daily timeframe — last ~3 years**")
    with st.spinner("Fetching Daily data..."):
        df_daily = load_data(ticker, daily_start, today, "1d")

    if df_daily.empty:
        st.warning("No daily data returned from Yahoo Finance.")
    else:
        df_daily = add_indicators(df_daily)
        plot_price_and_sma(df_daily, f"{stock_name} — Daily (3Y)")
        plot_rsi(df_daily, f"{stock_name} — Daily (3Y)")
        with st.expander("📄 Show Daily data (last 200 rows)"):
            st.dataframe(df_daily[["Close", "SMA20", "SMA50", "SMA200", "RSI"]].tail(200))

# ---- HOURLY TAB (60 days) ----
with tab_hourly:
    st.markdown("**Hourly timeframe — last 60 days**")
    with st.spinner("Fetching Hourly data..."):
        df_hourly = load_data(ticker, hourly_start, today, "1h")

    if df_hourly.empty:
        st.warning("No hourly data available (Yahoo limit or ticker restriction).")
    else:
        df_hourly = add_indicators(df_hourly)
        plot_price_and_sma(df_hourly, f"{stock_name} — Hourly (60D)")
        plot_rsi(df_hourly, f"{stock_name} — Hourly (60D)")
        with st.expander("📄 Show Hourly data (last 300 rows)"):
            st.dataframe(df_hourly[["Close", "SMA20", "SMA50", "SMA200", "RSI"]].tail(300))

# ---- WEEKLY TAB (10 years) ----
with tab_weekly:
    st.markdown("**Weekly timeframe — last ~10 years**")
    with st.spinner("Fetching Weekly data..."):
        df_weekly = load_data(ticker, weekly_start, today, "1wk")

    if df_weekly.empty:
        st.warning("No weekly data returned from Yahoo Finance.")
    else:
        df_weekly = add_indicators(df_weekly)
        plot_price_and_sma(df_weekly, f"{stock_name} — Weekly (10Y)")
        plot_rsi(df_weekly, f"{stock_name} — Weekly (10Y)")
        with st.expander("📄 Show Weekly data (all rows)"):
            st.dataframe(df_weekly[["Close", "SMA20", "SMA50", "SMA200", "RSI"]])
