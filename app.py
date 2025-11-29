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
    "Interactive dashboard for Nifty 50 stocks showing price with "
    "SMA 20/50/200 and RSI indicator."
)

# ------------- NIFTY 50 TICKERS -------------
# You can edit/update this list anytime if constituents change
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
    "JSW Energy": "JSWENERGY.NS"  # you can replace if needed
}

# ------------- SIDEBAR CONTROLS -------------
st.sidebar.header("⚙️ Settings")

stock_name = st.sidebar.selectbox(
    "Select Nifty 50 Stock",
    options=list(NIFTY50_TICKERS.keys()),
    index=0
)

ticker = NIFTY50_TICKERS[stock_name]

# Date range: default last 1 year
today = date.today()
default_start = today - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1h", "30m", "15m"],
    index=0,
    help="For intraday intervals, Yahoo may return last 60-90 days only."
)

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ using Streamlit & yfinance")

# ------------- DATA DOWNLOAD FUNCTION -------------
@st.cache_data(show_spinner=True)
def load_data(ticker_symbol, start, end, interval="1d"):
    df = yf.download(
        ticker_symbol,
        start=start,
        end=end + timedelta(days=1),  # include end date
        interval=interval,
        auto_adjust=True
    )
    return df

# ------------- INDICATOR CALCULATION -------------
def add_indicators(df):
    df = df.copy()
    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # RSI using ta library
    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    return df

# ------------- MAIN LOGIC -------------
if start_date > end_date:
    st.error("❌ Start date must be before end date.")
else:
    with st.spinner("Fetching data from Yahoo Finance..."):
        data = load_data(ticker, start_date, end_date, interval)

    if data.empty:
        st.warning("No data returned. Try changing date range or interval.")
    else:
        data = add_indicators(data)

        st.subheader(f"📈 {stock_name} ({ticker}) Price & SMA")

        # ------------ PRICE + SMA CHART ------------
        fig_price = go.Figure()

        # Candlestick
        fig_price.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        ))

        # SMA lines
        fig_price.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA20"],
            mode="lines",
            name="SMA20"
        ))
        fig_price.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA50"],
            mode="lines",
            name="SMA50"
        ))
        fig_price.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA200"],
            mode="lines",
            name="SMA200"
        ))

        fig_price.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            xaxis_rangeslider_visible=False,
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # ------------ RSI CHART ------------
        st.subheader("📉 RSI (14)")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data.index,
            y=data["RSI"],
            mode="lines",
            name="RSI"
        ))

        # Overbought / oversold reference lines
        fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold (30)")

        fig_rsi.update_layout(
            xaxis_title="Date",
            yaxis_title="RSI",
            height=350,
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

        # ------------ DATA TABLE ------------
        with st.expander("📄 Show raw data with indicators"):
            st.dataframe(
                data[["Close", "SMA20", "SMA50", "SMA200", "RSI"]].tail(200)
            )
