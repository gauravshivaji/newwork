import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TradingView Style Multi-Timeframe Dashboard",
    layout="wide",
)

st.title("📈 TradingView-Style Stock Dashboard")
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Wave 0 & 5")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    period: '60d', '3y', '10y', etc.
    interval: '1h', '1d', '1wk'
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (some yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMA(20/50/200) + RSI(14).
    """
    df = df.copy()
    if df.empty or "Close" not in df.columns:
        return df

    # Simple Moving Averages
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(window=win).mean()

    # RSI(14)
    window = 14
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(window=window, min_periods=window).mean()
    roll_down = loss.rolling(window=window, min_periods=window).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["RSI_14"] = rsi

    return df


# ---------------- WAVE 0 & 5 DETECTION ----------------
def add_wave_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wave 0:

      Base logic:
        Condition 1 (RSI + price):
          - RSI < 20
          - RSI rising (rsi[i] > rsi[i-1])
          - Close rising (close[i] > close[i-1])

        OR

        Condition 2 (structural low):
          - Low is lowest in centered 201-bar window (100 back + 100 forward)

      NEW extra rule for 0:
        - After 7 candles, price must be higher than at 0:
              Close[i+7] > Close[i]

      Final Wave0:
        Wave0 = base_zero AND future_7_higher

    Wave 5:

      - Take all Wave0 positions
      - For each pair of consecutive 0s:
          look between them (exclusive of second 0)
          find highest High in that segment
          mark that bar as Wave 5
    """
    df = df.copy()

    df["Wave0"] = False
    df["Wave5"] = False

    needed_cols = {"RSI_14", "Low", "High", "Close"}
    if df.empty or not needed_cols.issubset(df.columns):
        return df

    rsi = df["RSI_14"]
    close = df["Close"]

    # ---- Wave 0 base conditions ----

    # Condition 1: RSI < 20 AND RSI increasing AND price increasing (on that candle)
    cond_rsi_price = (rsi < 20) & (rsi > rsi.shift(1)) & (close > close.shift(1))

    # Condition 2: Extreme 100-bar low (centered 201 bars)
    rolling_min_low = df["Low"].rolling(window=201, center=True, min_periods=1).min()
    eps = 1e-8
    cond_extreme_low = df["Low"] <= (rolling_min_low + eps)

    base_zero = cond_rsi_price | cond_extreme_low

    # ---- NEW: after 7 candles, price should be greater than at 0 ----
    n = len(df)
    future_7_higher = np.zeros(n, dtype=bool)

    # Only valid if we have at least 7 candles ahead
    for i in range(n - 30):
        if close.iloc[i + 30] > close.iloc[i]:
            future_7_higher[i] = True

    # Final Wave0 mask
    df["Wave0"] = base_zero & future_7_higher

    # ---- Wave 5: highest High between two 0s ----
    wave0_positions = np.where(df["Wave0"].values)[0]

    if len(wave0_positions) >= 2:
        for k in range(len(wave0_positions) - 1):
            start_i = wave0_positions[k]
            end_i = wave0_positions[k + 1]

            # Segment is strictly between the two 0s → (start_i, end_i)
            if end_i - start_i <= 1:
                continue

            seg = df.iloc[start_i + 1 : end_i]
            if seg.empty:
                continue

            idx_max_high = seg["High"].idxmax()
            df.loc[idx_max_high, "Wave5"] = True

    return df


# ---------------- BIG CHART ----------------
def make_tv_style_chart(df: pd.DataFrame, title: str):
    """
    Bigger TradingView-style layout:
    Row 1: Candlestick (larger) + SMA + Wave 0 + Wave 5
    Row 2: Volume
    Row 3: RSI
    """
    if df is None or df.empty:
        return go.Figure()

    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.15, 0.13],
        vertical_spacing=0.02,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
    )

    # --- Candles ---
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- SMAs ---
    for win, name in zip([20, 50, 200], ["SMA 20", "SMA 50", "SMA 200"]):
        col_name = f"SMA_{win}"
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col_name],
                    mode="lines",
                    name=name,
                ),
                row=1,
                col=1,
            )

    # --- Wave 0 labels (bold below lows) ---
    if "Wave0" in df.columns:
        wave0_df = df[df["Wave0"]]
        if not wave0_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave0_df.index,
                    y=wave0_df["Low"] * 0.995,
                    mode="text",
                    text=["<b>0</b>"] * len(wave0_df),
                    textposition="middle center",
                    name="Wave 0",
                ),
                row=1,
                col=1,
            )

    # --- Wave 5 labels (bold above highs) ---
    if "Wave5" in df.columns:
        wave5_df = df[df["Wave5"]]
        if not wave5_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=wave5_df.index,
                    y=wave5_df["High"] * 1.005,
                    mode="text",
                    text=["<b>5</b>"] * len(wave5_df),
                    textposition="middle center",
                    name="Wave 5",
                ),
                row=1,
                col=1,
            )

    # --- Volume ---
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["Volume"],
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # --- RSI ---
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["RSI_14"],
                mode="lines",
                name="RSI 14",
            ),
            row=3,
            col=1,
        )
        fig.add_hrect(
            y0=30,
            y1=70,
            line_width=0,
            fillcolor="LightGray",
            opacity=0.2,
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=900,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    return fig


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

default_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
"BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
"BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
"EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
"HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
"INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
"MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
"SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
"TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",

]

ticker = st.sidebar.selectbox(
    "Select Symbol",
    options=default_tickers,
    index=0,
)

custom = st.sidebar.text_input(
    "Or type custom symbol (Yahoo format, e.g. COFORGE.NS, AAPL, TSLA)",
    value="",
)

if custom.strip():
    ticker = custom.strip()

st.sidebar.write("---")
st.sidebar.markdown(
    """
**Note:**  
- Data from Yahoo Finance  
- Only trading days/hours are returned  
  (no Saturdays, Sundays, or exchange holidays)
"""
)

# ---------------- MAIN CONTENT ----------------
tabs = st.tabs(["⏱ Hourly", "📅 Daily", "📆 Weekly"])

# Hourly
with tabs[0]:
    st.subheader(f"⏱ Hourly — last 60 days — {ticker}")
    df_h = load_data(ticker, period="60d", interval="1h")
    df_h = add_indicators(df_h)
    df_h = add_wave_labels(df_h)

    if df_h.empty:
        st.warning("No hourly data found for this symbol.")
    else:
        fig_h = make_tv_style_chart(df_h, f"{ticker} — Hourly (60D)")
        st.plotly_chart(fig_h, use_container_width=True)

# Daily
with tabs[1]:
    st.subheader(f"📅 Daily — last 3 years — {ticker}")
    df_d = load_data(ticker, period="3y", interval="1d")
    df_d = add_indicators(df_d)
    df_d = add_wave_labels(df_d)

    if df_d.empty:
        st.warning("No daily data found for this symbol.")
    else:
        fig_d = make_tv_style_chart(df_d, f"{ticker} — Daily (3Y)")
        st.plotly_chart(fig_d, use_container_width=True)

# Weekly
with tabs[2]:
    st.subheader(f"📆 Weekly — last 10 years — {ticker}")
    df_w = load_data(ticker, period="10y", interval="1wk")
    df_w = add_indicators(df_w)
    df_w = add_wave_labels(df_w)

    if df_w.empty:
        st.warning("No weekly data found for this symbol.")
    else:
        fig_w = make_tv_style_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)
