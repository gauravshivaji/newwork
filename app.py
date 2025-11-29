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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Elliott (1-5, A, B, C)")


# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    period: '60d', '3y', '10y' etc.
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

    # If columns are MultiIndex (happens with some yfinance versions),
    # keep only the first level: Open, High, Low, Close, Volume, etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # Ensure index is datetime for plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA(20/50/200) + RSI(14) using pure pandas."""
    if df is None or df.empty:
        return df

    if "Close" not in df.columns:
        return df

    # ---- Simple Moving Averages ----
    for win in [20, 50, 200]:
        df[f"SMA_{win}"] = df["Close"].rolling(window=win).mean()

    # ---- RSI(14) ----
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


def add_elliott_wave_labels(df: pd.DataFrame, pivot_window: int = 5) -> pd.DataFrame:
    """
    Very simple Elliott-like labelling:
    1. Detect swing highs/lows using local window.
    2. Take the last 8 pivots.
    3. Label them as 1,2,3,4,5,A,B,C in order.

    This is heuristic & educational, not a full Elliott engine.
    """
    if df is None or df.empty:
        return df

    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df

    w = max(3, pivot_window)  # at least 3, must be odd for "center"
    if w % 2 == 0:
        w += 1  # make it odd

    # Local highest high / lowest low in rolling window
    df["HH"] = df["High"].rolling(w, center=True).max()
    df["LL"] = df["Low"].rolling(w, center=True).min()

    df["pivot_high"] = (df["High"] == df["HH"])
    df["pivot_low"] = (df["Low"] == df["LL"])

    pivots = df[(df["pivot_high"] | df["pivot_low"])].copy()

    # Need at least some pivots to label
    if len(pivots) < 5:
        df["Elliott_Label"] = np.nan
        df["Elliott_Pivot_Type"] = np.nan
        return df

    # Take last up-to-8 pivots
    labels_full = ["1", "2", "3", "4", "5", "A", "B", "C"]
    n = min(len(labels_full), len(pivots))
    pivots = pivots.iloc[-n:]
    labels = labels_full[-n:]  # align to last n points

    df["Elliott_Label"] = np.nan
    df["Elliott_Pivot_Type"] = np.nan

    for (idx, row), label in zip(pivots.iterrows(), labels):
        df.at[idx, "Elliott_Label"] = label
        if row["pivot_high"]:
            df.at[idx, "Elliott_Pivot_Type"] = "H"
        elif row["pivot_low"]:
            df.at[idx, "Elliott_Pivot_Type"] = "L"

    # You can drop helper cols if you want less clutter, but they don't hurt
    # df.drop(columns=["HH", "LL", "pivot_high", "pivot_low"], inplace=True)

    return df


def make_tv_style_chart(df: pd.DataFrame, title: str):
    """
    TradingView-style layout:
    Row 1: Candlestick + SMA20/50/200 + Elliott labels
    Row 2: Volume bars
    Row 3: RSI(14)
    """
    if df is None or df.empty:
        return go.Figure()

    # x-axis will be the index (DatetimeIndex)
    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        specs=[[{"type": "candlestick"}],
               [{"type": "bar"}],
               [{"type": "scatter"}]],
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

    # --- Elliott Labels (1-5, A, B, C) ---
    if "Elliott_Label" in df.columns:
        label_df = df[df["Elliott_Label"].notna()]
        if not label_df.empty:
            ys = []
            for idx, row in label_df.iterrows():
                if "Elliott_Pivot_Type" in row and row["Elliott_Pivot_Type"] == "L":
                    # place slightly below the low
                    ys.append(row["Low"] * 0.995)
                else:
                    # place slightly above the high
                    ys.append(row["High"] * 1.005)

            fig.add_trace(
                go.Scatter(
                    x=label_df.index,
                    y=ys,
                    mode="text",
                    text=label_df["Elliott_Label"],
                    textposition="middle center",
                    name="Elliott Waves",
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
            y0=30, y1=70,
            line_width=0,
            fillcolor="LightGray",
            opacity=0.2,
            row=3, col=1,
        )

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

default_tickers = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "NIFTY.NS",
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

# ---- HOURLY ----
with tabs[0]:
    st.subheader(f"⏱ Hourly — last 60 days — {ticker}")
    df_h = load_data(ticker, period="60d", interval="1h")
    df_h = add_indicators(df_h)
    df_h = add_elliott_wave_labels(df_h, pivot_window=5)

    if df_h.empty:
        st.warning("No hourly data found for this symbol.")
    else:
        fig_h = make_tv_style_chart(df_h, f"{ticker} — Hourly (60D)")
        st.plotly_chart(fig_h, use_container_width=True)

# ---- DAILY ----
with tabs[1]:
    st.subheader(f"📅 Daily — last 3 years — {ticker}")
    df_d = load_data(ticker, period="3y", interval="1d")
    df_d = add_indicators(df_d)
    df_d = add_elliott_wave_labels(df_d, pivot_window=5)

    if df_d.empty:
        st.warning("No daily data found for this symbol.")
    else:
        fig_d = make_tv_style_chart(df_d, f"{ticker} — Daily (3Y)")
        st.plotly_chart(fig_d, use_container_width=True)

# ---- WEEKLY ----
with tabs[2]:
    st.subheader(f"📆 Weekly — last 10 years — {ticker}")
    df_w = load_data(ticker, period="10y", interval="1wk")
    df_w = add_indicators(df_w)
    df_w = add_elliott_wave_labels(df_w, pivot_window=5)

    if df_w.empty:
        st.warning("No weekly data found for this symbol.")
    else:
        fig_w = make_tv_style_chart(df_w, f"{ticker} — Weekly (10Y)")
        st.plotly_chart(fig_w, use_container_width=True)
