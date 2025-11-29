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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Elliott (Loose Rules)")


# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.empty or "Close" not in df.columns:
        return df

    # SMAs
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


# ---------------- LOOSE ELLIOTT ENGINE ----------------
def add_elliott_wave_labels(df: pd.DataFrame, pivot_window: int = 5) -> pd.DataFrame:
    """
    Loose Elliott implementation based on your ML-friendly rules:

    - Detect pivots via rolling HH/LL (zigzag-like).
    - Find sequences of 6 pivots L-H-L-H-L-H (uptrend impulse).
    - Loose checks for 1–5:
        * Wave 2 deep retrace, may come close to W0 (max ~2% break).
        * Wave 3 breaks W1 high clearly and is reasonably strong.
        * Wave 4 may overlap W1 but only ~10% of W1 range.
        * Wave 5 makes new high or small truncation (3% below W3).
        * Higher highs + higher lows with tolerance.
        * RSI strongest on Wave 3, Wave 5 shows divergence.
    - Then optional loose ABC (L-H-L) correction after 5.
    """
    df = df.copy()
    needed = {"High", "Low", "Close"}
    df["Elliott_Label"] = np.nan
    df["Elliott_Pivot_Type"] = np.nan

    if df is None or df.empty or not needed.issubset(df.columns):
        return df

    # --- Pivot detection (zigzag style) ---
    w = max(3, pivot_window)
    if w % 2 == 0:
        w += 1

    df["HH"] = df["High"].rolling(w, center=True).max()
    df["LL"] = df["Low"].rolling(w, center=True).min()
    df["pivot_high"] = df["High"] == df["HH"]
    df["pivot_low"] = df["Low"] == df["LL"]

    pivots = []
    for idx, row in df.iterrows():
        if bool(row["pivot_high"]):
            pivots.append({"idx": idx, "type": "H", "price": float(row["High"])})
        elif bool(row["pivot_low"]):
            pivots.append({"idx": idx, "type": "L", "price": float(row["Low"])})

    if len(pivots) < 6:
        return df

    best_seq = None
    best_start = None

    # ---------- Scan for loose bullish impulse 1–5 ----------
    for j in range(len(pivots) - 5):
        seq = pivots[j:j + 6]
        types = [p["type"] for p in seq]

        # Uptrend structure: L-H-L-H-L-H
        if types != ["L", "H", "L", "H", "L", "H"]:
            continue

        w0, w1, w2, w3, w4, w5 = seq
        l0, l2, l4 = w0["price"], w2["price"], w4["price"]
        h1, h3, h5 = w1["price"], w3["price"], w5["price"]

        def safe_div(a, b):
            return a / b if (b is not None and b != 0) else np.nan

        len1 = h1 - l0
        len3 = h3 - l2
        len5 = h5 - l4

        if len1 <= 0:
            continue

        # Wave 2 retracement (loose: 20–95% of Wave1)
        retr2 = safe_div(h1 - l2, h1 - l0)
        if not (0.2 <= retr2 <= 0.95):
            continue

        # Wave 2 can pierce W0 by up to ~2%
        if l2 < l0 * 0.98:
            continue

        # Wave 3 breaks Wave1 high clearly (>=1% above)
        if h3 < h1 * 1.01:
            continue

        # Wave 4: shallow-ish, overlap with Wave1 allowed up to 10% of W1 range
        wave1_range = h1 - l0
        if wave1_range <= 0:
            continue
        overlap = max(0.0, h1 - l4)  # distance into Wave1 zone
        overlap_frac = overlap / wave1_range
        if overlap_frac > 0.1:
            continue

        # Higher highs / higher lows with tolerance
        if not (h3 >= h1 * 1.01 and h5 >= h3 * 0.97):
            continue
        if not (l2 >= l0 * 0.98 and l4 >= l2 * 0.98):
            continue

        # Wave 3 should be reasonably strong vs Wave1
        if len3 < len1 * 0.8:
            continue

        # Wave 5 length loose: 0.3–1.2 * Wave1
        len5_ratio = safe_div(len5, len1)
        if np.isnan(len5_ratio) or not (0.3 <= len5_ratio <= 1.2):
            continue

        # Wave 5: can be truncated but not more than 3% below Wave3
        if h5 < h3 * 0.97:
            continue

        # ----- RSI / momentum checks (loose) -----
        rsi1 = df.loc[w1["idx"], "RSI_14"] if "RSI_14" in df.columns else np.nan
        rsi3 = df.loc[w3["idx"], "RSI_14"] if "RSI_14" in df.columns else np.nan
        rsi5 = df.loc[w5["idx"], "RSI_14"] if "RSI_14" in df.columns else np.nan

        # Wave 3 should have strongest RSI among pivot highs, and above 50
        if not np.isnan(rsi3):
            if (not np.isnan(rsi1) and rsi3 < rsi1 - 1) or \
               (not np.isnan(rsi5) and rsi3 < rsi5 - 1):
                continue
            if rsi3 < 50:
                continue

        # Wave 5 divergence: RSI5 <= RSI3 (loose)
        if not np.isnan(rsi3) and not np.isnan(rsi5):
            if rsi5 > rsi3 + 2:
                continue

        # Keep LAST valid loose impulse found
        best_seq = seq
        best_start = j

    if not best_seq:
        return df

    # ---------- Assign 1–5 labels (skip Wave0) ----------
    impulse_labels = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    for i, p in enumerate(best_seq):
        if i == 0:
            continue
        lbl = impulse_labels.get(i)
        if lbl is None:
            continue
        df.loc[p["idx"], "Elliott_Label"] = lbl
        df.loc[p["idx"], "Elliott_Pivot_Type"] = p["type"]

    # ---------- Loose ABC correction after 5 ----------
    if best_start is not None and best_start + 6 < len(pivots):
        tail = pivots[best_start + 6:]
        if len(tail) >= 3:
            A, B, C = tail[0], tail[1], tail[2]
            tA, tB, tC = A["type"], B["type"], C["type"]

            # Expect down correction after bullish impulse: L-H-L
            if [tA, tB, tC] == ["L", "H", "L"]:
                w5 = best_seq[5]
                low5 = w5["price"]
                lowA, highB, lowC = A["price"], B["price"], C["price"]

                # A should represent real down move from wave5
                # (slightly loose: allow lowA not too far above low5)
                if lowA >= low5 * 1.02:
                    pass
                else:
                    high5 = w5["price"]

                    # B can exceed wave5 high by up to ~10% (expanded flat)
                    if highB > high5 * 1.10:
                        pass
                    else:
                        # C usually goes below A, but failed C allowed up to ~2% above
                        if lowC > lowA * 1.02:
                            pass
                        else:
                            # loose Fib: C 0.5–2.0 × A
                            lenA = high5 - lowA
                            lenC = highB - lowC
                            if lenA > 0:
                                ratio_CA = lenC / lenA
                                if 0.5 <= ratio_CA <= 2.0:
                                    for lbl, pivot in zip(["A", "B", "C"], [A, B, C]):
                                        df.loc[pivot["idx"], "Elliott_Label"] = lbl
                                        df.loc[pivot["idx"], "Elliott_Pivot_Type"] = pivot["type"]

    return df


# ---------------- CHART ----------------
def make_tv_style_chart(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return go.Figure()

    x = df.index

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        specs=[
            [{"type": "candlestick"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
        ],
    )

    # Candles
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

    # SMAs
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

    # Elliott labels
    if "Elliott_Label" in df.columns:
        label_df = df[df["Elliott_Label"].notna()]
        if not label_df.empty:
            ys = []
            for _, row in label_df.iterrows():
                if "Elliott_Pivot_Type" in row and row["Elliott_Pivot_Type"] == "L":
                    ys.append(row["Low"] * 0.995)   # below low
                else:
                    ys.append(row["High"] * 1.005)  # above high

            fig.add_trace(
                go.Scatter(
                    x=label_df.index,
                    y=ys,
                    mode="text",
                    text=label_df["Elliott_Label"],
                    textposition="middle center",
                    name="Elliott (Loose)",
                ),
                row=1,
                col=1,
            )

    # Volume
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

    # RSI
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

# Hourly
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

# Daily
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

# Weekly
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
