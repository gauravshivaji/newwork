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
st.caption("Hourly / Daily / Weekly — Candles + SMA + Volume + RSI + Elliott (Rules-Based)")


# ---------------- DATA LOADER ----------------
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

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    # Ensure index is datetime for plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


# ---------------- INDICATORS ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA(20/50/200) + RSI(14) using pure pandas."""
    df = df.copy()
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


# ---------------- ELLIOTT RULE ENGINE ----------------
def add_elliott_wave_labels(df: pd.DataFrame, pivot_window: int = 5, tol: float = 1e-8) -> pd.DataFrame:
    """
    Elliott wave labelling with core rules:

    - Detect pivots (swing highs/lows) using rolling window.
    - Search consecutive 6 pivots for an UP impulse: 0-1-2-3-4-5 with pattern L-H-L-H-L-H.
    - Enforce rules:

      Rule 1  — Wave 2 never breaks Wave 1 start:
                 low2 > low0
      Rule 2  — Wave 3 is never the shortest (vs 1 and 5).
      Rule 3  — Wave 4 never overlaps Wave 1:
                 low4 > high1
      Rule 4  — Wave 3 must break Wave 1 high:
                 high3 > high1
      Rule 5  — Wave 5 must make new high above Wave 3:
                 high5 >= high3
      Rule 13 — Higher highs / higher lows:
                 high1 < high3 < high5 and low0 < low2 < low4

    - After a valid 1–5, try to attach A-B-C correction:
      * Pattern of next pivots: L-H-L (A,B,C)
      * Rule 8  — C breaks A:
                   low_C < low_A
      * Rule 9  — B length not > 2× A length (approx):

    Returns df with:
      - Elliott_Label: {1,2,3,4,5,A,B,C}
      - Elliott_Pivot_Type: "H" or "L"
    """
    df = df.copy()
    needed_cols = {"High", "Low", "Close"}
    df["Elliott_Label"] = np.nan
    df["Elliott_Pivot_Type"] = np.nan

    if df is None or df.empty or not needed_cols.issubset(df.columns):
        return df

    # --- Detect pivots using rolling HH/LL ---
    w = max(3, pivot_window)
    if w % 2 == 0:
        w += 1  # make window odd so center pivot is well-defined

    df["HH"] = df["High"].rolling(w, center=True).max()
    df["LL"] = df["Low"].rolling(w, center=True).min()
    df["pivot_high"] = (df["High"] == df["HH"])
    df["pivot_low"] = (df["Low"] == df["LL"])

    pivots = []
    for idx, row in df.iterrows():
        if row["pivot_high"]:
            pivots.append({"idx": idx, "type": "H", "price": row["High"]})
        elif row["pivot_low"]:
            pivots.append({"idx": idx, "type": "L", "price": row["Low"]})

    if len(pivots) < 6:
        return df

    best_seq = None
    best_start = None

    # --- Search for valid impulse up (1–5) among consecutive 6 pivots ---
    for j in range(len(pivots) - 5):
        seq = pivots[j:j + 6]
        types = [p["type"] for p in seq]

        # Trend requirement + structural: L-H-L-H-L-H = uptrend impulse
        if types != ["L", "H", "L", "H", "L", "H"]:
            continue

        w0, w1, w2, w3, w4, w5 = seq
        l0, l2, l4 = w0["price"], w2["price"], w4["price"]
        h1, h3, h5 = w1["price"], w3["price"], w5["price"]

        # Rule 1 — Wave 2 never retraces 100% of Wave 1
        # (Wave 2 low must stay above Wave 0 low)
        if not (l2 > l0):
            continue

        # Rule 3 — Wave 4 never enters Wave 1 territory
        # (Wave 4 low > Wave 1 high in strong uptrend)
        if not (l4 > h1):
            continue

        # Rule 4 — Wave 3 must go beyond Wave 1 high
        if not (h3 > h1):
            continue

        # Rule 5 — Wave 5 must make new high above Wave 3
        if not (h5 >= h3):
            continue

        # Rule 13 — higher highs / higher lows
        if not (h1 < h3 < h5 and l0 < l2 < l4):
            continue

        # Rule 2 — Wave 3 is never the shortest vs 1 and 5
        len1 = h1 - l0
        len3 = h3 - l2
        len5 = h5 - l4
        min_others = min(len1, len5)
        if not (len3 + tol >= min_others):
            continue

        # If everything passes, keep this as latest valid impulse
        best_seq = seq
        best_start = j

    if not best_seq:
        return df  # no valid impulse pattern found

    # --- Assign impulse labels 1–5 (we skip Wave 0) ---
    labels_map = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    for i_wave, p in enumerate(best_seq):
        if i_wave == 0:
            continue  # Wave 0 is the starting pivot, not labelled
        lbl = labels_map.get(i_wave)
        if lbl is None:
            continue
        idx = p["idx"]
        df.loc[idx, "Elliott_Label"] = lbl
        df.loc[idx, "Elliott_Pivot_Type"] = p["type"]

    # --- Try to attach ABC correction after wave 5 ---
    if best_start is not None and best_start + 6 < len(pivots):
        tail = pivots[best_start + 6 :]
        if len(tail) >= 3:
            A, B, C = tail[0], tail[1], tail[2]
            tA, tB, tC = A["type"], B["type"], C["type"]

            # For uptrend correction:
            # Wave A: down leg -> L after 5
            # Wave B: bounce -> H
            # Wave C: final down -> L
            if [tA, tB, tC] == ["L", "H", "L"]:
                # Rule 8 — Wave C must break Wave A:
                if C["price"] < A["price"]:
                    # Rule 9 — Wave B cannot exceed 2× length of Wave A (approx)
                    w5 = best_seq[5]
                    lenA = w5["price"] - A["price"]
                    lenB = B["price"] - A["price"]
                    if lenA > 0 and lenB <= 2 * lenA + tol:
                        for lbl, pivot in zip(["A", "B", "C"], [A, B, C]):
                            idx = pivot["idx"]
                            df.loc[idx, "Elliott_Label"] = lbl
                            df.loc[idx, "Elliott_Pivot_Type"] = pivot["type"]

    return df


# ---------------- CHART ----------------
def make_tv_style_chart(df: pd.DataFrame, title: str):
    """
    TradingView-style layout:
    Row 1: Candlestick + SMA20/50/200 + Elliott labels
    Row 2: Volume bars
    Row 3: RSI(14)
    """
    if df is None or df.empty:
        return go.Figure()

    x = df.index  # DatetimeIndex

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
                    ys.append(row["Low"] * 0.995)   # below lows
                else:
                    ys.append(row["High"] * 1.005)  # above highs

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
