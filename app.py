import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Nifty500 Multi-Timeframe Buy/Sell Scanner",
    layout="wide"
)

st.title("📊 Nifty500 Multi-Timeframe Buy Scanner — Hourly / Daily / Weekly")
st.write(
    """
    - Data from Yahoo Finance (yfinance)  
    - Indicators: **RSI**, **SMA22**, **SMA50**, **SMA200**  
    - Simple **Elliott-style wave labels (0–5, A–C)**  
    - **RandomForestClassifier** + your rules to decide Buy = Yes/No  
    """
)

# ---------------- NIFTY 500 TICKERS ----------------
# Put your full NIFTY500 list here
NIFTY500_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "HINDUNILVR.NS",
    "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    # TODO: paste your full list here...
]

# ---------------- HELPER FUNCTIONS ----------------

@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """Download OHLC data for selected timeframe and flatten columns."""
    if timeframe == "Daily":
        interval = "1d"
        period = "3y"      # more history so SMA200 + RF works
    elif timeframe == "Weekly":
        interval = "1wk"
        period = "10y"     # more history for weekly
    elif timeframe == "Hourly":
        interval = "60m"
        period = "60d"
    else:
        interval = "1d"
        period = "3y"

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df.empty:
        return df

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    wanted_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[wanted_cols].copy()

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI + SMAs + distance features."""
    df = df.copy()
    if "Close" not in df.columns:
        return df

    close = df["Close"]

    # Core indicators
    df["rsi"] = ta.momentum.rsi(close=close, window=14)
    df["sma22"] = close.rolling(22).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()

    # Distance to SMAs (normalized)
    for w in [22, 50, 200]:
        sma_col = f"sma{w}"
        df[f"dist_sma{w}"] = (df["Close"] - df[sma_col]) / df[sma_col]

    return df


def find_pivots(prices: np.ndarray, order: int = 5):
    """Simple local high/low pivots for Elliott-like labelling."""
    pivots = []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        if prices[i] == window.min():
            pivots.append((i, "low"))
        elif prices[i] == window.max():
            pivots.append((i, "high"))
    return pivots


def add_elliott_labels(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """
    Very simplified Elliott-like wave:
    last few pivots labelled 0,1,2,3,4,5,A,B,C
    """
    df = df.copy()
    if "Close" not in df.columns:
        df["elliott_wave"] = np.nan
        return df

    prices = df["Close"].values
    pivots = find_pivots(prices, order=order)
    wave_names = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]

    df["elliott_wave"] = np.nan

    if not pivots:
        return df

    last_pivots = pivots[-len(wave_names):]
    for idx, (i, kind) in enumerate(last_pivots):
        label = wave_names[idx]
        df.at[df.index[i], "elliott_wave"] = label

    return df


def last_elliott_label(df: pd.DataFrame):
    if "elliott_wave" not in df.columns:
        return None
    s = df["elliott_wave"].dropna()
    if s.empty:
        return None
    return s.iloc[-1]


def detect_bullish_rsi_divergence(df: pd.DataFrame, order: int = 5) -> bool:
    """Price makes lower low, RSI makes higher low."""
    if "rsi" not in df.columns:
        return False
    df = df.dropna(subset=["rsi"]).copy()
    if len(df) < order * 2 + 2:
        return False

    closes = df["Close"].values
    rsis = df["rsi"].values
    lows = []

    for i in range(order, len(df) - order):
        window = closes[i - order:i + order + 1]
        if closes[i] == window.min():
            lows.append(i)

    if len(lows) < 2:
        return False

    i1, i2 = lows[-2], lows[-1]
    price_lower_low = closes[i2] < closes[i1]
    rsi_higher_low = rsis[i2] > rsis[i1]

    return bool(price_lower_low and rsi_higher_low)


def near_support(df: pd.DataFrame, tolerance: float = 0.03, order: int = 5) -> bool:
    """Current price close to last swing low (support)."""
    if len(df) < order * 2 + 1:
        return False

    closes = df["Close"].values
    lows = []

    for i in range(order, len(df) - order):
        window = closes[i - order:i + order + 1]
        if closes[i] == window.min():
            lows.append(i)

    if not lows:
        return False

    last_low_idx = lows[-1]
    support_price = closes[last_low_idx]
    current_price = closes[-1]
    return abs(current_price - support_price) / current_price <= tolerance


def rule_based_buy(df: pd.DataFrame) -> int:
    """
    Your rules for BUY (slightly relaxed):
    - RSI near 30 (30–45)
    - Bullish RSI divergence present
    - Price above at least 2 of (SMA22, SMA50, SMA200)
    - Price near support
    - Elliott wave in 0,2,4 OR no label yet
    """
    needed_cols = ["rsi", "sma22", "sma50", "sma200"]
    for c in needed_cols:
        if c not in df.columns:
            return 0

    df = df.dropna(subset=needed_cols).copy()
    if df.empty:
        return 0

    last = df.iloc[-1]
    rsi = last["rsi"]

    # RSI condition
    cond_rsi = 30 <= rsi <= 45

    # RSI divergence
    cond_div = detect_bullish_rsi_divergence(df)

    # SMA condition: price above at least 2 of 3 SMAs
    cond_smas = [
        last["Close"] >= last["sma22"],
        last["Close"] >= last["sma50"],
        last["Close"] >= last["sma200"],
    ]
    cond_sma = sum(cond_smas) >= 2

    # Support
    cond_support = near_support(df)

    # Elliott condition
    ell = last_elliott_label(df)
    cond_elliott = (ell is None) or (ell in ["0", "2", "4"])

    if cond_rsi and cond_div and cond_sma and cond_support and cond_elliott:
        return 1
    return 0


def elliott_code_series(df: pd.DataFrame) -> pd.Series:
    """Convert Elliott labels to numeric code for ML."""
    mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "A": 6, "B": 7, "C": 8}
    if "elliott_wave" not in df.columns:
        return pd.Series(index=df.index, data=-1)
    return df["elliott_wave"].map(mapping).fillna(-1)


def train_rf_and_predict(df: pd.DataFrame):
    """
    Train a small RandomForest on past data:
    target = 1 if next 5 bars return > 2%
    Features: RSI + distance to SMAs + Elliott code
    """
    needed_cols = ["rsi", "sma22", "sma50", "sma200"]
    for c in needed_cols:
        if c not in df.columns:
            return None

    df = df.dropna(subset=needed_cols).copy()
    if len(df) < 50:    # relaxed from 80
        return None

    df["fwd_return"] = df["Close"].shift(-5) / df["Close"] - 1
    df = df.iloc[:-5]  # drop last 5 (no future)

    df["target"] = (df["fwd_return"] > 0.02).astype(int)
    df["elliott_code"] = elliott_code_series(df)

    feature_cols = [
        "rsi",
        "dist_sma22",
        "dist_sma50",
        "dist_sma200",
        "elliott_code",
    ]

    df = df.dropna(subset=feature_cols + ["target"])

    if len(df) < 40 or df["target"].nunique() < 2:  # relaxed from 60
        return None

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    latest_features = X.iloc[[-1]]
    proba = model.predict_proba(latest_features)[0, 1]
    pred_cls = int(proba > 0.5)

    return pred_cls, proba


def combined_buy_signal(df: pd.DataFrame):
    """
    Final decision:
    - If RF available: BUY only if rules + RF both say bullish
    - If RF not available: fallback to ONLY rules
    """
    rule_sig = rule_based_buy(df)
    rf_res = train_rf_and_predict(df)

    if rf_res is None:
        # Fallback: only rules
        return "Yes" if rule_sig == 1 else "No"

    rf_pred, rf_proba = rf_res
    final = (rule_sig == 1) and (rf_pred == 1)
    return "Yes" if final else "No"


def plot_chart(df: pd.DataFrame, ticker: str, timeframe: str):
    needed_cols = ["Open", "High", "Low", "Close", "sma22", "sma50", "sma200", "rsi"]
    for c in needed_cols:
        if c not in df.columns:
            st.warning("Not enough data to plot indicators.")
            return

    df = df.dropna(subset=["sma22", "sma50", "sma200", "rsi"]).copy()
    if df.empty:
        st.warning("Not enough data to plot.")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    # Price + SMAs
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma22"], name="SMA 22"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma50"], name="SMA 50"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["sma200"], name="SMA 200"),
        row=1,
        col=1,
    )

    # Elliott labels on price
    if "elliott_wave" in df.columns:
        ell = df["elliott_wave"].dropna()
        for ts, label in ell.items():
            price = df.loc[ts, "Close"]
            fig.add_annotation(
                x=ts,
                y=price,
                text=label,
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-20,
                row=1,
                col=1,
            )

    # RSI panel
    fig.add_trace(
        go.Scatter(x=df.index, y=df["rsi"], name="RSI 14"),
        row=2,
        col=1,
    )

    # (30,70) guide lines
    fig.add_shape(
        type="line",
        x0=df.index[0],
        x1=df.index[-1],
        y0=30,
        y1=30,
        xref="x2",
        yref="y2",
        line=dict(dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=df.index[0],
        x1=df.index[-1],
        y0=70,
        y1=70,
        xref="x2",
        yref="y2",
        line=dict(dash="dash"),
    )

    fig.update_layout(
        title=f"{ticker} — {timeframe} view",
        xaxis_rangeslider_visible=False,
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- SIDEBAR UI ----------------
st.sidebar.header("⚙️ Scanner Settings")

chart_timeframe = st.sidebar.radio(
    "Chart timeframe",
    ["Hourly", "Daily", "Weekly"],
    index=1,
)

scan_timeframes = st.sidebar.multiselect(
    "Timeframes for Buy table",
    ["Hourly", "Daily", "Weekly"],
    default=["Hourly", "Daily", "Weekly"],
)

max_tickers = st.sidebar.slider(
    "How many tickers to scan (for performance)?",
    min_value=5,
    max_value=min(60, len(NIFTY500_TICKERS)),
    value=20,
    step=5,
)

tickers_to_scan = NIFTY500_TICKERS[:max_tickers]

focus_ticker = st.selectbox(
    "🔍 Select stock for detailed chart",
    options=tickers_to_scan,
)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Chart: {focus_ticker} ({chart_timeframe})")
    df_focus = load_price_data(focus_ticker, chart_timeframe)
    if not df_focus.empty:
        df_focus = add_indicators(df_focus)
        df_focus = add_elliott_labels(df_focus)
        plot_chart(df_focus, focus_ticker, chart_timeframe)
    else:
        st.warning("No data for this ticker / timeframe.")

with col2:
    st.subheader("Current signal (selected ticker)")
    if not df_focus.empty:
        df_for_sig = add_elliott_labels(add_indicators(df_focus))
        signal = combined_buy_signal(df_for_sig)
        st.metric(
            label=f"Buy signal ({chart_timeframe})",
            value=signal,
        )
    else:
        st.write("Signal not available.")

st.markdown("---")
st.subheader("📋 Multi-Timeframe Buy Table")

if st.button("Run Scan"):
    rows = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, tck in enumerate(tickers_to_scan, start=1):
        row = {"Stock Name": tck}

        for tf in ["Hourly", "Daily", "Weekly"]:
            col_name = f"Buy at {tf} (Yes/No)"
            if tf not in scan_timeframes:
                row[col_name] = "-"
                continue

            df_tf = load_price_data(tck, tf)
            if df_tf.empty:
                row[col_name] = "No data"
                continue

            df_tf = add_indicators(df_tf)
            df_tf = add_elliott_labels(df_tf)
            decision = combined_buy_signal(df_tf)
            row[col_name] = decision

        rows.append(row)
        progress.progress(i / len(tickers_to_scan))
        status.text(f"Scanning {tck}... ({i}/{len(tickers_to_scan)})")

    progress.empty()
    status.empty()
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)
    st.caption(
        "✅ 'Yes' = RandomForest predicts upside (if enough data) AND your buy rules "
        "(RSI near 30 + divergence + SMA zone + support + Elliott 0/2/4/None) are satisfied. "
        "If RF has no data, result is based only on rules."
    )
else:
    st.info("Click **Run Scan** to build the multi-timeframe Buy table.")
