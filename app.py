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
    page_title="Nifty500 Multi-Timeframe Buy Scanner",
    layout="wide"
)

st.title("📊 Nifty500 Multi-Timeframe Buy Scanner — Hourly / Daily / Weekly")
st.write(
    """
    - Data from Yahoo Finance (yfinance)  
    - Indicators: **RSI**, **SMA22**, **SMA50**, **SMA200**  
    - Simple **Elliott-style wave (single cycle 0–5, A–C)**  
    - **RandomForestClassifier** + timeframe-specific rules to decide Buy = Yes/No  
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
        period = "3y"      # 3 years for swing trading
    elif timeframe == "Weekly":
        interval = "1wk"
        period = "10y"     # 10 years for position trading
    elif timeframe == "Hourly":
        interval = "60m"
        period = "60d"     # last 60 days intraday
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
    if "Close" not in df.columns or df.empty:
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


def add_elliott_labels(df: pd.DataFrame, timeframe: str, order_map=None) -> pd.DataFrame:
    """
    Elliott-style labelling for ONE cycle that covers MAXIMUM timeframe:

    - Uses all candles available for that timeframe:
        Hourly: last 60 days
        Daily: last 3 years
        Weekly: last 10 years
    - Finds ALL pivot points on this data.
    - Selects up to 9 pivots that are evenly spread from start to end,
      so the cycle (0–5, A–C) covers the maximum possible time range.
    - Only those pivots get labels 0,1,2,3,4,5,A,B,C.
    - Everything else stays NaN.
    """

    df = df.copy()
    df["elliott_wave"] = np.nan

    if "Close" not in df.columns or df.empty:
        return df

    # Pivot sensitivity based on timeframe
    if order_map is None:
        order_map = {
            "Hourly": 8,   # more smoothing on noisy intraday
            "Daily": 5,
            "Weekly": 3,
        }

    order = order_map.get(timeframe, 5)

    prices = df["Close"].values
    pivots = find_pivots(prices, order=order)  # list of (bar_index, 'low'/'high')

    m = len(pivots)
    if m < 5:
        # Not enough structure to define a clear cycle
        return df

    # We want up to 9 pivots that span the whole pivot range.
    if m <= 9:
        selected = pivots
    else:
        # Evenly spaced indices across [0, m-1]
        step = (m - 1) / (9 - 1)  # float step for 8 intervals
        idxs = [int(round(k * step)) for k in range(9)]
        # Keep them unique, in range, and sorted
        idxs = sorted(set(max(0, min(m - 1, i)) for i in idxs))

        # If due to rounding we got fewer than 9, fill with remaining pivots
        while len(idxs) < min(9, m):
            for j in range(m):
                if j not in idxs:
                    idxs.append(j)
                    if len(idxs) == min(9, m):
                        break
        idxs = sorted(idxs)

        selected = [pivots[j] for j in idxs]

    # Assign a single cycle 0,1,2,3,4,5,A,B,C to the selected pivots
    wave_pattern = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]

    for idx, (bar_idx, kind) in enumerate(selected):
        if idx >= len(wave_pattern):
            break
        if 0 <= bar_idx < len(df.index):
            df.at[df.index[bar_idx], "elliott_wave"] = wave_pattern[idx]

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
    if "rsi" not in df.columns or "Close" not in df.columns:
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
    if "Close" not in df.columns or len(df) < order * 2 + 1:
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


# --------- TIMEFRAME-SPECIFIC RULES (return decision + reasons) ---------

def hourly_rules(df: pd.DataFrame):
    """
    Short-Term Trading (Hourly)
    - RSI 25–40
    - Price above SMA22 OR SMA50
    - Support tight: within 3%
    - Elliott wave: 0 or 2 (or no label)
    """
    df = df.dropna(subset=["rsi", "sma22", "sma50"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 25 <= rsi <= 40
    cond_sma = (last["Close"] >= last["sma22"]) or (last["Close"] >= last["sma50"])
    cond_support = near_support(df, tolerance=0.03)
    ell = last_elliott_label(df)
    cond_ell = (ell in ["0", "2"]) or (ell is None)

    flags = {
        "RSI between 25–40 (oversold zone)": cond_rsi,
        "Price above fast SMA (22 or 50)": cond_sma,
        "Price near recent support (≤ 3%)": cond_support,
        "Elliott phase is 0/2 or not labeled": cond_ell,
    }

    score = sum(flags.values())
    decision = 1 if score >= 2 else 0   # need at least 2/4
    return decision, flags


def daily_rules(df: pd.DataFrame):
    """
    Swing Trading (Daily)
    - RSI 30–50
    - At least 2 of SMA22, SMA50, SMA200
    - Support within 5%
    - Elliott: 0,2,4
    - Bullish RSI divergence preferred
    """
    df = df.dropna(subset=["rsi", "sma22", "sma50", "sma200"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 30 <= rsi <= 50
    cond_sma = (
        (last["Close"] >= last["sma22"]).astype(int) +
        (last["Close"] >= last["sma50"]).astype(int) +
        (last["Close"] >= last["sma200"]).astype(int)
    ) >= 2
    cond_support = near_support(df, tolerance=0.05)
    ell = last_elliott_label(df)
    cond_ell = ell in ["0", "2", "4"]
    cond_div = detect_bullish_rsi_divergence(df)

    flags = {
        "RSI between 30–50 (healthy pullback)": cond_rsi,
        "Price above at least 2 of SMA22/50/200": cond_sma,
        "Price near support (≤ 5%)": cond_support,
        "Elliott phase in 0 / 2 / 4": cond_ell,
        "Bullish RSI divergence present": cond_div,
    }

    score = sum(flags.values())
    decision = 1 if score >= 3 else 0   # need 3/5
    return decision, flags


def weekly_rules(df: pd.DataFrame):
    """
    Long-Term Position Trading (Weekly)
    - RSI 35–55
    - MUST be above SMA200
    - Support within 10%
    - Elliott: 2 or 4 (big corrective waves)
    """
    df = df.dropna(subset=["rsi", "sma200"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 35 <= rsi <= 55
    cond_sma200 = last["Close"] >= last["sma200"]
    cond_support = near_support(df, tolerance=0.10)
    ell = last_elliott_label(df)
    cond_ell = ell in ["2", "4"]

    flags = {
        "RSI between 35–55 (accumulation zone)": cond_rsi,
        "Price above long-term SMA200": cond_sma200,
        "Price near major support (≤ 10%)": cond_support,
        "Elliott phase in 2 / 4": cond_ell,
    }

    score = sum(flags.values())
    decision = 1 if score >= 2 else 0   # need 2/4
    return decision, flags


def rule_based_buy(df: pd.DataFrame, timeframe: str):
    """Return (decision, flags) depending on timeframe."""
    if timeframe == "Hourly":
        return hourly_rules(df)
    elif timeframe == "Daily":
        return daily_rules(df)
    elif timeframe == "Weekly":
        return weekly_rules(df)
    return 0, {"Invalid timeframe": False}


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
    if len(df) < 50:    # relaxed
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

    if len(df) < 40 or df["target"].nunique() < 2:
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


def combined_buy_signal(df: pd.DataFrame, timeframe: str, return_reason: bool = False):
    """
    Final decision:
    - Rules give a decision + detailed flags
    - RF model (if available) gives additional confirmation
    """
    rule_sig, flags = rule_based_buy(df, timeframe)
    rf_res = train_rf_and_predict(df)

    lines = []

    # ---- Rules explanation ----
    lines.append(f"**Rule-based checks ({timeframe})**")
    for desc, ok in flags.items():
        mark = "✅" if ok else "❌"
        lines.append(f"- {mark} {desc}")

    # ---- RF explanation ----
    if rf_res is None:
        lines.append("")
        lines.append("_RandomForest model not used (not enough clean history). Decision based only on rules._")
        final = "Yes" if rule_sig == 1 else "No"
    else:
        rf_pred, rf_proba = rf_res
        lines.append("")
        lines.append("**Machine Learning (RandomForest)**")
        lines.append(f"- Model bullish probability: `{rf_proba:.2f}`")
        lines.append(f"- Model class prediction: `{'Bullish' if rf_pred == 1 else 'Not Bullish'}`")

        # Combine rules + ML
        if rf_pred == 1 and rule_sig == 1:
            final = "Yes"
            lines.append("- ✅ Both rules and model agree on bullish setup.")
        elif rf_pred == 1 and rule_sig == 0 and rf_proba > 0.65:
            final = "Yes"
            lines.append("- ⚠️ Model strongly bullish, but rules are weak (some conditions failed).")
        else:
            final = "No"
            lines.append("- ❌ Either rules are not satisfied or model is not bullish enough.")

    lines.append("")
    lines.append(f"**Final decision:** **{final}** for **{timeframe}** timeframe.")

    reason_text = "\n".join(lines)

    if return_reason:
        return final, reason_text
    else:
        return final


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

    # Elliott labels on price (single cycle)
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
        df_focus = add_elliott_labels(df_focus, chart_timeframe)
        plot_chart(df_focus, focus_ticker, chart_timeframe)
    else:
        st.warning("No data for this ticker / timeframe.")

with col2:
    st.subheader("Current signal (selected ticker)")
    if not df_focus.empty:
        df_for_sig = add_indicators(df_focus)
        df_for_sig = add_elliott_labels(df_for_sig, chart_timeframe)
        signal, reason = combined_buy_signal(df_for_sig, chart_timeframe, return_reason=True)
        st.metric(
            label=f"Buy signal ({chart_timeframe})",
            value=signal,
        )
        st.markdown("### 🧠 Why this signal?")
        st.markdown(reason)
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
            df_tf = add_elliott_labels(df_tf, tf)
            decision = combined_buy_signal(df_tf, tf)
            row[col_name] = decision

        rows.append(row)
        progress.progress(i / len(tickers_to_scan))
        status.text(f"Scanning {tck}... ({i}/{len(tickers_to_scan)})")

    progress.empty()
    status.empty()
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)
    st.caption(
        "✅ 'Yes' = RandomForest (if enough data) + timeframe-specific rules "
        "(RSI, SMA, support, Elliott) agree on bullish setup. "
        "If RF has no data, result is based only on rules."
    )
else:
    st.info("Click **Run Scan** to build the multi-timeframe Buy table.")
