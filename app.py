import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Elliott Wave Detector (Uptrend 1-5 + A-B-C)",
    layout="wide"
)

st.title("📈 Elliott Wave 1–5 + A–B–C Detector (Uptrend Only)")
st.markdown(
    "Uses ZigZag pivots + simple rule-based scoring to approximate an Elliott impulse (0–5) + correction (A–B–C)."
)

# ---------------- TIMEFRAME SETTINGS ----------------
TIMEFRAME_CONFIG = {
    "Hourly (1H)": {
        "interval": "1h",
        "period": "60d",
        "zigzag_pct": 4.0,   # 3–5% typical
    },
    "Daily (1D)": {
        "interval": "1d",
        "period": "3y",
        "zigzag_pct": 6.0,   # 5–8% typical
    },
    "Weekly (1W)": {
        "interval": "1wk",
        "period": "10y",
        "zigzag_pct": 12.0,  # 10–14% typical
    },
}


# ---------------- DATA LOADER ----------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ---------------- ZIGZAG PIVOT DETECTOR ----------------
def zigzag_pivots(df: pd.DataFrame, pct: float):
    """
    Simple percent-based ZigZag on Close.
    Returns list of pivots: [{'idx': index, 'price': float, 'type': 'high'/'low'}]
    """
    closes = df["Close"]
    if len(closes) < 3:
        return []

    pivots = []

    # initialize
    it = list(zip(closes.index, closes.values))
    last_pivot_idx, last_pivot_price = it[0]
    direction = 0  # 0 = unknown, 1 = up leg, -1 = down leg

    candidate_idx = last_pivot_idx
    candidate_price = last_pivot_price

    for idx, price in it[1:]:
        if direction == 0:
            change_pct = (price - last_pivot_price) / last_pivot_price * 100.0
            if change_pct >= pct:
                # first up leg; last pivot assumed low
                pivots.append({"idx": last_pivot_idx, "price": last_pivot_price, "type": "low"})
                direction = 1
                candidate_idx, candidate_price = idx, price
            elif change_pct <= -pct:
                # first down leg; last pivot assumed high
                pivots.append({"idx": last_pivot_idx, "price": last_pivot_price, "type": "high"})
                direction = -1
                candidate_idx, candidate_price = idx, price
        elif direction == 1:
            # up leg, searching for high
            if price > candidate_price:
                candidate_idx, candidate_price = idx, price

            drawdown_pct = (price - candidate_price) / candidate_price * 100.0
            if drawdown_pct <= -pct:
                # high pivot confirmed
                pivots.append({"idx": candidate_idx, "price": float(candidate_price), "type": "high"})
                direction = -1
                last_pivot_idx, last_pivot_price = candidate_idx, candidate_price
                candidate_idx, candidate_price = idx, price
        elif direction == -1:
            # down leg, searching for low
            if price < candidate_price:
                candidate_idx, candidate_price = idx, price

            bounce_pct = (price - candidate_price) / candidate_price * 100.0
            if bounce_pct >= pct:
                # low pivot confirmed
                pivots.append({"idx": candidate_idx, "price": float(candidate_price), "type": "low"})
                direction = 1
                last_pivot_idx, last_pivot_price = candidate_idx, candidate_price
                candidate_idx, candidate_price = idx, price

    # add final candidate as last pivot
    if len(pivots) == 0 or pivots[-1]["idx"] != candidate_idx:
        # decide type based on last direction
        p_type = "high" if direction == 1 else "low"
        pivots.append({"idx": candidate_idx, "price": float(candidate_price), "type": p_type})

    # sort by index just in case
    pivots.sort(key=lambda x: x["idx"])
    return pivots


# ---------------- ELLIOTT DETECTOR (RULE-BASED SCORE) ----------------
def detect_elliott_5_abc(pivots, min_score: int = 9):
    """
    Try to detect uptrend 0–5 + A–B–C pattern on last 9 pivots.
    Returns dict with waves or None.
    """
    if len(pivots) < 9:
        return None

    # take last 9 pivots
    cand = pivots[-9:]
    types = [p["type"] for p in cand]

    # we expect: low, high, low, high, low, high, low, high, low
    expected = ["low", "high", "low", "high", "low", "high", "low", "high", "low"]
    if types != expected:
        return None

    # Map them
    P0, P1, P2, P3, P4, P5, PA, PB, PC = cand
    p0, p1, p2, p3, p4, p5, pa, pb, pc = [x["price"] for x in cand]

    def safe_ratio(num, den):
        if den == 0:
            return np.nan
        return num / den

    score = 0
    reasons = []

    # Structural: rising lows / highs
    if p0 < p2 < p4:
        score += 1
        reasons.append("Rising lows 0 < 2 < 4")
    if p1 < p3 < p5:
        score += 1
        reasons.append("Rising highs 1 < 3 < 5")

    # Wave 2 retracement of Wave 1
    base_1 = p1 - p0
    R2 = safe_ratio(p1 - p2, base_1)
    if 0.4 <= R2 <= 0.85:
        score += 1
        reasons.append(f"Wave2 retracement R2={R2:.2f} in [0.4,0.85]")

    # Wave 3 extension vs Wave 1
    E3 = safe_ratio(p3 - p2, base_1)
    if 1.0 <= E3 <= 2.5:
        score += 1
        reasons.append(f"Wave3 extension E3={E3:.2f} in [1.0,2.5]")

    # Wave 4 retracement of Wave 3
    base_3 = p3 - p2
    R4 = safe_ratio(p3 - p4, base_3)
    if 0.2 <= R4 <= 0.5:
        score += 1
        reasons.append(f"Wave4 retracement R4={R4:.2f} in [0.2,0.5]")

    # Wave 5 extension vs Wave 1
    E5 = safe_ratio(p5 - p4, base_1)
    if 0.3 <= E5 <= 1.2:
        score += 1
        reasons.append(f"Wave5 extension E5={E5:.2f} in [0.3,1.2]")

    # A/B/C structure
    whole = p5 - p0
    RA = safe_ratio(p5 - pa, whole)
    if 0.1 <= RA <= 0.4:
        score += 1
        reasons.append(f"WaveA retracement RA={RA:.2f} in [0.1,0.4]")

    RC = safe_ratio(p5 - pc, whole)
    if 0.3 <= RC <= 0.8:
        score += 1
        reasons.append(f"WaveC retracement RC={RC:.2f} in [0.3,0.8]")

    # A/B/C position checks
    if pa < p5 and pa < pb < p5:
        score += 1
        reasons.append("A and B between 0 and 5 (uptrend correction)")
    if pc < pa:
        score += 1
        reasons.append("C below A")

    if score < min_score:
        return None

    waves = {
        "0": P0,
        "1": P1,
        "2": P2,
        "3": P3,
        "4": P4,
        "5": P5,
        "A": PA,
        "B": PB,
        "C": PC,
    }
    return {"waves": waves, "score": score, "reasons": reasons}


# ---------------- PLOTTING ----------------
def make_elliott_chart(df: pd.DataFrame, pivots, pattern=None, ticker: str = "", tf_label: str = ""):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=False,
        )
    )

    # Plot all pivots as small markers
    if pivots:
        fig.add_trace(
            go.Scatter(
                x=[p["idx"] for p in pivots],
                y=[p["price"] for p in pivots],
                mode="markers+text",
                text=["" for _ in pivots],
                marker=dict(size=6),
                name="Pivots",
                showlegend=False,
            )
        )

    # If Elliott pattern found, label 0–5 + A–B–C
    if pattern is not None:
        waves = pattern["waves"]
        labels = []
        xs = []
        ys = []
        for label in ["0", "1", "2", "3", "4", "5", "A", "B", "C"]:
            p = waves[label]
            xs.append(p["idx"])
            # Slight offset for text visibility
            y_offset = 0.003 * df["Close"].iloc[-1]
            y_val = p["price"] + (y_offset if label in ["1", "3", "5", "B"] else -y_offset)
            ys.append(y_val)
            labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="text+markers",
                text=labels,
                textposition="top center",
                marker=dict(size=9),
                name="Elliott Waves",
            )
        )

    fig.update_layout(
        title=f"Elliott Wave Detection — {ticker} ({tf_label})",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=700,
    )

    return fig


# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Yahoo Ticker (e.g. RELIANCE.NS, TCS.NS, AAPL)", value="RELIANCE.NS")

    tf_label = st.radio(
        "Timeframe",
        list(TIMEFRAME_CONFIG.keys()),
        index=1,  # default Daily
    )

    cfg = TIMEFRAME_CONFIG[tf_label]

    zigzag_pct = st.slider(
        "ZigZag deviation (%)",
        min_value=1.0,
        max_value=20.0,
        value=float(cfg["zigzag_pct"]),
        step=0.5,
        help="Minimum % move required to form a pivot (higher = fewer pivots, cleaner waves).",
    )

    min_score = st.slider(
        "Min Elliott score (0–10)",
        min_value=5,
        max_value=10,
        value=9,
        step=1,
        help="Higher score = stricter Elliott pattern validation.",
    )

# ---------------- MAIN APP ----------------
df = load_price_data(ticker, cfg["interval"], cfg["period"])

if df.empty:
    st.error("No data returned. Check ticker or internet connection.")
else:
    st.write(f"Data points: {len(df)} from {df.index.min().date()} to {df.index.max().date()}.")

    pivots = zigzag_pivots(df, pct=zigzag_pct)

    st.write(f"Detected pivots: {len(pivots)} (using {zigzag_pct:.1f}% ZigZag threshold).")

    pattern = detect_elliott_5_abc(pivots, min_score=min_score)

    if pattern is None:
        st.warning("❌ No strong 0–5 + A–B–C uptrend pattern detected on the last pivots with current settings.")
    else:
        st.success(f"✅ Elliott 0–5 + A–B–C pattern detected! Score = {pattern['score']}")
        with st.expander("Show rule checks / reasons"):
            for r in pattern["reasons"]:
                st.markdown(f"- {r}")

    fig = make_elliott_chart(df, pivots, pattern, ticker=ticker, tf_label=tf_label)
    st.plotly_chart(fig, use_container_width=True)
