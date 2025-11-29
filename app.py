import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Elliott Wave Detector (ML + ZigZag)",
    layout="wide"
)

st.title("📈 Elliott Wave 1-5 + A-B-C Detection (ML + ZigZag)")
st.write(
    "This demo uses a simple ML model + zigzag pivots to detect a "
    "possible Elliott 1-2-3-4-5-A-B-C pattern and label it on the chart."
)

# -------------------------------------------------------------------
# DOWNLOAD DATA
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    df.dropna(inplace=True)
    return df


# -------------------------------------------------------------------
# ZIGZAG PIVOTS (simple % move version)
# -------------------------------------------------------------------
def zigzag_pivots(series: pd.Series, pct: float = 3.0):
    """
    Return pivot indices and prices using a simple percentage zigzag.
    pct = minimum percentage move to confirm a pivot.
    """
    close = series.values
    n = len(close)
    if n < 3:
        return [], []

    pct = pct / 100.0
    pivots_idx = []
    pivots_price = []

    last_pivot_idx = 0
    last_pivot_price = close[0]
    last_extreme_idx = 0
    last_extreme_price = close[0]
    direction = 0  # 0 unknown, 1 up, -1 down

    for i in range(1, n):
        price = close[i]

        # update extreme
        if direction >= 0:  # up or unknown -> track highs
            if price > last_extreme_price:
                last_extreme_price = price
                last_extreme_idx = i
        if direction <= 0:  # down or unknown -> track lows
            if price < last_extreme_price:
                last_extreme_price = price
                last_extreme_idx = i

        change = (price - last_pivot_price) / last_pivot_price

        # reversal big enough?
        if direction >= 0 and change <= -pct:
            # up -> down, pivot high
            pivots_idx.append(last_extreme_idx)
            pivots_price.append(close[last_extreme_idx])
            last_pivot_idx = last_extreme_idx
            last_pivot_price = close[last_pivot_idx]
            direction = -1
            last_extreme_idx = i
            last_extreme_price = price

        elif direction <= 0 and change >= pct:
            # down -> up, pivot low
            pivots_idx.append(last_extreme_idx)
            pivots_price.append(close[last_extreme_idx])
            last_pivot_idx = last_extreme_idx
            last_pivot_price = close[last_pivot_idx]
            direction = 1
            last_extreme_idx = i
            last_extreme_price = price

    # add last extreme as final pivot
    if last_extreme_idx not in pivots_idx:
        pivots_idx.append(last_extreme_idx)
        pivots_price.append(close[last_extreme_idx])

    return pivots_idx, pivots_price


# -------------------------------------------------------------------
# ML: SYNTHETIC TRAIN DATA FOR ELLIOTT SHAPE
# -------------------------------------------------------------------
def pattern_features(y):
    """Convert 8 pivot prices to 7 normalized deltas."""
    y = np.array(y, dtype=float)
    mn = y.min()
    mx = y.max()
    if mx == mn:
        norm = np.zeros_like(y)
    else:
        norm = (y - mn) / (mx - mn)
    diffs = np.diff(norm)
    return diffs


def build_synthetic_dataset(n_elliott: int = 2000, n_random: int = 2000):
    X = []
    y = []

    base = np.array([0.1, 0.4, 0.15, 0.75, 0.5, 1.0, 0.7, 0.3])
    rng = np.random.default_rng(42)

    # Elliott-like patterns
    for _ in range(n_elliott):
        noise = rng.normal(0, 0.05, size=base.shape)
        scale = rng.uniform(0.8, 1.2)
        shift = rng.uniform(-0.1, 0.1)
        sample = base * scale + shift + noise
        X.append(pattern_features(sample))
        y.append(1)

    # Non-Elliott patterns
    for _ in range(n_random):
        style = rng.integers(0, 3)
        if style == 0:
            steps = rng.normal(0, 0.2, size=8)
            sample = np.cumsum(steps)
        elif style == 1:
            sample = np.sort(rng.normal(0, 1, size=8))
        else:
            half = rng.normal(0, 0.5, size=4)
            if rng.random() < 0.5:
                sample = np.concatenate([half, half[::-1]])
            else:
                sample = np.concatenate([half[::-1], half])

        X.append(pattern_features(sample))
        y.append(0)

    X = np.vstack(X)
    y = np.array(y)
    return X, y


@st.cache_resource(show_spinner=False)
def get_elliott_model():
    X, y = build_synthetic_dataset()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=0
    )
    model.fit(X, y)
    return model


def find_best_elliott_window(piv_idx, piv_price, model, prob_threshold=0.6):
    """Slide over pivots and find best 8-pivot Elliott window."""
    piv_idx = np.array(piv_idx)
    piv_price = np.array(piv_price, dtype=float)

    if len(piv_idx) < 8:
        return None

    best_prob = 0.0
    best_window = None

    for start in range(0, len(piv_idx) - 7):
        end = start + 8
        window_prices = piv_price[start:end]
        feats = pattern_features(window_prices).reshape(1, -1)
        prob = model.predict_proba(feats)[0, 1]
        if prob > best_prob:
            best_prob = prob
            best_window = (start, end, prob)

    if best_window is None or best_prob < prob_threshold:
        return None
    return best_window


# -------------------------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker (Yahoo symbol)", "RELIANCE.NS")

tf_label = st.sidebar.selectbox(
    "Timeframe",
    ["Hourly", "Daily", "Weekly"]
)

user_pct = st.sidebar.slider(
    "Base ZigZag % (pivot sensitivity)",
    min_value=0.5, max_value=10.0, value=3.0, step=0.5
)

if tf_label == "Hourly":
    interval = "1h"
    period = "90d"
elif tf_label == "Daily":
    interval = "1d"
    period = "3y"
else:
    interval = "1wk"
    period = "10y"

st.sidebar.markdown(
    "ℹ️ If no pattern is found, try changing timeframe or ZigZag %."
)

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
try:
    df = load_price_data(ticker, interval, period)
except Exception as e:
    st.error(f"Error downloading data for {ticker}: {e}")
    st.stop()

if df.empty:
    st.error("No data downloaded. Try another ticker or timeframe.")
    st.stop()

close = df["Close"]

# -------------------------------------------------------------------
# AUTO-ADJUST ZIGZAG TO GET ENOUGH PIVOTS
# -------------------------------------------------------------------
pct_candidates = [user_pct, max(user_pct * 0.7, 0.5), max(user_pct * 0.5, 0.5), 1.0, 0.8, 0.5]
used_pct = None
piv_idx = []
piv_price = []

for pct in pct_candidates:
    piv_idx, piv_price = zigzag_pivots(close, pct=pct)
    if len(piv_idx) >= 8:
        used_pct = pct
        break

if used_pct is None:
    st.warning(
        "Even with very sensitive ZigZag (0.5%) there are still less than 8 pivots.\n\n"
        "Try a longer period or different timeframe / ticker."
    )

    # Still show price chart without labels
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"))
    fig.update_layout(
        height=600,
        title=f"{ticker} — {tf_label} (not enough pivots for Elliott detection)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: This is an educational prototype. Real Elliott wave analysis is "
        "highly subjective; this simple ML model only tries to match a generic 1-5-A-B-C shape."
    )
    st.stop()

# debug/info
st.write(
    f"🔹 Using ZigZag **{used_pct:.2f}%** → Found **{len(piv_idx)}** pivots."
)

# -------------------------------------------------------------------
# ELLIOTT ML DETECTION
# -------------------------------------------------------------------
model = get_elliott_model()
best = find_best_elliott_window(piv_idx, piv_price, model)

labels = {}  # index -> text label

if best is not None:
    start, end, prob = best
    selected_idx = np.array(piv_idx[start:end])
    elliot_labels = ["1", "2", "3", "4", "5", "A", "B", "C"]

    for idx, lab in zip(selected_idx, elliot_labels):
        labels[idx] = lab

    st.success(
        f"✅ Possible Elliott 1-5-A-B-C pattern found "
        f"(probability ≈ {prob:.2f}) on timeframe: **{tf_label}**"
    )
    st.write(
        f"Pattern window: **{df.index[selected_idx[0]].date()}** → "
        f"**{df.index[selected_idx[-1]].date()}**"
    )
else:
    st.info(
        "ML model did not find a strong Elliott 1-5-A-B-C pattern with current settings "
        "(but pivots are still shown)."
    )

# -------------------------------------------------------------------
# PLOT CHART WITH LABELS
# -------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=close,
        mode="lines",
        name="Close"
    )
)

pivot_dates = df.index[piv_idx]
fig.add_trace(
    go.Scatter(
        x=pivot_dates,
        y=piv_price,
        mode="markers+text",
        name="Pivots",
        text=[labels.get(i, "") for i in piv_idx],
        textposition="top center",
        marker=dict(size=8)
    )
)

fig.update_layout(
    height=650,
    margin=dict(l=40, r=40, t=60, b=40),
    title=f"{ticker} — {tf_label} with ZigZag pivots & Elliott labels (if detected)"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: This is an educational prototype. Real Elliott wave analysis is highly subjective; "
    "this simple ML model only tries to match a generic 1-5-A-B-C shape."
)
