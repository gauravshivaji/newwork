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
    - **ML-based Elliott Wave detector** (9 pivots: 0–5, A, B, C)  
    - Timeframe-specific BUY rules (RSI, divergence, SMA, support, Elliott)  
    - RandomForest on returns used only as **advisory**, not final decision  
    """
)

# ---------------- NIFTY 500 TICKERS ----------------
NIFTY500_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "HINDUNILVR.NS",
    "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    # paste your full list here...
]

# ---------------- DATA & INDICATORS ----------------

@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """Download OHLC data for selected timeframe and flatten columns."""
    if timeframe == "Daily":
        interval = "1d"
        period = "3y"
    elif timeframe == "Weekly":
        interval = "1wk"
        period = "10y"
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    wanted_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[wanted_cols].copy()

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

    df["rsi"] = ta.momentum.rsi(close=close, window=14)
    df["sma22"] = close.rolling(22).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()

    for w in [22, 50, 200]:
        sma_col = f"sma{w}"
        df[f"dist_sma{w}"] = (df["Close"] - df[sma_col]) / df[sma_col]

    return df


# ---------- ELLIOTT ML ENGINE (9-PIVOT MODEL) ----------

def get_timeframe_params(timeframe: str):
    if timeframe == "Hourly":
        return {"order": 6, "min_pct": 0.02}
    elif timeframe == "Daily":
        return {"order": 5, "min_pct": 0.04}
    elif timeframe == "Weekly":
        return {"order": 4, "min_pct": 0.06}
    return {"order": 5, "min_pct": 0.04}


def find_pivots(prices: np.ndarray, order: int = 5):
    pivots = []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        if prices[i] == window.min():
            pivots.append((i, "low"))
        elif prices[i] == window.max():
            pivots.append((i, "high"))
    return pivots


def filter_pivots_min_move(prices: np.ndarray, pivots, min_pct: float):
    if not pivots:
        return []
    filtered = [pivots[0]]
    last_idx, _ = pivots[0]
    last_price = prices[last_idx]
    for idx, kind in pivots[1:]:
        p = prices[idx]
        if abs(p - last_price) / last_price >= min_pct:
            filtered.append((idx, kind))
            last_idx, last_price = idx, p
    return filtered


def is_alternating(seq):
    if len(seq) < 2:
        return False
    last_kind = seq[0][1]
    for _, kind in seq[1:]:
        if kind == last_kind:
            return False
        last_kind = kind
    return True


def fib_ratio(a, b):
    if b == 0:
        return None
    return abs(a / b)


def simple_ew_rule_check(prices, seq):
    """
    Light-weight rule filter based on your Elliott rules.
    Returns (valid: bool, trend: 'up'/'down'/None)
    """
    idxs = [i for i, _ in seq]
    p = [prices[i] for i in idxs]

    # Trend from first two pivots
    if p[1] > p[0]:
        trend = "up"
    elif p[1] < p[0]:
        trend = "down"
    else:
        return False, None

    # work on "up" version (mirror for down)
    if trend == "down":
        p = [-x for x in p]

    W0, W1, W2, W3, W4, W5, W6, W7, W8 = p

    # Wave 2 not below start of 1
    if W2 <= W0:
        return False, None

    # Wave 3 > wave 1 high
    if W3 <= W1:
        return False, None

    len1 = W1 - W0
    len3 = W3 - W2
    len5 = W5 - W4

    if len3 <= 0 or len3 <= min(len1, len5):
        return False, None   # wave 3 never shortest

    # W4 above W1 (no overlap)
    if W4 <= W1:
        return False, None

    # W5 higher than W3 (allow small truncation)
    if W5 < W3 * 0.98:
        return False, None

    # A breaks below W4
    if W6 >= W4:
        return False, None

    # B not way above 5
    if W7 > W5 * 1.02:
        return False, None

    # C below A
    if W8 >= W6:
        return False, None

    return True, trend


# ---------- ML: TRAIN A PATTERN CLASSIFIER ON SYNTHETIC DATA ----------

def make_elliott_features(points):
    """
    points: list of 9 prices [W0..W8]
    Return feature vector: normalized shape + ratios.
    """
    p = np.array(points, dtype=float)
    base = p[0]
    span = p.max() - p.min()
    if span == 0:
        span = 1.0
    norm = (p - base) / span

    # length ratios between waves
    W0, W1, W2, W3, W4, W5, W6, W7, W8 = p
    l1 = abs(W1 - W0)
    l2 = abs(W2 - W1)
    l3 = abs(W3 - W2)
    l4 = abs(W4 - W3)
    l5 = abs(W5 - W4)
    lA = abs(W6 - W5)
    lB = abs(W7 - W6)
    lC = abs(W8 - W7)

    ratios = []
    for a, b in [(l2, l1), (l3, l1), (l3, l5), (l4, l3), (l5, l3), (lA, lC), (lC, lA)]:
        ratios.append(0 if b == 0 else a / b)

    return np.concatenate([norm, np.array(ratios)])


def synth_elliott_up():
    """
    Create one synthetic valid Elliott 0-5,A,B,C uptrend pattern (9 points).
    Very rough but respects main rules + fib-ish ratios.
    """
    # W0 = 0
    W0 = 0.0
    # choose base size
    L1 = np.random.uniform(1.0, 2.0)          # wave 1
    retr2 = np.random.uniform(0.4, 0.7)       # wave 2 retrace
    L3 = L1 * np.random.uniform(1.4, 2.0)     # wave 3
    retr4 = np.random.uniform(0.2, 0.5)       # wave 4 retrace
    L5 = L1 * np.random.uniform(0.5, 1.0)     # wave 5

    # build impulse
    W1 = W0 + L1
    W2 = W1 - retr2 * L1
    W3 = W2 + L3
    W4 = W3 - retr4 * L3
    W5 = W4 + L5

    # correction: A,B,C
    LA = np.random.uniform(0.3, 0.8) * (W5 - W4)
    W6 = W5 - LA
    LB_ratio = np.random.uniform(0.3, 0.8)
    W7 = W6 + LB_ratio * LA
    LC_ratio = np.random.choice([1.0, 1.6]) * np.random.uniform(0.8, 1.2)
    W8 = W7 - LC_ratio * LA

    pts = np.array([W0, W1, W2, W3, W4, W5, W6, W7, W8], dtype=float)

    # add slight noise
    noise = np.random.normal(scale=0.05 * L1, size=pts.shape)
    return pts + noise


def synth_random_zigzag():
    """
    Negative sample: random 9-point zigzag that likely breaks rules.
    """
    pts = [0.0]
    for _ in range(8):
        step = np.random.uniform(-2.0, 2.0)
        pts.append(pts[-1] + step)
    return np.array(pts, dtype=float)


@st.cache_resource(show_spinner=False)
def train_elliott_ml_model(n_samples: int = 1500):
    """
    Train a RandomForest to classify "valid Elliott-like pattern" vs random.
    Uses synthetic data only (no internet needed).
    """
    X = []
    y = []

    # positive class
    for _ in range(n_samples):
        up = synth_elliott_up()
        X.append(make_elliott_features(up))
        y.append(1)

        # also add mirrored downtrend version
        down = -up
        X.append(make_elliott_features(down))
        y.append(1)

    # negative class: random
    for _ in range(n_samples):
        neg = synth_random_zigzag()
        X.append(make_elliott_features(neg))
        y.append(0)

    X = np.vstack(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # optional: could compute test accuracy and print; skipped in UI
    return model


ELLIOTT_ML_MODEL = train_elliott_ml_model()


def ml_score_pattern(prices, seq):
    """Return ML probability that this 9-pivot sequence is Elliott-like."""
    idxs = [i for i, _ in seq]
    pts = [prices[i] for i in idxs]
    feats = make_elliott_features(pts).reshape(1, -1)
    proba = ELLIOTT_ML_MODEL.predict_proba(feats)[0, 1]
    return proba


def detect_best_elliott_cycle(prices: np.ndarray, timeframe: str):
    """
    Full ML+rules based detector:
    1) build pivots
    2) filter small moves
    3) slide 9-pivot window
    4) apply light rule filter + orientation
    5) score with ML model
    6) choose candidate with max span * proba
    """
    params = get_timeframe_params(timeframe)
    order = params["order"]
    min_pct = params["min_pct"]

    raw_pivots = find_pivots(prices, order=order)
    pivots = filter_pivots_min_move(prices, raw_pivots, min_pct=min_pct)

    m = len(pivots)
    if m < 9:
        return {}

    best = None  # (score_weighted, span, ml_proba, labels_map)

    for start in range(0, m - 9 + 1):
        window = pivots[start:start + 9]
        if not is_alternating(window):
            continue

        # quick rule check
        valid, trend = simple_ew_rule_check(prices, window)
        if not valid or trend is None:
            continue

        kinds = [k for _, k in window]
        if trend == "up":
            expected = ["low", "high", "low", "high", "low", "high", "low", "high", "low"]
        else:
            expected = ["high", "low", "high", "low", "high", "low", "high", "low", "high"]
        if kinds != expected:
            continue

        # ML probability
        proba = ml_score_pattern(prices, window)
        if proba < 0.55:
            continue  # too unlikely

        idxs = [i for i, _ in window]
        span = idxs[-1] - idxs[0]

        labels_map = {}
        wave_pattern = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]
        for w_idx, (bar_idx, _) in enumerate(window):
            labels_map[bar_idx] = wave_pattern[w_idx]

        score_weighted = proba * (span + 1)

        cand = (score_weighted, span, proba, labels_map)
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return {}
    return best[3]


def fallback_elliott_cycle(prices: np.ndarray, timeframe: str):
    """
    Fallback when ML+rules find nothing:
    - Spread 9 pivots over full timeframe
    - Basic orientation only.
    """
    params = get_timeframe_params(timeframe)
    order = params["order"]
    min_pct = params["min_pct"]

    raw_pivots = find_pivots(prices, order=order)
    pivots = filter_pivots_min_move(prices, raw_pivots, min_pct=min_pct)

    m = len(pivots)
    if m < 2:
        return {}

    trend = "up" if prices[-1] >= prices[0] else "down"
    if trend == "up":
        pattern_kinds = ["low", "high", "low", "high", "low", "high", "low", "high", "low"]
    else:
        pattern_kinds = ["high", "low", "high", "low", "high", "low", "high", "low", "high"]

    if m <= 9:
        candidate = pivots
    else:
        step = (m - 1) / (9 - 1)
        idxs = [int(round(k * step)) for k in range(9)]
        idxs = sorted(set(max(0, min(m - 1, i)) for i in idxs))
        while len(idxs) < min(9, m):
            for j in range(m):
                if j not in idxs:
                    idxs.append(j)
                    if len(idxs) == min(9, m):
                        break
        idxs = sorted(idxs)
        candidate = [pivots[j] for j in idxs]

    kinds = [k for _, k in candidate]
    expected = pattern_kinds[:len(candidate)]
    # don't reject if mismatch, just accept as crude

    labels_map = {}
    wave_pattern = ["0", "1", "2", "3", "4", "5", "A", "B", "C"]
    for k, (bar_idx, _) in enumerate(candidate):
        if k >= len(wave_pattern):
            break
        labels_map[bar_idx] = wave_pattern[k]
    return labels_map


def add_elliott_labels(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    df["elliott_wave"] = np.nan
    if "Close" not in df.columns or df.empty:
        return df

    prices = df["Close"].values
    labels_map = detect_best_elliott_cycle(prices, timeframe)
    if not labels_map:
        labels_map = fallback_elliott_cycle(prices, timeframe)

    for bar_idx, label in labels_map.items():
        if 0 <= bar_idx < len(df.index):
            df.at[df.index[bar_idx], "elliott_wave"] = label
    return df


def last_elliott_label(df: pd.DataFrame):
    if "elliott_wave" not in df.columns:
        return None
    s = df["elliott_wave"].dropna()
    if s.empty:
        return None
    return s.iloc[-1]


# ---------- SUPPORT / DIVERGENCE HELPERS ----------

def detect_bullish_rsi_divergence(df: pd.DataFrame, order: int = 5) -> bool:
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


# --------- BUY RULES (ONLY RULES DECIDE YES/NO) ---------

def hourly_rules(df: pd.DataFrame):
    df = df.dropna(subset=["rsi", "sma22"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 25 <= rsi <= 35
    cond_div = detect_bullish_rsi_divergence(df, order=5)
    cond_support = near_support(df, tolerance=0.02, order=5)
    cond_sma = last["Close"] >= last["sma22"]
    ell = last_elliott_label(df)
    cond_ell = ell in ["0", "2"]

    flags = {
        "RSI between 25–35 (near 30)": cond_rsi,
        "Bullish RSI divergence": cond_div,
        "Price near intraday support (≤ 2%)": cond_support,
        "Price above SMA22": cond_sma,
        "Elliott phase in 0 / 2": cond_ell,
    }
    decision = 1 if all(flags.values()) else 0
    return decision, flags


def daily_rules(df: pd.DataFrame):
    df = df.dropna(subset=["rsi", "sma22", "sma200"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 25 <= rsi <= 40
    cond_div = detect_bullish_rsi_divergence(df, order=5)
    cond_support = near_support(df, tolerance=0.05, order=5)

    price = last["Close"]
    sma22 = last["sma22"]
    sma200 = last["sma200"]

    cond_sma_short = price >= sma22
    cond_sma_long = (price >= 0.95 * sma200)
    ell = last_elliott_label(df)
    cond_ell = ell in ["0", "2", "4"]

    flags = {
        "RSI between 25–40 (near 30)": cond_rsi,
        "Bullish RSI divergence": cond_div,
        "Price near support (≤ 5%)": cond_support,
        "Price above SMA22": cond_sma_short,
        "Price not far below SMA200 (≥95%)": cond_sma_long,
        "Elliott phase in 0 / 2 / 4": cond_ell,
    }
    decision = 1 if all(flags.values()) else 0
    return decision, flags


def weekly_rules(df: pd.DataFrame):
    df = df.dropna(subset=["rsi", "sma200"]).copy()
    if df.empty:
        return 0, {"Not enough data": False}

    last = df.iloc[-1]
    rsi = last["rsi"]

    cond_rsi = 35 <= rsi <= 50
    price = last["Close"]
    sma200 = last["sma200"]
    cond_sma200 = price >= sma200
    cond_support = near_support(df, tolerance=0.10, order=5)
    ell = last_elliott_label(df)
    cond_ell = ell in ["2", "4"]

    flags = {
        "RSI between 35–50 (accumulation)": cond_rsi,
        "Price above SMA200": cond_sma200,
        "Price near major support (≤10%)": cond_support,
        "Elliott phase in 2 / 4": cond_ell,
    }
    decision = 1 if all(flags.values()) else 0
    return decision, flags


def rule_based_buy(df: pd.DataFrame, timeframe: str):
    if timeframe == "Hourly":
        return hourly_rules(df)
    elif timeframe == "Daily":
        return daily_rules(df)
    elif timeframe == "Weekly":
        return weekly_rules(df)
    return 0, {"Invalid timeframe": False}


def elliott_code_series(df: pd.DataFrame) -> pd.Series:
    mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "A": 6, "B": 7, "C": 8}
    if "elliott_wave" not in df.columns:
        return pd.Series(index=df.index, data=-1)
    return df["elliott_wave"].map(mapping).fillna(-1)


# ---------- RANDOM FOREST ON RETURNS (ADVISORY ONLY) ----------

def train_rf_and_predict(df: pd.DataFrame):
    needed_cols = ["rsi", "sma22", "sma50", "sma200"]
    for c in needed_cols:
        if c not in df.columns:
            return None

    df = df.dropna(subset=needed_cols).copy()
    if len(df) < 50:
        return None

    df["fwd_return"] = df["Close"].shift(-5) / df["Close"] - 1
    df = df.iloc[:-5]

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
    Final decision comes ONLY from rules.
    RF output is advisory.
    """
    rule_sig, flags = rule_based_buy(df, timeframe)
    rf_res = train_rf_and_predict(df)

    final = "Yes" if rule_sig == 1 else "No"

    lines = []
    lines.append(f"**Rule-based checks ({timeframe})**")
    for desc, ok in flags.items():
        mark = "✅" if ok else "❌"
        lines.append(f"- {mark} {desc}")

    if rf_res is None:
        lines.append("")
        lines.append("_RandomForest not used (insufficient history). Final decision is rules only._")
    else:
        rf_pred, rf_proba = rf_res
        lines.append("")
        lines.append("**Machine Learning (RandomForest on returns) — advisory**")
        lines.append(f"- Model bullish probability: `{rf_proba:.2f}`")
        lines.append(f"- Model class: `{'Bullish' if rf_pred == 1 else 'Not Bullish'}`")
        lines.append("⚠ Final BUY = rules only, ML is just extra info.")

    lines.append("")
    lines.append(f"**Final decision:** **{final}** for **{timeframe}** timeframe (rules only).")

    reason_text = "\n".join(lines)
    if return_reason:
        return final, reason_text
    else:
        return final


# ---------- PLOTTING ----------

def plot_chart(df: pd.DataFrame, ticker: str, timeframe: str):
    needed_cols = ["Open", "High", "Low", "Close", "sma22", "sma50", "sma200", "rsi"]
    for c in needed_cols:
        if c not in df.columns:
            st.warning("Not enough data to plot indicators.")
            return

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

    fig.add_trace(go.Scatter(x=df.index, y=df["sma22"], name="SMA 22"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="SMA 200"), row=1, col=1)

    if "elliott_wave" in df.columns:
        ell = df["elliott_wave"].dropna()
        for ts, label in ell.items():
            price = df.loc[ts, "Close"]
            fig.add_annotation(
                x=ts,
                y=price,
                text=str(label),
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-20,
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI 14"), row=2, col=1)

    fig.add_shape(
        type="line", x0=df.index[0], x1=df.index[-1],
        y0=30, y1=30, xref="x2", yref="y2",
        line=dict(dash="dash"),
    )
    fig.add_shape(
        type="line", x0=df.index[0], x1=df.index[-1],
        y0=70, y1=70, xref="x2", yref="y2",
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
    value=15,
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
        st.metric(label=f"Buy signal ({chart_timeframe})", value=signal)
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
        "✅ 'Yes' = ALL timeframe-specific rules satisfied "
        "(RSI, divergence, SMA, support, Elliott). "
        "RandomForest on returns is only advisory."
    )
else:
    st.info("Click **Run Scan** to build the multi-timeframe Buy table.")
