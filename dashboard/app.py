import sys
from pathlib import Path
import textwrap

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data.live_ohlcv import fetch_live_ohlcv
from features.volatility_features import (
    calculate_features,
    FEATURE_COLUMNS,
)
from volatility_predictor import predict_volatility


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CryptoSentimentML",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_REFRESH_SECONDS = 60

INITIAL_CAPITAL = 10_000.00

OOS_FINAL_CAPITAL = 10_614.48
OOS_RETURN = 6.14
OOS_PROFIT_FACTOR = 1.353
OOS_MAX_DRAWDOWN = -8.24
OOS_WIN_RATE = 36.36
OOS_TRADES = 22

WALK_FORWARD_ACCURACY = 62.40
PREDICTION_HORIZON = 6

MIN_CONFIDENCE = 0.85


# ============================================================
# CUSTOM HTML HELPER
# ============================================================
# IMPORTANT:
# Streamlit's st.markdown() can interpret indented HTML
# as a Markdown code block.
#
# st.html() renders the HTML directly.
# textwrap.dedent() removes unwanted indentation.
# ============================================================

def render_html(html):
    st.html(
        textwrap.dedent(html).strip()
    )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1700px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0.55rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080b12 0%,
            #0b0f18 50%,
            #080b12 100%
        );

    border-right: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.3rem;
}

.sidebar-brand {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.sidebar-subtitle {
    color: #7f8a9c;
    font-size: 0.75rem;
    margin-top: -5px;
}

.sidebar-section {
    color: #687386;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin-top: 18px;
    margin-bottom: 7px;
}


/* =========================================================
   TOP BAR
   ========================================================= */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 10px 0 18px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.07);

    margin-bottom: 22px;
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.market-dot {
    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: #22c55e;

    display: inline-block;

    box-shadow:
        0 0 10px rgba(34,197,94,0.7);
}

.market-live {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #86efac;
}

.market-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #dbe3ef;
}

.topbar-right {
    color: #718096;
    font-size: 0.75rem;
}


/* =========================================================
   PAGE HEADER
   ========================================================= */

.page-title {
    font-size: 2rem;
    line-height: 1.1;

    font-weight: 850;

    letter-spacing: -1px;

    margin-bottom: 3px;
}

.page-subtitle {
    color: #7f8a9c;
    font-size: 0.84rem;
}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi {
    position: relative;

    min-height: 128px;

    padding: 19px 20px;

    border-radius: 15px;

    border:
        1px solid rgba(255,255,255,0.075);

    background:
        linear-gradient(
            145deg,
            rgba(20,26,38,0.96),
            rgba(12,16,25,0.96)
        );

    overflow: hidden;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

.kpi:hover {
    transform: translateY(-2px);

    border-color:
        rgba(96,165,250,0.25);
}

.kpi:after {
    content: "";

    position: absolute;

    top: -40px;
    right: -40px;

    width: 100px;
    height: 100px;

    border-radius: 50%;

    background:
        rgba(59,130,246,0.05);
}

.kpi-label {
    color: #7f8a9c;

    font-size: 0.68rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}

.kpi-value {
    color: #f4f7fb;

    font-size: 1.55rem;

    font-weight: 800;

    margin-top: 10px;
}

.kpi-sub {
    color: #687386;

    font-size: 0.72rem;

    margin-top: 5px;
}

.kpi-positive {
    color: #4ade80;
}

.kpi-negative {
    color: #f87171;
}

.kpi-neutral {
    color: #60a5fa;
}


/* =========================================================
   PANELS
   ========================================================= */

.panel {
    background:
        linear-gradient(
            145deg,
            rgba(17,23,34,0.97),
            rgba(11,15,23,0.97)
        );

    border:
        1px solid rgba(255,255,255,0.075);

    border-radius: 16px;

    padding: 20px;
}

.panel-title {
    font-size: 0.95rem;

    font-weight: 750;

    color: #e7edf6;
}

.panel-subtitle {
    font-size: 0.73rem;

    color: #6f7b8e;

    margin-top: 3px;

    margin-bottom: 12px;
}


/* =========================================================
   SIGNAL CARD
   ========================================================= */

.signal-card {
    border-radius: 17px;

    padding: 22px;

    min-height: 230px;

    border:
        1px solid rgba(255,255,255,0.08);

    background:
        radial-gradient(
            circle at top right,
            rgba(59,130,246,0.10),
            transparent 40%
        ),
        #0e141f;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

.signal-card:hover {
    transform: translateY(-2px);

    border-color:
        rgba(96,165,250,0.25);
}

.signal-label {
    font-size: 0.68rem;

    text-transform: uppercase;

    letter-spacing: 1px;

    color: #748095;

    font-weight: 700;
}

.signal-value {
    font-size: 2rem;

    font-weight: 850;

    margin-top: 10px;
}

.signal-desc {
    color: #8994a5;

    font-size: 0.78rem;

    line-height: 1.55;

    margin-top: 8px;
}


/* =========================================================
   REGIME
   ========================================================= */

.regime {
    border-radius: 14px;

    padding: 17px;

    border:
        1px solid rgba(255,255,255,0.08);
}

.regime-expand {
    background:
        rgba(245,158,11,0.075);

    border-color:
        rgba(245,158,11,0.25);
}

.regime-contract {
    background:
        rgba(59,130,246,0.075);

    border-color:
        rgba(59,130,246,0.25);
}

.regime-title {
    font-size: 0.68rem;

    color: #8994a5;

    font-weight: 700;

    letter-spacing: 0.8px;
}

.regime-value {
    font-size: 1.65rem;

    font-weight: 850;

    margin-top: 5px;
}

.regime-text {
    color: #7d8899;

    font-size: 0.72rem;

    margin-top: 4px;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section {
    margin-top: 25px;

    margin-bottom: 12px;
}

.section-title {
    font-size: 1.05rem;

    font-weight: 800;
}

.section-subtitle {
    color: #6f7b8e;

    font-size: 0.75rem;

    margin-top: 3px;
}


/* =========================================================
   TABLE
   ========================================================= */

.dataframe {
    border-radius: 12px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #4f5969;

    font-size: 0.68rem;

    padding-top: 35px;

    padding-bottom: 15px;
}


/* =========================================================
   STREAMLIT CLEANUP
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .page-title {
        font-size: 1.55rem;
    }

    .topbar {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }

}

</style>
"""
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"${value:,.2f}"


def pct(value):
    return f"{value:+.2f}%"


# ============================================================
# PRICE CHART
# ============================================================

def create_price_chart(chart_df):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["timestamp"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC/USDT",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema20"],
            mode="lines",
            name="EMA 20",
            line=dict(width=1.2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema50"],
            mode="lines",
            name="EMA 50",
            line=dict(width=1.2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["ema200"],
            mode="lines",
            name="EMA 200",
            line=dict(width=1.2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=chart_df["timestamp"],
            y=chart_df["volume"],
            name="Volume",
            opacity=0.45,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=610,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.045)",
    )

    return fig


# ============================================================
# EQUITY CHART
# ============================================================

def create_equity_chart():

    results_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "current_signal_realistic_results.csv"
    )

    if not results_path.exists():
        return None

    try:

        trades = pd.read_csv(results_path)

        if "capital_after" not in trades.columns:
            return None

        if "exit_timestamp" in trades.columns:

            x = pd.to_datetime(
                trades["exit_timestamp"],
                errors="coerce",
            )

        else:

            x = range(len(trades))

        equity = trades["capital_after"].astype(float)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(width=2),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
            )
        )

        fig.update_layout(
            height=390,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=15,
                b=10,
            ),
            yaxis_title="Capital",
            xaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            showgrid=False
        )

        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.045)"
        )

        return fig

    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            ₿ CryptoSentimentML
        </div>

        <div class="sidebar-subtitle">
            AI Trading Intelligence Terminal
        </div>
        """
    )

    st.divider()

    render_html(
        '<div class="sidebar-section">Workspace</div>'
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Live Intelligence",
            "Backtesting",
            "Strategy Analytics",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    render_html(
        '<div class="sidebar-section">Market</div>'
    )

    asset = st.selectbox(
        "Asset",
        ["BTC/USDT"],
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["1H"],
    )

    render_html(
        '<div class="sidebar-section">Live Engine</div>'
    )

    auto_refresh = st.toggle(
        "Auto refresh",
        value=True,
    )

    refresh_seconds = st.slider(
        "Refresh interval",
        min_value=30,
        max_value=300,
        value=DEFAULT_REFRESH_SECONDS,
        step=30,
        disabled=not auto_refresh,
    )

    if auto_refresh:

        st_autorefresh(
            interval=refresh_seconds * 1000,
            key="crypto_terminal_refresh",
        )

    if st.button(
        "↻  Refresh now",
        width="stretch",
    ):

        st.rerun()

    render_html(
        '<div class="sidebar-section">Model</div>'
    )

    st.caption(
        "XGBoost Volatility Classifier"
    )

    st.caption(
        "6-hour prediction horizon"
    )

    st.caption(
        "22 engineered features"
    )

    render_html(
        '<div class="sidebar-section">Validation</div>'
    )

    st.metric(
        "Walk-forward accuracy",
        f"{WALK_FORWARD_ACCURACY:.2f}%",
    )

    st.caption(
        "True out-of-sample walk-forward validation."
    )

    st.divider()

    st.caption(
        "BTC/USDT · Binance · CCXT"
    )


# ============================================================
# FETCH LIVE DATA
# ============================================================

try:

    with st.spinner(
        "Loading market intelligence..."
    ):

        df = fetch_live_ohlcv()

except Exception as e:

    st.error(
        f"Unable to fetch live Binance data: {e}"
    )

    st.stop()


# ============================================================
# MARKET VALIDATION
# ============================================================

required_market_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

missing_market_columns = [
    c
    for c in required_market_columns
    if c not in df.columns
]

if missing_market_columns:

    st.error(
        "Missing market columns: "
        + ", ".join(missing_market_columns)
    )

    st.stop()


if len(df) < 220:

    st.error(
        f"Not enough candles. Received {len(df)}; "
        "minimum required is 220."
    )

    st.stop()


# ============================================================
# FEATURES
# ============================================================

try:

    df = calculate_features(df)

except Exception as e:

    st.error(
        f"Feature calculation failed: {e}"
    )

    st.stop()


# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing_features:

    st.error(
        "Missing model features: "
        + ", ".join(missing_features)
    )

    st.stop()


latest_feature_row = df[
    FEATURE_COLUMNS
].tail(1)

if latest_feature_row.isnull().any().any():

    bad_features = (
        latest_feature_row
        .isnull()
        .sum()
    )

    bad_features = bad_features[
        bad_features > 0
    ].index.tolist()

    st.error(
        "Latest candle contains missing features: "
        + ", ".join(bad_features)
    )

    st.stop()


# ============================================================
# MODEL
# ============================================================

try:

    result = predict_volatility(df)

except Exception as e:

    st.error(
        f"Model prediction failed: {e}"
    )

    st.stop()


# ============================================================
# CURRENT VALUES
# ============================================================

current_price = float(
    df["close"].iloc[-1]
)

previous_price = float(
    df["close"].iloc[-2]
)

price_change_pct = (
    (current_price - previous_price)
    / previous_price
) * 100

confidence = float(
    result["confidence"]
)

contract_probability = float(
    result["contract_probability"]
)

expand_probability = float(
    result["expand_probability"]
)

regime = str(
    result["regime"]
)

signal_strength = str(
    result["signal_strength"]
)

horizon = int(
    result["prediction_horizon_hours"]
)

latest_timestamp = df[
    "timestamp"
].iloc[-1]

rsi = float(
    df["rsi"].iloc[-1]
)

atr = float(
    df["atr"].iloc[-1]
)

volatility_24h = float(
    df["volatility_24h"].iloc[-1]
)

volume_ratio = float(
    df["volume_ratio"].iloc[-1]
)

bb_width = float(
    df["bb_width"].iloc[-1]
)

macd = float(
    df["macd"].iloc[-1]
)


# ============================================================
# TOP STATUS BAR
# ============================================================

render_html(
    f"""
    <div class="topbar">

        <div class="topbar-left">

            <span class="market-dot"></span>

            <span class="market-live">
                LIVE MARKET
            </span>

            <span class="market-name">
                {asset}
            </span>

            <span style="color:#596579;">·</span>

            <span style="color:#8994a5;">
                {timeframe}
            </span>

        </div>

        <div class="topbar-right">
            Last candle: {latest_timestamp}
            &nbsp; · &nbsp;
            Horizon: {horizon}H
        </div>

    </div>
    """
)


# ============================================================
# PAGE HEADER
# ============================================================

render_html(
    f"""
    <div class="page-title">
        {page}
    </div>

    <div class="page-subtitle">
        AI-powered Bitcoin market and volatility intelligence
    </div>
    """
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    # ========================================================
    # KPI ROW
    # ========================================================

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:

        change_class = (
            "kpi-positive"
            if price_change_pct >= 0
            else "kpi-negative"
        )

        render_html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    BTC / USDT
                </div>

                <div class="kpi-value">
                    {money(current_price)}
                </div>

                <div class="kpi-sub">

                    <span class="{change_class}">
                        {price_change_pct:+.2f}%
                    </span>

                    &nbsp; 1H change

                </div>

            </div>
            """
        )

    with k2:

        regime_class = (
            "kpi-negative"
            if regime == "EXPAND"
            else "kpi-neutral"
        )

        render_html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Volatility Regime
                </div>

                <div class="kpi-value {regime_class}">
                    {regime}
                </div>

                <div class="kpi-sub">
                    Next {horizon} hours
                </div>

            </div>
            """
        )

    with k3:

        confidence_class = (
            "kpi-positive"
            if confidence >= MIN_CONFIDENCE
            else "kpi-neutral"
        )

        render_html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Model Confidence
                </div>

                <div class="kpi-value {confidence_class}">
                    {confidence * 100:.1f}%
                </div>

                <div class="kpi-sub">
                    Threshold {MIN_CONFIDENCE * 100:.0f}%
                </div>

            </div>
            """
        )

    with k4:

        render_html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    OOS Return
                </div>

                <div class="kpi-value kpi-positive">
                    +{OOS_RETURN:.2f}%
                </div>

                <div class="kpi-sub">
                    Realistic holdout
                </div>

            </div>
            """
        )

    with k5:

        render_html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Profit Factor
                </div>

                <div class="kpi-value">
                    {OOS_PROFIT_FACTOR:.3f}
                </div>

                <div class="kpi-sub">
                    {OOS_TRADES} OOS trades
                </div>

            </div>
            """
        )


    # ========================================================
    # MARKET INTELLIGENCE
    # ========================================================

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Market Intelligence
            </div>

            <div class="section-subtitle">
                BTC/USDT price action, trend structure and live model state
            </div>

        </div>
        """
    )

    chart_col, signal_col = st.columns(
        [2.7, 1]
    )

    with chart_col:

        render_html(
            """
            <div class="panel-title">
                BTC/USDT · 1H
            </div>

            <div class="panel-subtitle">
                100 latest candles · EMA trend structure · volume
            </div>
            """
        )

        chart_df = df.tail(100).copy()

        fig = create_price_chart(
            chart_df
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

    with signal_col:

        signal_color = (
            "#f59e0b"
            if regime == "EXPAND"
            else "#60a5fa"
        )

        render_html(
            f"""
            <div class="signal-card">

                <div class="signal-label">
                    Current Model State
                </div>

                <div
                    class="signal-value"
                    style="color:{signal_color};"
                >
                    {regime}
                </div>

                <div class="signal-desc">
                    XGBoost volatility classifier
                    currently favors the
                    <b>{regime}</b> regime.
                </div>

                <br>

                <div class="signal-label">
                    Confidence
                </div>

                <div
                    class="signal-value"
                    style="font-size:1.45rem;"
                >
                    {confidence * 100:.2f}%
                </div>

                <div class="signal-desc">
                    Prediction horizon:
                    <b>{horizon} hours</b>
                </div>

            </div>
            """
        )

        st.info(
            "Important: volatility regime is not a BUY/SELL signal."
        )

        if confidence >= MIN_CONFIDENCE:

            st.success(
                f"Model confidence is above the "
                f"{MIN_CONFIDENCE * 100:.0f}% research threshold."
            )

        else:

            st.warning(
                f"Model confidence is below the "
                f"{MIN_CONFIDENCE * 100:.0f}% research threshold."
            )


    # ========================================================
    # VOLATILITY PROBABILITIES
    # ========================================================

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Volatility Probability
            </div>

            <div class="section-subtitle">
                Six-hour ahead regime classification
            </div>

        </div>
        """
    )

    p1, p2 = st.columns(2)

    with p1:

        render_html(
            f"""
            <div class="regime regime-contract">

                <div class="regime-title">
                    CONTRACT
                </div>

                <div class="regime-value">
                    {contract_probability * 100:.2f}%
                </div>

                <div class="regime-text">
                    Probability of lower / contracting volatility.
                </div>

            </div>
            """
        )

        st.progress(
            contract_probability
        )

    with p2:

        render_html(
            f"""
            <div class="regime regime-expand">

                <div class="regime-title">
                    EXPAND
                </div>

                <div class="regime-value">
                    {expand_probability * 100:.2f}%
                </div>

                <div class="regime-text">
                    Probability of higher / expanding volatility.
                </div>

            </div>
            """
        )

        st.progress(
            expand_probability
        )


    # ========================================================
    # TECHNICAL SNAPSHOT
    # ========================================================

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Technical Snapshot
            </div>

            <div class="section-subtitle">
                Current values from the production feature pipeline
            </div>

        </div>
        """
    )

    t1, t2, t3, t4, t5, t6 = st.columns(6)

    with t1:

        st.metric(
            "RSI",
            f"{rsi:.2f}",
        )

    with t2:

        st.metric(
            "ATR",
            f"{atr:,.2f}",
        )

    with t3:

        st.metric(
            "24H Volatility",
            f"{volatility_24h * 100:.3f}%",
        )

    with t4:

        st.metric(
            "Volume Ratio",
            f"{volume_ratio:.2f}x",
        )

    with t5:

        st.metric(
            "BB Width",
            f"{bb_width * 100:.2f}%",
        )

    with t6:

        st.metric(
            "MACD",
            f"{macd:.2f}",
        )


# ============================================================
# LIVE INTELLIGENCE
# ============================================================

elif page == "Live Intelligence":

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Live ML Intelligence
            </div>

            <div class="section-subtitle">
                Real-time interpretation of the current market state
            </div>

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        render_html(
            """
            <div class="panel">

                <div class="panel-title">
                    Volatility Model
                </div>

                <div class="panel-subtitle">
                    XGBoost · 6-hour forecast
                </div>

            </div>
            """
        )

        st.metric(
            "Regime",
            regime,
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
        )

        st.metric(
            "Prediction Horizon",
            f"{horizon} Hours",
        )

    with c2:

        render_html(
            """
            <div class="panel">

                <div class="panel-title">
                    Probability Distribution
                </div>

                <div class="panel-subtitle">
                    Model class probabilities
                </div>

            </div>
            """
        )

        st.write(
            f"CONTRACT — "
            f"{contract_probability * 100:.2f}%"
        )

        st.progress(
            contract_probability
        )

        st.write(
            f"EXPAND — "
            f"{expand_probability * 100:.2f}%"
        )

        st.progress(
            expand_probability
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Feature Intelligence
            </div>

            <div class="section-subtitle">
                Same feature pipeline used by the trained model
            </div>

        </div>
        """
    )

    feature_display = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Current Value": [
                float(
                    df[feature].iloc[-1]
                )
                for feature in FEATURE_COLUMNS
            ],
        }
    )

    feature_display["Current Value"] = (
        feature_display["Current Value"]
        .round(6)
    )

    st.dataframe(
        feature_display,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# BACKTESTING
# ============================================================

elif page == "Backtesting":

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Out-of-Sample Backtesting
            </div>

            <div class="section-subtitle">
                Realistic holdout performance ·
                2026-01-19 → 2026-08-30
            </div>

        </div>
        """
    )

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:

        st.metric(
            "Final Capital",
            money(OOS_FINAL_CAPITAL),
        )

    with b2:

        st.metric(
            "OOS Return",
            f"+{OOS_RETURN:.2f}%",
        )

    with b3:

        st.metric(
            "Profit Factor",
            f"{OOS_PROFIT_FACTOR:.3f}",
        )

    with b4:

        st.metric(
            "Max Drawdown",
            f"{OOS_MAX_DRAWDOWN:.2f}%",
        )

    with b5:

        st.metric(
            "Win Rate",
            f"{OOS_WIN_RATE:.2f}%",
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Equity Curve
            </div>

            <div class="section-subtitle">
                Realistic execution model with 2× maximum leverage
            </div>

        </div>
        """
    )

    equity_fig = create_equity_chart()

    if equity_fig is not None:

        st.plotly_chart(
            equity_fig,
            width="stretch",
            config={
                "displaylogo": False,
            },
        )

    else:

        st.info(
            "Backtest result CSV was not found or does not contain "
            "the required capital_after column."
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Risk Configuration
            </div>

        </div>
        """
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "Risk / Trade",
            "1.00%",
        )

    with r2:

        st.metric(
            "Max Leverage",
            "2.00×",
        )

    with r3:

        st.metric(
            "Fee / Side",
            "0.10%",
        )

    with r4:

        st.metric(
            "Slippage",
            "0.02%",
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Backtest Summary
            </div>

        </div>
        """
    )

    st.write(
        f"""
        **Initial capital:** {money(INITIAL_CAPITAL)}

        **Final capital:** {money(OOS_FINAL_CAPITAL)}

        **Total trades:** {OOS_TRADES}

        **Winning trades:** 8

        **Losing trades:** 14

        **Win rate:** {OOS_WIN_RATE:.2f}%

        **Profit factor:** {OOS_PROFIT_FACTOR:.3f}

        **Maximum drawdown:** {OOS_MAX_DRAWDOWN:.2f}%

        **Maximum actual leverage:** 1.91×

        **Average actual leverage:** 0.69×

        **Execution:** realistic fee + slippage model
        """
    )


# ============================================================
# STRATEGY ANALYTICS
# ============================================================

elif page == "Strategy Analytics":

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Strategy Analytics
            </div>

            <div class="section-subtitle">
                Locked research configuration and holdout performance
            </div>

        </div>
        """
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Confidence Filter",
            "0.85",
        )

    with s2:

        st.metric(
            "Total OOS Trades",
            "22",
        )

    with s3:

        st.metric(
            "SELL Trades",
            "15",
        )

    with s4:

        st.metric(
            "BUY Trades",
            "7",
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Strategy Architecture
            </div>

        </div>
        """
    )

    render_html(
        """
        <div class="panel">

            <b>1. Market Data</b><br>
            BTC/USDT 1H OHLCV from Binance

            <br><br>

            ↓

            <br><br>

            <b>2. Technical Feature Pipeline</b><br>
            Returns · EMA · RSI · MACD · Bollinger Bands ·
            ATR · Volume · Volatility

            <br><br>

            ↓

            <br><br>

            <b>3. Walk-Forward XGBoost</b><br>
            Volatility regime prediction

            <br><br>

            ↓

            <br><br>

            <b>4. Confidence Filter</b><br>
            Minimum research threshold = 0.85

            <br><br>

            ↓

            <br><br>

            <b>5. Signal Engine</b><br>
            Structure-based BUY / SELL logic

            <br><br>

            ↓

            <br><br>

            <b>6. Risk Management</b><br>
            1% risk per trade · 2× maximum leverage

            <br><br>

            ↓

            <br><br>

            <b>7. Realistic Execution</b><br>
            0.10% fee per side · 0.02% slippage

        </div>
        """
    )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Holdout Interpretation
            </div>

        </div>
        """
    )

    st.warning(
        """
        The holdout contains only 22 trades. The result is encouraging,
        but the sample is too small to claim robust long-term profitability.
        In particular, the final two BUY trades contributed a large portion
        of the total holdout profit.
        """
    )

    st.success(
        """
        The strategy still outperformed BTC buy-and-hold over the same
        holdout period: approximately +6.14% strategy return versus
        -15.67% BTC buy-and-hold, with substantially lower drawdown.
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    render_html(
        """
        <div class="section">

            <div class="section-title">
                Model Performance
            </div>

            <div class="section-subtitle">
                True out-of-sample walk-forward volatility classification
            </div>

        </div>
        """
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Walk-forward Accuracy",
            "62.40%",
        )

    with m2:

        st.metric(
            "Validation Folds",
            "5",
        )

    with m3:

        st.metric(
            "Prediction Horizon",
            "6H",
        )

    with m4:

        st.metric(
            "Features",
            "22",
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Model Interpretation
            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html(
            """
            <div class="panel">

                <div class="panel-title">
                    Current Prediction
                </div>

                <br>

            </div>
            """
        )

        st.metric(
            "Predicted Regime",
            regime,
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
        )

    with col2:

        render_html(
            """
            <div class="panel">

                <div class="panel-title">
                    Probability Split
                </div>

                <br>

            </div>
            """
        )

        st.write(
            f"CONTRACT: "
            f"{contract_probability * 100:.2f}%"
        )

        st.progress(
            contract_probability
        )

        st.write(
            f"EXPAND: "
            f"{expand_probability * 100:.2f}%"
        )

        st.progress(
            expand_probability
        )


    render_html(
        """
        <div class="section">

            <div class="section-title">
                Model Methodology
            </div>

        </div>
        """
    )

    render_html(
        """
        <div class="panel">

            <b>Model:</b> XGBoost classifier

            <br><br>

            <b>Task:</b> Predict BTC volatility regime

            <br><br>

            <b>Classes:</b> CONTRACT / EXPAND

            <br><br>

            <b>Forecast horizon:</b> 6 hours

            <br><br>

            <b>Validation:</b> True walk-forward out-of-sample testing

            <br><br>

            <b>Current confidence:</b>
            Probability of the model's selected class

            <br><br>

            <b>Important:</b>
            Model accuracy and prediction confidence
            are different measurements.

        </div>
        """
    )


# ============================================================
# GLOBAL MODEL EXPLANATION
# ============================================================

with st.expander(
    "ℹ️  About CryptoSentimentML"
):

    st.markdown(
        """
### System architecture

**Market Data**

BTC/USDT 1-hour OHLCV candles are collected from Binance.

↓

**Feature Engineering**

The same production feature pipeline calculates:

- Returns
- EMA 20 / 50 / 200
- EMA distances
- RSI
- MACD
- Bollinger Bands
- ATR
- Volume metrics
- 24-hour volatility

↓

**XGBoost**

The trained model predicts the next 6-hour volatility regime:

- CONTRACT
- EXPAND

↓

**Confidence**

The highest class probability becomes the current
prediction confidence.

### Important distinction

The volatility model does **not** directly predict:

- BUY
- SELL
- Price direction

Those belong to the separate strategy/signal layer.

### Research configuration

Current locked strategy research configuration:

- Confidence threshold: 0.85
- Risk per trade: 1%
- Maximum leverage: 2×
- Fee: 0.10% per side
- Slippage: 0.02% entry + exit
- BUY + SELL enabled
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

render_html(
    """
    <div class="footer">

        CryptoSentimentML · AI Trading Intelligence Terminal<br>

        BTC/USDT · 1H · XGBoost Volatility Model ·
        Walk-Forward Validation

        <br><br>

        Research and educational software only.
        Not financial advice.

    </div>
    """
)