"""
CryptoSentimentML
Advanced Live Trading Signal Engine

Signal states:
    BUY
    SELL
    HOLD
    NO TRADE

Core principles:
    - Never force a trade
    - BUY/SELL requires R:R > 2.5
    - Structure + trend + momentum must align
    - Volatility model is used as a filter
    - Closed candles are used for market structure
    - Current/live price is used as entry
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MIN_RR = 2.5

MIN_VOLATILITY_CONFIDENCE = 0.60

SWING_LOOKBACK = 50

ATR_PERIOD = 14

ATR_STOP_MULTIPLIER = 2.2

MIN_TREND_SCORE = 3

MIN_TRADE_SCORE = 5


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class TradeSignal:

    signal: str

    entry: Optional[float]

    stop_loss: Optional[float]

    target: Optional[float]

    risk_reward: Optional[float]

    confidence: float

    setup: str

    reason: str


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df):

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) < 50:

        raise ValueError(
            "At least 50 candles are required."
        )


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=ATR_PERIOD):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(
        period
    ).mean()


# ============================================================
# SWING DETECTION
# ============================================================

def find_swing_highs(df):

    highs = df["high"]

    swing_high = (
        (highs > highs.shift(1))
        &
        (highs >= highs.shift(-1))
    )

    return df.loc[swing_high].copy()


def find_swing_lows(df):

    lows = df["low"]

    swing_low = (
        (lows < lows.shift(1))
        &
        (lows <= lows.shift(-1))
    )

    return df.loc[swing_low].copy()


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure(df):

    swing_highs = find_swing_highs(df)

    swing_lows = find_swing_lows(df)

    if (
        len(swing_highs) < 2
        or
        len(swing_lows) < 2
    ):

        return {
            "structure": "UNKNOWN",
            "last_swing_high": None,
            "last_swing_low": None
        }

    previous_high = float(
        swing_highs["high"].iloc[-2]
    )

    latest_high = float(
        swing_highs["high"].iloc[-1]
    )

    previous_low = float(
        swing_lows["low"].iloc[-2]
    )

    latest_low = float(
        swing_lows["low"].iloc[-1]
    )

    if (
        latest_high > previous_high
        and
        latest_low > previous_low
    ):

        structure = "BULLISH"

    elif (
        latest_high < previous_high
        and
        latest_low < previous_low
    ):

        structure = "BEARISH"

    else:

        structure = "RANGE"

    return {

        "structure": structure,

        "last_swing_high": latest_high,

        "last_swing_low": latest_low
    }


# ============================================================
# TREND
# ============================================================

def detect_trend(df):

    close = df["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False
    ).mean()

    price = float(close.iloc[-1])

    e20 = float(ema20.iloc[-1])

    e50 = float(ema50.iloc[-1])

    e200 = float(ema200.iloc[-1])

    if (
        price > e20
        and
        e20 > e50
        and
        e50 > e200
    ):

        return "BULLISH"

    if (
        price < e20
        and
        e20 < e50
        and
        e50 < e200
    ):

        return "BEARISH"

    return "RANGE"


# ============================================================
# MOMENTUM
# ============================================================

def detect_momentum(df):

    close = df["close"]

    returns = close.pct_change()

    recent_return = float(
        returns.tail(5).mean()
    )

    if recent_return > 0.0005:

        return "BULLISH"

    if recent_return < -0.0005:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# RSI MOMENTUM
# ============================================================

def detect_rsi(df):

    close = df["close"]

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    value = float(rsi.iloc[-1])

    if value >= 55:

        return "BULLISH"

    if value <= 45:

        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# SUPPORT
# ============================================================

def find_support(df):

    data = df.tail(
        SWING_LOOKBACK
    )

    swing_lows = find_swing_lows(
        data
    )

    if swing_lows.empty:

        return float(
            data["low"].min()
        )

    return float(
        swing_lows["low"].iloc[-1]
    )


# ============================================================
# RESISTANCE
# ============================================================

def find_resistance(df):

    data = df.tail(
        SWING_LOOKBACK
    )

    swing_highs = find_swing_highs(
        data
    )

    if swing_highs.empty:

        return float(
            data["high"].max()
        )

    return float(
        swing_highs["high"].iloc[-1]
    )


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_rr(
    signal,
    entry,
    stop_loss,
    target
):

    if signal == "BUY":

        risk = entry - stop_loss

        reward = target - entry

    elif signal == "SELL":

        risk = stop_loss - entry

        reward = entry - target

    else:

        return 0.0

    if risk <= 0:

        return 0.0

    if reward <= 0:

        return 0.0

    return reward / risk


# ============================================================
# NO TRADE
# ============================================================

def no_trade(
    confidence,
    setup,
    reason
):

    return TradeSignal(

        signal="NO TRADE",

        entry=None,

        stop_loss=None,

        target=None,

        risk_reward=None,

        confidence=confidence,

        setup=setup,

        reason=reason
    )


# ============================================================
# HOLD
# ============================================================

def hold_signal(
    confidence,
    setup,
    reason
):

    return TradeSignal(

        signal="HOLD",

        entry=None,

        stop_loss=None,

        target=None,

        risk_reward=None,

        confidence=confidence,

        setup=setup,

        reason=reason
    )


# ============================================================
# MAIN SIGNAL ENGINE
# ============================================================

def generate_signal(
    df,
    volatility_result
):

    validate_dataframe(df)

    # --------------------------------------------------------
    # Use only CLOSED candles for analysis
    # --------------------------------------------------------

    if len(df) < 3:

        return no_trade(
            0.0,
            "INSUFFICIENT DATA",
            "Not enough candles."
        )

    data = df.iloc[:-1].copy()

    if len(data) < 50:

        return no_trade(
            0.0,
            "INSUFFICIENT DATA",
            "Not enough closed candles."
        )

    # --------------------------------------------------------
    # Volatility model
    # --------------------------------------------------------

    volatility_regime = volatility_result.get(
        "regime",
        "UNKNOWN"
    )

    volatility_confidence = float(
        volatility_result.get(
            "confidence",
            0.0
        )
    )

    # --------------------------------------------------------
    # Market analysis
    # --------------------------------------------------------

    structure_data = detect_structure(
        data
    )

    structure = structure_data[
        "structure"
    ]

    trend = detect_trend(
        data
    )

    momentum = detect_momentum(
        data
    )

    rsi_state = detect_rsi(
        data
    )

    atr_series = calculate_atr(
        data
    )

    atr = float(
        atr_series.iloc[-1]
    )

    # --------------------------------------------------------
    # LIVE ENTRY
    # --------------------------------------------------------

    entry = float(
        df["close"].iloc[-1]
    )

    # --------------------------------------------------------
    # ATR validation
    # --------------------------------------------------------

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):

        return no_trade(
            volatility_confidence,
            f"{structure} / {trend}",
            "ATR is invalid."
        )

    # --------------------------------------------------------
    # Volatility confidence
    # --------------------------------------------------------

    if (
        volatility_confidence
        < MIN_VOLATILITY_CONFIDENCE
    ):

        return no_trade(
            volatility_confidence,
            f"{structure} / {trend}",
            (
                "Volatility model confidence "
                "is below 60%."
            )
        )

    # ========================================================
    # BULLISH SCORE
    # ========================================================

    bullish_score = 0

    if structure == "BULLISH":

        bullish_score += 2

    if trend == "BULLISH":

        bullish_score += 2

    if momentum == "BULLISH":

        bullish_score += 1

    if rsi_state == "BULLISH":

        bullish_score += 1

    if volatility_regime == "EXPAND":

        bullish_score += 1

    # ========================================================
    # BEARISH SCORE
    # ========================================================

    bearish_score = 0

    if structure == "BEARISH":

        bearish_score += 2

    if trend == "BEARISH":

        bearish_score += 2

    if momentum == "BEARISH":

        bearish_score += 1

    if rsi_state == "BEARISH":

        bearish_score += 1

    if volatility_regime == "EXPAND":

        bearish_score += 1

    # ========================================================
    # BUY SETUP
    # ========================================================

    if (
        bullish_score >= MIN_TRADE_SCORE
        and
        trend == "BULLISH"
        and
        structure == "BULLISH"
    ):

        support = find_support(
            data
        )

        resistance = find_resistance(
            data
        )

        atr_stop = (
            entry
            -
            ATR_STOP_MULTIPLIER * atr
        )

        stop_loss = min(
            support,
            atr_stop
        )

        target = resistance

        # ----------------------------------------------------
        # Target must actually be above entry
        # ----------------------------------------------------

        if target <= entry:

            target = entry + (
                3.0 * (
                    entry - stop_loss
                )
            )

        rr = calculate_rr(
            "BUY",
            entry,
            stop_loss,
            target
        )

        # ----------------------------------------------------
        # Strict R:R requirement
        # ----------------------------------------------------

        if rr <= MIN_RR:

            return no_trade(
                volatility_confidence,
                "BULLISH SETUP",
                (
                    "BUY setup detected, but "
                    f"R:R is only {rr:.2f}. "
                    "Required R:R is greater than 2.5."
                )
            )

        confidence = (
            0.45
            +
            (bullish_score * 0.06)
            +
            (volatility_confidence * 0.20)
        )

        confidence = min(
            confidence,
            0.95
        )

        return TradeSignal(

            signal="BUY",

            entry=entry,

            stop_loss=stop_loss,

            target=target,

            risk_reward=rr,

            confidence=confidence,

            setup="BULLISH STRUCTURE",

            reason=(
                "Bullish structure + bullish trend + "
                "momentum confirmation + valid risk/reward."
            )
        )

    # ========================================================
    # SELL SETUP
    # ========================================================

    if (
        bearish_score >= MIN_TRADE_SCORE
        and
        trend == "BEARISH"
        and
        structure == "BEARISH"
    ):

        resistance = find_resistance(
            data
        )

        support = find_support(
            data
        )

        atr_stop = (
            entry
            +
            ATR_STOP_MULTIPLIER * atr
        )

        stop_loss = max(
            resistance,
            atr_stop
        )

        target = support

        # ----------------------------------------------------
        # Target must actually be below entry
        # ----------------------------------------------------

        if target >= entry:

            target = entry - (
                3.0 * (
                    stop_loss - entry
                )
            )

        rr = calculate_rr(
            "SELL",
            entry,
            stop_loss,
            target
        )

        # ----------------------------------------------------
        # Strict R:R requirement
        # ----------------------------------------------------

        if rr <= MIN_RR:

            return no_trade(
                volatility_confidence,
                "BEARISH SETUP",
                (
                    "SELL setup detected, but "
                    f"R:R is only {rr:.2f}. "
                    "Required R:R is greater than 2.5."
                )
            )

        confidence = (
            0.45
            +
            (bearish_score * 0.06)
            +
            (volatility_confidence * 0.20)
        )

        confidence = min(
            confidence,
            0.95
        )

        return TradeSignal(

            signal="SELL",

            entry=entry,

            stop_loss=stop_loss,

            target=target,

            risk_reward=rr,

            confidence=confidence,

            setup="BEARISH STRUCTURE",

            reason=(
                "Bearish structure + bearish trend + "
                "momentum confirmation + valid risk/reward."
            )
        )

    # ========================================================
    # HOLD
    # ========================================================

    if (
        structure == "BULLISH"
        and
        trend == "BULLISH"
    ):

        return hold_signal(

            volatility_confidence,

            "BULLISH STRUCTURE",

            (
                "Bullish market structure and trend exist, "
                "but complete trade confirmation is missing."
            )
        )

    if (
        structure == "BEARISH"
        and
        trend == "BEARISH"
    ):

        return hold_signal(

            volatility_confidence,

            "BEARISH STRUCTURE",

            (
                "Bearish market structure and trend exist, "
                "but complete trade confirmation is missing."
            )
        )

    # ========================================================
    # RANGE / NO TRADE
    # ========================================================

    return no_trade(

        volatility_confidence,

        f"{structure} / {trend}",

        (
            "Market structure, trend and momentum "
            "are not sufficiently aligned for a "
            "high-quality trade."
        )
    )