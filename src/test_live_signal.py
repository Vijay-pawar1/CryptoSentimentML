import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from data.live_ohlcv import fetch_live_ohlcv
from features.volatility_features import calculate_features
from volatility_predictor import predict_volatility
from signal_engine import generate_signal


# --------------------------------------------------
# MAIN
# --------------------------------------------------

print("=" * 70)
print("        LIVE BTC/USDT TRADING SIGNAL TEST")
print("=" * 70)

# --------------------------------------------------
# 1. FETCH LIVE DATA
# --------------------------------------------------

print("\nFetching live Binance data...")

df = fetch_live_ohlcv()

if df is None or df.empty:
    raise ValueError("Live market data could not be loaded.")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(f"Live candles: {len(df)}")
print(f"Latest candle: {df['timestamp'].iloc[-1]}")
print(
    f"BTC price: ${df['close'].iloc[-1]:,.2f}"
)


# --------------------------------------------------
# 2. CALCULATE FEATURES
# --------------------------------------------------

print("\nCalculating model features...")

df_features = calculate_features(df)

print("Feature calculation complete.")


# --------------------------------------------------
# 3. VOLATILITY MODEL
# --------------------------------------------------

print("\nRunning XGBoost volatility model...")

volatility_result = predict_volatility(
    df_features
)
# --------------------------------------------------
# 3b. DEBUG: INTERMEDIATE SIGNAL STATE
# --------------------------------------------------

from signal_engine import (
    detect_structure,
    detect_trend,
    detect_momentum,
    detect_rsi,
    calculate_atr,
    MIN_TRADE_SCORE,
)

debug_data = df_features.iloc[:-1].copy()

structure_info = detect_structure(debug_data)
trend_state = detect_trend(debug_data)
momentum_state = detect_momentum(debug_data)
rsi_state = detect_rsi(debug_data)
atr_value = float(calculate_atr(debug_data).iloc[-1])

bullish_score = (
    (2 if structure_info["structure"] == "BULLISH" else 0)
    + (2 if trend_state == "BULLISH" else 0)
    + (1 if momentum_state == "BULLISH" else 0)
    + (1 if rsi_state == "BULLISH" else 0)
    + (1 if volatility_result.get("regime") == "EXPAND" else 0)
)

bearish_score = (
    (2 if structure_info["structure"] == "BEARISH" else 0)
    + (2 if trend_state == "BEARISH" else 0)
    + (1 if momentum_state == "BEARISH" else 0)
    + (1 if rsi_state == "BEARISH" else 0)
    + (1 if volatility_result.get("regime") == "EXPAND" else 0)
)

print("\n" + "-" * 70)
print("DEBUG: SIGNAL COMPONENTS")
print("-" * 70)
print(f"Structure:          {structure_info['structure']}")
print(f"Trend:               {trend_state}")
print(f"Momentum:            {momentum_state}")
print(f"RSI state:           {rsi_state}")
print(f"ATR:                 {atr_value:.2f}")
print(f"Bullish score:       {bullish_score} / {MIN_TRADE_SCORE} needed")
print(f"Bearish score:       {bearish_score} / {MIN_TRADE_SCORE} needed")
print("-" * 70)

# --------------------------------------------------
# 4. TRADING SIGNAL
# --------------------------------------------------

print("\nRunning trading signal engine...")

signal = generate_signal(
    df_features,
    volatility_result
)


# --------------------------------------------------
# 5. DISPLAY RESULT
# --------------------------------------------------

print("\n")
print("=" * 70)
print("             FINAL TRADING SIGNAL")
print("=" * 70)

print()

print(f"Signal:             {signal.signal}")

print(
    f"Entry:              "
    f"{'N/A' if signal.entry is None else f'${signal.entry:,.2f}'}"
)

print(
    f"Stop Loss:          "
    f"{'N/A' if signal.stop_loss is None else f'${signal.stop_loss:,.2f}'}"
)

print(
    f"Target:             "
    f"{'N/A' if signal.target is None else f'${signal.target:,.2f}'}"
)

print(
    f"Risk / Reward:      "
    f"{'N/A' if signal.risk_reward is None else f'1 : {signal.risk_reward:.2f}'}"
)

print(
    f"Confidence:         "
    f"{signal.confidence * 100:.2f}%"
)

print(
    f"Setup:              {signal.setup}"
)

print(
    f"Reason:             {signal.reason}"
)

print()
print("-" * 70)

print(
    f"Volatility regime:  "
    f"{volatility_result['regime']}"
)

print(
    f"Volatility confidence: "
    f"{volatility_result['confidence'] * 100:.2f}%"
)

print(
    f"Contract probability: "
    f"{volatility_result['contract_probability'] * 100:.2f}%"
)

print(
    f"Expand probability:   "
    f"{volatility_result['expand_probability'] * 100:.2f}%"
)

print(
    f"Horizon:             "
    f"{volatility_result['prediction_horizon_hours']} hours"
)

print("=" * 70)
print("              TEST COMPLETE")
print("=" * 70)