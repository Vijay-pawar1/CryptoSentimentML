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
from features.volatility_features import (
    calculate_features,
    FEATURE_COLUMNS
)
from volatility_predictor import predict_volatility


# --------------------------------------------------
# FETCH LIVE DATA
# --------------------------------------------------

print("=" * 70)
print("        LIVE BTC VOLATILITY PREDICTION TEST")
print("=" * 70)

print("\nFetching live Binance data...")

df = fetch_live_ohlcv()

print(f"Live candles: {len(df)}")
print(f"Latest candle: {df['timestamp'].iloc[-1]}")
print(f"BTC price: ${df['close'].iloc[-1]:,.2f}")


# --------------------------------------------------
# CALCULATE FEATURES
# --------------------------------------------------

print("\nCalculating 22 model features...")

df = calculate_features(df)

print(f"Feature count: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# CHECK FEATURES
# --------------------------------------------------

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print("-", feature)

    sys.exit(1)


latest_features = df[FEATURE_COLUMNS].tail(1)

if latest_features.isnull().any().any():

    print("\nERROR: Latest feature row contains NaN values.")

    print(
        latest_features.isna()
        .sum()
        .loc[lambda x: x > 0]
    )

    sys.exit(1)


print("All 22 features calculated successfully.")


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

print("\nRunning XGBoost prediction...")

result = predict_volatility(df)


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print()
print("=" * 70)
print("             LIVE MODEL RESULT")
print("=" * 70)

print(f"Timestamp:          {result['timestamp']}")
print(f"BTC Price:          ${df['close'].iloc[-1]:,.2f}")
print(f"Regime:             {result['regime']}")
print(
    f"Confidence:         "
    f"{result['confidence'] * 100:.2f}%"
)

print(
    f"Signal strength:    "
    f"{result['signal_strength']}"
)

print(
    f"CONTRACT probability: "
    f"{result['contract_probability'] * 100:.2f}%"
)

print(
    f"EXPAND probability:   "
    f"{result['expand_probability'] * 100:.2f}%"
)

print(
    f"Horizon:             "
    f"{result['prediction_horizon_hours']} hours"
)

print()
print("=" * 70)
print("LIVE PREDICTION TEST COMPLETE")
print("=" * 70)