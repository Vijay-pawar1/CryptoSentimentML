import sys
from pathlib import Path

import pandas as pd

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from volatility_predictor import predict_volatility


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "btc_usdt_1h_volatility.csv"
)


df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


result = predict_volatility(df)


print("===================================")
print("      VOLATILITY PREDICTOR TEST")
print("===================================")

print(f"Timestamp: {result['timestamp']}")
print(f"Regime: {result['regime']}")
print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)
print(
    f"Signal strength: "
    f"{result['signal_strength']}"
)
print(
    f"CONTRACT probability: "
    f"{result['contract_probability'] * 100:.2f}%"
)
print(
    f"EXPAND probability: "
    f"{result['expand_probability'] * 100:.2f}%"
)
print(
    f"Horizon: "
    f"{result['prediction_horizon_hours']} hours"
)