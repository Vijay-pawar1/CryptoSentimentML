import sys
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from features.volatility_features import calculate_features
from volatility_predictor import predict_volatility
from signal_engine import generate_signal

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
WINDOW_SIZE = 250          # same as live: candles per snapshot
TOTAL_CANDLES = 3000       # how far back to test (~125 days on 1h)
STEP = 4                   # slide window by N candles each iteration (speed vs coverage)

# --------------------------------------------------
# FETCH LARGE HISTORICAL DATASET (paginated)
# --------------------------------------------------

def fetch_historical_ohlcv(total_candles):
    exchange = ccxt.binance({"enableRateLimit": True})
    all_candles = []
    limit = 1000
    since = None

    # ccxt fetch_ohlcv returns oldest-first when using 'since' pagination forward
    # Simpler approach: fetch backwards from now
    end_time = exchange.milliseconds()
    ms_per_candle = exchange.parse_timeframe(TIMEFRAME) * 1000

    while len(all_candles) < total_candles:
        since = end_time - (limit * ms_per_candle)
        batch = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=limit)
        if not batch:
            break
        all_candles = batch + all_candles
        end_time = batch[0][0] - ms_per_candle

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df.tail(total_candles).reset_index(drop=True)

# --------------------------------------------------
# RUN BACKTEST
# --------------------------------------------------

print("Fetching historical data...")
full_df = fetch_historical_ohlcv(TOTAL_CANDLES)
print(f"Fetched {len(full_df)} candles: {full_df['timestamp'].iloc[0]} -> {full_df['timestamp'].iloc[-1]}")

results = []

for start in range(0, len(full_df) - WINDOW_SIZE, STEP):
    window = full_df.iloc[start:start + WINDOW_SIZE].copy().reset_index(drop=True)

    try:
        features = calculate_features(window)
        vol_result = predict_volatility(features)
        signal = generate_signal(features, vol_result)

        results.append({
            "timestamp": window["timestamp"].iloc[-1],
            "signal": signal.signal,
            "setup": signal.setup,
            "confidence": signal.confidence,
            "risk_reward": signal.risk_reward,
            "reason": signal.reason,
        })
    except Exception as e:
        results.append({
            "timestamp": window["timestamp"].iloc[-1],
            "signal": "ERROR",
            "setup": "",
            "confidence": None,
            "risk_reward": None,
            "reason": str(e),
        })

results_df = pd.DataFrame(results)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("BACKTEST SUMMARY")
print("=" * 70)
print(f"Total snapshots tested: {len(results_df)}")
print()
print(results_df["signal"].value_counts())
print()

trades = results_df[results_df["signal"].isin(["BUY", "SELL"])]
if not trades.empty:
    print(f"Trades fired: {len(trades)}")
    print(f"Average R:R on fired trades: {trades['risk_reward'].mean():.2f}")
else:
    print("No BUY/SELL trades fired in this backtest window.")

output_path = PROJECT_ROOT / "backtest_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nFull results saved to: {output_path}")