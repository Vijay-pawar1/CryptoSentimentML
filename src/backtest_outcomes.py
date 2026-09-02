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
WINDOW_SIZE = 250
TOTAL_CANDLES = 3000
STEP = 4
MAX_HOLD_CANDLES = 200      # how far forward to look for TP/SL before giving up

# --------------------------------------------------
# FETCH HISTORICAL DATA
# --------------------------------------------------

def fetch_historical_ohlcv(total_candles):
    exchange = ccxt.binance({"enableRateLimit": True})
    all_candles = []
    limit = 1000
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
# STEP 1: GENERATE RAW SIGNALS (same as before)
# --------------------------------------------------

print("Fetching historical data...")
full_df = fetch_historical_ohlcv(TOTAL_CANDLES)
print(f"Fetched {len(full_df)} candles: {full_df['timestamp'].iloc[0]} -> {full_df['timestamp'].iloc[-1]}")

raw_signals = []

for start in range(0, len(full_df) - WINDOW_SIZE, STEP):
    window = full_df.iloc[start:start + WINDOW_SIZE].copy().reset_index(drop=True)
    end_index = start + WINDOW_SIZE - 1  # index in full_df of the entry candle

    try:
        features = calculate_features(window)
        vol_result = predict_volatility(features)
        signal = generate_signal(features, vol_result)

        if signal.signal in ("BUY", "SELL"):
            raw_signals.append({
                "entry_index": end_index,
                "timestamp": window["timestamp"].iloc[-1],
                "signal": signal.signal,
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "target": signal.target,
                "risk_reward": signal.risk_reward,
                "confidence": signal.confidence,
            })
    except Exception:
        continue

print(f"\nRaw signals (before dedup): {len(raw_signals)}")

# --------------------------------------------------
# STEP 2: DEDUPLICATE OVERLAPPING SIGNALS
# --------------------------------------------------
# Consecutive snapshots with the same direction are very likely the
# same setup detected repeatedly as the window slides. Collapse
# consecutive same-direction signals into a single trade, keeping
# only the FIRST occurrence (earliest entry).

deduped = []
prev_signal = None
prev_index = None
GAP_TOLERANCE = WINDOW_SIZE  # if signals are further apart than this, treat as new trade

for s in raw_signals:
    if (
        prev_signal == s["signal"]
        and prev_index is not None
        and (s["entry_index"] - prev_index) <= GAP_TOLERANCE
    ):
        continue  # same ongoing setup, skip
    deduped.append(s)
    prev_signal = s["signal"]
    prev_index = s["entry_index"]

print(f"Deduplicated trades: {len(deduped)}")

# --------------------------------------------------
# STEP 3: SIMULATE FORWARD OUTCOMES
# --------------------------------------------------

def simulate_outcome(trade, full_df):
    idx = trade["entry_index"]
    future = full_df.iloc[idx + 1: idx + 1 + MAX_HOLD_CANDLES]

    if future.empty:
        return "NO_DATA", None

    for _, candle in future.iterrows():
        if trade["signal"] == "BUY":
            hit_sl = candle["low"] <= trade["stop_loss"]
            hit_tp = candle["high"] >= trade["target"]
        else:  # SELL
            hit_sl = candle["high"] >= trade["stop_loss"]
            hit_tp = candle["low"] <= trade["target"]

        # if both hit in same candle, assume worst case (SL hit first)
        if hit_sl and hit_tp:
            return "LOSS", candle["timestamp"]
        if hit_sl:
            return "LOSS", candle["timestamp"]
        if hit_tp:
            return "WIN", candle["timestamp"]

    return "TIMEOUT", None  # neither hit within MAX_HOLD_CANDLES


outcomes = []
for trade in deduped:
    result, exit_time = simulate_outcome(trade, full_df)
    trade["result"] = result
    trade["exit_time"] = exit_time
    outcomes.append(trade)

outcomes_df = pd.DataFrame(outcomes)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("OUTCOME BACKTEST SUMMARY")
print("=" * 70)

if outcomes_df.empty:
    print("No trades to evaluate.")
else:
    print(outcomes_df["result"].value_counts())
    print()

    decided = outcomes_df[outcomes_df["result"].isin(["WIN", "LOSS"])]
    if not decided.empty:
        win_rate = (decided["result"] == "WIN").mean()
        avg_rr = decided["risk_reward"].mean()
        expectancy = (win_rate * avg_rr) - (1 - win_rate)

        print(f"Decided trades (excl. timeouts): {len(decided)}")
        print(f"Win rate:            {win_rate * 100:.1f}%")
        print(f"Average R:R:         1 : {avg_rr:.2f}")
        print(f"Expectancy (R units): {expectancy:.2f} per trade")
        print()
        print("Expectancy > 0 means the strategy is profitable in R terms")
        print("over this sample, assuming R:R and win rate hold going forward.")
    else:
        print("No decided (WIN/LOSS) trades — all timed out or no data.")

output_path = PROJECT_ROOT / "backtest_outcomes.csv"
outcomes_df.to_csv(output_path, index=False)
print(f"\nFull results saved to: {output_path}")