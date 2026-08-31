"""
fetch_ohlcv.py
Fetches historical OHLCV (Open, High, Low, Close, Volume) data
for BTC/USDT from Binance using ccxt, and saves it as a CSV.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime

# ---- CONFIG ----
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SINCE_DATE = "2023-01-01T00:00:00Z"   # start date for historical data
OUTPUT_PATH = "data/raw/btc_usdt_1h.csv"
LIMIT = 1000  # max candles per request (Binance limit)


def fetch_ohlcv():
    exchange = ccxt.binance()
    since = exchange.parse8601(SINCE_DATE)
    all_candles = []

    print(f"Fetching {SYMBOL} {TIMEFRAME} data starting from {SINCE_DATE}...")

    while True:
        candles = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=LIMIT)

        if not candles:
            break

        all_candles += candles
        since = candles[-1][0] + 1  # move to next batch

        last_time = datetime.utcfromtimestamp(candles[-1][0] / 1000)
        print(f"Fetched up to {last_time} | total candles: {len(all_candles)}")

        # Stop once we reach current time
        if len(candles) < LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)  # respect rate limits

    return all_candles


def save_to_csv(candles):
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    candles = fetch_ohlcv()
    save_to_csv(candles)