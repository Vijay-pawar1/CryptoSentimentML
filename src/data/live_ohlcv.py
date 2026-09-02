import ccxt
import pandas as pd


SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 250


def fetch_live_ohlcv():

    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    candles = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT
    )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    return df


if __name__ == "__main__":

    df = fetch_live_ohlcv()

    print("=" * 60)
    print("LIVE BTC/USDT DATA")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Latest candle: {df['timestamp'].iloc[-1]}")
    print(f"BTC Price: ${df['close'].iloc[-1]:,.2f}")

    print()
    print(df.tail(5).to_string(index=False))