
import ccxt
import pandas as pd

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 250


def fetch_live_ohlcv():
    """Fetch live BTC/USDT spot candles from OKX."""

    exchange = ccxt.okx({
        "enableRateLimit": True,
        "timeout": 15000,
    })

    try:
        candles = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            limit=LIMIT,
        )

        if not candles:
            raise RuntimeError("OKX returned no OHLCV candles.")

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        # Convert timestamps to UTC.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )

        # Ensure market data is numeric and valid.
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        df = (
            df.dropna(subset=["timestamp"] + numeric_columns)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(df) < 220:
            raise RuntimeError(
                f"Insufficient valid candles from OKX: {len(df)}. "
                "At least 220 candles are required by the dashboard."
            )

        return df

    finally:
        exchange.close()


if __name__ == "__main__":
    df = fetch_live_ohlcv()

    print("=" * 60)
    print("LIVE BTC/USDT DATA — OKX")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Latest candle timestamp (UTC): {df['timestamp'].iloc[-1]}")
    print(f"Latest returned close: ${df['close'].iloc[-1]:,.2f}")
    print()
    print(df.tail(5).to_string(index=False))
