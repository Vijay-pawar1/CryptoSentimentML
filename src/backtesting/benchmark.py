import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_usdt_1h_clean.csv"
)

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

initial_capital = 10_000

first_price = df.iloc[0]["close"]
last_price = df.iloc[-1]["close"]

buy_hold_return = (
    last_price / first_price - 1
) * 100

buy_hold_final = (
    initial_capital
    * last_price
    / first_price
)

print("=" * 70)
print("BTC/USDT BUY & HOLD BENCHMARK")
print("=" * 70)

print(f"Period:")
print(f"{df['timestamp'].min()} → {df['timestamp'].max()}")

print()

print(f"Starting BTC price: ${first_price:,.2f}")
print(f"Ending BTC price:   ${last_price:,.2f}")

print()

print(f"Initial capital: ${initial_capital:,.2f}")
print(f"Final capital:   ${buy_hold_final:,.2f}")
print(f"Total return:    {buy_hold_return:.2f}%")

print("=" * 70)