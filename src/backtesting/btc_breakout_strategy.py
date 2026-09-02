import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "btc_usdt_1h_clean.csv"
RESULT_FILE = PROJECT_ROOT / "data" / "processed" / "btc_breakout_results.csv"


# ============================================================
# SETTINGS
# ============================================================

BREAKOUT_LOOKBACK = 20
EMA_PERIOD = 200
ATR_PERIOD = 14

ATR_STOP_MULTIPLIER = 2.0
RISK_REWARD = 2.0

# Binance-style approximate taker fee.
# We will make this conservative for testing.
FEE_RATE = 0.001

INITIAL_CAPITAL = 10_000.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BTC/USDT BREAKOUT STRATEGY V2")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Loaded rows: {len(df):,}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")


# ============================================================
# INDICATORS
# ============================================================

# EMA 200
df["ema200"] = df["close"].ewm(
    span=EMA_PERIOD,
    adjust=False
).mean()


# True Range
previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]
tr2 = (df["high"] - previous_close).abs()
tr3 = (df["low"] - previous_close).abs()

df["true_range"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)


# ATR
df["atr"] = df["true_range"].rolling(
    ATR_PERIOD
).mean()


# ============================================================
# BREAKOUT LEVELS
# ============================================================

# IMPORTANT:
# shift(1) means the current candle cannot use its own high/low.
# This prevents look-ahead bias.

df["previous_high"] = (
    df["high"]
    .rolling(BREAKOUT_LOOKBACK)
    .max()
    .shift(1)
)

df["previous_low"] = (
    df["low"]
    .rolling(BREAKOUT_LOOKBACK)
    .min()
    .shift(1)
)


# ============================================================
# SIGNALS
# ============================================================

df["long_signal"] = (
    (df["close"] > df["previous_high"]) &
    (df["close"] > df["ema200"])
)

df["short_signal"] = (
    (df["close"] < df["previous_low"]) &
    (df["close"] < df["ema200"])
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

capital = INITIAL_CAPITAL

position = None

entry_price = None
stop_price = None
target_price = None
entry_time = None

trades = []


def close_trade(exit_price, exit_time, reason):
    global capital
    global position
    global entry_price
    global stop_price
    global target_price
    global entry_time

    if position is None:
        return

    # Position sizing:
    # Risk 1% of current capital.
    risk_amount = capital * 0.01

    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        position = None
        return

    quantity = risk_amount / stop_distance

    if position == "LONG":
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    # Approximate round-trip trading fee.
    fees = (
        (entry_price * quantity)
        + (exit_price * quantity)
    ) * FEE_RATE

    net_pnl = gross_pnl - fees

    capital += net_pnl

    trades.append({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": position,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "capital_after_trade": capital,
        "exit_reason": reason,
    })

    position = None
    entry_price = None
    stop_price = None
    target_price = None
    entry_time = None


# ============================================================
# MAIN LOOP
# ============================================================

for i in range(max(EMA_PERIOD, BREAKOUT_LOOKBACK, ATR_PERIOD), len(df)):

    row = df.iloc[i]

    current_time = row["timestamp"]

    # --------------------------------------------------------
    # MANAGE EXISTING POSITION
    # --------------------------------------------------------

    if position == "LONG":

        # Conservative assumption:
        # If both stop and target are touched in same candle,
        # assume STOP happened first.
        if row["low"] <= stop_price:

            close_trade(
                stop_price,
                current_time,
                "STOP"
            )
            continue

        if row["high"] >= target_price:

            close_trade(
                target_price,
                current_time,
                "TARGET"
            )
            continue


    elif position == "SHORT":

        if row["high"] >= stop_price:

            close_trade(
                stop_price,
                current_time,
                "STOP"
            )
            continue

        if row["low"] <= target_price:

            close_trade(
                target_price,
                current_time,
                "TARGET"
            )
            continue


    # --------------------------------------------------------
    # OPEN NEW POSITION
    # --------------------------------------------------------

    if position is not None:
        continue

    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        continue


    # LONG
    if row["long_signal"]:

        position = "LONG"

        entry_price = row["close"]

        stop_price = (
            entry_price
            - ATR_STOP_MULTIPLIER * atr
        )

        risk = entry_price - stop_price

        target_price = (
            entry_price
            + RISK_REWARD * risk
        )

        entry_time = current_time


    # SHORT
    elif row["short_signal"]:

        position = "SHORT"

        entry_price = row["close"]

        stop_price = (
            entry_price
            + ATR_STOP_MULTIPLIER * atr
        )

        risk = stop_price - entry_price

        target_price = (
            entry_price
            - RISK_REWARD * risk
        )

        entry_time = current_time


# ============================================================
# SAVE RESULTS
# ============================================================

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:

    trades_df.to_csv(
        RESULT_FILE,
        index=False
    )


# ============================================================
# PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final capital:   ${capital:,.2f}")

total_return = (
    (capital / INITIAL_CAPITAL) - 1
) * 100

print(f"Total return:    {total_return:.2f}%")

print(f"Total trades:    {len(trades_df):,}")


if len(trades_df) > 0:

    wins = trades_df[
        trades_df["net_pnl"] > 0
    ]

    losses = trades_df[
        trades_df["net_pnl"] <= 0
    ]

    win_rate = (
        len(wins) / len(trades_df)
    ) * 100

    gross_profit = wins["net_pnl"].sum()

    gross_loss = abs(
        losses["net_pnl"].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = np.inf

    print(f"Winning trades: {len(wins):,}")
    print(f"Losing trades:  {len(losses):,}")
    print(f"Win rate:       {win_rate:.2f}%")
    print(f"Profit factor:  {profit_factor:.2f}")

    print()
    print("Exit reasons:")
    print(
        trades_df["exit_reason"]
        .value_counts()
        .to_string()
    )

    # Equity curve
    equity = trades_df["capital_after_trade"]

    running_max = equity.cummax()

    drawdown = (
        (equity - running_max)
        / running_max
    ) * 100

    max_drawdown = drawdown.min()

    print()
    print(f"Max drawdown:   {max_drawdown:.2f}%")

    print()
    print("Last 5 trades:")
    print(
        trades_df.tail(5).to_string(
            index=False
        )
    )

    print()
    print(f"Results saved to:")
    print(RESULT_FILE)

else:

    print()
    print("No trades were generated.")

print("=" * 70)