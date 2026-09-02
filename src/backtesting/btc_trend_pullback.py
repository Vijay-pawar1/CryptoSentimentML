import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# BTC/USDT TREND PULLBACK STRATEGY V3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "btc_usdt_1h_clean.csv"
RESULT_FILE = PROJECT_ROOT / "data" / "processed" / "btc_trend_pullback_results.csv"


# ============================================================
# SETTINGS
# ============================================================

EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14

ATR_STOP_MULTIPLIER = 1.5
RISK_REWARD = 2.0

# 0.1% fee per side
FEE_RATE = 0.001

# Risk only 1% of current capital per trade
RISK_PER_TRADE = 0.01

INITIAL_CAPITAL = 10_000.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BTC/USDT TREND PULLBACK STRATEGY V3")
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

df["ema20"] = df["close"].ewm(
    span=EMA_FAST,
    adjust=False
).mean()

df["ema200"] = df["close"].ewm(
    span=EMA_TREND,
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
# PULLBACK CONDITIONS
# ============================================================

# Previous candle tells us whether price was on the
# correct side of EMA20.

previous_close_value = df["close"].shift(1)
previous_ema20 = df["ema20"].shift(1)


# LONG:
# 1. Price is above EMA200.
# 2. Previous candle was below/touching EMA20.
# 3. Current candle closes back above EMA20.
# 4. Current candle is bullish.

df["long_signal"] = (
    (df["close"] > df["ema200"]) &
    (previous_close_value <= previous_ema20) &
    (df["close"] > df["ema20"]) &
    (df["close"] > df["open"])
)


# SHORT:
# 1. Price is below EMA200.
# 2. Previous candle was above/touching EMA20.
# 3. Current candle closes back below EMA20.
# 4. Current candle is bearish.

df["short_signal"] = (
    (df["close"] < df["ema200"]) &
    (previous_close_value >= previous_ema20) &
    (df["close"] < df["ema20"]) &
    (df["close"] < df["open"])
)


# ============================================================
# BACKTEST
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

    risk_amount = capital * RISK_PER_TRADE

    stop_distance = abs(
        entry_price - stop_price
    )

    if stop_distance <= 0:
        position = None
        return

    quantity = risk_amount / stop_distance

    if position == "LONG":
        gross_pnl = (
            exit_price - entry_price
        ) * quantity
    else:
        gross_pnl = (
            entry_price - exit_price
        ) * quantity

    # Entry + exit fees
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

start_index = max(
    EMA_TREND,
    ATR_PERIOD
)

for i in range(start_index, len(df)):

    row = df.iloc[i]

    current_time = row["timestamp"]


    # --------------------------------------------------------
    # MANAGE LONG
    # --------------------------------------------------------

    if position == "LONG":

        # Conservative assumption:
        # If stop AND target occur in the same candle,
        # stop is assumed to happen first.

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


    # --------------------------------------------------------
    # MANAGE SHORT
    # --------------------------------------------------------

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
    # ONLY ONE POSITION AT A TIME
    # --------------------------------------------------------

    if position is not None:
        continue


    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        continue


    # --------------------------------------------------------
    # LONG ENTRY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SHORT ENTRY
    # --------------------------------------------------------

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
# RESULTS
# ============================================================

print()
print("=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final capital:   ${capital:,.2f}")

total_return = (
    capital / INITIAL_CAPITAL - 1
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
        len(wins)
        / len(trades_df)
    ) * 100

    gross_profit = wins["net_pnl"].sum()

    gross_loss = abs(
        losses["net_pnl"].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = np.inf


    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    equity = trades_df[
        "capital_after_trade"
    ]

    running_max = equity.cummax()

    drawdown = (
        (equity - running_max)
        / running_max
    ) * 100

    max_drawdown = drawdown.min()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(f"Winning trades: {len(wins):,}")
    print(f"Losing trades:  {len(losses):,}")
    print(f"Win rate:       {win_rate:.2f}%")
    print(f"Profit factor:  {profit_factor:.2f}")
    print(f"Max drawdown:   {max_drawdown:.2f}%")


    print()
    print("Exit reasons:")

    print(
        trades_df[
            "exit_reason"
        ].value_counts().to_string()
    )


    print()
    print("Long / Short:")

    print(
        trades_df[
            "side"
        ].value_counts().to_string()
    )


    print()
    print("Last 5 trades:")

    print(
        trades_df.tail(5).to_string(
            index=False
        )
    )


    print()
    print("Results saved to:")

    print(RESULT_FILE)


else:

    print("No trades were generated.")


print("=" * 70)