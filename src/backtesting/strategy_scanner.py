import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# BTC/USDT STRATEGY SCANNER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_usdt_1h_clean.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_scan_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000.0

FEE_RATE = 0.001

RISK_PER_TRADE = 0.01

EMA_FAST = 20
EMA_SLOW = 200

RSI_PERIOD = 14

ATR_PERIOD = 14

LOOKBACKS = [10, 20, 30, 50]

RSI_OVERSOLD = [25, 30, 35]

RSI_OVERBOUGHT = [65, 70, 75]

RISK_REWARD_VALUES = [1.5, 2.0, 2.5, 3.0]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BTC/USDT STRATEGY SCANNER")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Rows: {len(df):,}")
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
    span=EMA_SLOW,
    adjust=False
).mean()


# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(RSI_PERIOD).mean()

avg_loss = loss.rolling(RSI_PERIOD).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["rsi"] = 100 - (
    100 / (1 + rs)
)


# ATR
previous_close = df["close"].shift(1)

tr1 = df["high"] - df["low"]

tr2 = (
    df["high"] - previous_close
).abs()

tr3 = (
    df["low"] - previous_close
).abs()

df["true_range"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df["atr"] = (
    df["true_range"]
    .rolling(ATR_PERIOD)
    .mean()
)


# MACD
ema12 = df["close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = df["close"].ewm(
    span=26,
    adjust=False
).mean()

df["macd"] = ema12 - ema26

df["macd_signal"] = (
    df["macd"]
    .ewm(span=9, adjust=False)
    .mean()
)


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

strategies = []


# ------------------------------------------------------------
# 1. EMA MOMENTUM
# ------------------------------------------------------------

strategies.append({
    "name": "EMA_Momentum",
    "long": (
        (df["close"] > df["ema200"]) &
        (df["ema20"] > df["ema200"]) &
        (df["close"] > df["ema20"])
    ),
    "short": (
        (df["close"] < df["ema200"]) &
        (df["ema20"] < df["ema200"]) &
        (df["close"] < df["ema20"])
    )
})


# ------------------------------------------------------------
# 2. MACD MOMENTUM
# ------------------------------------------------------------

strategies.append({
    "name": "MACD_Momentum",
    "long": (
        (df["macd"] > df["macd_signal"]) &
        (df["close"] > df["ema200"])
    ),
    "short": (
        (df["macd"] < df["macd_signal"]) &
        (df["close"] < df["ema200"])
    )
})


# ------------------------------------------------------------
# 3. RSI REVERSAL
# ------------------------------------------------------------

for oversold in RSI_OVERSOLD:

    for overbought in RSI_OVERBOUGHT:

        strategies.append({
            "name": (
                f"RSI_Reversal_"
                f"{oversold}_{overbought}"
            ),

            "long": (
                (df["rsi"] < oversold) &
                (df["close"] > df["close"].shift(1))
            ),

            "short": (
                (df["rsi"] > overbought) &
                (df["close"] < df["close"].shift(1))
            )
        })


# ------------------------------------------------------------
# 4. BREAKOUT STRATEGIES
# ------------------------------------------------------------

for lookback in LOOKBACKS:

    previous_high = (
        df["high"]
        .rolling(lookback)
        .max()
        .shift(1)
    )

    previous_low = (
        df["low"]
        .rolling(lookback)
        .min()
        .shift(1)
    )

    strategies.append({
        "name": f"Breakout_{lookback}",

        "long": (
            df["close"] > previous_high
        ),

        "short": (
            df["close"] < previous_low
        )
    })


# ------------------------------------------------------------
# 5. EMA + RSI
# ------------------------------------------------------------

for oversold in RSI_OVERSOLD:

    for overbought in RSI_OVERBOUGHT:

        strategies.append({
            "name": (
                f"EMA_RSI_"
                f"{oversold}_{overbought}"
            ),

            "long": (
                (df["close"] > df["ema200"]) &
                (df["rsi"] < oversold) &
                (df["close"] > df["open"])
            ),

            "short": (
                (df["close"] < df["ema200"]) &
                (df["rsi"] > overbought) &
                (df["close"] < df["open"])
            )
        })


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def backtest(
    long_signal,
    short_signal,
    strategy_name,
    rr
):

    capital = INITIAL_CAPITAL

    position = None

    entry_price = None

    stop_price = None

    target_price = None

    trades = []

    start_index = max(
        EMA_SLOW,
        ATR_PERIOD,
        50
    )

    for i in range(
        start_index,
        len(df)
    ):

        row = df.iloc[i]

        # ----------------------------------------------------
        # MANAGE POSITION
        # ----------------------------------------------------

        if position == "LONG":

            if row["low"] <= stop_price:

                exit_price = stop_price

                reason = "STOP"

            elif row["high"] >= target_price:

                exit_price = target_price

                reason = "TARGET"

            else:

                continue

            risk_amount = (
                capital * RISK_PER_TRADE
            )

            stop_distance = abs(
                entry_price - stop_price
            )

            if stop_distance > 0:

                quantity = (
                    risk_amount
                    / stop_distance
                )

                gross_pnl = (
                    exit_price
                    - entry_price
                ) * quantity

                fees = (
                    entry_price * quantity
                    + exit_price * quantity
                ) * FEE_RATE

                net_pnl = (
                    gross_pnl - fees
                )

                capital += net_pnl

                trades.append({
                    "strategy": strategy_name,
                    "rr": rr,
                    "side": "LONG",
                    "entry": entry_price,
                    "exit": exit_price,
                    "net_pnl": net_pnl,
                    "fees": fees,
                    "capital": capital,
                    "reason": reason
                })

            position = None

            continue


        elif position == "SHORT":

            if row["high"] >= stop_price:

                exit_price = stop_price

                reason = "STOP"

            elif row["low"] <= target_price:

                exit_price = target_price

                reason = "TARGET"

            else:

                continue

            risk_amount = (
                capital * RISK_PER_TRADE
            )

            stop_distance = abs(
                entry_price - stop_price
            )

            if stop_distance > 0:

                quantity = (
                    risk_amount
                    / stop_distance
                )

                gross_pnl = (
                    entry_price
                    - exit_price
                ) * quantity

                fees = (
                    entry_price * quantity
                    + exit_price * quantity
                ) * FEE_RATE

                net_pnl = (
                    gross_pnl - fees
                )

                capital += net_pnl

                trades.append({
                    "strategy": strategy_name,
                    "rr": rr,
                    "side": "SHORT",
                    "entry": entry_price,
                    "exit": exit_price,
                    "net_pnl": net_pnl,
                    "fees": fees,
                    "capital": capital,
                    "reason": reason
                })

            position = None

            continue


        # ----------------------------------------------------
        # OPEN LONG
        # ----------------------------------------------------

        if long_signal.iloc[i]:

            atr = row["atr"]

            if pd.isna(atr) or atr <= 0:

                continue

            position = "LONG"

            entry_price = row["close"]

            stop_price = (
                entry_price
                - 1.5 * atr
            )

            risk = (
                entry_price
                - stop_price
            )

            target_price = (
                entry_price
                + rr * risk
            )


        # ----------------------------------------------------
        # OPEN SHORT
        # ----------------------------------------------------

        elif short_signal.iloc[i]:

            atr = row["atr"]

            if pd.isna(atr) or atr <= 0:

                continue

            position = "SHORT"

            entry_price = row["close"]

            stop_price = (
                entry_price
                + 1.5 * atr
            )

            risk = (
                stop_price
                - entry_price
            )

            target_price = (
                entry_price
                - rr * risk
            )


    # ========================================================
    # METRICS
    # ========================================================

    if not trades:

        return None

    trades_df = pd.DataFrame(trades)

    wins = trades_df[
        trades_df["net_pnl"] > 0
    ]

    losses = trades_df[
        trades_df["net_pnl"] <= 0
    ]

    total_trades = len(trades_df)

    win_rate = (
        len(wins)
        / total_trades
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


    equity = trades_df["capital"]

    running_max = equity.cummax()

    drawdown = (
        (equity - running_max)
        / running_max
    ) * 100

    max_drawdown = drawdown.min()

    total_return = (
        capital / INITIAL_CAPITAL - 1
    ) * 100


    return {
        "strategy": strategy_name,
        "rr": rr,
        "trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "final_capital": capital
    }


# ============================================================
# RUN SCAN
# ============================================================

results = []

total_tests = (
    len(strategies)
    * len(RISK_REWARD_VALUES)
)

completed = 0

print()
print(
    f"Testing {total_tests} strategy configurations..."
)

for strategy in strategies:

    for rr in RISK_REWARD_VALUES:

        result = backtest(
            strategy["long"],
            strategy["short"],
            strategy["name"],
            rr
        )

        if result is not None:

            results.append(result)

        completed += 1

        if completed % 10 == 0:

            print(
                f"Progress: "
                f"{completed}/{total_tests}"
            )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

if len(results_df) == 0:

    print("No valid strategies found.")

else:

    results_df = results_df.sort_values(
        [
            "profit_factor",
            "win_rate"
        ],
        ascending=False
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("TOP STRATEGIES")
    print("=" * 70)

    print(
        results_df.head(20).to_string(
            index=False
        )
    )

    print()
    print("Results saved to:")

    print(RESULT_FILE)

    print("=" * 70)