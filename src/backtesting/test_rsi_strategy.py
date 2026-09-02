import pandas as pd

from backtest_engine import load_data, run_backtest


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# SETTINGS
# ============================================================

EMA_PERIOD = 200
RSI_PERIOD = 14
ATR_PERIOD = 14


# ============================================================
# EMA 200
# ============================================================

df["ema200"] = (
    df["close"]
    .ewm(
        span=EMA_PERIOD,
        adjust=False
    )
    .mean()
)


# ============================================================
# RSI
# ============================================================

delta = df["close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(
    RSI_PERIOD
).mean()

avg_loss = loss.rolling(
    RSI_PERIOD
).mean()

rs = (
    avg_gain
    / avg_loss.replace(0, pd.NA)
)

df["rsi"] = (
    100
    - (
        100
        / (1 + rs)
    )
)


# ============================================================
# ATR
# ============================================================

previous_close = df["close"].shift(1)

tr1 = (
    df["high"]
    - df["low"]
)

tr2 = (
    df["high"]
    - previous_close
).abs()

tr3 = (
    df["low"]
    - previous_close
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


# ============================================================
# RSI MEAN REVERSION
# ============================================================

# LONG:
# Previous RSI was below 30
# Current RSI recovered to 30+
# Price is above EMA200

long_signal = (
    (df["close"] > df["ema200"]) &
    (df["rsi"].shift(1) < 30) &
    (df["rsi"] >= 30)
)


# SHORT:
# Previous RSI was above 70
# Current RSI fell to 70-
# Price is below EMA200

short_signal = (
    (df["close"] < df["ema200"]) &
    (df["rsi"].shift(1) > 70) &
    (df["rsi"] <= 70)
)


# ============================================================
# SIGNAL CHECK
# ============================================================

print()
print("Signal statistics")
print("-" * 70)

print(
    f"Long signals:  {long_signal.sum()}"
)

print(
    f"Short signals: {short_signal.sum()}"
)

print(
    f"ATR values:    {df['atr'].notna().sum()}"
)


# ============================================================
# RUN BACKTEST
# ============================================================

result = run_backtest(
    df=df,
    long_signal=long_signal,
    short_signal=short_signal,
    stop_atr=2.0,
    reward_risk=2.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("BTC/USDT RSI MEAN REVERSION")
print("=" * 70)

print(
    f"Final capital: "
    f"${result['final_capital']:,.2f}"
)

print(
    f"Return: "
    f"{result['return_pct']:.2f}%"
)

print(
    f"Trades: "
    f"{result['trades']}"
)

print(
    f"Win rate: "
    f"{result['win_rate']:.2f}%"
)

print(
    f"Profit factor: "
    f"{result['profit_factor']:.3f}"
)

print(
    f"Max drawdown: "
    f"{result['max_drawdown_pct']:.2f}%"
)

print(
    f"Sharpe: "
    f"{result['sharpe']:.3f}"
)

print()

if len(result["trades_df"]) > 0:

    print("Exit reasons:")

    print(
        result["trades_df"][
            "reason"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print("Last 5 trades:")

    print(
        result["trades_df"]
        .tail(5)
        .to_string(index=False)
    )

print("=" * 70)