import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# GENERIC BTC/USDT BACKTEST ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_usdt_1h_clean.csv"
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000.0

FEE_RATE = 0.001          # 0.10% per side
SLIPPAGE_RATE = 0.0002    # 0.02% per side

RISK_PER_TRADE = 0.01     # 1%


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# BUY & HOLD
# ============================================================

def buy_and_hold(df):

    start_price = df.iloc[0]["open"]

    end_price = df.iloc[-1]["close"]

    capital = INITIAL_CAPITAL

    quantity = capital / start_price

    final_value = quantity * end_price

    return {
        "initial_capital": capital,
        "final_capital": final_value,
        "return_pct": (
            final_value / capital - 1
        ) * 100
    }


# ============================================================
# STRATEGY BACKTEST
# ============================================================

def run_backtest(
    df,
    long_signal,
    short_signal,
    stop_atr=2.0,
    reward_risk=2.0
):

    capital = INITIAL_CAPITAL

    equity_curve = []

    trades = []

    position = None

    entry_price = None
    stop_price = None
    target_price = None
    quantity = None
    entry_time = None

    # --------------------------------------------------------
    # START AFTER INDICATORS HAVE ENOUGH HISTORY
    # --------------------------------------------------------

    for i in range(1, len(df)):

        row = df.iloc[i]

        previous = df.iloc[i - 1]

        current_time = row["timestamp"]


        # ====================================================
        # MANAGE EXISTING POSITION
        # ====================================================

        if position == "LONG":

            exit_price = None
            exit_reason = None


            # Stop first if both stop and target are hit.
            if row["low"] <= stop_price:

                exit_price = stop_price

                exit_reason = "STOP"


            elif row["high"] >= target_price:

                exit_price = target_price

                exit_reason = "TARGET"


            if exit_price is not None:

                # Apply slippage to exit
                execution_price = (
                    exit_price
                    * (1 - SLIPPAGE_RATE)
                )

                gross_pnl = (
                    execution_price
                    - entry_price
                ) * quantity

                entry_value = (
                    entry_price
                    * quantity
                )

                exit_value = (
                    execution_price
                    * quantity
                )

                fees = (
                    entry_value
                    + exit_value
                ) * FEE_RATE

                net_pnl = (
                    gross_pnl
                    - fees
                )

                capital += net_pnl

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": "LONG",
                    "entry_price": entry_price,
                    "exit_price": execution_price,
                    "quantity": quantity,
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "capital": capital,
                    "reason": exit_reason
                })

                position = None

                entry_price = None
                stop_price = None
                target_price = None
                quantity = None
                entry_time = None


        elif position == "SHORT":

            exit_price = None
            exit_reason = None


            if row["high"] >= stop_price:

                exit_price = stop_price

                exit_reason = "STOP"


            elif row["low"] <= target_price:

                exit_price = target_price

                exit_reason = "TARGET"


            if exit_price is not None:

                execution_price = (
                    exit_price
                    * (1 + SLIPPAGE_RATE)
                )

                gross_pnl = (
                    entry_price
                    - execution_price
                ) * quantity

                entry_value = (
                    entry_price
                    * quantity
                )

                exit_value = (
                    execution_price
                    * quantity
                )

                fees = (
                    entry_value
                    + exit_value
                ) * FEE_RATE

                net_pnl = (
                    gross_pnl
                    - fees
                )

                capital += net_pnl

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": execution_price,
                    "quantity": quantity,
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "capital": capital,
                    "reason": exit_reason
                })

                position = None

                entry_price = None
                stop_price = None
                target_price = None
                quantity = None
                entry_time = None


        # ====================================================
        # RECORD EQUITY
        # ====================================================

        equity_curve.append({
            "timestamp": current_time,
            "equity": capital
        })


        # ====================================================
        # DON'T OPEN ANOTHER POSITION
        # ====================================================

        if position is not None:

            continue


        # ====================================================
        # CHECK ATR
        # ====================================================

        atr = row.get("atr", np.nan)

        if pd.isna(atr) or atr <= 0:

            continue


        # ====================================================
        # SIGNAL IS GENERATED USING PREVIOUS CLOSED CANDLE
        #
        # ENTRY HAPPENS AT CURRENT CANDLE OPEN
        # ====================================================

        previous_long = bool(
            long_signal.iloc[i - 1]
        )

        previous_short = bool(
            short_signal.iloc[i - 1]
        )


        # ====================================================
        # LONG
        # ====================================================

        if previous_long:

            position = "LONG"

            entry_price = (
                row["open"]
                * (1 + SLIPPAGE_RATE)
            )

            stop_price = (
                entry_price
                - stop_atr * atr
            )

            risk_per_unit = (
                entry_price
                - stop_price
            )

            target_price = (
                entry_price
                + reward_risk
                * risk_per_unit
            )

            risk_amount = (
                capital
                * RISK_PER_TRADE
            )

            quantity = (
                risk_amount
                / risk_per_unit
            )

            entry_time = current_time


        # ====================================================
        # SHORT
        # ====================================================

        elif previous_short:

            position = "SHORT"

            entry_price = (
                row["open"]
                * (1 - SLIPPAGE_RATE)
            )

            stop_price = (
                entry_price
                + stop_atr * atr
            )

            risk_per_unit = (
                stop_price
                - entry_price
            )

            target_price = (
                entry_price
                - reward_risk
                * risk_per_unit
            )

            risk_amount = (
                capital
                * RISK_PER_TRADE
            )

            quantity = (
                risk_amount
                / risk_per_unit
            )

            entry_time = current_time


    # ========================================================
    # RESULTS
    # ========================================================

    trades_df = pd.DataFrame(trades)

    equity_df = pd.DataFrame(
        equity_curve
    )


    if len(trades_df) == 0:

        return {
            "final_capital": INITIAL_CAPITAL,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "trades_df": trades_df,
            "equity_df": equity_df
        }


    # ========================================================
    # METRICS
    # ========================================================

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


    gross_profit = wins[
        "net_pnl"
    ].sum()

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
    # MAX DRAWDOWN
    # --------------------------------------------------------

    equity = equity_df["equity"]

    running_max = equity.cummax()

    drawdown = (
        equity
        / running_max
        - 1
    )

    max_drawdown_pct = (
        drawdown.min()
        * 100
    )


    # --------------------------------------------------------
    # SHARPE
    # --------------------------------------------------------

    returns = equity.pct_change()

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()


    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
        ) * np.sqrt(24 * 365)

    else:

        sharpe = 0.0


    # --------------------------------------------------------
    # TOTAL RETURN
    # --------------------------------------------------------

    return_pct = (
        capital
        / INITIAL_CAPITAL
        - 1
    ) * 100


    return {
        "final_capital": capital,
        "return_pct": return_pct,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "trades_df": trades_df,
        "equity_df": equity_df
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BACKTEST ENGINE SELF TEST")
    print("=" * 70)

    df = load_data()

    benchmark = buy_and_hold(df)

    print()
    print("BUY & HOLD")
    print("-" * 70)

    print(
        f"Initial: ${benchmark['initial_capital']:,.2f}"
    )

    print(
        f"Final:   ${benchmark['final_capital']:,.2f}"
    )

    print(
        f"Return:  {benchmark['return_pct']:.2f}%"
    )

    print()
    print("Engine loaded successfully.")
    print("=" * 70)