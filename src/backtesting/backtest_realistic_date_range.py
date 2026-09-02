import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORTS
# ============================================================

from features.volatility_features import calculate_features
from signal_engine import generate_signal


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_usdt_1h_clean.csv"
)

WALK_FORWARD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "volatility_walkforward_results.csv"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_signal_realistic_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000.0

# Risk 1% of current capital per trade
RISK_PER_TRADE = 0.01

# 0.1% fee per side
FEE_RATE = 0.001

# 0.02% slippage
SLIPPAGE_RATE = 0.0002

# Maximum futures leverage
MAX_LEVERAGE = 2.0

# Minimum historical bars required before signals
MIN_BARS = 250

# Confidence threshold
MIN_CONFIDENCE = (
    float(sys.argv[1])
    if len(sys.argv) > 1
    else 0.80
)

# Optional start date
START_DATE = (
    pd.Timestamp(sys.argv[2])
    if len(sys.argv) > 2
    else None
)

# Optional end date
END_DATE = (
    pd.Timestamp(sys.argv[3])
    if len(sys.argv) > 3
    else None
)


# ============================================================
# LOAD BTC DATA
# ============================================================

print("=" * 70)
print("BTC/USDT REALISTIC EXECUTION BACKTEST")
print("=" * 70)

print("\nLoading BTC/USDT data...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(
    f"Rows:  {len(df):,}"
)

print(
    f"Start: {df['timestamp'].iloc[0]}"
)

print(
    f"End:   {df['timestamp'].iloc[-1]}"
)


# ============================================================
# CALCULATE FEATURES
# ============================================================

print("\nCalculating features...")

df_features = calculate_features(
    df.copy()
)

df_features["timestamp"] = pd.to_datetime(
    df_features["timestamp"]
)

print(
    f"Feature rows: {len(df_features):,}"
)


# ============================================================
# LOAD TRUE WALK-FORWARD PREDICTIONS
# ============================================================

print(
    "\nLoading walk-forward volatility predictions..."
)

if not WALK_FORWARD_PATH.exists():

    raise FileNotFoundError(
        f"\nWalk-forward results not found:\n"
        f"{WALK_FORWARD_PATH}"
    )


wf = pd.read_csv(
    WALK_FORWARD_PATH
)

wf["timestamp"] = pd.to_datetime(
    wf["timestamp"]
)

wf = (
    wf
    .sort_values("timestamp")
    .reset_index(drop=True)
)


required_columns = [
    "timestamp",
    "predicted",
    "confidence",
]


missing = [
    column
    for column in required_columns
    if column not in wf.columns
]


if missing:

    raise ValueError(
        f"Missing columns in walk-forward results: {missing}"
    )


print(
    f"Walk-forward rows: {len(wf):,}"
)

print(
    f"Walk-forward start: "
    f"{wf['timestamp'].iloc[0]}"
)

print(
    f"Walk-forward end:   "
    f"{wf['timestamp'].iloc[-1]}"
)


# ============================================================
# ALIGN WALK-FORWARD PREDICTIONS TO MARKET DATA
# ============================================================

print(
    "\nAligning predictions with BTC candles..."
)


wf_lookup = (
    wf[
        [
            "timestamp",
            "predicted",
            "confidence",
        ]
    ]
    .drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )
    .set_index("timestamp")
)


df_features["vol_regime"] = (
    df_features["timestamp"]
    .map(wf_lookup["predicted"])
)


df_features["vol_confidence"] = (
    df_features["timestamp"]
    .map(wf_lookup["confidence"])
)


matched = (
    df_features["vol_regime"]
    .notna()
    .sum()
)


print(
    f"Matched candles: {matched:,}"
)


print(
    f"Unmatched candles: "
    f"{len(df_features) - matched:,}"
)


# ============================================================
# APPLY BACKTEST DATE RANGE
# ============================================================

if START_DATE is not None or END_DATE is not None:

    mask = pd.Series(
        True,
        index=df_features.index
    )

    if START_DATE is not None:

        mask &= (
            df_features["timestamp"]
            >= START_DATE
        )

    if END_DATE is not None:

        mask &= (
            df_features["timestamp"]
            <= END_DATE
        )

    df_features = (
        df_features
        .loc[mask]
        .reset_index(drop=True)
    )


print(
    f"Backtest rows: {len(df_features):,}"
)


if len(df_features):

    print(
        f"Backtest start: "
        f"{df_features['timestamp'].iloc[0]}"
    )

    print(
        f"Backtest end:   "
        f"{df_features['timestamp'].iloc[-1]}"
    )


# ============================================================
# BACKTEST STATE
# ============================================================

capital = INITIAL_CAPITAL

trades = []

position = None


# ============================================================
# HELPER: EXIT POSITION
# ============================================================

def execute_exit(
    position,
    exit_time,
    exit_price,
    reason
):

    global capital

    side = position["side"]

    entry_price = position["entry_price"]

    quantity = position["quantity"]


    # --------------------------------------------------------
    # APPLY EXIT SLIPPAGE
    # --------------------------------------------------------

    if side == "BUY":

        # Selling a BUY position
        # receives a slightly worse price.

        execution_exit_price = (
            exit_price
            * (1 - SLIPPAGE_RATE)
        )

    else:

        # Buying back a SELL position
        # pays a slightly worse price.

        execution_exit_price = (
            exit_price
            * (1 + SLIPPAGE_RATE)
        )


    # --------------------------------------------------------
    # GROSS PNL
    # --------------------------------------------------------

    if side == "BUY":

        gross_pnl = (
            execution_exit_price
            - entry_price
        ) * quantity

    else:

        gross_pnl = (
            entry_price
            - execution_exit_price
        ) * quantity


    # --------------------------------------------------------
    # TRADE VALUES
    # --------------------------------------------------------

    entry_value = (
        entry_price
        * quantity
    )

    exit_value = (
        execution_exit_price
        * quantity
    )


    # --------------------------------------------------------
    # FEES
    # --------------------------------------------------------

    fees = (
        entry_value
        + exit_value
    ) * FEE_RATE


    # --------------------------------------------------------
    # NET PNL
    # --------------------------------------------------------

    net_pnl = (
        gross_pnl
        - fees
    )


    # --------------------------------------------------------
    # UPDATE CAPITAL
    # --------------------------------------------------------

    capital += net_pnl


    # --------------------------------------------------------
    # SAVE TRADE
    # --------------------------------------------------------

    trades.append(
        {
            "entry_time":
                position["entry_time"],

            "exit_time":
                exit_time,

            "side":
                side,

            "entry_price":
                entry_price,

            "exit_price":
                execution_exit_price,

            "stop_loss":
                position["stop_loss"],

            "target":
                position["target"],

            "quantity":
                quantity,

            "notional":
                position["notional"],

            "effective_leverage":
                position["effective_leverage"],

            "risk_amount":
                position["risk_amount"],

            "gross_pnl":
                gross_pnl,

            "fees":
                fees,

            "net_pnl":
                net_pnl,

            "capital":
                capital,

            "reason":
                reason
        }
    )


# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

print(
    "\nRunning realistic execution backtest..."
)

for i in range(
    MIN_BARS,
    len(df_features) - 1
):

    row = df_features.iloc[i]

    next_row = df_features.iloc[i + 1]

    current_time = row["timestamp"]


    # --------------------------------------------------------
    # MANAGE EXISTING POSITION
    # --------------------------------------------------------

    if position is not None:

        high = float(
            next_row["high"]
        )

        low = float(
            next_row["low"]
        )

        stop = position[
            "stop_loss"
        ]

        target = position[
            "target"
        ]

        side = position[
            "side"
        ]


        # ----------------------------------------------------
        # BUY POSITION
        # ----------------------------------------------------

        if side == "BUY":

            if low <= stop:

                execute_exit(
                    position,
                    next_row["timestamp"],
                    stop,
                    "STOP"
                )

                position = None

            elif high >= target:

                execute_exit(
                    position,
                    next_row["timestamp"],
                    target,
                    "TARGET"
                )

                position = None


        # ----------------------------------------------------
        # SELL POSITION
        # ----------------------------------------------------

        else:

            if high >= stop:

                execute_exit(
                    position,
                    next_row["timestamp"],
                    stop,
                    "STOP"
                )

                position = None

            elif low <= target:

                execute_exit(
                    position,
                    next_row["timestamp"],
                    target,
                    "TARGET"
                )

                position = None


        # ----------------------------------------------------
        # ONE POSITION AT A TIME
        # ----------------------------------------------------

        # Never open another position during
        # the same iteration after an exit.

        continue


    # --------------------------------------------------------
    # WALK-FORWARD VOLATILITY PREDICTION
    # --------------------------------------------------------

    regime = row["vol_regime"]

    vol_confidence = row[
        "vol_confidence"
    ]


    if pd.isna(regime):

        continue


    if pd.isna(vol_confidence):

        continue


    vol_confidence = float(
        vol_confidence
    )


    # --------------------------------------------------------
    # CONFIDENCE FILTER
    # --------------------------------------------------------

    if vol_confidence < MIN_CONFIDENCE:

        continue


    # --------------------------------------------------------
    # BUILD VOLATILITY RESULT
    # --------------------------------------------------------

    volatility_result = {

        "timestamp":
            current_time,

        "regime":
            str(regime),

        "confidence":
            vol_confidence,

        "signal_strength":
            (
                "HIGH"
                if vol_confidence >= 0.80
                else
                "MODERATE"
                if vol_confidence >= 0.70
                else
                "LOW"
            ),

        "contract_probability":
            np.nan,

        "expand_probability":
            np.nan,

        "prediction_horizon_hours":
            6
    }


    # --------------------------------------------------------
    # GENERATE SIGNAL
    # --------------------------------------------------------

    historical_data = (
        df_features
        .iloc[: i + 1]
        .copy()
    )


    signal = generate_signal(
        historical_data,
        volatility_result
    )


    # --------------------------------------------------------
    # ONLY TRADE BUY / SELL
    # --------------------------------------------------------

    if signal.signal not in (
        "BUY",
        "SELL"
    ):

        continue


    # --------------------------------------------------------
    # VALIDATE SIGNAL
    # --------------------------------------------------------

    if (
        signal.entry is None
        or
        signal.stop_loss is None
        or
        signal.target is None
    ):

        continue


    # --------------------------------------------------------
    # ENTRY WITH SLIPPAGE
    # --------------------------------------------------------

    raw_entry = float(
        signal.entry
    )


    if signal.signal == "BUY":

        entry_price = (
            raw_entry
            * (1 + SLIPPAGE_RATE)
        )

    else:

        entry_price = (
            raw_entry
            * (1 - SLIPPAGE_RATE)
        )


    stop_loss = float(
        signal.stop_loss
    )

    target = float(
        signal.target
    )


    # --------------------------------------------------------
    # RISK PER UNIT
    # --------------------------------------------------------

    risk_per_unit = abs(
        entry_price
        - stop_loss
    )


    if (
        not np.isfinite(
            risk_per_unit
        )
        or
        risk_per_unit <= 0
    ):

        continue


    # ========================================================
    # POSITION SIZING
    # ========================================================
    #
    # 1. Start with 1% account-risk sizing.
    #
    # 2. Apply maximum 2x leverage cap.
    #
    # 3. Use the smaller of the two quantities.
    #
    # ========================================================

    risk_amount = (
        capital
        * RISK_PER_TRADE
    )


    # Quantity required to risk 1% of capital
    risk_based_quantity = (
        risk_amount
        / risk_per_unit
    )


    # Maximum position notional at 2x leverage
    max_notional = (
        capital
        * MAX_LEVERAGE
    )


    # Maximum quantity allowed by leverage
    max_quantity = (
        max_notional
        / entry_price
    )


    # Final quantity
    quantity = min(
        risk_based_quantity,
        max_quantity
    )


    if quantity <= 0:

        continue


    # --------------------------------------------------------
    # ACTUAL POSITION INFORMATION
    # --------------------------------------------------------

    notional = (
        entry_price
        * quantity
    )


    effective_leverage = (
        notional
        / capital
    )


    actual_risk_amount = (
        risk_per_unit
        * quantity
    )


    # --------------------------------------------------------
    # OPEN POSITION
    # --------------------------------------------------------

    position = {

        "entry_time":
            current_time,

        "side":
            signal.signal,

        "entry_price":
            entry_price,

        "stop_loss":
            stop_loss,

        "target":
            target,

        "quantity":
            quantity,

        "notional":
            notional,

        "effective_leverage":
            effective_leverage,

        "risk_amount":
            actual_risk_amount
    }


# ============================================================
# CLOSE OPEN POSITION
# ============================================================

if position is not None:

    last_row = df_features.iloc[-1]

    exit_price = float(
        last_row["close"]
    )

    execute_exit(
        position,
        last_row["timestamp"],
        exit_price,
        "END"
    )


# ============================================================
# RESULTS
# ============================================================

trades_df = pd.DataFrame(
    trades
)


if trades_df.empty:

    print(
        "\nNO TRADES GENERATED."
    )

    print("=" * 70)

    raise SystemExit


final_capital = capital


return_pct = (
    (
        final_capital
        / INITIAL_CAPITAL
    )
    - 1
) * 100


# ============================================================
# TRADE STATISTICS
# ============================================================

wins = (
    trades_df["net_pnl"] > 0
)

losses = (
    trades_df["net_pnl"] <= 0
)


winning_trades = int(
    wins.sum()
)

losing_trades = int(
    losses.sum()
)


win_rate = (
    winning_trades
    / len(trades_df)
) * 100


gross_profit = (
    trades_df.loc[
        wins,
        "net_pnl"
    ].sum()
)


gross_loss = abs(
    trades_df.loc[
        losses,
        "net_pnl"
    ].sum()
)


profit_factor = (
    gross_profit
    / gross_loss
    if gross_loss > 0
    else np.inf
)


# ============================================================
# MAX DRAWDOWN
# ============================================================

equity = (
    trades_df["capital"]
)

peak = (
    equity.cummax()
)

drawdown = (
    equity - peak
) / peak


max_drawdown_pct = (
    drawdown.min()
    * 100
)


# ============================================================
# LEVERAGE STATISTICS
# ============================================================

max_actual_leverage = (
    trades_df[
        "effective_leverage"
    ].max()
)

average_leverage = (
    trades_df[
        "effective_leverage"
    ].mean()
)

capped_trades = (
    trades_df[
        "effective_leverage"
    ]
    >= MAX_LEVERAGE * 0.999999
).sum()


# ============================================================
# SAVE RESULTS
# ============================================================

trades_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print("=" * 70)

print(
    "REALISTIC EXECUTION BACKTEST RESULTS"
)

print("=" * 70)


print(
    f"Initial capital: "
    f"${INITIAL_CAPITAL:,.2f}"
)


print(
    f"Final capital:   "
    f"${final_capital:,.2f}"
)


print(
    f"Total return:    "
    f"{return_pct:.2f}%"
)


print(
    f"Total trades:    "
    f"{len(trades_df)}"
)


print(
    f"Winning trades:  "
    f"{winning_trades}"
)


print(
    f"Losing trades:   "
    f"{losing_trades}"
)


print(
    f"Win rate:        "
    f"{win_rate:.2f}%"
)


print(
    f"Profit factor:   "
    f"{profit_factor:.3f}"
)


print(
    f"Max drawdown:    "
    f"{max_drawdown_pct:.2f}%"
)


print()

print(
    "Leverage:"
)


print(
    f"Maximum allowed: "
    f"{MAX_LEVERAGE:.2f}x"
)


print(
    f"Maximum actual:  "
    f"{max_actual_leverage:.2f}x"
)


print(
    f"Average actual:  "
    f"{average_leverage:.2f}x"
)


print(
    f"Trades at cap:   "
    f"{capped_trades}"
)


print()

print(
    "Exit reasons:"
)


print(
    trades_df[
        "reason"
    ]
    .value_counts()
    .to_string()
)


print()

print(
    "Long / Short:"
)


print(
    trades_df[
        "side"
    ]
    .value_counts()
    .to_string()
)


print()

print(
    "Last 5 trades:"
)


print(
    trades_df
    .tail(5)
    .to_string(index=False)
)


print()

print(
    "Results saved to:"
)


print(
    RESULT_PATH
)