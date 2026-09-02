import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1h",
    "return_3h",
    "return_6h",
    "return_24h",
    "ema20",
    "ema50",
    "ema200",
    "ema20_dist",
    "ema50_dist",
    "ema200_dist",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr",
    "volume_change",
    "volume_ratio",
    "volatility_24h",
]


def calculate_features(df):
    """
    Calculate the 22 features required by the volatility XGBoost model.

    Input columns required:
        timestamp, open, high, low, close, volume
    """

    df = df.copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------
    # RETURNS
    # --------------------------------------------------

    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)
    df["return_24h"] = df["close"].pct_change(24)

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------
    # DISTANCE FROM EMA
    # --------------------------------------------------

    df["ema20_dist"] = (
        df["close"] - df["ema20"]
    ) / df["ema20"]

    df["ema50_dist"] = (
        df["close"] - df["ema50"]
    ) / df["ema50"]

    df["ema200_dist"] = (
        df["close"] - df["ema200"]
    ) / df["ema200"]

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------
    # MACD
    # --------------------------------------------------

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # --------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------

    df["bb_middle"] = df["close"].rolling(20).mean()

    bb_std = df["close"].rolling(20).std()

    df["bb_upper"] = (
        df["bb_middle"] + 2 * bb_std
    )

    df["bb_lower"] = (
        df["bb_middle"] - 2 * bb_std
    )

    df["bb_width"] = (
        df["bb_upper"] - df["bb_lower"]
    ) / df["bb_middle"]

    # --------------------------------------------------
    # ATR
    # --------------------------------------------------

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # --------------------------------------------------
    # VOLUME FEATURES
    # --------------------------------------------------

    df["volume_change"] = (
        df["volume"].pct_change()
    )

    volume_mean = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] / volume_mean
    )

    # --------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------

    df["volatility_24h"] = (
        df["return_1h"].rolling(24).std()
    )

    # --------------------------------------------------
    # CLEAN
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df