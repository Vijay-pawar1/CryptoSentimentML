import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/btc_usdt_1h_volatility.csv"
OUTPUT_PATH = "data/processed/volatility_walkforward_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

print("=" * 70)
print("       TRUE OUT-OF-SAMPLE VOLATILITY BACKTEST")
print("=" * 70)

print(f"Dataset rows: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

feature_cols = [
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
    "volatility_24h"
]


# ============================================================
# CLEAN DATA
# ============================================================

X = df[feature_cols].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_mask = X.notnull().all(axis=1)

df = df.loc[valid_mask].reset_index(drop=True)

X = df[feature_cols]

y = df["vol_target"]


print(f"Usable rows: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Classes: {sorted(y.unique())}")


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

n_splits = 5

total_len = len(df)

fold_size = total_len // (n_splits + 1)


results = []


# ============================================================
# WALK-FORWARD TEST
# ============================================================

for fold in range(1, n_splits + 1):

    train_end = fold_size * fold
    test_end = fold_size * (fold + 1)

    train_data = df.iloc[:train_end]
    test_data = df.iloc[train_end:test_end]

    X_train = train_data[feature_cols]
    y_train = train_data["vol_target"]

    X_test = test_data[feature_cols]
    y_test = test_data["vol_target"]


    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)

    y_test_encoded = encoder.transform(y_test)


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )


    model.fit(
        X_train,
        y_train_encoded
    )


    # --------------------------------------------------------
    # Predict unseen data
    # --------------------------------------------------------

    probabilities = model.predict_proba(X_test)

    predicted_encoded = probabilities.argmax(axis=1)

    predicted_labels = encoder.inverse_transform(
        predicted_encoded
    )

    confidence = probabilities.max(axis=1)


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    fold_result = pd.DataFrame({

        "timestamp": test_data["timestamp"].values,

        "close": test_data["close"].values,

        "actual": y_test.values,

        "predicted": predicted_labels,

        "confidence": confidence,

        "correct": (
            predicted_labels == y_test.values
        ),

        "fold": fold

    })


    results.append(fold_result)


    # --------------------------------------------------------
    # Fold statistics
    # --------------------------------------------------------

    accuracy = (
        predicted_labels == y_test.values
    ).mean()


    print()
    print("-" * 70)

    print(f"FOLD {fold}")

    print(
        f"Train: {train_data['timestamp'].min()} "
        f"→ {train_data['timestamp'].max()}"
    )

    print(
        f"Test:  {test_data['timestamp'].min()} "
        f"→ {test_data['timestamp'].max()}"
    )

    print(
        f"Train rows: {len(train_data)}"
    )

    print(
        f"Test rows: {len(test_data)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

results_df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

overall_accuracy = (
    results_df["correct"].mean()
)


print()
print("=" * 70)
print("TRUE OUT-OF-SAMPLE RESULTS")
print("=" * 70)

print()
print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        results_df["actual"],
        results_df["predicted"]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix")
print("(rows = actual, columns = predicted)")
print()

labels = ["CONTRACT", "EXPAND"]

cm = confusion_matrix(
    results_df["actual"],
    results_df["predicted"],
    labels=labels
)

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("OUT-OF-SAMPLE CONFIDENCE ANALYSIS")
print("=" * 70)


thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


for threshold in thresholds:

    subset = results_df[
        results_df["confidence"] >= threshold
    ]


    if len(subset) == 0:
        continue


    accuracy = subset["correct"].mean()

    coverage = (
        len(subset)
        / len(results_df)
    )


    avg_confidence = (
        subset["confidence"].mean()
    )


    print()

    print(
        f"Confidence >= {threshold:.0%}"
    )

    print(
        f"Predictions: {len(subset)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Coverage: {coverage * 100:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{avg_confidence * 100:.2f}%"
    )


# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE CALIBRATION")
print("=" * 70)


bins = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.01
]


labels_conf = [
    "50-55%",
    "55-60%",
    "60-65%",
    "65-70%",
    "70-75%",
    "75-80%",
    "80-85%",
    "85-90%",
    "90-95%",
    "95%+"
]


results_df["confidence_band"] = pd.cut(
    results_df["confidence"],
    bins=bins,
    labels=labels_conf,
    include_lowest=True
)


calibration = (
    results_df
    .groupby(
        "confidence_band",
        observed=False
    )
    .agg(
        predictions=("correct", "count"),
        accuracy=("correct", "mean"),
        average_confidence=("confidence", "mean")
    )
)


calibration["accuracy"] *= 100

calibration["average_confidence"] *= 100


print(
    calibration.round(2)
)


# ============================================================
# PERFORMANCE BY FOLD
# ============================================================

print()
print("=" * 70)
print("PERFORMANCE BY FOLD")
print("=" * 70)


fold_summary = (
    results_df
    .groupby("fold")
    .agg(
        predictions=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean")
    )
)


fold_summary["accuracy"] *= 100

fold_summary["avg_confidence"] *= 100


print(
    fold_summary.round(2)
)


# ============================================================
# PERFORMANCE BY REGIME
# ============================================================

print()
print("=" * 70)
print("PERFORMANCE BY ACTUAL REGIME")
print("=" * 70)


regime_summary = (
    results_df
    .groupby("actual")
    .agg(
        predictions=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean")
    )
)


regime_summary["accuracy"] *= 100

regime_summary["avg_confidence"] *= 100


print(
    regime_summary.round(2)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)

print()
print(
    f"Saved results to:"
)

print(
    OUTPUT_PATH
)