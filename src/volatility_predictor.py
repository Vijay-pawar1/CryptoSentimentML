import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "volatility_xgb_final.pkl"
FEATURES_PATH = MODEL_DIR / "volatility_xgb_final_features.pkl"
ENCODER_PATH = MODEL_DIR / "volatility_label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "volatility_xgb_metadata.pkl"


# --------------------------------------------------
# LOAD MODEL PACKAGE
# --------------------------------------------------

def load_model_package():

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = joblib.load(METADATA_PATH)

    return model, feature_cols, encoder, metadata


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df, feature_cols):

    X = df[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# --------------------------------------------------
# PREDICT VOLATILITY REGIME
# --------------------------------------------------

def predict_volatility(df):

    model, feature_cols, encoder, metadata = load_model_package()

    X = prepare_features(df, feature_cols)

    valid_rows = X.dropna()

    if valid_rows.empty:
        raise ValueError(
            "No valid feature row available for prediction."
        )

    latest_X = valid_rows.tail(1)

    probabilities = model.predict_proba(latest_X)[0]

    predicted_index = probabilities.argmax()

    regime = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = probabilities[predicted_index]

    class_probabilities = dict(
        zip(
            encoder.classes_,
            probabilities
        )
    )

    # Signal classification
    if confidence < 0.60:
        signal_strength = "NO SIGNAL"
    elif confidence < 0.70:
        signal_strength = "LOW"
    elif confidence < 0.80:
        signal_strength = "MODERATE"
    else:
        signal_strength = "HIGH"

    latest_index = latest_X.index[0]

    return {
        "timestamp": df.loc[latest_index, "timestamp"],
        "regime": regime,
        "confidence": confidence,
        "signal_strength": signal_strength,
        "contract_probability": class_probabilities.get(
            "CONTRACT", 0
        ),
        "expand_probability": class_probabilities.get(
            "EXPAND", 0
        ),
        "prediction_horizon_hours": metadata[
            "prediction_horizon_hours"
        ]
    }