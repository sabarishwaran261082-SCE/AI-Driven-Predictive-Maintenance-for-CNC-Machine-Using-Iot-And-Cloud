import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/nema17_like_from_cmapss.csv"
MODEL_OUT = "models/rul_base_stepper.pkl"
META_OUT  = "models/rul_base_stepper_meta.json"

FEATURES = ["temperature_C", "vibration_g", "current_A", "voltage_V", "rpm"]
TARGET = "RUL"

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        out[f"{c}__missing"] = out[c].isna().astype(int)
    return out

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Basic checks
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Keep target present; allow missing sensor values
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(float)

    # Optional: clip extreme RUL (helps stability)
    df[TARGET] = df[TARGET].clip(lower=0, upper=200)

    X = df[FEATURES].copy()
    y = df[TARGET].values

    X = add_missing_indicators(X)
    all_features = list(X.columns)

    preprocess = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), all_features)],
        remainder="drop"
    )

    # Strong regressor for tabular data, stable, good with non-linearities
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=700,
        random_state=42
    )

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("\n✅ Base RUL model trained (sensor-only)")
    print(f"MAE : {mae:.3f} cycles")
    print(f"RMSE: {rmse:.3f} cycles")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)

    meta = {
        "features": FEATURES,
        "all_features_after_missing_indicators": all_features,
        "target": TARGET
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved model: {MODEL_OUT}")
    print(f"✅ Saved meta : {META_OUT}")

if __name__ == "__main__":
    main()
