import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/ai4i2020.csv"
MODEL_OUT = "models/failure_model.pkl"
META_OUT  = "models/failure_model_meta.json"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET = "Machine failure"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk fusion config (anomaly -> failure probability boost)
MAX_ANOMALY_BOOST = 0.30   # at most +0.30 added probability
ANOMALY_THRESHOLD_DEFAULT = 0.5  # if you only pass anomaly_score 0..1


# ==============================
# UTILITIES
# ==============================
def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add _missing columns to indicate missing sensor values."""
    out = df.copy()
    for col in FEATURES:
        out[f"{col}__missing"] = out[col].isna().astype(int)
    return out


def tune_threshold_for_recall(y_true, p_pred, min_recall=0.85):
    """
    Choose a probability threshold that tries to achieve at least min_recall
    for the positive class while keeping precision reasonable.
    If not possible, fall back to best F1 threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, p_pred)
    # thresholds has length = len(precision)-1
    best_thr = 0.5
    best_f1 = -1.0

    # Evaluate thresholds
    for i, thr in enumerate(thresholds):
        pr = precision[i+1]
        rc = recall[i+1]
        if pr + rc == 0:
            f1 = 0.0
        else:
            f1 = 2 * pr * rc / (pr + rc)

        # Track best f1
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    # Try to find a threshold satisfying minimum recall
    candidate = None
    candidate_f1 = -1.0
    for i, thr in enumerate(thresholds):
        rc = recall[i+1]
        pr = precision[i+1]
        if rc >= min_recall:
            if pr + rc == 0:
                f1 = 0.0
            else:
                f1 = 2 * pr * rc / (pr + rc)
            if f1 > candidate_f1:
                candidate_f1 = f1
                candidate = float(thr)

    if candidate is not None:
        return candidate, {"mode": "min_recall", "min_recall": float(min_recall), "f1": float(candidate_f1)}
    else:
        return best_thr, {"mode": "best_f1", "f1": float(best_f1)}


def fuse_anomaly_with_failure_prob(p_fail: float, anomaly_detected: bool = False,
                                  anomaly_score: float | None = None,
                                  anomaly_threshold: float = ANOMALY_THRESHOLD_DEFAULT,
                                  max_boost: float = MAX_ANOMALY_BOOST) -> float:
    """
    Increase failure probability if anomaly is detected.
    - If anomaly_detected=True: apply a boost.
    - If anomaly_score is provided, boost scales with severity.
    """
    p = float(p_fail)
    if anomaly_detected or (anomaly_score is not None and anomaly_score >= anomaly_threshold):
        # severity from 0..1
        sev = 1.0
        if anomaly_score is not None:
            # normalize: below threshold => 0 severity, max at 1.0
            sev = (float(anomaly_score) - float(anomaly_threshold)) / max(1e-9, (1.0 - float(anomaly_threshold)))
            sev = float(np.clip(sev, 0.0, 1.0))
        boost = max_boost * sev
        return float(np.clip(p + boost, 0.0, 1.0))
    return p


def state_from_prob(p_final: float, anomaly: bool = False) -> str:
    """
    Converts probability + anomaly into an interpretable machine state.
    Tune thresholds based on your safety preference.
    """
    p = float(p_final)
    if anomaly and p >= 0.60:
        return "CRITICAL"
    if p >= 0.70:
        return "FAILURE_RISK"
    if anomaly or p >= 0.40:
        return "WARNING"
    return "NORMAL"


# ==============================
# TRAINING
# ==============================
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Keep only needed columns if present
    needed = FEATURES + [TARGET]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"These columns are missing in CSV: {missing_cols}")

    # Allow missing values in features; only ensure target exists
    df = df.dropna(subset=[TARGET])

    # Convert target to int
    y = df[TARGET].astype(int).values

    # Add missing indicators
    X = add_missing_indicators(df[FEATURES])

    # Now total feature columns = original FEATURES + missing flags
    feature_cols = list(X.columns)

    # Preprocess: impute numeric with median
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_cols)
        ],
        remainder="drop"
    )

    base_model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2
    )

    # Full pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", base_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit base pipeline
    pipe.fit(X_train, y_train)

    # Calibrate for better probabilities (important for dashboard)
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # Predict probabilities
    p_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_default = (p_test >= 0.5).astype(int)

    # Tune threshold focusing on recall for failures
    best_thr, thr_info = tune_threshold_for_recall(y_test, p_test, min_recall=0.85)
    y_pred_tuned = (p_test >= best_thr).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred_tuned)
    cm = confusion_matrix(y_test, y_pred_tuned)

    try:
        roc = roc_auc_score(y_test, p_test)
    except Exception:
        roc = None
    try:
        pr_auc = average_precision_score(y_test, p_test)
    except Exception:
        pr_auc = None

    print("\n==============================")
    print("✅ Failure Model Evaluation")
    print("==============================")
    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {len(df)}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Default threshold: 0.50")
    print(f"Tuned threshold:   {best_thr:.4f}  ({thr_info})")
    print("\n--- Metrics (Tuned Threshold) ---")
    print(f"Accuracy: {acc:.6f}")
    if roc is not None:
        print(f"ROC-AUC:  {roc:.6f}")
    if pr_auc is not None:
        print(f"PR-AUC:   {pr_auc:.6f}")

    print("\n--- Confusion Matrix (Tuned) ---")
    print(cm)

    print("\n--- Classification Report (Tuned) ---")
    print(classification_report(y_test, y_pred_tuned, digits=4))

    # Save model + metadata
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(calibrated, MODEL_OUT)

    meta = {
        "features": FEATURES,
        "feature_columns_after_missing_indicators": feature_cols,
        "target": TARGET,
        "threshold": float(best_thr),
        "threshold_info": thr_info,
        "max_anomaly_boost": float(MAX_ANOMALY_BOOST),
        "anomaly_threshold_default": float(ANOMALY_THRESHOLD_DEFAULT),
        "random_state": RANDOM_STATE
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved calibrated model to: {MODEL_OUT}")
    print(f"✅ Saved metadata to:        {META_OUT}")

    # Quick demo: inference with missing sensor + anomaly fusion
    demo = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": np.nan,           # simulate sensor failure
        "Tool wear [min]": 80.0
    }
    demo_df = pd.DataFrame([demo])
    demo_df = add_missing_indicators(demo_df)
    p_fail = float(calibrated.predict_proba(demo_df)[:, 1][0])

    # fuse with anomaly
    p_final = fuse_anomaly_with_failure_prob(p_fail, anomaly_detected=True, anomaly_score=0.85, anomaly_threshold=0.6)
    state = state_from_prob(p_final, anomaly=True)

    print("\n--- Demo (missing Torque sensor + anomaly detected) ---")
    print(f"Raw failure probability:   {p_fail:.4f}")
    print(f"Final failure probability: {p_final:.4f}")
    print(f"State: {state}")


if __name__ == "__main__":
    main()

