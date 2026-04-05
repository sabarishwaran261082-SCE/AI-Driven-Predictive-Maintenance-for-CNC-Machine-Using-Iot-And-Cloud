import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


NORMAL_DIR = "data/bearing/normal"
FAULTY_DIR = "data/bearing/faulty"

MODEL_OUT = "models/anomaly_bearing_vib.pkl"
META_OUT  = "models/anomaly_bearing_vib_meta.json"

RANDOM_STATE = 42

# Windowing (tune if needed)
WINDOW_SIZE = 2048
STEP_SIZE = 1024
MAX_WINDOWS_PER_FILE = 300

# IsolationForest params
CONTAMINATION = 0.08


# -----------------------------
# Feature extraction (vibration only)
# -----------------------------
def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)

def safe_skew(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    return float(np.mean(z**3))

def extract_features(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "rms": np.nan, "ptp": np.nan,
            "maxabs": np.nan, "crest": np.nan, "kurtosis": np.nan,
            "skew": np.nan, "zcr": np.nan, "mad": np.nan
        }

    mean = float(np.mean(x))
    std  = float(np.std(x) + 1e-12)
    rms  = float(np.sqrt(np.mean(x**2)))
    ptp  = float(np.ptp(x))
    maxabs = float(np.max(np.abs(x)))
    crest  = float(maxabs / (rms + 1e-12))
    kurt   = safe_kurtosis(x)
    skew   = safe_skew(x)

    # zero crossing rate (rough frequency indicator)
    zc = np.mean((x[:-1] * x[1:]) < 0) if x.size > 1 else 0.0
    zcr = float(zc)

    # median absolute deviation
    med = np.median(x)
    mad = float(np.median(np.abs(x - med)) + 1e-12)

    return {
        "mean": mean, "std": std, "rms": rms, "ptp": ptp,
        "maxabs": maxabs, "crest": crest,
        "kurtosis": kurt, "skew": skew,
        "zcr": zcr, "mad": mad
    }

def window_signal(sig: np.ndarray, win: int, step: int):
    n = len(sig)
    for start in range(0, max(0, n - win + 1), step):
        yield sig[start:start+win]

def load_windows_from_folder(folder: str, label: str):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    feats = []
    for fp in files:
        df = pd.read_csv(fp)
        if "vibration" not in df.columns:
            raise ValueError(f"'vibration' column not found in {fp}")

        sig = df["vibration"].astype(float).to_numpy()
        count = 0
        for w in window_signal(sig, WINDOW_SIZE, STEP_SIZE):
            feats.append(extract_features(w) | {"__file__": os.path.basename(fp), "__label__": label})
            count += 1
            if count >= MAX_WINDOWS_PER_FILE:
                break

    return pd.DataFrame(feats)


def main():
    os.makedirs("models", exist_ok=True)

    normal_feat = load_windows_from_folder(NORMAL_DIR, "normal")
    faulty_feat = load_windows_from_folder(FAULTY_DIR, "faulty")

    feature_cols = [c for c in normal_feat.columns if not c.startswith("__")]

    X_normal = normal_feat[feature_cols].copy()
    X_faulty = faulty_feat[feature_cols].copy()

    # Pipeline: impute + scale + isolation forest
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=400,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # ✅ Train only on NORMAL
    pipe.fit(X_normal)

    # decision_function: higher = more normal
    normal_scores = pipe.named_steps["iforest"].decision_function(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["imputer"].transform(X_normal)
        )
    )
    faulty_scores = pipe.named_steps["iforest"].decision_function(
        pipe.named_steps["scaler"].transform(
            pipe.named_steps["imputer"].transform(X_faulty)
        )
    )

    # Threshold = 5th percentile of normal scores (conservative)
    threshold = float(np.percentile(normal_scores, 5))

    normal_pred = (normal_scores < threshold).mean()
    faulty_pred = (faulty_scores < threshold).mean()

    print("\n✅ Bearing Vibration Anomaly Model Trained")
    print("Window:", WINDOW_SIZE, "Step:", STEP_SIZE)
    print("Feature count:", len(feature_cols))
    print("Threshold (5th pct normal):", threshold)
    print("Normal flagged as anomaly (%):", round(normal_pred * 100, 2))
    print("Faulty flagged as anomaly (%):", round(faulty_pred * 100, 2))

    joblib.dump(pipe, MODEL_OUT)
    meta = {
        "feature_cols": feature_cols,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "threshold": threshold,
        "contamination": CONTAMINATION,
        "note": "IsolationForest trained on vibration-window features from NORMAL bearing data."
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Saved model:", MODEL_OUT)
    print("✅ Saved meta :", META_OUT)


if __name__ == "__main__":
    main()
