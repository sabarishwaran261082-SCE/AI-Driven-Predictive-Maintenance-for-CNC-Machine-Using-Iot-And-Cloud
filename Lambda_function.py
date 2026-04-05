import os
import json
import time
import boto3
import joblib
import numpy as np
import warnings
from decimal import Decimal
from botocore.exceptions import ClientError

# ------------------------
# joblib safe on Lambda
# ------------------------
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")

# ------------------------
# CONFIG / ENV
# ------------------------
BUCKET = os.environ.get("MODEL_BUCKET", "cnc-ai-models")

# Failure model (AI4I trained, lambda-friendly)
FAILURE_MODEL_KEY = os.environ.get("FAILURE_MODEL_KEY", "failure/failure_model_lambda.pkl")
FAILURE_META_KEY  = os.environ.get("FAILURE_META_KEY",  "failure/failure_model_lambda_meta.json")

# RUL model (lambda-friendly)
RUL_MODEL_KEY = os.environ.get("RUL_MODEL_KEY", "rul/rul_base_stepper_lambda.pkl")
RUL_META_KEY  = os.environ.get("RUL_META_KEY",  "rul/rul_base_stepper_lambda_meta.json")

# Anomaly model + meta (10 feature)
ANOMALY_MODEL_KEY = os.environ.get("ANOMALY_MODEL_KEY", "anomaly/anomaly_bearing_vib.pkl")
ANOMALY_META_KEY  = os.environ.get("ANOMALY_META_KEY",  "anomaly/anomaly_bearing_vib_meta.json")

TABLE     = os.environ.get("LATEST_TABLE", "cnc_latest")
DEVICE_ID = os.environ.get("DEVICE_ID", "cnc-esp32-01")

# vibration buffer
VIB_BUFFER_SIZE = int(os.environ.get("VIB_BUFFER_SIZE", "256"))

# ---- Influence tuning knobs (safe defaults) ----
FAILURE_BOOST_MAX = float(os.environ.get("FAILURE_BOOST_MAX", "0.15"))   # max increase when anomaly true
FAILURE_DECAY     = float(os.environ.get("FAILURE_DECAY", "0.05"))       # decrease when anomaly false
FAILURE_FLAG_TH   = float(os.environ.get("FAILURE_FLAG_TH", "0.50"))     # failure_prob >= this => "failure_flag"

RUL_DROP_MILD     = float(os.environ.get("RUL_DROP_MILD", "0.08"))       # mild drop fraction
RUL_DROP_STRONG   = float(os.environ.get("RUL_DROP_STRONG", "0.22"))     # strong drop fraction
RUL_SEV_GAIN      = float(os.environ.get("RUL_SEV_GAIN", "0.10"))        # extra drop per anomaly severity unit
RUL_MAX_DROP      = float(os.environ.get("RUL_MAX_DROP", "0.80"))        # clamp

# ---- Warm-up guard for anomaly (prevents false anomaly at startup) ----
# how many samples needed before we trust anomaly output.
# You can set this env var, ex: ANOM_MIN_READY=200 (recommended) or 2048 (strict).
ANOM_MIN_READY = int(os.environ.get("ANOM_MIN_READY", "200"))

# ---- Torque estimation constant (Nm per Amp) ----
KT_NM_PER_AMP = float(os.environ.get("KT_NM_PER_AMP", "0.35"))

# ------------------------
# CACHED MODELS (warm start)
# ------------------------
_failure_model = None
_failure_meta  = None

_rul_model = None
_rul_meta  = None

_anomaly_model = None
_anomaly_meta  = None
_anomaly_threshold = None
_anomaly_feature_cols = None

# ------------------------
# Tool-wear state (warm Lambda memory)
# NOTE: This resets on cold-start. For persistent tool wear, store in DynamoDB.
# ------------------------
_last_ts = None
_tool_wear_min = 0.0


# ==========================
# HTTP helpers
# ==========================
def _cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
    }

def _response(status_code: int, body_obj):
    return {"statusCode": status_code, "headers": _cors_headers(), "body": json.dumps(body_obj)}

def _is_http_event(event: dict) -> bool:
    return isinstance(event, dict) and "requestContext" in event

def _http_method(event: dict) -> str:
    try:
        return (event.get("requestContext", {}).get("http", {}).get("method", "")).upper()
    except Exception:
        return ""

def _http_path(event: dict) -> str:
    p = event.get("rawPath") or event.get("path")
    if p:
        return str(p)
    try:
        return str(event.get("requestContext", {}).get("http", {}).get("path", ""))
    except Exception:
        return ""

def _parse_event_body(event: dict) -> dict:
    if not isinstance(event, dict):
        return {}
    if "body" in event and event["body"]:
        body = event["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except Exception:
                return {}
        if isinstance(body, dict):
            return body
    return event


# ==========================
# S3 download + model load
# ==========================
def _download(key: str) -> str:
    safe_key = key.replace("/", "_")
    local_path = f"/tmp/{safe_key}"
    if os.path.exists(local_path):
        return local_path
    try:
        s3.download_file(BUCKET, key, local_path)
        return local_path
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        raise FileNotFoundError(f"S3 object not found: s3://{BUCKET}/{key} (code={code})")

def _load_json(key: str) -> dict:
    p = _download(key)
    with open(p, "r") as f:
        return json.load(f)

def _load_models():
    global _failure_model, _failure_meta
    global _rul_model, _rul_meta
    global _anomaly_model, _anomaly_meta, _anomaly_threshold, _anomaly_feature_cols

    if _failure_model is None:
        _failure_model = joblib.load(_download(FAILURE_MODEL_KEY))
    if _failure_meta is None:
        _failure_meta = _load_json(FAILURE_META_KEY)

    if _rul_model is None:
        _rul_model = joblib.load(_download(RUL_MODEL_KEY))
    if _rul_meta is None:
        _rul_meta = _load_json(RUL_META_KEY)

    if _anomaly_model is None:
        _anomaly_model = joblib.load(_download(ANOMALY_MODEL_KEY))
    if _anomaly_meta is None:
        _anomaly_meta = _load_json(ANOMALY_META_KEY)
        thr = _anomaly_meta.get("threshold", 0.0)
        if thr is None:
            thr = 0.0
        _anomaly_threshold = float(thr)
        _anomaly_feature_cols = _anomaly_meta.get(
            "feature_cols",
            ["mean","std","rms","ptp","maxabs","crest","kurtosis","skew","zcr","mad"]
        )


# ==========================
# DynamoDB safe converters
# ==========================
def _to_ddb_safe(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_ddb_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb_safe(v) for v in obj]
    return obj

def _to_json_safe(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


# ==========================
# DynamoDB read/write (vibration buffer)
# ==========================
def _table():
    return ddb.Table(TABLE)

def _read_item():
    return _table().get_item(Key={"device_id": DEVICE_ID}).get("Item")

def _read_latest():
    item = _read_item()
    if not item:
        return None
    return item.get("resp")

def _read_vib_buffer():
    item = _read_item()
    if not item:
        return []
    vib_buf = _to_json_safe(item.get("vib_buf", []))
    return vib_buf if isinstance(vib_buf, list) else []

def _save_latest(resp: dict, vibration_value: float):
    old = _read_item() or {}
    vib_buf = _to_json_safe(old.get("vib_buf", []))
    vib_buf = vib_buf if isinstance(vib_buf, list) else []
    vib_buf = [float(x) for x in vib_buf]

    vib_buf.append(float(vibration_value))
    vib_buf = vib_buf[-VIB_BUFFER_SIZE:]

    _table().put_item(Item={
        "device_id": DEVICE_ID,
        "ts": int(time.time()),
        "vib_buf": _to_ddb_safe(vib_buf),
        "resp": _to_ddb_safe(resp),
    })


# ==========================
# Numeric helpers
# ==========================
def _get_float(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _get_int(d, k, default=0):
    try:
        v = d.get(k, default)
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


# ==========================
# ANOMALY feature extraction (10 features) - matches your trained model
# ==========================
def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)

def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std() + 1e-12
    z = (x - m) / s
    return float(np.mean(z**3))

def _extract_anom_features_10(vib_buf):
    x = np.asarray(vib_buf, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: 0.0 for k in ["mean","std","rms","ptp","maxabs","crest","kurtosis","skew","zcr","mad"]}

    mean = float(np.mean(x))
    std  = float(np.std(x) + 1e-12)
    rms  = float(np.sqrt(np.mean(x**2)))
    ptp  = float(np.ptp(x))
    maxabs = float(np.max(np.abs(x)))
    crest  = float(maxabs / (rms + 1e-12))
    kurt   = _safe_kurtosis(x)
    skew   = _safe_skew(x)
    zcr    = float(np.mean((x[:-1] * x[1:]) < 0)) if x.size > 1 else 0.0
    med    = np.median(x)
    mad    = float(np.median(np.abs(x - med)) + 1e-12)

    return {
        "mean": mean, "std": std, "rms": rms, "ptp": ptp,
        "maxabs": maxabs, "crest": crest,
        "kurtosis": kurt, "skew": skew,
        "zcr": zcr, "mad": mad
    }

def _anomaly_score_from_vibbuf(vib_buf):
    feats = _extract_anom_features_10(vib_buf)
    cols = _anomaly_feature_cols
    # safe float conversion
    row = []
    for k in cols:
        v = feats.get(k, 0.0)
        if v is None or v == "":
            v = 0.0
        row.append(float(v))
    X = np.array([row], dtype=np.float32)

    if hasattr(_anomaly_model, "decision_function"):
        score = float(_anomaly_model.decision_function(X)[0])
    elif hasattr(_anomaly_model, "score_samples"):
        score = float(_anomaly_model.score_samples(X)[0])
    else:
        score = float(_anomaly_model.predict(X)[0])

    anomaly = bool(score < float(_anomaly_threshold))
    severity = max(0.0, float(_anomaly_threshold) - float(score)) if anomaly else 0.0
    return anomaly, score, float(_anomaly_threshold), severity, feats


# ==========================
# FAILURE inference (uses failure_meta full_feature_order)
# ==========================
def _build_failure_payload_from_iot(payload: dict) -> dict:
    """
    Maps your 5 sensor values into AI4I-style inputs.
    AI4I expects:
      Air temperature [K], Process temperature [K],
      Rotational speed [rpm], Torque [Nm], Tool wear [min]
    We estimate Torque + Tool wear from your sensors/time so they are NOT null.
    """
    temperature_c = payload.get("temperature", None)
    rpm = payload.get("rpm", None)

    out = {}

    # temperature C -> Kelvin
    if temperature_c is None:
        out["Air temperature [K]"] = None
        out["Process temperature [K]"] = None
    else:
        out["Air temperature [K]"] = float(temperature_c) + 273.15
        out["Process temperature [K]"] = float(temperature_c) + 273.15

    out["Rotational speed [rpm]"] = None if rpm is None else float(rpm)

    # our estimated fields (already added in _run_inference)
    out["Torque [Nm]"] = payload.get("torque_Nm", None)
    out["Tool wear [min]"] = payload.get("tool_wear_min", None)

    return out

def _predict_failure_probability(payload: dict) -> dict:
    meta = _failure_meta
    order = meta.get("full_feature_order")
    if not order:
        raise ValueError("failure meta missing full_feature_order")

    base = _build_failure_payload_from_iot(payload)

    row = []
    for col in order:
        if col.endswith("__missing"):
            real = col.replace("__missing", "")
            v = base.get(real, None)
            row.append(1.0 if v is None else 0.0)
        else:
            v = base.get(col, None)
            row.append(np.nan if v is None else float(v))

    X = np.array([row], dtype=np.float32)
    p = float(_failure_model.predict_proba(X)[0][1])
    return {"p_base": p, "mapped_inputs": base}

def _fuse_failure_with_anomaly(p_base: float, anomaly: bool, severity: float) -> dict:
    p = float(np.clip(p_base, 0.0, 1.0))

    if anomaly:
        boost = min(FAILURE_BOOST_MAX, 0.05 + 0.20 * float(severity))
        p_adj = float(np.clip(p + boost, 0.0, 1.0))
        mode = "boosted_by_anomaly"
    else:
        p_adj = float(np.clip(p * (1.0 - FAILURE_DECAY), 0.0, 1.0))
        boost = 0.0
        mode = "decreased_no_anomaly"

    return {"p_adjusted": p_adj, "mode": mode, "boost_used": float(boost)}


# ==========================
# RUL inference (uses rul_meta feature order)
# ==========================
def _predict_rul_base(payload: dict) -> dict:
    meta = _rul_meta
    feats = meta.get("features", [])
    all_feats = meta.get("all_features_after_missing_indicators", [])

    if not feats or not all_feats:
        raise ValueError("RUL meta missing features/all_features_after_missing_indicators")

    base = {
        "temperature_C": payload.get("temperature_C", payload.get("temperature", None)),
        "vibration_g": payload.get("vibration_g", payload.get("vibration", None)),
        "current_A": payload.get("current_A", payload.get("current", None)),
        "voltage_V": payload.get("voltage_V", payload.get("voltage", None)),
        "rpm": payload.get("rpm", None),
    }

    row = []
    for c in feats:
        v = base.get(c, None)
        row.append(np.nan if v is None else float(v))
    for c in feats:
        v = base.get(c, None)
        row.append(1.0 if v is None else 0.0)

    X = np.array([row], dtype=np.float32)
    rul = float(_rul_model.predict(X)[0])
    rul = max(0.0, rul)

    return {"rul_base": float(rul), "rul_inputs": base}

def _adjust_rul(rul_base: float, failure_prob_adj: float, anomaly: bool, severity: float) -> dict:
    p = float(np.clip(failure_prob_adj, 0.0, 1.0))
    failure_flag = bool(p >= FAILURE_FLAG_TH)

    if failure_flag and anomaly:
        drop_frac = RUL_DROP_STRONG + (0.20 * p) + (RUL_SEV_GAIN * float(severity))
        mode = "strong_drop_both_true"
    elif failure_flag or anomaly:
        drop_frac = RUL_DROP_MILD + (0.12 * p) + (0.5 * RUL_SEV_GAIN * float(severity))
        mode = "mild_drop_one_true"
    else:
        drop_frac = 0.0
        mode = "stable_both_false"

    drop_frac = float(np.clip(drop_frac, 0.0, RUL_MAX_DROP))
    rul_adj = float(max(0.0, rul_base * (1.0 - drop_frac)))

    return {
        "rul_adjusted": rul_adj,
        "drop_fraction": drop_frac,
        "mode": mode,
        "failure_flag_used": failure_flag
    }


# ==========================
# CORE INFERENCE
# ==========================
def _run_inference(payload: dict) -> dict:
    _load_models()

    # ---- NEMA17 IoT expected inputs ----
    vibration   = _get_float(payload, "vibration", 0.0)
    temperature = _get_float(payload, "temperature", 0.0)
    current     = _get_float(payload, "current", 0.0)
    voltage     = _get_float(payload, "voltage", 0.0)
    rpm         = _get_float(payload, "rpm", 0.0)

    # event timestamp (prefer payload ts if present)
    now_ts = int(time.time())
    ts_in = _get_int(payload, "ts", now_ts)
    if ts_in <= 0:
        ts_in = now_ts

    # ---- Estimate Torque + Tool wear from your 5 sensors/time ----
    # Torque estimate (Nm) using a simple motor constant
    torque_nm = float(KT_NM_PER_AMP * current)

    # Tool wear estimate (minutes) accumulates with time, weighted by vibration stress
    global _last_ts, _tool_wear_min
    if _last_ts is not None and ts_in > _last_ts:
        delta_min = (ts_in - _last_ts) / 60.0
        vib_ref = 0.25
        # stress: 1 to 7 (clamped)
        stress = 1.0 + 3.0 * min(max(vibration / vib_ref, 0.0), 2.0)
        _tool_wear_min += float(delta_min * stress)
    _last_ts = ts_in

    # write these into payload so failure mapping picks them up (no nulls)
    payload["torque_Nm"] = torque_nm
    payload["tool_wear_min"] = float(_tool_wear_min)

    # ---- Anomaly using vib buffer from DynamoDB ----
    vib_buf = _read_vib_buffer()
    vib_buf.append(float(vibration))
    vib_buf = vib_buf[-VIB_BUFFER_SIZE:]

    # warm-up guard: do not run anomaly until buffer has enough samples
    if len(vib_buf) < min(ANOM_MIN_READY, int((_anomaly_meta or {}).get("window_size", 2048))):
        anomaly = False
        anom_score = None
        anom_thr = float(_anomaly_threshold)
        severity = 0.0
        anom_feats = _extract_anom_features_10(vib_buf)
        anomaly_state = "warming_up"
    else:
        anomaly, anom_score, anom_thr, severity, anom_feats = _anomaly_score_from_vibbuf(vib_buf)
        anomaly_state = "ready"

    # ---- Failure (AI4I) base + anomaly influence ----
    f = _predict_failure_probability(payload)
    p_fail_base = float(f["p_base"])

    f_fused = _fuse_failure_with_anomaly(p_fail_base, anomaly, severity)
    p_fail = float(f_fused["p_adjusted"])

    # ---- RUL base + influence from both ----
    r = _predict_rul_base(payload)
    rul_base = float(r["rul_base"])

    rul_adj = _adjust_rul(rul_base, p_fail, anomaly, severity)

    resp = {
        "device_id": DEVICE_ID,
        "ts": now_ts,
        "inputs": {
            "vibration": vibration,
            "temperature": temperature,
            "current": current,
            "voltage": voltage,
            "rpm": rpm,
            "vib_buffer_len": len(vib_buf),
            "motor": "NEMA17",
            "torque_Nm_est": torque_nm,
            "tool_wear_min_est": float(_tool_wear_min),
            "ts_in": ts_in
        },

        # anomaly outputs
        "anomaly_state": anomaly_state,
        "anomaly": bool(anomaly),
        "anomaly_threshold": float(anom_thr),
        "anomaly_severity": float(severity),
        "anom_features_used": anom_feats,
        "anom_feature_cols": list(_anomaly_feature_cols),

        # anomaly_score can be None during warm-up
        "anomaly_score": (None if anom_score is None else float(anom_score)),

        # failure outputs
        "failure_probability_base": float(p_fail_base),
        "failure_probability": float(p_fail),
        "failure_fusion": {
            "mode": f_fused["mode"],
            "boost_used": f_fused["boost_used"],
            "failure_flag_threshold": FAILURE_FLAG_TH
        },
        "failure_inputs_mapped_to_ai4i": f["mapped_inputs"],

        # rul outputs
        "rul_base": float(rul_base),
        "rul": float(rul_adj["rul_adjusted"]),
        "rul_adjustment": {
            "mode": rul_adj["mode"],
            "drop_fraction": rul_adj["drop_fraction"],
            "failure_flag_used": rul_adj["failure_flag_used"]
        },
    }

    _save_latest(resp, vibration)
    return resp


# ==========================
# MAIN HANDLER
# ==========================
def lambda_handler(event, context):
    try:
        # HTTP API
        if _is_http_event(event):
            method = _http_method(event)
            path = _http_path(event) or "/"

            if method == "OPTIONS":
                return _response(200, {"ok": True})

            if method == "GET" and path.endswith("/health"):
                return _response(200, {"status": "ok"})

            if method == "GET" and path.endswith("/latest"):
                latest = _read_latest()
                if latest is None:
                    return _response(404, {"message": "No latest prediction yet"})
                return _response(200, _to_json_safe(latest))

            if method == "POST" and path.endswith("/predict"):
                payload = _parse_event_body(event)
                if not isinstance(payload, dict) or not payload:
                    return _response(400, {"error": "Invalid JSON body"})
                resp = _run_inference(payload)
                return _response(200, resp)

            return _response(404, {"error": f"Route not found: {method} {path}"})

        # IoT direct invoke
        payload = _parse_event_body(event)
        if not isinstance(payload, dict) or not payload:
            return _response(400, {"error": "Invalid payload"})

        resp = _run_inference(payload)
        return _response(200, resp)

    except Exception as e:
        return _response(500, {"error": str(e), "bucket": BUCKET})