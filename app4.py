# app_fixed.py
# CNC Industrial Dashboard (SCADA/HMI style) + RPM support
# - Fixes: sidebar input visibility, system-normal strip visibility, plot titles visibility
# - Fixes: StreamlitDuplicateElementId by adding unique keys
# - Fixes: streamlit_autorefresh missing -> fallback to HTML meta refresh
# Run:
#   streamlit run app_fixed.py --server.port 8501 --server.address 0.0.0.0

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px


# =========================
# Optional autorefresh (safe)
# =========================
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False
    st_autorefresh = None


# =========================
# Config
# =========================
DEFAULT_API_BASE = "https://f23uoqxe23.execute-api.us-east-1.amazonaws.com/prod"
DEFAULT_REFRESH_SECONDS = 8
MAX_POINTS_DEFAULT = 600  # ~80 minutes at 8s cadence


@dataclass
class Thresholds:
    anomaly_severity_warning: float = 0.05
    anomaly_severity_critical: float = 0.15
    failure_prob_warning: float = 0.30
    failure_prob_critical: float = 0.50
    rul_warning: float = 40.0
    rul_critical: float = 20.0


# Core sensors (always show if present)
SENSOR_KEYS = [
    ("vibration", "Vibration", "g"),
    ("temperature", "Temperature", "°C"),
    ("current", "Current", "A"),
    ("voltage", "Voltage", "V"),
]
# Optional sensors (show if present in API)
OPTIONAL_KEYS = [
    ("rpm", "RPM", "rpm"),
]

# Trend threshold lines (industrial soft limits)
SENSOR_LIMITS = {
    "vibration": (0.30, 0.60),      # warn, crit
    "temperature": (60.0, 80.0),
    "current": (2.5, 3.5),
    "voltage": (24.5, 26.0),
    # rpm is optional; set basic defaults (edit if needed)
    "rpm": (2000.0, 3000.0),
}


# =========================
# UI Styling (Industrial SCADA/HMI) - FIXED VISIBILITY
# =========================
def apply_industrial_css():
    st.markdown(
        """
<style>
:root{
  --bg0:#070B14;
  --bg1:#0B1220;
  --stroke:rgba(255,255,255,.12);
  --stroke2:rgba(255,255,255,.08);
  --text:#EAF2FF;
  --muted:#B6C3D8;
  --muted2:#8EA0BE;
  --ok:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
  --shadow: 0 18px 55px rgba(0,0,0,.55);
  --shadow2: 0 10px 28px rgba(0,0,0,.35);
  --r:18px;
}

/* App background */
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% 0%, rgba(56,189,248,.14), transparent 60%),
              radial-gradient(1000px 600px at 80% 10%, rgba(167,139,250,.12), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
  color: var(--text) !important;
}

/* Container spacing */
.block-container{
  padding-top: 1.0rem;
  padding-bottom: 1.4rem;
  max-width: 1320px;
}

/* -------- Sidebar -------- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(2,6,23,.90)) !important;
  border-right: 1px solid var(--stroke2);
}
section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

/* Fix: make inputs readable (no white washed bars) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{
  color: var(--text) !important;
  opacity: 0.98 !important;
}

/* Text input + number input background */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input{
  background: rgba(255,255,255,.10) !important;
  border: 1px solid rgba(255,255,255,.18) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  padding: 0.65rem 0.8rem !important;
}

/* Fix: number-input container sometimes shows a bright strip; force dark background */
section[data-testid="stSidebar"] div[data-baseweb="input"]{
  background: rgba(255,255,255,.10) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="base-input"]{
  background: transparent !important;
}

/* Fix: +/- buttons */
section[data-testid="stSidebar"] .stNumberInput button{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] .stNumberInput button:hover{
  background: rgba(255,255,255,.10) !important;
}

/* -------- Alarm strip (SYSTEM NORMAL visibility) -------- */
.alarm-strip{
  border-radius: 18px;
  padding: 12px 14px;
  border: 1px solid rgba(255,255,255,.14);
  background: linear-gradient(135deg, rgba(15,23,42,.86), rgba(2,6,23,.80));
  box-shadow: var(--shadow2);
  margin-bottom: 12px;
}
.alarm-strip .alarm-title{
  font-weight: 950;
  letter-spacing: .5px;
  color: var(--text);
  font-size: 14px;
}
.alarm-strip .alarm-sub{
  color: rgba(234,242,255,.90);
  font-size: 12px;
  margin-top: 3px;
}
.alarm-strip.crit{
  border: 1px solid rgba(239,68,68,.55);
  background: radial-gradient(700px 240px at 10% 0%, rgba(239,68,68,.25), transparent 55%),
              linear-gradient(180deg, rgba(239,68,68,.14), rgba(2,6,23,.70));
  animation: pulse 1.2s infinite;
}
.alarm-strip.warn{
  border: 1px solid rgba(245,158,11,.55);
  background: radial-gradient(700px 240px at 10% 0%, rgba(245,158,11,.22), transparent 55%),
              linear-gradient(180deg, rgba(245,158,11,.12), rgba(2,6,23,.70));
}
@keyframes pulse{
  0% { box-shadow: 0 0 0 rgba(239,68,68,.0), var(--shadow2); transform: translateY(0px); }
  50% { box-shadow: 0 0 18px rgba(239,68,68,.30), var(--shadow2); transform: translateY(-1px); }
  100% { box-shadow: 0 0 0 rgba(239,68,68,.0), var(--shadow2); transform: translateY(0px); }
}

/* -------- Header -------- */
.hmi-header{
  position: relative;
  display:flex; align-items:center; justify-content:space-between;
  padding: 16px 18px;
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(2,6,23,.88));
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow);
  overflow:hidden;
  margin-bottom: 14px;
}
.hmi-title{
  font-size: 26px;
  font-weight: 950;
  letter-spacing: .6px;
  color: var(--text);
}
.hmi-sub{
  font-size: 13px;
  margin-top: 5px;
  color: var(--muted);
}
.pills{ display:flex; gap:10px; align-items:center; }
.pill{
  font-size: 12px; font-weight: 900;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
}
.pill.ok{ background: rgba(34,197,94,.14); color: #a7f3d0; }
.pill.bad{ background: rgba(239,68,68,.16); color: #fecaca; }

/* Section headings */
.section-title{
  font-size: 13px;
  font-weight: 950;
  letter-spacing: 1px;
  color: rgba(234,242,255,.94);
  margin: 14px 0 10px 2px;
  text-transform: uppercase;
}

/* Cards */
.card{
  border-radius: var(--r);
  padding: 14px 14px;
  border: 1px solid var(--stroke2);
  background: linear-gradient(180deg, rgba(15,23,42,.62), rgba(2,6,23,.62));
  box-shadow: var(--shadow2);
}
.card-title{ font-size: 12px; color: var(--muted); font-weight: 800; }
.card-value{ font-size: 30px; font-weight: 950; margin-top: 8px; color: var(--text); }
.card-unit{ font-size: 12px; color: var(--muted2); font-weight: 800; margin-left: 6px; }
.card-band{
  height: 8px; border-radius: 999px; margin-top: 12px;
  background: rgba(148,163,184,.24);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);
}
.band-ok{ background: linear-gradient(90deg, rgba(34,197,94,.90), rgba(34,197,94,.45)) !important; }
.band-warn{ background: linear-gradient(90deg, rgba(245,158,11,.95), rgba(245,158,11,.45)) !important; }
.band-bad{ background: linear-gradient(90deg, rgba(239,68,68,.95), rgba(239,68,68,.45)) !important; }

/* Plotly rounding */
.js-plotly-plot .plotly .main-svg{ border-radius: 16px; }
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# Data fetch & parsing
# =========================
def api_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def unwrap_lambda_response(obj: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(obj, dict) and "body" in obj and isinstance(obj["body"], str):
        try:
            return json.loads(obj["body"])
        except Exception:
            return obj
    return obj


def fetch_latest(base: str, timeout_s: float = 4.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = api_url(base, "/latest")
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        obj = r.json()
        data = unwrap_lambda_response(obj)
        if not isinstance(data, dict) or not data:
            return None, "Empty/invalid JSON response"
        return data, None
    except Exception as e:
        return None, str(e)


def ts_to_dt(ts: Any) -> datetime:
    try:
        t = float(ts)
        return datetime.fromtimestamp(t, tz=timezone.utc).astimezone()
    except Exception:
        return datetime.now().astimezone()


def get_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    inputs = data.get("inputs", {}) if isinstance(data.get("inputs", {}), dict) else {}
    return inputs


# =========================
# Risk classification
# =========================
def band_from_value(value: Optional[float], warn: float, crit: float, higher_is_worse: bool = True) -> str:
    if value is None:
        return "neutral"
    v = float(value)
    if higher_is_worse:
        if v >= crit:
            return "bad"
        if v >= warn:
            return "warn"
        return "ok"
    else:
        if v <= crit:
            return "bad"
        if v <= warn:
            return "warn"
        return "ok"


def should_alarm(data: Dict[str, Any], th: Thresholds) -> Tuple[bool, List[Dict[str, Any]], str]:
    items: List[Dict[str, Any]] = []

    anomaly_state = (data.get("anomaly_state") or "").lower()
    anomaly = bool(data.get("anomaly", False))
    severity = data.get("anomaly_severity", 0.0)
    failure_p = data.get("failure_probability", 0.0)
    rul = data.get("rul", None)

    if anomaly_state == "ready":
        sev_band = band_from_value(float(severity), th.anomaly_severity_warning, th.anomaly_severity_critical, True)
        if anomaly or sev_band in ("warn", "bad"):
            items.append({"type": "ANOMALY", "band": sev_band, "severity": float(severity)})

    fail_band = band_from_value(float(failure_p), th.failure_prob_warning, th.failure_prob_critical, True)
    if fail_band in ("warn", "bad"):
        items.append({"type": "FAILURE RISK", "band": fail_band, "failure_probability": float(failure_p)})

    if rul is not None:
        rul_band = band_from_value(float(rul), th.rul_warning, th.rul_critical, higher_is_worse=False)
        if rul_band in ("warn", "bad"):
            items.append({"type": "LOW RUL", "band": rul_band, "rul": float(rul)})

    active = any(i.get("band") in ("warn", "bad") for i in items)
    level = "ok"
    if any(i.get("band") == "bad" for i in items):
        level = "crit"
    elif any(i.get("band") == "warn" for i in items):
        level = "warn"
    return active, items, level


# =========================
# Plot helpers (FIX: title visibility)
# =========================
def _hline(fig: go.Figure, y: float, label: str, color_rgba: str):
    fig.add_hline(
        y=y,
        line_width=2,
        line_dash="dash",
        line_color=color_rgba,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_color="rgba(234,242,255,0.92)",
    )


def style_plot(fig: go.Figure):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(234,242,255,0.94)"),
        title_font=dict(color="rgba(234,242,255,0.98)", size=18),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(148,163,184,0.14)",
        tickfont=dict(color="rgba(234,242,255,0.85)", size=12),
        title_font=dict(color="rgba(234,242,255,0.90)", size=13),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(148,163,184,0.14)",
        tickfont=dict(color="rgba(234,242,255,0.85)", size=12),
        title_font=dict(color="rgba(234,242,255,0.90)", size=13),
    )
    return fig


def gauge(title: str, value: Optional[float], vmin: float, vmax: float,
          warn: float, crit: float, suffix: str = "", invert: bool = False) -> go.Figure:
    if value is None:
        value = vmin

    if not invert:
        steps = [
            {"range": [vmin, warn], "color": "rgba(34,197,94,0.55)"},
            {"range": [warn, crit], "color": "rgba(245,158,11,0.65)"},
            {"range": [crit, vmax], "color": "rgba(239,68,68,0.65)"},
        ]
        threshold_value = crit
    else:
        steps = [
            {"range": [vmin, crit], "color": "rgba(239,68,68,0.65)"},
            {"range": [crit, warn], "color": "rgba(245,158,11,0.65)"},
            {"range": [warn, vmax], "color": "rgba(34,197,94,0.55)"},
        ]
        threshold_value = crit

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": f" {suffix}" if suffix else ""},
        title={"text": title, "font": {"size": 16, "color": "rgba(234,242,255,0.98)"}},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": "rgba(148,163,184,0.60)"},
            "steps": steps,
            "threshold": {"line": {"color": "rgba(239,68,68,0.9)", "width": 3}, "value": threshold_value},
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=18, r=18, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(234,242,255,0.94)"),
        template="plotly_dark",
    )
    return fig


def line_chart(df: pd.DataFrame, y: str, title: str, unit: str,
               warn: Optional[float] = None, crit: Optional[float] = None) -> go.Figure:
    fig = px.line(df, x="dt", y=y, title=title)
    fig.update_traces(mode="lines", line=dict(width=3))

    fig.update_layout(
        height=270,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=f"{unit}",
        legend=dict(orientation="h"),
    )
    style_plot(fig)

    if warn is not None:
        _hline(fig, warn, "WARN", "rgba(245,158,11,0.85)")
    if crit is not None:
        _hline(fig, crit, "CRIT", "rgba(239,68,68,0.90)")

    return fig


def sparkline(values: List[float]) -> go.Figure:
    if not values:
        values = [0.0]
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(width=2)))
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark",
    )
    return fig


# =========================
# Streamlit State
# =========================
def init_state():
    all_cols = ["dt", "ts"] + [k for k, _, _ in SENSOR_KEYS] + [k for k, _, _ in OPTIONAL_KEYS]
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=all_cols)
    if "last_ts" not in st.session_state:
        st.session_state.last_ts = None
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "last_toast_key" not in st.session_state:
        st.session_state.last_toast_key = None


def append_point(data: Dict[str, Any], max_points: int):
    inputs = get_inputs(data)
    ts = data.get("ts", inputs.get("ts_in", time.time()))
    dt = ts_to_dt(ts)

    if st.session_state.last_ts is not None and float(ts) == float(st.session_state.last_ts):
        return

    row = {"dt": dt, "ts": float(ts)}
    # core sensors
    for key, _, _ in SENSOR_KEYS:
        row[key] = float(inputs.get(key, 0.0) or 0.0)
    # optional sensors
    for key, _, _ in OPTIONAL_KEYS:
        if key in inputs:
            row[key] = float(inputs.get(key, 0.0) or 0.0)
        else:
            row[key] = None

    df = st.session_state.df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    if len(df) > max_points:
        df = df.iloc[-max_points:].reset_index(drop=True)

    st.session_state.df = df
    st.session_state.last_ts = float(ts)


def add_alerts(dt: datetime, alarm_items: List[Dict[str, Any]]):
    for it in alarm_items:
        st.session_state.alerts.append({
            "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "type": it.get("type"),
            "band": it.get("band"),
            "details": json.dumps({k: v for k, v in it.items() if k not in ("message",)}, ensure_ascii=False),
        })
    if len(st.session_state.alerts) > 200:
        st.session_state.alerts = st.session_state.alerts[-200:]


# =========================
# UI Blocks
# =========================
def render_alarm_strip(online: bool, alarm_active: bool, alarm_items: List[Dict[str, Any]], level: str):
    if not online:
        st.markdown(
            """
<div class="alarm-strip">
  <div class="alarm-title">SYSTEM OFFLINE</div>
  <div class="alarm-sub">API not reachable. Showing last cached data.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    if not alarm_active:
        st.markdown(
            """
<div class="alarm-strip">
  <div class="alarm-title">SYSTEM NORMAL</div>
  <div class="alarm-sub">No active alarms. All monitored conditions are within limits.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    cls = "crit" if level == "crit" else "warn"
    title = "CRITICAL ALARM" if level == "crit" else "WARNING"
    details = " • ".join([f"{i.get('type')} ({str(i.get('band','')).upper()})" for i in alarm_items])

    st.markdown(
        f"""
<div class="alarm-strip {cls}">
  <div class="alarm-title">⚠️ {title}</div>
  <div class="alarm-sub">{details}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_header(device_id: str, online: bool, last_update: Optional[datetime], refresh_s: int):
    status_class = "ok" if online else "bad"
    status_text = "ONLINE" if online else "OFFLINE"
    last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S") if last_update else "—"

    st.markdown(
        f"""
<div class="hmi-header">
  <div>
    <div class="hmi-title">CNC Health Monitoring Dashboard</div>
    <div class="hmi-sub">Device: <b>{device_id}</b> • Last update: <b>{last_update_str}</b> • Refresh: every <b>{refresh_s}s</b></div>
  </div>
  <div class="pills">
    <div class="pill {status_class}">{status_text}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sensor_cards(df: pd.DataFrame):
    st.markdown('<div class="section-title">LIVE SENSORS</div>', unsafe_allow_html=True)

    # decide which sensors to show (add rpm only if any non-null exists)
    present = list(SENSOR_KEYS)
    for key, label, unit in OPTIONAL_KEYS:
        if key in df.columns and df[key].notna().any():
            present.append((key, label, unit))

    cols = st.columns(min(5, max(1, len(present))))
    for i, (key, label, unit) in enumerate(present):
        latest = float(df[key].dropna().iloc[-1]) if (len(df) and df[key].notna().any()) else 0.0
        series = df[key].dropna().tail(40).tolist() if (len(df) and df[key].notna().any()) else [0.0]

        warn, crit = SENSOR_LIMITS.get(key, (None, None))
        band = "neutral"
        if warn is not None and crit is not None:
            band = band_from_value(latest, float(warn), float(crit), higher_is_worse=True)
        band_class = {"ok": "band-ok", "warn": "band-warn", "bad": "band-bad"}.get(band, "")

        with cols[i % len(cols)]:
            fmt = "{:.3f}" if key in ("vibration",) else "{:.2f}"
            if key == "rpm":
                fmt = "{:.0f}"
            st.markdown(
                f"""
<div class="card">
  <div class="card-title">{label}</div>
  <div class="card-value">{fmt.format(latest)}<span class="card-unit">{unit}</span></div>
  <div class="card-band {band_class}"></div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(sparkline(series), use_container_width=True, config={"displayModeBar": False})


def render_sensor_charts(df: pd.DataFrame):
    st.markdown('<div class="section-title">SENSOR TRENDS (REAL-TIME)</div>', unsafe_allow_html=True)

    # Always show 4 core charts
    c1, c2 = st.columns(2)
    with c1:
        w, c = SENSOR_LIMITS["vibration"]
        st.plotly_chart(line_chart(df, "vibration", "Vibration (g)", "g", warn=w, crit=c), use_container_width=True)
        w, c = SENSOR_LIMITS["current"]
        st.plotly_chart(line_chart(df, "current", "Current (A)", "A", warn=w, crit=c), use_container_width=True)
    with c2:
        w, c = SENSOR_LIMITS["temperature"]
        st.plotly_chart(line_chart(df, "temperature", "Temperature (°C)", "°C", warn=w, crit=c), use_container_width=True)
        w, c = SENSOR_LIMITS["voltage"]
        st.plotly_chart(line_chart(df, "voltage", "Voltage (V)", "V", warn=w, crit=c), use_container_width=True)

    # RPM chart (only if exists)
    if "rpm" in df.columns and df["rpm"].notna().any():
        st.markdown('<div class="section-title">RPM TREND</div>', unsafe_allow_html=True)
        w, c = SENSOR_LIMITS["rpm"]
        st.plotly_chart(line_chart(df, "rpm", "RPM", "rpm", warn=w, crit=c), use_container_width=True)


def render_models(data: Dict[str, Any], th: Thresholds):
    st.markdown('<div class="section-title">MODEL INSIGHTS (HMI VIEW)</div>', unsafe_allow_html=True)

    anomaly_state = (data.get("anomaly_state") or "unknown").upper()
    anomaly = bool(data.get("anomaly", False))
    severity = data.get("anomaly_severity", 0.0)
    score = data.get("anomaly_score", None)
    thr = data.get("anomaly_threshold", None)

    fp_base = data.get("failure_probability_base", 0.0)
    fp = data.get("failure_probability", 0.0)
    fusion_mode = (data.get("failure_fusion", {}) or {}).get("mode", "—")

    rul_base = data.get("rul_base", None)
    rul = data.get("rul", None)
    rul_mode = (data.get("rul_adjustment", {}) or {}).get("mode", "—")
    drop_frac = (data.get("rul_adjustment", {}) or {}).get("drop_fraction", 0.0)

    left, mid, right = st.columns(3)

    with left:
        st.markdown("**ANOMALY (IsolationForest)**")
        sev_max = max(0.25, float(th.anomaly_severity_critical) * 2.0)
        st.plotly_chart(
            gauge("Severity", float(severity), 0.0, sev_max, th.anomaly_severity_warning, th.anomaly_severity_critical),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(f"State: **{anomaly_state}** • Anomaly: **{anomaly}**")
        if score is None or thr is None:
            st.caption("Score vs Threshold: **N/A (warming up)**")
        else:
            st.caption(f"Score: **{float(score):.4f}** • Threshold: **{float(thr):.4f}**")

    with mid:
        st.markdown("**FAILURE RISK (AI4I model)**")
        st.plotly_chart(
            gauge("Failure Probability", float(fp), 0.0, 1.0, th.failure_prob_warning, th.failure_prob_critical),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(f"Base: **{float(fp_base):.6f}** • Fused: **{float(fp):.6f}**")
        st.caption(f"Fusion Mode: **{fusion_mode}**")

    with right:
        st.markdown("**RUL (Remaining Useful Life)**")
        rul_v = float(rul) if rul is not None else 0.0
        vmax = max(100.0, rul_v * 1.4)
        st.plotly_chart(
            gauge("RUL", rul_v, 0.0, vmax, th.rul_warning, th.rul_critical, invert=True),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(f"Base: **{(float(rul_base) if rul_base is not None else 0.0):.2f}** • Adjusted: **{rul_v:.2f}**")
        st.caption(f"Mode: **{rul_mode}** • Drop: **{float(drop_frac):.3f}**")


def render_alert_log():
    st.markdown('<div class="section-title">ALERT LOG</div>', unsafe_allow_html=True)
    if not st.session_state.alerts:
        st.caption("Alert log is empty.")
        return
    df = pd.DataFrame(st.session_state.alerts)
    st.dataframe(df.tail(30), use_container_width=True, hide_index=True)


# =========================
# Refresh helper (no dependency required)
# =========================
def do_refresh(refresh_s: int, pause: bool):
    if pause:
        return

    if HAS_AUTOREFRESH:
        st_autorefresh(interval=int(refresh_s * 1000), key="poller-fixed")
    else:
        # fallback: HTML meta refresh
        st.markdown(
            f"""<meta http-equiv="refresh" content="{int(refresh_s)}">""",
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Auto refresh (fallback): using browser meta refresh (no extra package).")


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="CNC Industrial Dashboard", layout="wide", initial_sidebar_state="expanded")
    apply_industrial_css()
    init_state()

    # -------- Sidebar (all widgets have UNIQUE keys)
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("### Control Panel")
    st.sidebar.caption("API + thresholds • SCADA mode")

    api_base = st.sidebar.text_input(
        "API Gateway Base URL",
        value=DEFAULT_API_BASE,
        key="sb_api_base",
    )
    refresh_s = st.sidebar.number_input(
        "Refresh interval (seconds)",
        min_value=2, max_value=60,
        value=DEFAULT_REFRESH_SECONDS,
        step=1,
        key="sb_refresh_s",
    )
    max_points = st.sidebar.number_input(
        "Max points to keep (rolling buffer)",
        min_value=100, max_value=5000,
        value=MAX_POINTS_DEFAULT,
        step=50,
        key="sb_max_points",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Thresholds")
    th = Thresholds(
        anomaly_severity_warning=st.sidebar.number_input(
            "Anomaly severity WARNING", 0.0, 10.0, 0.05, 0.01, key="th_anom_warn"
        ),
        anomaly_severity_critical=st.sidebar.number_input(
            "Anomaly severity CRITICAL", 0.0, 10.0, 0.15, 0.01, key="th_anom_crit"
        ),
        failure_prob_warning=st.sidebar.number_input(
            "Failure prob WARNING", 0.0, 1.0, 0.30, 0.05, key="th_fail_warn"
        ),
        failure_prob_critical=st.sidebar.number_input(
            "Failure prob CRITICAL", 0.0, 1.0, 0.50, 0.05, key="th_fail_crit"
        ),
        rul_warning=st.sidebar.number_input(
            "RUL WARNING (low)", 0.0, 100000.0, 40.0, 5.0, key="th_rul_warn"
        ),
        rul_critical=st.sidebar.number_input(
            "RUL CRITICAL (low)", 0.0, 100000.0, 20.0, 5.0, key="th_rul_crit"
        ),
    )

    st.sidebar.markdown("---")
    enable_alerts = st.sidebar.toggle("Enable alerts", value=True, key="sb_enable_alerts")
    enable_beep = st.sidebar.toggle("Enable beep", value=False, key="sb_enable_beep")  # kept for your UI
    pause_refresh = st.sidebar.toggle("Pause refresh", value=False, key="sb_pause_refresh")
    st.sidebar.caption("Tip: If refresh package missing, the app uses a safe fallback refresh.")

    # -------- Refresh
    do_refresh(int(refresh_s), pause_refresh)

    # -------- Fetch data
    data, err = fetch_latest(api_base)
    online = (err is None and data is not None)

    if online:
        device_id = data.get("device_id", "unknown")
        dt = ts_to_dt(data.get("ts", time.time()))
        append_point(data, max_points=int(max_points))
        alarm_active, alarm_items, alarm_level = should_alarm(data, th)
    else:
        device_id = "unknown"
        dt = None
        alarm_active, alarm_items, alarm_level = (False, [], "ok")

    # -------- Top status strip (FIXED)
    render_alarm_strip(
        online=online,
        alarm_active=(enable_alerts and alarm_active),
        alarm_items=alarm_items,
        level=alarm_level,
    )

    # -------- Header
    render_header(device_id=device_id, online=online, last_update=dt if online else None, refresh_s=int(refresh_s))

    # If offline and no cached data
    df = st.session_state.df
    if not online and len(df) == 0:
        st.error(f"API is offline and no cached data yet.\n\nError: {err}")
        st.stop()

    # -------- Sensors
    render_sensor_cards(df)
    render_sensor_charts(df)

    # -------- Models (keep as your good images)
    if online:
        render_models(data, th)

        if enable_alerts and alarm_active and dt is not None:
            add_alerts(dt, alarm_items)

        render_alert_log()
    else:
        st.info("Models/alerts paused because API is offline.")
        st.caption(f"API error: {err}")

    st.caption("Data source: API Gateway `/latest` • Refresh: automatic")


if __name__ == "__main__":
    main()
