import math
from pathlib import Path
from textwrap import dedent

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from streamlit_folium import st_folium

from semantic_search import render_qa_assistant


st.set_page_config(page_title="إشارة | ESHARAH AI", layout="wide")


# محاذاة عناصر الواجهة العربية من اليمين، مع إبقاء خريطة Leaflet كما هي.
st.markdown(
    """<style>
    /* ESHARAH_RTL_V3 */
    [data-testid="stMainBlockContainer"] { direction: rtl !important; }
    [data-testid="stAppViewContainer"] .stMainBlockContainer h1,
    [data-testid="stAppViewContainer"] .stMainBlockContainer h2,
    [data-testid="stAppViewContainer"] .stMainBlockContainer h3,
    [data-testid="stAppViewContainer"] .stMainBlockContainer h4,
    [data-testid="stAppViewContainer"] .stMainBlockContainer [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] .stMainBlockContainer [data-testid="stMarkdownContainer"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stSelectbox"],
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stCheckbox"],
    [data-testid="stCheckbox"] label,
    [data-testid="stTextInput"],
    [data-testid="stTextInput"] input,
    [data-testid="stPills"],
    [data-testid="stPills"] [role="group"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stCheckbox"] label {
        flex-direction: row !important;
        justify-content: flex-start !important;
    }
    [data-testid="stMetric"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        direction: rtl !important;
        text-align: right !important;
        justify-content: flex-start !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stFormSubmitButton"] {
        direction: rtl !important;
        display: flex !important;
        justify-content: flex-start !important;
    }
    [data-testid="stPills"] [role="group"] {
        justify-content: flex-start !important;
    }
    /* لا نعكس خريطة Leaflet أو أزرار تكبيرها */
    .leaflet-container { direction: ltr !important; }

    .esharah-brand {
        direction: rtl; text-align:right; display:flex;
        align-items:center; gap:12px; margin-bottom:8px;
    }
    .esharah-brand-name {
        font-size:clamp(1.8rem,3vw,2.6rem); font-weight:750;
    }
    .esharah-satellite {
        position:relative; display:inline-block;
        font-size:2.6rem; animation:esharah-hover 2.7s ease-in-out infinite;
    }
    .esharah-satellite::after {
        content:'◔'; position:absolute; top:-4px; right:-8px;
        color:#37c9ae; font-size:1.1rem;
        animation:esharah-signal 2s ease-in-out infinite;
    }
    @keyframes esharah-hover {
        0%,100% {transform:translateY(0) rotate(-6deg)}
        50% {transform:translateY(-5px) rotate(0deg)}
    }
    @keyframes esharah-signal {
        0%,100% {opacity:.2;transform:scale(.7)}
        50% {opacity:1;transform:scale(1.5)}
    }
    @media (prefers-reduced-motion: reduce) {
        .esharah-satellite,.esharah-satellite::after {animation:none!important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="esharah-brand" dir="rtl">'
    '<span class="esharah-satellite" aria-hidden="true">🛰️</span>'
    '<span class="esharah-brand-name">إشارة | '
    '<bdi dir="ltr">ESHARAH AI</bdi></span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption(
    "النظام الذكي لتحليل جودة تغطية شبكات الاتصالات والإنترنت على الطرق"
    " وتقدير جودة المسارات"
)

PLACES = {
    "المويه الجديدة": (22.4298749, 41.7602315),
    "الخرمة": (21.90807, 42.03179),
    "ظلم": (22.72337, 42.16871),
}

ROADS = [
    ("المويه الجديدة", "الخرمة"),
    ("الخرمة", "ظلم"),
    ("ظلم", "المويه الجديدة"),
]

# نفس مسافة المقاطع المستخدمة في المعاينة التجريبية.
SEGMENT_LENGTH_KM = 5.0
MODEL_PATH = Path(__file__).resolve().parent / "models" / "weak_speed_risk_model.joblib"


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def vectorized_haversine_km(lat1, lon1, lats2, lons2):
    R = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)
    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def weighted_route_metrics(
    points,
    lats,
    lons,
    speeds,
    latencies,
    fallback_speed,
    fallback_lat,
    max_radius_km=20.0,
):
    """حساب متوسط السرعة وزمن الاستجابة المرجحين بقرب قياسات Ookla."""
    if len(lats) == 0 or len(speeds) == 0:
        return float(fallback_speed), float(fallback_lat)

    sampled = points[::10] if len(points) >= 10 else points
    weighted_speeds = []
    weighted_lats = []
    weights = []

    for pt in sampled:
        dists = vectorized_haversine_km(pt[0], pt[1], lats, lons)
        mask = dists <= max_radius_km
        if not np.any(mask):
            continue
        close_dists = dists[mask]
        close_speeds = speeds[mask]
        w = 1.0 / np.maximum(close_dists, 0.25)
        w_sum = float(np.sum(w))
        pt_speed = float(np.sum(close_speeds * w) / w_sum)

        if len(latencies) == len(speeds):
            close_lats = latencies[mask]
            pt_lat = float(np.sum(close_lats * w) / w_sum)
        else:
            pt_lat = float(fallback_lat)

        pt_weight = 1.0 / max(float(np.min(close_dists)), 0.5)
        weighted_speeds.append(pt_speed)
        weighted_lats.append(pt_lat)
        weights.append(pt_weight)

    if not weighted_speeds:
        return float(fallback_speed), float(fallback_lat)

    w_arr = np.array(weights, dtype=float)
    s_arr = np.array(weighted_speeds, dtype=float)
    l_arr = np.array(weighted_lats, dtype=float)
    avg_s = float(np.sum(s_arr * w_arr) / np.sum(w_arr))
    avg_l = float(np.sum(l_arr * w_arr) / np.sum(w_arr))
    return avg_s, avg_l


def find_lat_lon_columns(df):
    lat_col = next(
        (c for c in ["latitude", "lat", "Latitude", "LAT"] if c in df.columns),
        None,
    )
    lon_col = next(
        (c for c in ["longitude", "lon", "lng", "Longitude", "LON"] if c in df.columns),
        None,
    )
    return lat_col, lon_col


@st.cache_data(ttl=86400, show_spinner=False)
def get_route(start, end):
    coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    url = f"https://router.project-osrm.org/route/v1/driving/{coordinates}"

    response = requests.get(
        url, params={"overview": "full", "geometries": "geojson"}, timeout=30
    )
    response.raise_for_status()
    result = response.json()

    if result.get("code") != "Ok" or not result.get("routes"):
        raise ValueError("ما لقينا طريق")

    route = result["routes"][0]
    points = [[lat, lon] for lon, lat in route["geometry"]["coordinates"]]
    distance_km = route["distance"] / 1000
    return points, distance_km


def resolve_data_file(filename):
    candidates = [
        Path("data") / filename,
        Path(filename),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_real_metrics():
    p = resolve_data_file("target_area_metrics.csv")
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"تعذر تحميل بيانات القياسات: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ksa_cells():
    p = resolve_data_file("ksa_network_data.csv")
    if p is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        lat_col, lon_col = find_lat_lon_columns(df)
        if lat_col and lon_col:
            df = df.dropna(subset=[lat_col, lon_col]).copy()
            df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
            df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
            return df.dropna(subset=["lat", "lon"])
    except Exception as e:
        st.error(f"تعذر تحميل بيانات الخلايا: {e}")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_temporal_and_mlab_signals():
    """قراءة المؤشرات الزمنية والمرجع الإقليمي للعرض والتحليل القديم."""
    summary = {
        "grid_lats": np.array([]),
        "grid_lons": np.array([]),
        "grid_cov_latest": np.array([]),
        "grid_trend": np.array([]),
        "grid_dist_city": np.array([]),
        "regional_speed_ref": 30.4,
        "regional_latency_ref": 70.0,
        "loaded_files": [],
    }

    comp_p = resolve_data_file("E-2.csv") or resolve_data_file(
        "ESHARAH_4G_FOCUS_COMPARISON_2024_2026.csv"
    )
    f24_p = resolve_data_file("E-3.csv") or resolve_data_file(
        "ESHARAH_4G_FOCUS_2024.csv"
    )
    f25_p = resolve_data_file("E-4.csv") or resolve_data_file(
        "ESHARAH_4G_FOCUS_2025.csv"
    )
    f26_p = resolve_data_file("E-5.csv") or resolve_data_file(
        "ESHARAH_4G_FOCUS_2026 2.csv"
    )
    mlab_p = resolve_data_file("E-1.csv") or resolve_data_file(
        "ESHARAH_MLAB_Saudi_2026_09.csv"
    )

    comp_df = None
    if comp_p is not None:
        try:
            comp_df = pd.read_csv(comp_p)
            summary["loaded_files"].append(f"{comp_p.name} ({len(comp_df):,} صف)")
        except Exception:
            comp_df = None

    if comp_df is None and (f24_p is not None or f26_p is not None):
        frames = []
        for p_item in [f24_p, f25_p, f26_p]:
            if p_item is not None:
                try:
                    d = pd.read_csv(p_item)
                    summary["loaded_files"].append(f"{p_item.name} ({len(d):,} صف)")
                    frames.append(d)
                except Exception:
                    pass
        if len(frames) >= 2:
            base = frames[0][
                ["grid_id", "latitude", "longitude", "distance_to_nearest_city_km"]
            ].copy()
            p_first = pd.to_numeric(frames[0]["coverage_probability"], errors="coerce")
            p_last = pd.to_numeric(frames[-1]["coverage_probability"], errors="coerce")
            base["probability_2026"] = p_last
            base["change_2024_to_2026_pp"] = (p_last - p_first) * 100.0
            comp_df = base
        elif len(frames) == 1:
            d0 = frames[0].copy()
            d0["probability_2026"] = pd.to_numeric(
                d0["coverage_probability"], errors="coerce"
            )
            d0["change_2024_to_2026_pp"] = 0.0
            comp_df = d0

    if comp_df is not None and not comp_df.empty:
        lat_col, lon_col = find_lat_lon_columns(comp_df)
        if lat_col and lon_col:
            valid = comp_df.dropna(subset=[lat_col, lon_col]).copy()
            prob_col = (
                "probability_2026"
                if "probability_2026" in valid.columns
                else "coverage_probability"
            )
            chg_col = (
                "change_2024_to_2026_pp"
                if "change_2024_to_2026_pp" in valid.columns
                else None
            )
            summary["grid_lats"] = pd.to_numeric(
                valid[lat_col], errors="coerce"
            ).to_numpy(dtype=float)
            summary["grid_lons"] = pd.to_numeric(
                valid[lon_col], errors="coerce"
            ).to_numpy(dtype=float)
            summary["grid_cov_latest"] = (
                pd.to_numeric(valid[prob_col], errors="coerce")
                .fillna(0.18)
                .to_numpy(dtype=float)
            )
            summary["grid_trend"] = (
                pd.to_numeric(valid[chg_col], errors="coerce")
                .fillna(2.4)
                .to_numpy(dtype=float)
                if chg_col
                else np.full(len(valid), 2.4, dtype=float)
            )
            summary["grid_dist_city"] = (
                pd.to_numeric(valid["distance_to_nearest_city_km"], errors="coerce")
                .fillna(40.0)
                .to_numpy(dtype=float)
                if "distance_to_nearest_city_km" in valid.columns
                else np.full(len(valid), 40.0, dtype=float)
            )

    if mlab_p is not None:
        try:
            mlab_df = pd.read_csv(mlab_p)
            summary["loaded_files"].append(f"{mlab_p.name} ({len(mlab_df):,} صف)")
            if "subdivision1_name" in mlab_df.columns:
                mecca_df = mlab_df[
                    mlab_df["subdivision1_name"]
                    .astype(str)
                    .str.contains("Mecca", case=False, na=False)
                ]
                target_mlab = mecca_df if not mecca_df.empty else mlab_df
            else:
                target_mlab = mlab_df
            if "download_p50" in target_mlab.columns:
                val_s = pd.to_numeric(
                    target_mlab["download_p50"], errors="coerce"
                ).median()
                if pd.notna(val_s) and val_s > 0:
                    summary["regional_speed_ref"] = float(val_s)
            if "latency_p50" in target_mlab.columns:
                val_l = pd.to_numeric(
                    target_mlab["latency_p50"], errors="coerce"
                ).median()
                if pd.notna(val_l) and val_l > 0:
                    summary["regional_latency_ref"] = float(val_l)
        except Exception:
            pass

    return summary


@st.cache_data(show_spinner=False)
def load_extra_datasets():
    extra_files = [
        "ESHARAH_Ookla_Saudi_2024_Q4.csv",
        "real_towers.csv",
        "saudi_raw.csv",
        "towers.csv",
    ]
    loaded = {}
    metric_rows = []
    tower_rows = []

    for file_str in extra_files:
        p = resolve_data_file(file_str)
        if p is None:
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            loaded[p.name] = len(df)
            lat_col, lon_col = find_lat_lon_columns(df)
            if not lat_col or not lon_col:
                continue

            temp = df.copy()
            temp["lat"] = pd.to_numeric(temp[lat_col], errors="coerce")
            temp["lon"] = pd.to_numeric(temp[lon_col], errors="coerce")
            temp = temp.dropna(subset=["lat", "lon"])
            temp = temp[
                temp["lat"].between(21.5, 23.2) & temp["lon"].between(41.3, 42.6)
            ]
            if temp.empty:
                continue

            speed_col = next(
                (
                    c
                    for c in [
                        "avg_download_mbps",
                        "download_mbps",
                        "download",
                        "avg_d_kbps",
                    ]
                    if c in temp.columns
                ),
                None,
            )
            latency_col = next(
                (
                    c
                    for c in ["avg_latency_ms", "latency_ms", "latency", "avg_rtt"]
                    if c in temp.columns
                ),
                None,
            )

            if speed_col:
                speeds = pd.to_numeric(temp[speed_col], errors="coerce")
                if "kbps" in speed_col.lower():
                    speeds = speeds / 1000.0
                latencies = (
                    pd.to_numeric(temp[latency_col], errors="coerce").fillna(35.0)
                    if latency_col
                    else pd.Series(35.0, index=temp.index)
                )
                m_df = pd.DataFrame({
                    "latitude": temp["lat"],
                    "longitude": temp["lon"],
                    "avg_download_mbps": speeds,
                    "avg_latency_ms": latencies,
                }).dropna(subset=["avg_download_mbps"])
                if not m_df.empty:
                    metric_rows.append(m_df)
            else:
                t_df = pd.DataFrame({
                    "lat": temp["lat"],
                    "lon": temp["lon"],
                    "radio": temp["radio"] if "radio" in temp.columns else "Tower",
                    "net": temp["net"] if "net" in temp.columns else "-",
                    "range": temp["range"] if "range" in temp.columns else 0,
                    "samples": temp["samples"] if "samples" in temp.columns else 0,
                })
                tower_rows.append(t_df)
        except Exception:
            continue

    extra_metrics = (
        pd.concat(metric_rows, ignore_index=True)
        if metric_rows
        else pd.DataFrame()
    )
    extra_towers = (
        pd.concat(tower_rows, ignore_index=True)
        if tower_rows
        else pd.DataFrame()
    )
    return loaded, extra_metrics, extra_towers


# =========================================================
# الجديد: تحميل نموذج خطر ضعف السرعة الذي دربناه بالفعل
# =========================================================
@st.cache_resource(show_spinner=False)
def load_weak_speed_model():
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("ملف النموذج لا يحتوي على البيانات المطلوبة")
    return bundle


def point_along_route(coords, cumulative, target):
    """إحداثيات نقطة على خط الطريق عند مسافة معينة، دون تغيير شكل الطريق."""
    if target <= 0:
        return coords[0].tolist()
    if target >= cumulative[-1]:
        return coords[-1].tolist()

    index = int(np.searchsorted(cumulative, target, side="right") - 1)
    index = max(0, min(index, len(coords) - 2))
    while index < len(coords) - 2 and cumulative[index + 1] <= cumulative[index]:
        index += 1
    edge = cumulative[index + 1] - cumulative[index]
    t = float((target - cumulative[index]) / edge) if edge > 0 else 0.0
    return (coords[index] * (1 - t) + coords[index + 1] * t).tolist()


def split_road_into_segments(points, step_km=SEGMENT_LENGTH_KM):
    """يقسم شكل طريق OSRM الحقيقي إلى مقاطع طول كل منها نحو 5 كم."""
    coords = np.asarray(points, dtype=float)
    if len(coords) < 2:
        return []
    edges = vectorized_haversine_km(
        coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]
    )
    cumulative = np.concatenate([[0.0], np.cumsum(edges)])
    total = float(cumulative[-1])
    if total <= 0:
        return []

    boundaries = list(np.arange(0.0, total, step_km))
    if not boundaries or boundaries[0] != 0.0:
        boundaries.insert(0, 0.0)
    boundaries.append(total)

    segments = []
    for start_km, end_km in zip(boundaries[:-1], boundaries[1:]):
        if end_km - start_km < 0.0001:
            continue
        start_pt = point_along_route(coords, cumulative, start_km)
        end_pt = point_along_route(coords, cumulative, end_km)
        first_inner = int(np.searchsorted(cumulative, start_km, side="right"))
        last_inner = int(np.searchsorted(cumulative, end_km, side="left"))
        path = [start_pt] + coords[first_inner:last_inner].tolist() + [end_pt]
        mid_pt = point_along_route(coords, cumulative, (start_km + end_km) / 2.0)
        segments.append({
            "points": path,
            "latitude": float(mid_pt[0]),
            "longitude": float(mid_pt[1]),
            "length_km": float(end_km - start_km),
        })
    return segments


def risk_color(score):
    # نفس ألوان وحدود خريطة المعاينة التي جربتها.
    if score >= 0.30:
        return "red"
    if score >= 0.15:
        return "orange"
    return "green"


def route_label_from_mean_risk(mean_score):
    # هذا وصف تلخيصي للطريق؛ تنبيهات المقاطع لا تختفي لو كان المتوسط جيداً.
    if mean_score >= 0.30:
        return "ضعيف"
    if mean_score >= 0.20:
        return "جيد"
    if mean_score >= 0.15:
        return "جيد جداً"
    return "ممتاز"


# التصنيف القديم يبقى احتياطياً فقط إذا تعذر استخدام النموذج المدرّب.
@st.cache_resource(show_spinner=False)
def train_ai_classifier():
    X_train = np.array([
        [5.0, 95.0, 0.08, 1.0, 0.0],
        [9.0, 85.0, 0.10, 1.2, 1.0],
        [13.0, 75.0, 0.14, 1.6, 1.0],
        [20.0, 145.0, 0.18, 1.8, 1.0],
        [16.0, 65.0, 0.14, 1.8, 1.0],
        [22.0, 58.0, 0.16, 2.0, 2.0],
        [28.0, 52.0, 0.18, 2.2, 2.0],
        [34.0, 48.0, 0.16, 2.3, 2.0],
        [38.0, 38.0, 0.18, 2.4, 3.0],
        [44.0, 35.0, 0.22, 2.5, 3.0],
        [49.0, 32.0, 0.25, 2.6, 4.0],
        [55.0, 30.0, 0.18, 2.4, 2.0],
        [70.0, 25.0, 0.20, 2.6, 3.0],
        [90.0, 20.0, 0.22, 2.8, 4.0],
        [110.0, 18.0, 0.25, 3.0, 5.0],
    ])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf


# =========================================================
# البيانات الأصلية والواجهة نفسها
# =========================================================
df_metrics_base = load_real_metrics()
df_cells_base = load_ksa_cells()
temporal_signals = load_temporal_and_mlab_signals()
loaded_extra_counts, extra_metrics_df, extra_towers_df = load_extra_datasets()

df_metrics = df_metrics_base.copy()
if not df_metrics.empty and not extra_metrics_df.empty:
    df_metrics = pd.concat([df_metrics, extra_metrics_df], ignore_index=True)
elif df_metrics.empty and not extra_metrics_df.empty:
    df_metrics = extra_metrics_df.copy()

df_cells = df_cells_base.copy()
if not extra_towers_df.empty:
    if df_cells.empty:
        df_cells = extra_towers_df.copy()
    else:
        df_cells = pd.concat(
            [
                df_cells[["lat", "lon", "radio", "net", "range", "samples"]],
                extra_towers_df,
            ],
            ignore_index=True,
        )

df_cells_area = pd.DataFrame()
if not df_cells.empty:
    df_cells_area = df_cells[
        df_cells["lat"].between(21.5, 23.2) & df_cells["lon"].between(41.3, 42.6)
    ]

model_bundle = None
model_problem = None
try:
    model_bundle = load_weak_speed_model()
except Exception as error:
    model_problem = str(error)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("عدد الطرق الرئيسية", len(ROADS))
col2.metric("نقاط القياس المرصودة", len(df_metrics))
col3.metric("خلايا الشبكة في المنطقة", len(df_cells_area))
col4.metric("إجمالي خلايا الشبكة", len(df_cells))
col5.metric(
    "حالة النموذج الذكي",
    "يعمل" if model_bundle is not None else "غير متاح"
)

# موضع عرض الخريطة أعلى الإعدادات، مع تنفيذ حساباتها بعد قراءة الإعدادات.
map_section = st.container()

st.subheader("إعدادات عرض الطرق وتحليل التغطية")
route_names = [f"{a} إلى {b}" for a, b in ROADS]
selected_route = st.selectbox(
    "اختر الطريق المطلوب فحصه:", ["جميع الطرق"] + route_names
)
show_cells = st.checkbox("إظهار خلايا/أبراج الشبكة على الخريطة", value=True)
only_target_area_cells = st.checkbox(
    "تقييد الخلايا بنطاق المنطقة المستهدفة فقط", value=True
)

# موضع بطاقات الطرق بعد الإعدادات، حتى لو حُسبت البطاقات لاحقاً.
route_cards_section = st.container()

# نفس خريطة Folium الأصلية، ونفس مواقع المدن وألوان العلامات.
m = folium.Map(location=PLACES["المويه الجديدة"], zoom_start=9)

for name, coordinates in PLACES.items():
    folium.Marker(
        location=coordinates,
        tooltip=name,
        popup=name,
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

if not df_metrics.empty:
    for _, row in df_metrics.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        dl_speed = row.get("avg_download_mbps", 0)
        latency = row.get("avg_latency_ms", 0)

        if dl_speed >= 50:
            color = "green"
            status = "ممتاز"
        elif dl_speed >= 15:
            color = "orange"
            status = "متوسط"
        else:
            color = "red"
            status = "ضعيف / انقطاع محتمل"

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=(
                f"<b>سرعة التنزيل:</b> {dl_speed:.1f} Mbps<br><b>زمن الاستجابة:"
                f"</b> {latency:.1f} ms<br><b>الحالة:</b> {status}"
            ),
        ).add_to(m)

if show_cells and not df_cells.empty:
    cells_to_plot = (
        df_cells_area if only_target_area_cells else df_cells.head(500)
    )
    for _, row in cells_to_plot.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=(
                    '<div title="سجل خلية اتصال" '
                    'style="font-size:22px;text-align:center;'
                    'width:30px;height:30px">📡</div>'
                ),
                icon_size=(30, 30),
                icon_anchor=(15, 15),
            ),
            popup=(
                f"<b>التقنية:</b> {row.get('radio', '-')}<br>"
                f"<b>المشغل (net):</b> {row.get('net', '-')}<br>"
                f"<b>النطاق (range):</b> {row.get('range', 0)} m<br>"
                f"<b>العينات (samples):</b> {row.get('samples', 0)}"
            ),
        ).add_to(m)

all_points = []
route_risk_detected = False
route_summaries = []

metric_lats = (
    df_metrics["latitude"].to_numpy(dtype=float)
    if not df_metrics.empty else np.array([])
)
metric_lons = (
    df_metrics["longitude"].to_numpy(dtype=float)
    if not df_metrics.empty else np.array([])
)
metric_speeds = (
    df_metrics["avg_download_mbps"].to_numpy(dtype=float)
    if not df_metrics.empty else np.array([])
)
metric_latencies = (
    df_metrics["avg_latency_ms"].to_numpy(dtype=float)
    if (not df_metrics.empty and "avg_latency_ms" in df_metrics.columns)
    else np.array([])
)

cell_lats = (
    df_cells["lat"].to_numpy(dtype=float) if not df_cells.empty else np.array([])
)
cell_lons = (
    df_cells["lon"].to_numpy(dtype=float) if not df_cells.empty else np.array([])
)

grid_lats = temporal_signals["grid_lats"]
grid_lons = temporal_signals["grid_lons"]
grid_cov = temporal_signals["grid_cov_latest"]
grid_trend = temporal_signals["grid_trend"]
reg_speed_ref = temporal_signals["regional_speed_ref"]
reg_latency_ref = temporal_signals["regional_latency_ref"]

# نجهز الطرق بنفس الحسابات القديمة، ثم نضيف التنبؤ على مقاطعها.
calculated_routes = []
for start_name, end_name in ROADS:
    route_name = f"{start_name} إلى {end_name}"
    if selected_route != "جميع الطرق" and selected_route != route_name:
        continue

    try:
        points, distance_km = get_route(PLACES[start_name], PLACES[end_name])

        route_covs = []
        route_trends = []
        sampled_for_grid = points[::10] if len(points) >= 10 else points
        for pt in sampled_for_grid:
            if len(grid_lats) > 0:
                g_dists = vectorized_haversine_km(pt[0], pt[1], grid_lats, grid_lons)
                g_idx = int(np.argmin(g_dists))
                if g_dists[g_idx] <= 15.0:
                    route_covs.append(float(grid_cov[g_idx]))
                    route_trends.append(float(grid_trend[g_idx]))

        nearby_cells_count = 0
        if len(cell_lats) > 0:
            sampled_pts = points[::15] if len(points) >= 15 else points
            matched_indices = set()
            for pt in sampled_pts:
                dists = vectorized_haversine_km(pt[0], pt[1], cell_lats, cell_lons)
                matched_indices.update(np.where(dists <= 15.0)[0].tolist())
            nearby_cells_count = len(matched_indices)

        avg_cov = float(np.mean(route_covs)) if route_covs else 0.18
        avg_trend = float(np.mean(route_trends)) if route_trends else 2.4
        cov_factor = np.clip(avg_cov / 0.25, 0.5, 1.35)
        fallback_speed = float(reg_speed_ref * cov_factor)

        avg_route_speed, avg_route_latency = weighted_route_metrics(
            points=points,
            lats=metric_lats,
            lons=metric_lons,
            speeds=metric_speeds,
            latencies=metric_latencies,
            fallback_speed=fallback_speed,
            fallback_lat=reg_latency_ref,
            max_radius_km=20.0,
        )

        calculated_routes.append({
            "route_name": route_name,
            "points": points,
            "distance": distance_km,
            "speed": avg_route_speed,
            "latency": avg_route_latency,
            "coverage": avg_cov,
            "trend": avg_trend,
            "cells": nearby_cells_count,
            "segments": split_road_into_segments(points),
        })
        all_points.extend(points)
    except Exception as error:
        st.error(f"ما قدرنا نحمل الطريق {route_name}: {error}")

# عند توفر النموذج الجديد، نتنبأ لكل مقطع في عملية واحدة.
segment_scores = None
segment_locations = []
if model_bundle is not None and calculated_routes:
    for route in calculated_routes:
        for segment in route["segments"]:
            segment_locations.append((segment["latitude"], segment["longitude"]))

    if segment_locations:
        try:
            # نفس دالة تجهيز الخصائص المستخدمة أثناء التدريب (Ookla/Zenodo/OpenCellID).
            from train_esharah_ai import FEATURES, prepare_features

            expected_features = list(model_bundle["features"])
            if expected_features != list(FEATURES):
                raise ValueError("أسماء ميزات النموذج لا تطابق ملف التدريب")

            points_df = pd.DataFrame(
                segment_locations, columns=["latitude", "longitude"]
            )
            feature_df = prepare_features(points_df)
            segment_scores = np.asarray(
                model_bundle["model"].predict_proba(
                    feature_df[expected_features]
                )[:, 1],
                dtype=float,
            )
            if not np.all(np.isfinite(segment_scores)):
                raise ValueError("النموذج أعاد قيماً غير صالحة")
        except Exception as error:
            model_problem = str(error)
            segment_scores = None

if segment_scores is None:
    if model_problem:
        st.warning(f"تعذر عرض تنبؤ المقاطع، فُعّل العرض القديم مؤقتًا: {model_problem}")
    elif model_bundle is None:
        st.info(
            "نموذج المقاطع غير موجود في models. "
            "شغّل python train_weak_signal.py لاستخدام التنبؤ الجديد."
        )
    ai_classifier = train_ai_classifier()

# الرسم: نفس الخريطة، لكن خط الطريق يُقسم إلى مقاطع متصلة.
score_index = 0
for route in calculated_routes:
    route_name = route["route_name"]
    segments = route["segments"]
    weak_count = 0
    total_segments = len(segments)
    mean_risk = None

    if segment_scores is not None and total_segments > 0:
        road_scores = segment_scores[score_index:score_index + total_segments]
        score_index += total_segments
        weak_count = int(np.sum(road_scores >= 0.30))
        mean_risk = float(np.mean(road_scores))
        route_eval = route_label_from_mean_risk(mean_risk)
        route_risk_detected = route_risk_detected or weak_count > 0

        for number, (segment, score) in enumerate(zip(segments, road_scores), start=1):
            color = risk_color(float(score))
            tooltip = (
                f"{route_name} | المقطع {number} | "
                f"مؤشر توقع ضعف السرعة: {score:.0%}"
            )
            folium.PolyLine(
                locations=segment["points"],
                color=color,
                weight=5,
                opacity=0.85,
                tooltip=tooltip,
                popup=folium.Popup(
                    f"<div dir='rtl' style='font-size:14px'>"
                    f"<b>{route_name}</b><br>"
                    f"المقطع رقم {number}<br>"
                    f"طول المقطع: {segment['length_km']:.1f} كم<br>"
                    f"مؤشر توقع سرعة 15 Mbps أو أقل: {score:.0%}<br>"
                    f"<small>توقع من بيانات الشبكات وليس قياسًا ميدانيًا مباشرًا.</small>"
                    f"</div>",
                    max_width=300,
                ),
            ).add_to(m)
    else:
        # احتياط: نفس أسلوب الرسم والتصنيف القديم بدون الحكم اليدوي على أي طريق.
        prediction = train_ai_classifier().predict([[
            route["speed"], route["latency"], route["coverage"],
            route["trend"], float(route["cells"]),
        ]])[0]
        if prediction == 3:
            route_eval, line_color = "ممتاز", "green"
        elif prediction == 2:
            route_eval, line_color = "جيد جداً", "green"
        elif prediction == 1:
            route_eval, line_color = "جيد", "orange"
        else:
            route_eval, line_color = "ضعيف", "red"
            route_risk_detected = True
        folium.PolyLine(
            route["points"],
            color=line_color,
            weight=5,
            opacity=0.85,
            tooltip=f"{route_name} (متوسط السرعة: {route['speed']:.1f} Mbps)",
        ).add_to(m)

    route_summaries.append({
        "route_name": route_name,
        "distance": route["distance"],
        "speed": route["speed"],
        "cells": route["cells"],
        "eval": route_eval,
        # إضافات لا تؤثر في المساعد الذكي القديم، لكنها مفيدة للتوسع لاحقًا.
        "latency": route["latency"],
        "coverage": route["coverage"],
        "weak_segments": weak_count if mean_risk is not None else None,
        "total_segments": total_segments if mean_risk is not None else None,
        "mean_risk": mean_risk,
    })

    # نفس ملخص الطريق الأصلي، مع عداد المقاطع الضعيفة عند استخدام النموذج الجديد.
    # ترتيب معلومات الطريق فقط، مع إبقاء نتائج النموذج كما هي.
    with route_cards_section:
        with st.container(border=True):
            st.markdown(
                f'<h4 dir="rtl" style="text-align:right">🛣️ {route_name}</h4>',
                unsafe_allow_html=True,
            )

            info1, info2, info3, info4 = st.columns(4)

            info1.metric(
                "تقييم النموذج",
                route_eval
            )
            info2.metric(
                "المسافة",
                f"{route['distance']:.1f} كم"
            )
            info3.metric(
                "متوسط السرعة",
                f"{route['speed']:.1f} Mbps"
            )
            info4.metric(
                "الخلايا القريبة",
                route["cells"]
            )

            if mean_risk is not None:
                st.markdown(
                    f'<div dir="rtl" style="text-align:right">'
                    f'مقاطع متوقع ضعف سرعتها: '
                    f'<strong>{weak_count} من {total_segments}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

if all_points:
    m.fit_bounds(all_points)
else:
    st.warning("ما قدرنا نرسم الطريق. تأكد من الإنترنت وجرب مرة ثانية.")

# عرض الخريطة والتوضيحات في موضعها المحجوز أعلى الإعدادات.
with map_section:
    st.subheader("خريطة تحليل أداء الشبكة والتغطية الفعلية")
    st_folium(
        m,
        width="100%",
        height=550,
        returned_objects=[],
        key=f"map_{selected_route}_{show_cells}_{only_target_area_cells}",
    )

    st.markdown(
        '<div dir="rtl" style="text-align:right;line-height:1.9;font-size:.9rem">'
        '🟢 🟠 🔴 <b>دوائر القياس:</b> توضح سرعات الإنترنت المقاسة. '
        '&nbsp; 📡 <b>رمز الاتصال:</b> يوضح سجلات خلايا الشبكة.'
        '</div>',
        unsafe_allow_html=True,
    )

    if segment_scores is not None:
        st.markdown(
            '<div dir="rtl" style="text-align:right;line-height:1.9;font-size:.9rem">'
            '<b>ألوان خط الطريق:</b> '
            '🟢 أقل من 15% &nbsp;|&nbsp; '
            '🟠 من 15% إلى أقل من 30% &nbsp;|&nbsp; '
            '🔴 30% أو أكثر. '
            '<div>هذه درجات تنبؤية.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

if route_risk_detected:
    if segment_scores is not None:
        st.warning("⚠️ تنبيه ذكي: توجد مقاطع يتوقع النموذج ارتفاع خطر ضعف سرعة الإنترنت فيها.")
    else:
        st.warning("⚠️ تنبيه تجريبي: أظهر التصنيف القديم ضعفًا محتملاً على بعض المسارات.")
else:
    if segment_scores is not None:
        st.success("✅ لا توجد مقاطع تجاوزت حد التنبيه الأحمر في الطرق المحددة.")
    else:
        st.success("✅ أداء السرعات على المسارات المحددة يقع ضمن المستويات المقبولة وفق التصنيف التجريبي.")

# المساعد الذكي كما هو تمامًا، مع تمرير المعلومات المضافة إن احتاجها لاحقًا.
render_qa_assistant(route_summaries=route_summaries)

# قائمة المصادر والبيانات — أسماء الجهات الأصلية والملفات المشتقة.
st.markdown("---")
with st.expander("📚 قائمة المصادر والبيانات المستفاد منها (اضغط للعرض)"):
    # HTML مكتوب في أسطر بايثون متصلة دون مسافات بادئة قبل الوسوم،
    # لتجنب تحويله إلى كتلة كود رمادية داخل Streamlit.
    st.markdown(
        '<div dir="rtl" style="direction:rtl;text-align:right;line-height:2;">'
        '<ul dir="rtl" style="direction:rtl;text-align:right;'
        'list-style-position:outside;padding-right:1.5rem;padding-left:0;">'

        '<li><strong>قياسات سرعة الإنترنت — Ookla Open Data:</strong> '
        'ملفات <code dir="ltr">target_area_metrics.csv</code> و'
        '<code dir="ltr">ESHARAH_Ookla_Saudi_2024_Q4.csv</code> '
        'تتضمن متوسطات سرعة التنزيل وزمن الاستجابة من اختبارات منشورة. '
        'استُخدمت بيانات Ookla للربع الرابع 2024 في تدريب نموذج ضعف السرعة.</li>'

        '<li><strong>القياسات الإقليمية — Measurement Lab (M-Lab):</strong> '
        'الملف المختصر <code dir="ltr">E-1.csv</code>، '
        'واسمه الأصلي في المشروع '
        '<code dir="ltr">ESHARAH_MLAB_Saudi_2026_09.csv</code>، '
        'مع وجود ملف خاص بمنطقة مكة '
        '<code dir="ltr">ESHARAH_MLAB_Makkah_Region_2026_09.csv</code>. '
        'الجهة: <a href="https://www.measurementlab.net/data/" target="_blank">'
        'Measurement Lab</a>، وبياناتها تُستخدم للمراجع الإقليمية.</li>'

        '<li><strong>تقديرات تغطية 4G — Zenodo (مقارنة السنوات):</strong> '
        'الملف المختصر <code dir="ltr">E-2.csv</code>، '
        'واسمه الأصلي '
        '<code dir="ltr">ESHARAH_4G_FOCUS_COMPARISON_2024_2026.csv</code>. '
        'ملف معالجة جغرافية من مجموعة البيانات البحثية '
        '<em>A global mobile network coverage raster product at 1km resolution, 1999–2030</em>. '
        'يعرض تقديرات 2024 وإسقاطات 2025 و2026، '
        'وليس قياسات اتصال فعلية. '
        '<a href="https://doi.org/10.5281/zenodo.21594337" target="_blank">'
        'المصدر الأصلي على Zenodo</a>.</li>'

        '<li><strong>Zenodo — تغطية 4G لعام 2024:</strong> '
        '<code dir="ltr">E-3.csv</code>، '
        'واسمه الأصلي <code dir="ltr">ESHARAH_4G_FOCUS_2024.csv</code>، '
        'وهو تقدير جغرافي مشتق من المجموعة البحثية نفسها.</li>'

        '<li><strong>Zenodo — تغطية 4G لعام 2025:</strong> '
        '<code dir="ltr">E-4.csv</code>، '
        'واسمه الأصلي <code dir="ltr">ESHARAH_4G_FOCUS_2025.csv</code>، '
        'ويمثل إسقاطًا نموذجيًا لا قياسًا ميدانيًا مباشرًا.</li>'

        '<li><strong>Zenodo — تغطية 4G لعام 2026:</strong> '
        '<code dir="ltr">E-5.csv</code>، '
        'واسمه الأصلي في المشروع '
        '<code dir="ltr">ESHARAH_4G_FOCUS_2026 2.csv</code>، '
        'وهو إسقاط نموذجي لتغطية 4G وليس رصدًا مباشرًا.</li>'

        '<li><strong>سجلات خلايا الاتصالات — OpenCellID:</strong> '
        '<code dir="ltr">ksa_network_data.csv</code> و'
        '<code dir="ltr">real_towers.csv</code> و'
        '<code dir="ltr">saudi_raw.csv</code> و'
        '<code dir="ltr">towers.csv</code>. '
        'تعرض سجلات مواقع الخلايا المتاحة، ولا تعني بالضرورة '
        'مواقع أبراج مستقلة مؤكدة.</li>'

        '<li><strong>بيانات إضافية:</strong> '
        '<code dir="ltr">420.csv</code> موجود ضمن ملفات المشروع '
        'لكنه لا يُقرأ مباشرة في هذا الإصدار من <code dir="ltr">app.py</code>.</li>'

        '<li><strong>مسارات القيادة — Open Source Routing Machine (OSRM):</strong> '
        'يُستخدم للحصول على شكل الطريق الفعلي ومسافته بين المدن، '
        'ثم تقسيمه إلى مقاطع تقارب 5 كم.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

    all_loaded_labels = list(temporal_signals["loaded_files"]) + [
        f"{name} ({count:,} صف)" for name, count in loaded_extra_counts.items()
    ]
    if all_loaded_labels:
        st.caption(
            "الملفات المحمّلة تلقائياً في النظام: " + " | ".join(all_loaded_labels)
        )