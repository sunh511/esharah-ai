"""ESHARAH AI: experimental road-segment low-speed risk preview.

Place next to app.py and train_esharah_ai.py, then run:
    python predict_road_segments.py

Reads the locally trained model and 2024 data features, and requests road
geometry from public OSRM. Does not change app.py or source datasets.
"""

from pathlib import Path
import math

import folium
import joblib
import numpy as np
import pandas as pd
import requests

from train_esharah_ai import FEATURES, prepare_features

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "weak_speed_risk_model.joblib"
OUTPUT_FILE = ROOT / "models" / "esharah_road_risk_preview.html"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
STEP_KM = 5.0
EARTH_RADIUS_KM = 6371.0

# Approximate city centers. Actual road geometry comes from OSRM.
PLACES = {
    "المويه الجديدة": (22.4298749, 41.7602315),
    "الخرمة": (21.90807, 42.03179),
    "ظلم": (22.72337, 42.16871),
}

ROUTES = [
    ("المويه الجديدة", "الخرمة"),
    ("الخرمة", "ظلم"),
    ("ظلم", "المويه الجديدة"),
]


def distance_km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def get_road_geometry(start, end):
    lat1, lon1 = PLACES[start]
    lat2, lon2 = PLACES[end]
    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"

    response = requests.get(
        url,
        params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        },
        timeout=35,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("code") != "Ok" or not result.get("routes"):
        raise RuntimeError(
            f"OSRM returned no valid road: {result.get('code')}"
        )

    coordinates = result["routes"][0]["geometry"]["coordinates"]
    if len(coordinates) < 2:
        raise RuntimeError("OSRM returned an empty road geometry")

    return [
        (float(lat), float(lon))
        for lon, lat in coordinates
    ]


def split_into_segments(points, target_km=STEP_KM):
    """Split a detailed road polyline into ~5 km continuous pieces."""
    segments = []
    current = [points[0]]
    length_current = 0.0

    for destination in points[1:]:
        origin = current[-1]
        remaining_edge = distance_km(origin, destination)

        if remaining_edge < 1e-9:
            continue

        while length_current + remaining_edge >= target_km - 1e-9:
            need = target_km - length_current
            fraction = max(
                0.0,
                min(1.0, need / remaining_edge)
            )
            pivot = (
                origin[0] + (destination[0] - origin[0]) * fraction,
                origin[1] + (destination[1] - origin[1]) * fraction,
            )

            if distance_km(current[-1], pivot) > 1e-9:
                current.append(pivot)

            if len(current) >= 2:
                segments.append(current)

            current = [pivot]
            origin = pivot
            length_current = 0.0
            remaining_edge = distance_km(origin, destination)

            if remaining_edge < 1e-9:
                break

        if remaining_edge > 1e-9:
            current.append(destination)
            length_current += remaining_edge

    if (
        len(current) >= 2
        and distance_km(current[0], current[-1]) > 1e-8
    ):
        segments.append(current)

    return segments


def middle_point(points):
    """Find a point at half the segment distance."""
    edge_lengths = [
        distance_km(a, b)
        for a, b in zip(points[:-1], points[1:])
    ]
    half = sum(edge_lengths) / 2.0
    walked = 0.0

    for a, b, length in zip(
        points[:-1],
        points[1:],
        edge_lengths
    ):
        if walked + length >= half and length > 0:
            fraction = (half - walked) / length
            return (
                a[0] + fraction * (b[0] - a[0]),
                a[1] + fraction * (b[1] - a[1]),
            )
        walked += length

    return points[-1]


def risk_color(score):
    # Provisional risk bands, NOT measured coverage.
    if score >= 0.30:
        return "#d64545"

    if score >= 0.15:
        return "#e4a328"

    return "#258867"


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing trained model. "
            "Run: python train_weak_signal.py"
        )

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    expected_features = list(bundle["features"])

    if expected_features != list(FEATURES):
        raise ValueError(
            "Training features have changed. "
            "Retrain before preview."
        )

    route_data = []
    segment_rows = []

    for start, end in ROUTES:
        label = f"{start} إلى {end}"
        print(f"Fetching road: {label}")

        try:
            geometry = get_road_geometry(start, end)
        except (
            requests.RequestException,
            ValueError,
            RuntimeError
        ) as exc:
            print(f"Unable to fetch real road: {exc}")
            continue

        parts = split_into_segments(geometry)

        if not parts:
            print("No road segments could be generated")
            continue

        route_data.append((label, parts))

        for number, part in enumerate(parts, 1):
            lat, lon = middle_point(part)

            segment_rows.append({
                "route": label,
                "segment": number,
                "latitude": lat,
                "longitude": lon,
                "path": part,
            })

        print(f"Segments: {len(parts)}")

    if not segment_rows:
        print(
            "No roads retrieved. "
            "Check your internet connection and retry."
        )
        return

    # Prepare the same features used during training.
    points_df = pd.DataFrame(segment_rows)
    X = prepare_features(points_df)

    scores = np.asarray(
        model.predict_proba(
            X[expected_features]
        )
    )[:, 1]

    view = folium.Map(
        location=[22.36, 42.0],
        zoom_start=9,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    for place, (lat, lon) in PLACES.items():
        folium.Marker(
            [lat, lon],
            tooltip=place,
            popup=place,
            icon=folium.Icon(
                color="blue",
                icon="info-sign"
            ),
        ).add_to(view)

    all_points = []

    for index, row in points_df.iterrows():
        score = float(scores[index])
        color = risk_color(score)

        has_coverage = bool(
            X.iloc[index]["coverage_grid_nearby"]
        )

        near_cells = int(
            X.iloc[index]["cell_records_within_15km"]
        )

        label = (
            f"{row['route']} | "
            f"مقطع {row['segment']} | "
            f"مؤشر النموذج: {score:.0%} | "
            f"تقدير تغطية 2024: "
            f"{'متاح' if has_coverage else 'غير متاح'} | "
            f"سجلات خلايا ضمن 15 كم: {near_cells}"
        )

        folium.PolyLine(
            locations=row["path"],
            color=color,
            weight=7,
            opacity=0.90,
            tooltip=label,
        ).add_to(view)

        all_points.extend(row["path"])

    legend = """
    <div style="
        position:fixed;
        bottom:18px;
        right:18px;
        z-index:9999;
        background:white;
        border:1px solid #bbb;
        border-radius:10px;
        padding:12px;
        font:14px Arial;
        direction:rtl;
        max-width:280px;
        box-shadow:0 2px 8px #7775;
    ">
        <b>إشارة | خريطة تجريبية</b><br>

        <span style="color:#d64545">●</span>
        خطر سرعة منخفضة أعلى (≥30%)<br>

        <span style="color:#e4a328">●</span>
        متوسط (15% إلى أقل من 30%)<br>

        <span style="color:#258867">●</span>
        أقل من 15%<br>

        <small>
        الألوان تعكس مخرجات نموذج تعلم آلي
        مدرب على Ookla 2024، وليست قياس
        تغطية حاليًا أو تأكيدًا لانقطاع الشبكة.
        القيم غير معايرة كاحتمالات ميدانية موثوقة.
        </small>
    </div>
    """

    view.get_root().html.add_child(
        folium.Element(legend)
    )

    view.fit_bounds(all_points)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    view.save(str(OUTPUT_FILE))

    print(f"\nFinished. Preview: {OUTPUT_FILE}")
    print(f"Segments predicted: {len(points_df)}")

    print(
        "Segments with 2024 coverage feature: "
        f"{int(X['coverage_grid_nearby'].sum())}"
    )

    print(
        "Experimental prediction only. "
        "app.py and data were not changed."
    )


if __name__ == "__main__":
    main()