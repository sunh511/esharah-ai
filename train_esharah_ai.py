"""Train the first measured-data speed predictor for ESHARAH AI.

Place beside app.py and run: python train_esharah_ai.py
Only reads data/*.csv. Does not alter app.py or source files.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree
from sklearn.compose import TransformedTargetRegressor


# =====================================
# 1. Project paths
# =====================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "models"

OOKLA_FILE = DATA / "ESHARAH_Ookla_Saudi_2024_Q4.csv"
COVERAGE_FILE = DATA / "E-2.csv"
CELLS_FILE = DATA / "ksa_network_data.csv"

EARTH_RADIUS_KM = 6371.0


# =====================================
# 2. AI model features
# =====================================

FEATURES = [
    "latitude",
    "longitude",
    "distance_to_study_cities_km",
    "coverage_probability_2024",
    "coverage_grid_nearby",
    "distance_to_known_cell_km",
    "cell_records_within_15km",
]


# =====================================
# 3. Study locations
# =====================================

STUDY_CITIES = np.array([
    [22.4298749, 41.7602315],  # New Muwayh
    [21.90807, 42.03179],       # Khurma
    [22.72337, 42.16871],      # Dhulm
])


# =====================================
# 4. Load real Ookla measurements
# =====================================

def load_ookla():

    if not OOKLA_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {OOKLA_FILE}"
        )

    df = pd.read_csv(
        OOKLA_FILE,
        low_memory=False,
        dtype={"quadkey": str}
    )

    needed = [
        "latitude",
        "longitude",
        "avg_download_mbps"
    ]

    for column in needed:

        if column not in df.columns:
            raise ValueError(
                f"Ookla column missing: {column}"
            )

    for column in needed:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=needed
    )

    df = df[
        df["latitude"].between(16, 33)
        & df["longitude"].between(34, 56)
        & df["avg_download_mbps"].between(0.01, 1500)
    ].copy()

    if "period" in df.columns:

        df = df[
            df["period"].astype(str).eq("2024-Q4")
        ]

    if "quadkey" in df.columns:

        df = df.drop_duplicates(
            subset=["quadkey"],
            keep="first"
        )

    if len(df) < 100:

        raise ValueError(
            f"Only {len(df)} usable measurements. "
            "Need at least 100."
        )

    return df.reset_index(drop=True)


# =====================================
# 5. Distance to study cities
# =====================================

def add_city_distance(df):

    tree = BallTree(
        np.radians(STUDY_CITIES),
        metric="haversine"
    )

    distances, _ = tree.query(
        np.radians(
            df[["latitude", "longitude"]].to_numpy()
        ),
        k=1
    )

    df["distance_to_study_cities_km"] = (
        distances[:, 0] * EARTH_RADIUS_KM
    )

    return df


# =====================================
# 6. Add Zenodo coverage features
# =====================================

def add_coverage_features(df):

    df["coverage_probability_2024"] = np.nan

    df["coverage_grid_nearby"] = 0

    if not COVERAGE_FILE.exists():

        print(
            "E-2.csv not found: "
            "training without 2024 coverage grid."
        )

        return df

    columns = [
        "latitude",
        "longitude",
        "probability_2024"
    ]

    coverage = pd.read_csv(
        COVERAGE_FILE,
        usecols=columns,
        low_memory=False
    )

    for col in columns:

        coverage[col] = pd.to_numeric(
            coverage[col],
            errors="coerce"
        )

    coverage = coverage.dropna(
        subset=columns
    )

    coverage = coverage[
        coverage["latitude"].between(-90, 90)
        & coverage["longitude"].between(-180, 180)
        & coverage["probability_2024"].between(0, 1)
    ]

    if coverage.empty:
        return df

    tree = BallTree(
        np.radians(
            coverage[
                ["latitude", "longitude"]
            ].to_numpy()
        ),
        metric="haversine"
    )

    distances, indices = tree.query(
        np.radians(
            df[
                ["latitude", "longitude"]
            ].to_numpy()
        ),
        k=1
    )

    close = (
        distances[:, 0] * EARTH_RADIUS_KM <= 2.0
    )

    values = coverage[
        "probability_2024"
    ].to_numpy()

    df.loc[
        close,
        "coverage_probability_2024"
    ] = values[
        indices[close, 0]
    ]

    df.loc[
        close,
        "coverage_grid_nearby"
    ] = 1

    print(
        f"2024 coverage grid near "
        f"{close.sum():,} Ookla points."
    )

    return df


# =====================================
# 7. Add OpenCellID cell features
# =====================================

def add_cell_features(df):

    df["distance_to_known_cell_km"] = np.nan

    df["cell_records_within_15km"] = 0

    if not CELLS_FILE.exists():

        print(
            "ksa_network_data.csv not found: "
            "training without cell features."
        )

        return df

    usecols = [
        "lat",
        "lon",
        "radio",
        "mcc",
        "created"
    ]

    cells = pd.read_csv(
        CELLS_FILE,
        usecols=usecols,
        low_memory=False
    )

    for col in [
        "lat",
        "lon",
        "mcc",
        "created"
    ]:

        cells[col] = pd.to_numeric(
            cells[col],
            errors="coerce"
        )

    # Include LTE / NR records first registered before 2025.
    # Coordinates may have been updated after the registration date.

    valid = (
        cells["mcc"].eq(420)
        & cells["radio"].astype(str)
        .str.upper()
        .isin(["LTE", "NR"])
        & cells["lat"].between(16, 33)
        & cells["lon"].between(34, 56)
        & cells["created"].between(
            1,
            1735689599
        )
    )

    cells = cells.loc[
        valid,
        ["lat", "lon"]
    ].drop_duplicates()

    if cells.empty:
        return df

    tree = BallTree(
        np.radians(
            cells.to_numpy()
        ),
        metric="haversine"
    )

    coords = np.radians(
        df[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    near, _ = tree.query(
        coords,
        k=1
    )

    counts = tree.query_radius(
        coords,
        r=15.0 / EARTH_RADIUS_KM,
        count_only=True
    )

    df["distance_to_known_cell_km"] = np.minimum(
        near[:, 0] * EARTH_RADIUS_KM,
        500
    )

    df["cell_records_within_15km"] = (
        counts.astype(int)
    )

    print(
        f"4G/5G cell positions used: "
        f"{len(cells):,} "
        "(approximate records)."
    )

    return df


# =====================================
# 8. Prepare AI features
# =====================================

def prepare_features(df):

    df = df.copy()

    df = add_city_distance(df)

    df = add_coverage_features(df)

    df = add_cell_features(df)

    return df[FEATURES].copy()


# =====================================
# 9. Train and evaluate AI model
# =====================================

def main():

    print(
        "=== ESHARAH AI: MEASURED-DATA TRAINING ==="
    )

    # Load real Ookla measurements
    df = load_ookla()

    print(
        f"Unique 2024-Q4 Ookla tiles: {len(df):,}"
    )

    # Combine geographic data
    X = prepare_features(df)

    # Target: observed download speed
    y = df[
        "avg_download_mbps"
    ].to_numpy(dtype=float)

    # Create geographic blocks for evaluation
    block_lat = np.floor(
        df["latitude"].to_numpy() * 2
    ).astype(int)

    block_lon = np.floor(
        df["longitude"].to_numpy() * 2
    ).astype(int)

    groups = pd.Series(
        block_lat.astype(str)
        + "_"
        + block_lon.astype(str)
    )

    if groups.nunique() < 8:

        raise ValueError(
            "Not enough different geographic "
            "blocks for an honest split."
        )

    # Separate training and testing regions
    train_index, test_index = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42
        ).split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    # Simple baseline for comparison
    baseline = DummyRegressor(
        strategy="median"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )

    # Real machine learning model
    estimator = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=180,
            max_leaf_nodes=25,
            min_samples_leaf=25,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=42
        ),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True
    )

    # Train AI
    print("\nTraining AI model...")

    estimator.fit(
        X_train,
        y_train
    )

    # Predict unseen geographic data
    predicted = np.maximum(
        0,
        estimator.predict(X_test)
    )

    # Evaluate predictions
    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    model_mae = mean_absolute_error(
        y_test,
        predicted
    )

    model_r2 = r2_score(
        y_test,
        predicted
    )

    improve = baseline_mae - model_mae

    # Generate training report
    lines = [
        "ESHARAH AI - Model evaluation (2024-Q4)",

        "Target: observed Ookla average download speed (Mbps)",

        "Coverage feature: 2024 modeled grid, not a measured speed",

        "Cells: approximate cell records from a newer export; historical bias possible",

        "Validation: geographically separated 0.5-degree blocks",

        f"Measurements total: {len(df):,}",

        f"Train rows: {len(train_index):,}",

        f"Test rows: {len(test_index):,}",

        f"Train geographic blocks: "
        f"{groups.iloc[train_index].nunique():,}",

        f"Test geographic blocks: "
        f"{groups.iloc[test_index].nunique():,}",

        f"Median baseline MAE: "
        f"{baseline_mae:.2f} Mbps",

        f"ML model MAE: "
        f"{model_mae:.2f} Mbps",

        f"ML model R2: "
        f"{model_r2:.3f}",

        f"MAE improvement over baseline: "
        f"{improve:+.2f} Mbps",

        "Quality check: "
        + (
            "model beats median baseline"
            if model_mae < baseline_mae
            else "model does NOT beat median baseline; do not deploy"
        ),

        "Deployment status: NOT CONNECTED TO app.py yet"
    ]

    # Create models folder
    OUTPUT.mkdir(
        exist_ok=True
    )

    # Save training report
    report_file = OUTPUT / "training_report.txt"

    report_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    # Save trained AI model
    joblib.dump(
        {
            "model": estimator,
            "features": FEATURES,
            "trained_period": "2024-Q4",
            "evaluation_mae_mbps": model_mae,
            "coverage_year": 2024
        },
        OUTPUT / "esharah_speed_model.joblib"
    )

    # Show results
    print("\n" + "\n".join(lines))

    print(
        f"\nSaved: {report_file}"
    )

    print(
        "Finished. Existing app.py has not been changed."
    )


# =====================================
# 10. Run training
# =====================================

if __name__ == "__main__":
    main()