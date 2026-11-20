"""Train a low-speed risk classifier for ESHARAH AI."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit

from train_esharah_ai import (
    load_ookla,
    prepare_features,
    FEATURES,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "models"

LOW_SPEED_MBPS = 15.0


def evaluate_threshold(y_true, probability, threshold):

    predicted = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1]
    ).ravel()

    return [
        f"Threshold: {threshold:.2f}",
        f"  Detected low-speed: {tp}",
        f"  Missed low-speed: {fn}",
        f"  False alerts: {fp}",
        f"  Correct normal-speed: {tn}",
        f"  Precision: {precision_score(y_true, predicted, zero_division=0):.3f}",
        f"  Recall: {recall_score(y_true, predicted, zero_division=0):.3f}",
        f"  F1: {f1_score(y_true, predicted, zero_division=0):.3f}",
    ]


def main():

    print("=== ESHARAH AI: LOW-SPEED RISK MODEL ===")

    # 1. Load real Ookla data
    df = load_ookla()

    print(f"Ookla records: {len(df):,}")

    # A low-speed record is 15 Mbps or less
    y = (
        df["avg_download_mbps"].to_numpy()
        <= LOW_SPEED_MBPS
    ).astype(int)

    print(
        f"Low-speed records: {y.sum():,} "
        f"({100 * y.mean():.2f}%)"
    )

    # 2. Combine geographic, coverage and cell features
    X = prepare_features(df)

    # 3. Separate geographic blocks for testing
    lat_block = np.floor(
        df["latitude"].to_numpy() * 2
    ).astype(int)

    lon_block = np.floor(
        df["longitude"].to_numpy() * 2
    ).astype(int)

    groups = pd.Series(
        lat_block.astype(str)
        + "_"
        + lon_block.astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    if (
        len(np.unique(y_train)) < 2
        or len(np.unique(y_test)) < 2
    ):
        raise ValueError(
            "Both speed classes are required "
            "in training and testing."
        )

    # 4. Train AI classifier
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=20,
        min_samples_leaf=40,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )

    print("\nTraining AI model...")

    model.fit(X_train, y_train)

    # 5. Predict probabilities
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # 6. Evaluate performance
    prevalence = y_test.mean()

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    # 7. Create report
    lines = [
        "ESHARAH AI - Low-speed prediction report",
        "",
        "Training period: 2024-Q4",
        "Target: Ookla download speed <= 15 Mbps",
        "Note: Low speed does not prove disconnection.",
        "Features: location, 2024 coverage, cell records",
        "Cell data export is newer than the Ookla period.",
        "Evaluation: geographically separated blocks",
        "",
        f"Total records: {len(df):,}",
        f"Training records: {len(train_idx):,}",
        f"Testing records: {len(test_idx):,}",
        f"Training low-speed share: {y_train.mean():.2%}",
        f"Testing low-speed share: {prevalence:.2%}",
        "",
        f"Random-ranking AP baseline: {prevalence:.3f}",
        f"Model Average Precision: {average_precision:.3f}",
        f"Model ROC-AUC: {roc_auc:.3f}",
        f"Model Brier Score: {brier:.3f}",
        "",
    ]

    lines.extend(
        evaluate_threshold(
            y_test,
            probabilities,
            0.30
        )
    )

    lines.append("")

    lines.extend(
        evaluate_threshold(
            y_test,
            probabilities,
            0.50
        )
    )

    lines.extend([
        "",
        "Status: EXPERIMENTAL",
        "Not connected to app.py yet.",
        "Road predictions require independent validation.",
    ])

    # 8. Save results
    OUTPUT.mkdir(exist_ok=True)

    report_path = OUTPUT / "weak_speed_report.txt"

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    model_path = OUTPUT / "weak_speed_risk_model.joblib"

    joblib.dump(
        {
            "model": model,
            "features": list(FEATURES),
            "threshold_mbps": LOW_SPEED_MBPS,
            "period": "2024-Q4",
        },
        model_path
    )

    print("\n" + "\n".join(lines))

    print(f"\nReport saved: {report_path}")

    print("Training finished successfully.")
    print("app.py and original data were not changed.")


if __name__ == "__main__":
    main()