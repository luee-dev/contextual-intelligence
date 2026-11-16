
import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from contextual_intelligence.classification.train import (
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression.joblib"
REPORT_DIR = ROOT_DIR / "reports" / "metrics"

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_model():
    """Load the trained classification pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return load(MODEL_PATH)


def prepare_validation_data():
    """Recreate the original stratified validation split."""

    df = load_training_data()

    X = df["text"]
    y = df["label"]

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_val.reset_index(drop=True), y_val.reset_index(drop=True)


def analyze_predictions(model, X_val, y_val):
    """Compare predictions against actual labels."""

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    results = pd.DataFrame({
        "text": X_val,
        "actual_label": y_val,
        "predicted_label": predictions,
        "prediction_confidence": probabilities.max(axis=1),
    })

    results["actual_category"] = (
        results["actual_label"].map(LABEL_NAMES)
    )

    results["predicted_category"] = (
        results["predicted_label"].map(LABEL_NAMES)
    )

    results["correct_prediction"] = (
        results["actual_label"] == results["predicted_label"]
    )

    return results


def save_error_analysis(results):
    """Save misclassified records and error summaries."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    errors = results.loc[
        ~results["correct_prediction"]
    ].copy()

    error_pairs = (
        errors.groupby(
            ["actual_category", "predicted_category"]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    category_summary = (
        results.groupby("actual_category")
        .agg(
            total_records=("correct_prediction", "size"),
            correct_predictions=("correct_prediction", "sum"),
        )
    )

    category_summary["errors"] = (
        category_summary["total_records"]
        - category_summary["correct_predictions"]
    )

    category_summary["error_rate"] = (
        category_summary["errors"]
        / category_summary["total_records"]
    )

    summary = {
        "validation_records": int(len(results)),
        "correct_predictions": int(
            results["correct_prediction"].sum()
        ),
        "misclassified_records": int(len(errors)),
        "overall_error_rate": float(
            len(errors) / len(results)
        ),
        "high_confidence_errors": int(
            (errors["prediction_confidence"] >= 0.80).sum()
        ),
        "errors_by_actual_category": (
            category_summary["errors"].astype(int).to_dict()
        ),
        "error_rate_by_actual_category": (
            category_summary["error_rate"].round(4).to_dict()
        ),
    }

    errors = errors.sort_values(
        "prediction_confidence",
        ascending=False,
    )

    errors_path = REPORT_DIR / "misclassified_records.csv"
    pairs_path = REPORT_DIR / "confusion_pairs.csv"
    summary_path = REPORT_DIR / "error_analysis_summary.json"

    errors.to_csv(errors_path, index=False)
    error_pairs.to_csv(pairs_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"Validation records: {summary['validation_records']:,}")
    print(f"Correct predictions: {summary['correct_predictions']:,}")
    print(f"Misclassified records: {summary['misclassified_records']:,}")
    print(f"Overall error rate: {summary['overall_error_rate']:.2%}")
    print(
        "High-confidence errors (80% or above): "
        f"{summary['high_confidence_errors']:,}"
    )

    print("\n--- ERRORS BY ACTUAL CATEGORY ---")
    print(category_summary.round(4))

    print("\n--- MOST COMMON CLASSIFICATION ERRORS ---")
    print(error_pairs.head(10).to_string(index=False))

    print("\n--- OUTPUT FILES ---")
    print(f"Misclassified records: {errors_path}")
    print(f"Confusion pairs: {pairs_path}")
    print(f"Summary: {summary_path}")


def main():
    """Run baseline error analysis."""

    print("\n--- LOADING MODEL ---")
    model = load_model()

    print("\n--- PREPARING VALIDATION DATA ---")
    X_val, y_val = prepare_validation_data()
    print(f"Validation records: {len(X_val):,}")

    print("\n--- ANALYZING PREDICTIONS ---")
    results = analyze_predictions(model, X_val, y_val)

    save_error_analysis(results)

    print("\n--- ERROR ANALYSIS COMPLETED ---")


if __name__ == "__main__":
    main()
