import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split

from contextual_intelligence.classification.train import (
    DATA_PATH,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression.joblib"
REPORT_DIR = ROOT_DIR / "reports" / "metrics"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

LABELS = [0, 1, 2, 3]

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]

def load_model():
    """Load the saved TF-IDF and Logistic Regression pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return load(MODEL_PATH)

def prepare_validation_data():
    """
    Recreate the same stratified validation split used

    The same random state and split size ensure that the validation
    records are identical to those used when trained the model.
    """

    df = load_training_data()

    X = df["text"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_val, y_val

def evaluate_model(model, X_val, y_val):
    """Generate classification metrics and predictions."""

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    report = classification_report(
        y_val,
        predictions,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=LABELS,
    )

    return predictions, accuracy, report, matrix


def save_metrics(accuracy, report):
    """Save evaluation metrics in JSON and CSV formats."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            class_name: report[class_name]
            for class_name in CLASS_NAMES
        },
    }

    json_path = REPORT_DIR / "baseline_evaluation.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    class_metrics = {
        name: values
        for name, values in report.items()
        if isinstance(values, dict)
    }

    report_df = pd.DataFrame.from_dict(
        class_metrics,
        orient="index",
    )

    csv_path = REPORT_DIR / "classification_report.csv"

    report_df.to_csv(csv_path, index_label="category")

    print("\n--- CLASSIFICATION REPORT ---")
    print(report_df.round(4))

    print("\n--- OVERALL PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")

    print(f"\nMetrics saved to: {json_path}")
    print(f"Classification report saved to: {csv_path}")


def save_confusion_matrix(matrix):
    """Generate and save the confusion matrix visualization."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Baseline Model - Confusion Matrix")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

    plt.tight_layout()

    figure_path = FIGURE_DIR / "baseline_confusion_matrix.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Confusion matrix saved to: {figure_path}")


def main():
    """Run the complete baseline evaluation."""

    print("\n--- LOADING MODEL ---")
    model = load_model()

    print("\n--- PREPARING VALIDATION DATA ---")
    X_val, y_val = prepare_validation_data()

    print(f"Validation records: {len(X_val):,}")

    print("\n--- EVALUATING MODEL ---")

    predictions, accuracy, report, matrix = evaluate_model(
        model,
        X_val,
        y_val,
    )

    save_metrics(accuracy, report)
    save_confusion_matrix(matrix)

    print("\n--- EVALUATION COMPLETED ---")


if __name__ == "__main__":
    main()
