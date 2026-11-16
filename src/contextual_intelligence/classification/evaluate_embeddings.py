
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from contextual_intelligence.classification.train import (
    RANDOM_STATE,
    VALIDATION_SIZE,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

EMBEDDING_DIR = ROOT_DIR / "models" / "embeddings"
MODEL_PATH = ROOT_DIR / "models" / "embedding_logistic_regression.joblib"

METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

EMBEDDING_PATH = EMBEDDING_DIR / "ag_news_embeddings.npy"
LABELS_PATH = EMBEDDING_DIR / "ag_news_labels.npy"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_data():
    """Load embeddings, labels and trained model."""

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABELS_PATH)
    model = joblib.load(MODEL_PATH)

    return embeddings, labels, model


def evaluate_model(model, X_val, y_val):
    """Generate classification metrics and predictions."""

    print("\n--- EVALUATING EMBEDDING CLASSIFIER ---")

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    report = classification_report(
        y_val,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
    )

    print("\n--- CLASSIFICATION REPORT ---")

    report_df = pd.DataFrame(
        {
            name: values
            for name, values in report.items()
            if isinstance(values, dict)
        }
    ).T

    print(report_df.round(4))

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-score: {report['weighted avg']['f1-score']:.4f}")

    return predictions, accuracy, report, matrix, report_df


def save_results(accuracy, report, matrix, report_df):
    """Save evaluation metrics and confusion matrix."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Logistic Regression with Sentence Embeddings",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "validation_records": int(matrix.sum()),
        "accuracy": float(accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
    }

    with open(
        METRICS_DIR / "embedding_evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    report_df.to_csv(
        METRICS_DIR / "embedding_classification_report.csv"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
    )

    ax.set_title("Sentence Embedding Classifier - Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "embedding_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n--- SAVING COMPLETED ---")
    print("Evaluation: reports/metrics/embedding_evaluation.json")
    print("Report: reports/metrics/embedding_classification_report.csv")
    print("Confusion matrix: reports/figures/embedding_confusion_matrix.png")


def main():
    embeddings, labels, model = load_data()

    print("\n--- CREATING VALIDATION SPLIT ---")

    _, X_val, _, y_val = train_test_split(
        embeddings,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Validation records: {len(X_val):,}")

    _, accuracy, report, matrix, report_df = evaluate_model(
        model,
        X_val,
        y_val,
    )

    save_results(
        accuracy,
        report,
        matrix,
        report_df,
    )

    print("\n--- EMBEDDING MODEL EVALUATION COMPLETED ---")


if __name__ == "__main__":
    main()
