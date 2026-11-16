
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from contextual_intelligence.classification.train import (
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

EMBEDDING_DIR = ROOT_DIR / "models" / "embeddings"
MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf_logistic_regression.joblib"
EMBEDDING_MODEL_PATH = MODEL_DIR / "embedding_logistic_regression.joblib"

EMBEDDING_PATH = EMBEDDING_DIR / "ag_news_embeddings.npy"
LABELS_PATH = EMBEDDING_DIR / "ag_news_labels.npy"

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_data_and_models():
    """Load data, embeddings and both trained classifiers."""

    print("\n--- LOADING DATA AND MODELS ---")

    df = load_training_data()

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABELS_PATH)

    if not np.array_equal(
        labels,
        df["label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Embedding labels do not match dataset labels.")

    tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    embedding_model = joblib.load(EMBEDDING_MODEL_PATH)

    print(f"Records: {len(df):,}")
    print("Both models loaded successfully.")

    return df, embeddings, labels, tfidf_model, embedding_model


def generate_predictions(
    df,
    embeddings,
    labels,
    tfidf_model,
    embedding_model,
):
    """Generate predictions using the same validation indices."""

    print("\n--- CREATING VALIDATION SPLIT ---")

    indices = np.arange(len(labels))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    X_validation_text = df["text"].iloc[validation_indices]
    X_validation_embeddings = embeddings[validation_indices]

    y_validation = labels[validation_indices]

    print(f"Validation records: {len(validation_indices):,}")

    print("\n--- GENERATING MODEL PREDICTIONS ---")

    tfidf_predictions = tfidf_model.predict(X_validation_text)
    embedding_predictions = embedding_model.predict(
        X_validation_embeddings
    )

    tfidf_probabilities = tfidf_model.predict_proba(
        X_validation_text
    )

    embedding_probabilities = embedding_model.predict_proba(
        X_validation_embeddings
    )

    results = pd.DataFrame({
        "text": df["text"].iloc[validation_indices].to_numpy(),
        "actual_label": y_validation,
        "tfidf_prediction": tfidf_predictions,
        "embedding_prediction": embedding_predictions,
        "tfidf_confidence": tfidf_probabilities.max(axis=1),
        "embedding_confidence": embedding_probabilities.max(axis=1),
    })

    return results


def create_comparison_summary(results):
    """Summarise model performance and prediction agreement."""

    tfidf_correct = (
        results["tfidf_prediction"] == results["actual_label"]
    )

    embedding_correct = (
        results["embedding_prediction"] == results["actual_label"]
    )

    summary = {
        "validation_records": int(len(results)),
        "tfidf_accuracy": float(
            accuracy_score(
                results["actual_label"],
                results["tfidf_prediction"],
            )
        ),
        "embedding_accuracy": float(
            accuracy_score(
                results["actual_label"],
                results["embedding_prediction"],
            )
        ),
        "tfidf_macro_f1": float(
            f1_score(
                results["actual_label"],
                results["tfidf_prediction"],
                average="macro",
            )
        ),
        "embedding_macro_f1": float(
            f1_score(
                results["actual_label"],
                results["embedding_prediction"],
                average="macro",
            )
        ),
        "prediction_agreement": float(
            (
                results["tfidf_prediction"]
                == results["embedding_prediction"]
            ).mean()
        ),
        "both_correct": int((tfidf_correct & embedding_correct).sum()),
        "both_wrong": int((~tfidf_correct & ~embedding_correct).sum()),
        "tfidf_only_correct": int(
            (tfidf_correct & ~embedding_correct).sum()
        ),
        "embedding_only_correct": int(
            (~tfidf_correct & embedding_correct).sum()
        ),
    }

    return summary


def create_confusion_pairs(results, model_column):
    """Return the most frequent classification error pairs."""

    errors = results[
        results[model_column] != results["actual_label"]
    ].copy()

    errors["actual_category"] = errors["actual_label"].map(
        LABEL_NAMES
    )

    errors["predicted_category"] = errors[model_column].map(
        LABEL_NAMES
    )

    pairs = (
        errors.groupby(
            ["actual_category", "predicted_category"]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    return pairs


def main():
    """Compare TF-IDF and embedding classifier predictions."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    (
        df,
        embeddings,
        labels,
        tfidf_model,
        embedding_model,
    ) = load_data_and_models()

    results = generate_predictions(
        df,
        embeddings,
        labels,
        tfidf_model,
        embedding_model,
    )

    print("\n--- CREATING COMPARISON SUMMARY ---")

    summary = create_comparison_summary(results)

    print(json.dumps(summary, indent=4))

    results["actual_category"] = results["actual_label"].map(
        LABEL_NAMES
    )

    results["tfidf_category"] = results["tfidf_prediction"].map(
        LABEL_NAMES
    )

    results["embedding_category"] = results[
        "embedding_prediction"
    ].map(LABEL_NAMES)

    results.to_csv(
        METRICS_DIR / "model_prediction_comparison.csv",
        index=False,
    )

    with open(
        METRICS_DIR / "model_comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    tfidf_pairs = create_confusion_pairs(
        results,
        "tfidf_prediction",
    )

    embedding_pairs = create_confusion_pairs(
        results,
        "embedding_prediction",
    )

    tfidf_pairs.to_csv(
        METRICS_DIR / "tfidf_confusion_pairs.csv",
        index=False,
    )

    embedding_pairs.to_csv(
        METRICS_DIR / "embedding_confusion_pairs.csv",
        index=False,
    )

    print("\n--- TOP TF-IDF CONFUSION PAIRS ---")
    print(tfidf_pairs.head(10).to_string(index=False))

    print("\n--- TOP EMBEDDING CONFUSION PAIRS ---")
    print(embedding_pairs.head(10).to_string(index=False))

    print("\n--- COMPARISON COMPLETED ---")
    print("Prediction comparison saved.")
    print("Comparison summary saved.")
    print("Confusion pair reports saved.")


if __name__ == "__main__":
    main()
