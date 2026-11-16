
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
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

EMBEDDING_PATH = EMBEDDING_DIR / "ag_news_embeddings.npy"
LABELS_PATH = EMBEDDING_DIR / "ag_news_labels.npy"

MODEL_PATH = MODEL_DIR / "embedding_logistic_regression.joblib"
METADATA_PATH = METRICS_DIR / "embedding_model_metadata.json"


def load_embeddings():
    """Load embeddings and verify alignment with the cleaned dataset."""

    print("\n--- LOADING EMBEDDINGS ---")

    embeddings = np.load(EMBEDDING_PATH)
    labels = np.load(LABELS_PATH)

    df = load_training_data()
    dataset_labels = df["label"].to_numpy(dtype=np.int64)

    if len(embeddings) != len(labels):
        raise ValueError("Embedding and label counts do not match.")

    if not np.array_equal(labels, dataset_labels):
        raise ValueError(
            "Embedding labels do not match the cleaned dataset."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    print(f"Records: {len(embeddings):,}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return embeddings, labels


def train_model(X_train, y_train):
    """Train Logistic Regression using sentence embeddings."""

    print("\n--- TRAINING EMBEDDING CLASSIFIER ---")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    print("Training completed.")
    print(f"Iterations: {model.n_iter_}")

    return model


def save_model(model, X_train, X_val):
    """Save the trained model and its metadata."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_type": "Logistic Regression",
        "input_features": "Sentence embeddings",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "data_source": "data/processed/ag_news_train_clean.csv",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n--- SAVING COMPLETED ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


def main():
    """Train the embedding-based classification model."""

    embeddings, labels = load_embeddings()

    print("\n--- CREATING VALIDATION SPLIT ---")

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")

    model = train_model(X_train, y_train)

    save_model(model, X_train, X_val)

    print("\n--- EMBEDDING MODEL TRAINING COMPLETED ---")


if __name__ == "__main__":
    main()
