
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT_DIR / "data/processed/ag_news_test.csv"
EMBEDDING_FILE = ROOT_DIR / "models/embeddings/ag_news_test_embeddings.npy"

TFIDF_MODEL_FILE = ROOT_DIR / "models/tfidf_logistic_regression.joblib"
MINILM_MODEL_FILE = ROOT_DIR / "models/embedding_logistic_regression.joblib"

TFIDF_WEIGHT = 0.70
MINILM_WEIGHT = 0.30

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def main():
    print("\n--- LOADING TEST DATA ---")

    df = pd.read_csv(DATA_FILE)
    embeddings = np.load(EMBEDDING_FILE)

    print(f"Test records: {len(df):,}")

    print("\n--- LOADING MODELS ---")

    tfidf_model = joblib.load(TFIDF_MODEL_FILE)
    minilm_model = joblib.load(MINILM_MODEL_FILE)

    if not np.array_equal(
        tfidf_model.classes_,
        minilm_model.classes_,
    ):
        raise ValueError("Model class ordering does not match.")

    classes = tfidf_model.classes_

    print("Class ordering:", classes)

    print("\n--- GENERATING PROBABILITIES ---")

    tfidf_probabilities = tfidf_model.predict_proba(df["text"])
    minilm_probabilities = minilm_model.predict_proba(embeddings)

    if tfidf_probabilities.shape != minilm_probabilities.shape:
        raise ValueError("Model probability dimensions do not match.")

    print("TF-IDF probabilities:", tfidf_probabilities.shape)
    print("MiniLM probabilities:", minilm_probabilities.shape)

    print("\n--- COMBINING PROBABILITIES ---")

    ensemble_probabilities = (
        TFIDF_WEIGHT * tfidf_probabilities
        + MINILM_WEIGHT * minilm_probabilities
    )

    predictions = classes[
        np.argmax(ensemble_probabilities, axis=1)
    ]

    confidence = ensemble_probabilities.max(axis=1)

    y_true = df["label"].to_numpy()

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )
    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
    )

    print("\n--- ENSEMBLE TEST EVALUATION ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Mean confidence: {confidence.mean():.4f}")

    print("\n--- CLASSIFICATION REPORT ---")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=LABEL_NAMES,
        )
    )

    print("\n--- COMPARISON ---")

    print(f"TF-IDF accuracy: 91.53%")
    print(f"MiniLM accuracy: 89.43%")
    print(f"Ensemble accuracy: {accuracy * 100:.2f}%")

    print(
        "Improvement over TF-IDF: "
        f"{(accuracy - 0.9153) * 100:.2f} percentage points"
    )


if __name__ == "__main__":
    main()
