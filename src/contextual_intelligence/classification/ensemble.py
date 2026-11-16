
import json
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from contextual_intelligence.classification.train import (
    DATA_PATH,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)


ROOT_DIR = Path(__file__).resolve().parents[3]

EMBEDDINGS_PATH = ROOT_DIR / "models" / "embeddings" / "ag_news_embeddings.npy"
LABELS_PATH = ROOT_DIR / "models" / "embeddings" / "ag_news_labels.npy"

TFIDF_MODEL_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression.joblib"
EMBEDDING_MODEL_PATH = ROOT_DIR / "models" / "embedding_logistic_regression.joblib"

OUTPUT_DIR = ROOT_DIR / "reports" / "metrics"
OUTPUT_PATH = OUTPUT_DIR / "ensemble_experiment.json"


def main():
    print("\n--- LOADING VALIDATION DATA ---")

    data = load_training_data()

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)

    if len(data) != len(embeddings) or len(data) != len(labels):
        raise ValueError("Dataset, embeddings and labels are not aligned.")

    if not np.array_equal(data["label"].to_numpy(), labels):
        raise ValueError("Labels do not match the cleaned dataset.")

    indices = np.arange(len(labels))

    _, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    validation_texts = data.iloc[validation_indices]["text"].to_numpy()
    X_validation_embeddings = embeddings[validation_indices]
    y_validation = labels[validation_indices]

    print(f"Validation records: {len(y_validation):,}")

    print("\n--- LOADING MODELS ---")

    tfidf_model = load(TFIDF_MODEL_PATH)
    embedding_model = load(EMBEDDING_MODEL_PATH)

    tfidf_classes = tfidf_model.classes_
    embedding_classes = embedding_model.classes_

    if not np.array_equal(tfidf_classes, embedding_classes):
        raise ValueError("The two models have different class orders.")

    classes = tfidf_classes

    print("Both models loaded successfully.")
    print(f"Classes: {classes}")

    print("\n--- GENERATING PREDICTION PROBABILITIES ---")

    tfidf_probabilities = tfidf_model.predict_proba(validation_texts)

    embedding_probabilities = embedding_model.predict_proba(
        X_validation_embeddings
    )

    print("Prediction probabilities generated.")

    print("\n--- TESTING ENSEMBLE WEIGHTS ---")

    weights = [0.7, 0.8, 0.9]

    results = []

    for tfidf_weight in weights:
        embedding_weight = 1 - tfidf_weight

        ensemble_probabilities = (
            tfidf_weight * tfidf_probabilities
            + embedding_weight * embedding_probabilities
        )

        predictions = classes[np.argmax(ensemble_probabilities, axis=1)]

        accuracy = accuracy_score(y_validation, predictions)

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro",
        )

        weighted_f1 = f1_score(
            y_validation,
            predictions,
            average="weighted",
        )

        result = {
            "tfidf_weight": tfidf_weight,
            "embedding_weight": embedding_weight,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }

        results.append(result)

        print(
            f"TF-IDF: {tfidf_weight:.1f} | "
            f"MiniLM: {embedding_weight:.1f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    best_result = max(results, key=lambda item: item["macro_f1"])

    output = {
        "validation_records": int(len(y_validation)),
        "experiment": "Soft-voting ensemble",
        "results": results,
        "best_configuration": best_result,
        "selection_metric": "macro_f1",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n--- BEST ENSEMBLE CONFIGURATION ---")
    print(f"TF-IDF weight: {best_result['tfidf_weight']}")
    print(f"MiniLM weight: {best_result['embedding_weight']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Macro F1: {best_result['macro_f1']:.4f}")

    print("\n--- SAVING COMPLETED ---")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
