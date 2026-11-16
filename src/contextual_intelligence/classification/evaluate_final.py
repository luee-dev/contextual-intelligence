
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from contextual_intelligence.preprocessing.text_cleaner import clean_text


ROOT_DIR = Path(__file__).resolve().parents[3]

TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "ag_news_test.csv"
EMBEDDINGS_PATH = ROOT_DIR / "models" / "embeddings" / "ag_news_embeddings.npy"

TFIDF_MODEL_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression.joblib"
EMBEDDING_MODEL_PATH = ROOT_DIR / "models" / "embedding_logistic_regression.joblib"

METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def main():
    print("\n--- LOADING ORIGINAL TEST DATA ---")

    test_data = pd.read_csv(TEST_DATA_PATH)

    required_columns = {"text", "label"}

    if not required_columns.issubset(test_data.columns):
        raise ValueError(
            f"Test data must contain columns: {required_columns}"
        )

    test_texts = test_data["text"].apply(clean_text).to_numpy()
    y_test = test_data["label"].to_numpy()

    print(f"Test records: {len(y_test):,}")

    print("\n--- LOADING MODELS ---")

    tfidf_model = load(TFIDF_MODEL_PATH)
    embedding_model = load(EMBEDDING_MODEL_PATH)

    if not np.array_equal(
        tfidf_model.classes_,
        embedding_model.classes_,
    ):
        raise ValueError("Model class orders do not match.")

    classes = tfidf_model.classes_

    if not np.array_equal(np.unique(y_test), classes):
        raise ValueError("Test labels do not match model classes.")

    print("Models loaded successfully.")

    print("\n--- GENERATING TEST EMBEDDINGS ---")

    embeddings_model = load(
        ROOT_DIR / "models" / "embedding_model_placeholder.joblib"
    ) if False else None

    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    test_embeddings = sentence_model.encode(
        test_texts.tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )

    print(f"Test embedding shape: {test_embeddings.shape}")

    print("\n--- GENERATING PREDICTIONS ---")

    tfidf_probabilities = tfidf_model.predict_proba(test_texts)

    embedding_probabilities = embedding_model.predict_proba(
        test_embeddings
    )

    ensemble_probabilities = (
        0.7 * tfidf_probabilities
        + 0.3 * embedding_probabilities
    )

    predictions = {
        "TF-IDF": classes[np.argmax(tfidf_probabilities, axis=1)],
        "MiniLM": classes[np.argmax(embedding_probabilities, axis=1)],
        "Ensemble 70/30": classes[
            np.argmax(ensemble_probabilities, axis=1)
        ],
    }

    print("\n--- FINAL TEST RESULTS ---")

    results = {}

    for model_name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
        )
        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
        )

        results[model_name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

    best_model_name = max(
        results,
        key=lambda name: results[name]["macro_f1"],
    )

    print(f"\nBest test model by macro F1: {best_model_name}")

    ensemble_predictions = predictions["Ensemble 70/30"]

    report = classification_report(
        y_test,
        ensemble_predictions,
        target_names=["World", "Sports", "Business", "Sci/Tech"],
        zero_division=0,
    )

    print("\n--- ENSEMBLE CLASSIFICATION REPORT ---")
    print(report)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "test_records": int(len(y_test)),
        "evaluation_type": "Original held-out test set",
        "models": results,
        "best_model_by_macro_f1": best_model_name,
    }

    with open(
        METRICS_DIR / "final_test_evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=4)

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            ensemble_predictions,
            target_names=["World", "Sports", "Business", "Sci/Tech"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    report_df.to_csv(
        METRICS_DIR / "final_ensemble_classification_report.csv"
    )

    matrix = confusion_matrix(
        y_test,
        ensemble_predictions,
        labels=classes,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["World", "Sports", "Business", "Sci/Tech"],
    )

    display.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
    )

    plt.title("Final Test Set - Ensemble 70/30")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "final_ensemble_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\n--- SAVING COMPLETED ---")
    print(f"Metrics: {METRICS_DIR / 'final_test_evaluation.json'}")
    print(
        "Classification report: "
        f"{METRICS_DIR / 'final_ensemble_classification_report.csv'}"
    )
    print(
        "Confusion matrix: "
        f"{FIGURES_DIR / 'final_ensemble_confusion_matrix.png'}"
    )


if __name__ == "__main__":
    main()
