
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT_DIR / "data" / "processed" / "ag_news_train_clean.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports" / "metrics"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def load_training_data() -> pd.DataFrame:
    """Load and validate the cleaned training dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError("Training data contains missing text or labels.")

    return df


def build_model() -> Pipeline:
    """Build the TF-IDF and Logistic Regression pipeline."""

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


def main() -> None:
    """Train and save the baseline classification model."""

    print("\n--- LOADING TRAINING DATA ---")

    df = load_training_data()

    print(f"Total records: {len(df):,}")
    print(f"Unique labels: {sorted(df['label'].unique())}")

    X = df["text"]
    y = df["label"]

    # Create a stratified training-validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n--- DATA SPLIT ---")
    print(f"Training records: {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")

    print("\n--- BUILDING MODEL ---")

    model = build_model()

    print("\n--- TRAINING MODEL ---")

    model.fit(X_train, y_train)

    # Save the trained pipeline.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    dump(model, model_path)

    # Save experiment configuration and metadata.
    metadata = {
        "model": "Logistic Regression",
        "feature_extraction": "TF-IDF",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 100000,
        "sublinear_tf": True,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "classes": sorted(int(label) for label in y.unique()),
        "features_learned": len(
            model.named_steps["tfidf"].get_feature_names_out()
        ),
    }

    metadata_path = REPORT_DIR / "baseline_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n--- TRAINING COMPLETED ---")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(
        "Learned TF-IDF features:",
        metadata["features_learned"],
    )


if __name__ == "__main__":
    main()
