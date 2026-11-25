
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sqlalchemy.dialects.postgresql import insert

from contextual_intelligence.database.connection import SessionLocal
from contextual_intelligence.database.models import Article, ArticlePrediction


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT_DIR / "data/processed/ag_news_test.csv"
EMBEDDING_FILE = ROOT_DIR / "models/embeddings/ag_news_test_embeddings.npy"
MODEL_FILE = ROOT_DIR / "models/embedding_logistic_regression.joblib"

LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def persist_minilm_predictions(session):
    """Generate and persist MiniLM predictions."""

    df = pd.read_csv(DATA_FILE)
    embeddings = np.load(EMBEDDING_FILE)
    model = joblib.load(MODEL_FILE)

    if len(df) != len(embeddings):
        raise ValueError("Dataset and embedding record counts do not match.")

    predictions = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings)

    records = []

    for index, (prediction, probability) in enumerate(
        zip(predictions, probabilities)
    ):
        source_id = f"AGNEWS-TEST-{index + 1:06d}"

        article = session.query(Article).filter_by(
            source_id=source_id
        ).one()

        predicted_category = LABEL_MAPPING[int(prediction)]

        records.append({
            "article_id": article.id,
            "model_name": "MiniLM Logistic Regression",
            "predicted_category": predicted_category,
            "confidence_score": float(probability.max()),
            "is_correct": (
                predicted_category == str(df.iloc[index]["label_name"])
            ),
        })

    statement = insert(ArticlePrediction).values(records)

    statement = statement.on_conflict_do_update(
        constraint="uq_article_model_prediction",
        set_={
            "predicted_category": statement.excluded.predicted_category,
            "confidence_score": statement.excluded.confidence_score,
            "is_correct": statement.excluded.is_correct,
        },
    )

    session.execute(statement)

    print(f"MiniLM predictions processed: {len(records)}")


def main():
    with SessionLocal() as session:
        try:
            persist_minilm_predictions(session)
            session.commit()
            print("MiniLM predictions persisted successfully!")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
