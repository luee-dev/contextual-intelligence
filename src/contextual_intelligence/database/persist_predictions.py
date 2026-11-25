
import joblib
import pandas as pd

from pathlib import Path
from sqlalchemy.dialects.postgresql import insert

from contextual_intelligence.database.connection import SessionLocal
from contextual_intelligence.database.models import Article, ArticlePrediction


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = ROOT_DIR / "data/processed/ag_news_test.csv"
MODEL_FILE = ROOT_DIR / "models/tfidf_logistic_regression.joblib"

LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def persist_tfidf_predictions(session):
    """Generate and persist TF-IDF predictions."""

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    predictions = model.predict(df["text"])
    probabilities = model.predict_proba(df["text"])

    records = []

    for index, (prediction, probability) in enumerate(
        zip(predictions, probabilities)
    ):
        source_id = f"AGNEWS-TEST-{index + 1:06d}"

        article = session.query(Article).filter_by(
            source_id=source_id
        ).one()

        records.append({
            "article_id": article.id,
            "model_name": "TF-IDF Logistic Regression",
            "predicted_category": LABEL_MAPPING[int(prediction)],
            "confidence_score": float(probability.max()),
            "is_correct": (
                LABEL_MAPPING[int(prediction)] == str(df.iloc[index]["label_name"])
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

    print(f"TF-IDF predictions processed: {len(records)}")


def main():
    with SessionLocal() as session:
        try:
            persist_tfidf_predictions(session)
            session.commit()
            print("TF-IDF predictions persisted successfully!")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
