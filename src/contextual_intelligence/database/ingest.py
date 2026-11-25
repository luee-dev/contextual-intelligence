
import pandas as pd

from sqlalchemy.dialects.postgresql import insert

from contextual_intelligence.database.connection import SessionLocal
from contextual_intelligence.database.models import Article, Campaign


ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[3]

ARTICLES_FILE = ROOT_DIR / "data/processed/ag_news_test.csv"
CAMPAIGNS_FILE = ROOT_DIR / "data/external/campaigns.csv"


def load_articles(session):
    """Load the AG News test dataset into PostgreSQL."""

    df = pd.read_csv(ARTICLES_FILE)

    records = []

    for index, row in df.iterrows():
        records.append({
            "source_id": f"AGNEWS-TEST-{index + 1:06d}",
            "article_text": str(row["text"]),
            "category": str(row["label_name"]),
            "dataset_split": "test",
        })

    statement = insert(Article).values(records)

    statement = statement.on_conflict_do_update(
        index_elements=["source_id"],
        set_={
            "article_text": statement.excluded.article_text,
            "category": statement.excluded.category,
            "dataset_split": statement.excluded.dataset_split,
        },
    )

    session.execute(statement)
    print(f"Articles processed: {len(records)}")


def load_campaigns(session):
    """Load campaign configurations into PostgreSQL."""

    df = pd.read_csv(CAMPAIGNS_FILE)

    text_columns = [
        "campaign_id",
        "advertiser",
        "campaign_name",
        "campaign_description",
        "target_categories",
        "target_keywords",
        "excluded_keywords",
        "excluded_categories",
    ]

    df[text_columns] = df[text_columns].fillna("")

    records = []
    for _, row in df.iterrows():
        records.append({
            "campaign_id": row["campaign_id"],
            "advertiser": row["advertiser"],
            "campaign_name": row["campaign_name"],
            "campaign_description": row["campaign_description"],
            "target_categories": row["target_categories"],
            "target_keywords": row["target_keywords"],
            "excluded_keywords": row["excluded_keywords"],
            "excluded_categories": row["excluded_categories"],
            "minimum_relevance_score": float(
                row["minimum_relevance_score"]
            ),
            "active": bool(row["active"]),
        })

    statement = insert(Campaign).values(records)

    statement = statement.on_conflict_do_update(
        index_elements=["campaign_id"],
        set_={
            "advertiser": statement.excluded.advertiser,
            "campaign_name": statement.excluded.campaign_name,
            "campaign_description": statement.excluded.campaign_description,
            "target_categories": statement.excluded.target_categories,
            "target_keywords": statement.excluded.target_keywords,
            "excluded_keywords": statement.excluded.excluded_keywords,
            "excluded_categories": statement.excluded.excluded_categories,
            "minimum_relevance_score": (
                statement.excluded.minimum_relevance_score
            ),
            "active": statement.excluded.active,
        },
    )

    session.execute(statement)
    print(f"Campaigns processed: {len(records)}")


def main():
    with SessionLocal() as session:
        try:
            load_articles(session)
            load_campaigns(session)
            session.commit()

            print("Data ingestion completed successfully!")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
