
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from contextual_intelligence.database.connection import SessionLocal
from contextual_intelligence.database.models import (
    Article,
    ArticlePrediction,
    Campaign,
    CampaignMatch,
)
from contextual_intelligence.matching.engine import ContextualMatcher
from contextual_intelligence.matching.campaign_loader import load_campaigns


ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_NAME = "Ensemble (70/30)"
BATCH_SIZE = 64
INSERT_BATCH_SIZE = 1000

CAMPAIGN_FILE = ROOT_DIR / "data/external/campaigns.csv"


def load_articles_with_predictions(session):
    """Load test articles and their ensemble-predicted categories."""

    statement = (
        select(
            Article.id,
            Article.source_id,
            Article.article_text,
            ArticlePrediction.predicted_category,
        )
        .join(
            ArticlePrediction,
            Article.id == ArticlePrediction.article_id,
        )
        .where(
            Article.dataset_split == "test",
            ArticlePrediction.model_name == MODEL_NAME,
        )
        .order_by(Article.id)
    )

    rows = session.execute(statement).all()

    if not rows:
        raise ValueError(
            "No articles with ensemble predictions were found."
        )

    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "source_id",
            "article_text",
            "predicted_category",
        ],
    )


def validate_inputs(articles, campaigns):
    """Validate article and campaign data before matching."""

    required_categories = {
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    }

    if articles.empty:
        raise ValueError("Article dataset is empty.")

    if articles["article_id"].duplicated().any():
        raise ValueError("Duplicate article IDs detected.")

    invalid_categories = (
        set(articles["predicted_category"].dropna())
        - required_categories
    )

    if invalid_categories:
        raise ValueError(
            f"Invalid predicted categories: {invalid_categories}"
        )

    if articles["predicted_category"].isna().any():
        raise ValueError(
            "Missing ensemble-predicted categories."
        )

    if articles["article_text"].isna().any():
        raise ValueError("Articles with missing text were found.")

    if campaigns.empty:
        raise ValueError("No campaigns were loaded.")

    if campaigns["campaign_id"].duplicated().any():
        raise ValueError("Duplicate campaign IDs detected.")


def build_campaign_id_map(session, matcher):
    """Map external campaign IDs to PostgreSQL primary keys."""

    database_campaigns = session.scalars(
        select(Campaign)
    ).all()

    database_map = {
        campaign.campaign_id: campaign.id
        for campaign in database_campaigns
    }

    matcher_ids = set(matcher.campaigns["campaign_id"])
    database_ids = set(database_map)

    if matcher_ids != database_ids:
        raise ValueError(
            "Campaign configurations in CSV and PostgreSQL do not match. "
            f"CSV-only: {matcher_ids - database_ids}; "
            f"database-only: {database_ids - matcher_ids}"
        )

    return database_map


def generate_article_embeddings(matcher, articles):
    """Generate normalized article embeddings in batches."""

    texts = articles["article_text"].tolist()

    embeddings = matcher.model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        len(articles),
        matcher.campaign_embeddings.shape[1],
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Non-finite article embeddings detected."
        )

    return embeddings


def evaluate_batch(
    articles,
    embeddings,
    matcher,
    campaign_id_map,
):
    """Evaluate each article against every campaign."""

    similarity_matrix = (
        embeddings @ matcher.campaign_embeddings.T
    )

    records = []

    for article_index, article in articles.iterrows():

        article_text = article["article_text"].lower()
        article_category = article["predicted_category"]
        article_id = int(article["article_id"])

        for campaign_index, (_, campaign) in enumerate(
            matcher.campaigns.iterrows()
        ):
            matched_keywords = matcher.find_keyword_matches(
                article_text,
                campaign["target_keywords"],
            )

            excluded_matches = matcher.find_keyword_matches(
                article_text,
                campaign["excluded_keywords"],
            )

            relevance_score = float(
                similarity_matrix[
                    article_index,
                    campaign_index,
                ]
            )

            suitability, reasons = matcher.evaluate_suitability(
                article_category,
                campaign,
                excluded_matches,
            )

            if suitability == "unsuitable":
                decision = "unsuitable"

            elif (
                relevance_score
                < campaign["minimum_relevance_score"]
            ):
                decision = "review"

                reasons.append(
                    "Semantic relevance is below the campaign threshold."
                )

            else:
                decision = "suitable"

                reasons.append(
                    "Category, exclusion and relevance checks passed."
                )

            records.append({
                "article_id": article_id,
                "campaign_id": campaign_id_map[
                    campaign["campaign_id"]
                ],
                "relevance_score": relevance_score,
                "suitability": suitability,
                "decision": decision,
                "matched_keywords": "; ".join(matched_keywords),
                "excluded_keywords_found": "; ".join(
                    excluded_matches
                ),
                "decision_reasons": json.dumps(reasons),
            })

    return records


def persist_records(session, records):
    """Insert or update campaign matching results."""

    if not records:
        return

    statement = insert(CampaignMatch).values(records)

    statement = statement.on_conflict_do_update(
        constraint="uq_article_campaign_match",
        set_={
            "relevance_score": statement.excluded.relevance_score,
            "suitability": statement.excluded.suitability,
            "decision": statement.excluded.decision,
            "matched_keywords": statement.excluded.matched_keywords,
            "excluded_keywords_found": (
                statement.excluded.excluded_keywords_found
            ),
            "decision_reasons": statement.excluded.decision_reasons,
            "evaluated_at": statement.excluded.evaluated_at,
        },
    )

    session.execute(statement)


def persist_campaign_matches(session):
    """Run and persist the complete campaign matching pipeline."""

    articles = load_articles_with_predictions(session)

    campaigns = load_campaigns(CAMPAIGN_FILE)

    matcher = ContextualMatcher(
        campaigns=campaigns
    )

    validate_inputs(articles, matcher.campaigns)

    campaign_id_map = build_campaign_id_map(
        session,
        matcher,
    )

    print(f"Articles loaded: {len(articles):,}")
    print(f"Campaigns loaded: {len(matcher.campaigns):,}")

    expected_records = (
        len(articles) * len(matcher.campaigns)
    )

    print(
        f"Expected article-campaign matches: "
        f"{expected_records:,}"
    )

    embeddings = generate_article_embeddings(
        matcher,
        articles,
    )

    total_records = 0

    for start in range(0, len(articles), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(articles),
        )

        batch_articles = (
            articles.iloc[start:end].reset_index(drop=True)
        )

        batch_embeddings = embeddings[start:end]

        records = evaluate_batch(
            batch_articles,
            batch_embeddings,
            matcher,
            campaign_id_map,
        )

        for insert_start in range(
            0,
            len(records),
            INSERT_BATCH_SIZE,
        ):
            insert_end = (
                insert_start + INSERT_BATCH_SIZE
            )

            persist_records(
                session,
                records[insert_start:insert_end],
            )

        total_records += len(records)

        print(
            f"Processed articles: {end:,}/{len(articles):,} "
            f"| Matches: {total_records:,}"
        )

    if total_records != expected_records:
        raise ValueError(
            f"Expected {expected_records} matches, "
            f"but generated {total_records}."
        )

    print(
        f"Total campaign matches processed: "
        f"{total_records:,}"
    )


def main():
    with SessionLocal() as session:
        try:
            persist_campaign_matches(session)

            session.commit()

            print(
                "Campaign matching results persisted successfully!"
            )

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
