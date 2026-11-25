
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    func,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
)

from contextual_intelligence.database.connection import Base


class Article(Base):
    """Stores news articles used by the contextual intelligence pipeline."""

    __tablename__ = "articles"

    __table_args__ = (
        CheckConstraint(
            "category IN ('World', 'Sports', 'Business', 'Sci/Tech')",
            name="check_article_category",
        ),
        CheckConstraint(
            "dataset_split IN ('train', 'test', 'external')",
            name="check_dataset_split",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    source_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    article_text = Column(Text, nullable=False)

    category = Column(String(50), nullable=False)

    dataset_split = Column(String(20), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<Article(id={self.id}, "
            f"source_id='{self.source_id}', "
            f"category='{self.category}')>"
        )


class ArticlePrediction(Base):
    """Stores article classification predictions from different models."""

    __tablename__ = "article_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    article_id = Column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_name = Column(String(100), nullable=False)

    predicted_category = Column(String(50), nullable=False)

    confidence_score = Column(Float, nullable=False)

    is_correct = Column(Boolean, nullable=True)

    predicted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    __table_args__ = (
    UniqueConstraint(
        "article_id",
        "model_name",
        name="uq_article_model_prediction",
    ),
)

    def __repr__(self):
        return (
            f"<ArticlePrediction(id={self.id}, "
            f"model='{self.model_name}', "
            f"category='{self.predicted_category}')>"
        )


class Campaign(Base):
    """Stores advertising campaign configurations."""

    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, autoincrement=True)

    campaign_id = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )

    advertiser = Column(String(150), nullable=False)

    campaign_name = Column(String(200), nullable=False)

    campaign_description = Column(Text, nullable=False)

    target_categories = Column(Text, nullable=False)

    target_keywords = Column(Text, nullable=False)

    excluded_keywords = Column(Text, nullable=False)

    excluded_categories = Column(Text, nullable=False)

    minimum_relevance_score = Column(Float, nullable=False)

    active = Column(Boolean, nullable=False, default=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<Campaign(campaign_id='{self.campaign_id}', "
            f"advertiser='{self.advertiser}')>"
        )


class CampaignMatch(Base):
    """Stores campaign-to-article matching and suitability results."""

    __tablename__ = "campaign_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)

    article_id = Column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    campaign_id = Column(
        Integer,
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    relevance_score = Column(Float, nullable=False)

    suitability = Column(String(30), nullable=False)

    decision = Column(String(30), nullable=False)

    matched_keywords = Column(Text, nullable=True)

    evaluated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<CampaignMatch(article_id={self.article_id}, "
            f"campaign_id={self.campaign_id}, "
            f"decision='{self.decision}')>"
        )
