
from contextual_intelligence.database.connection import Base, engine
from contextual_intelligence.database.models import (
    Article,
    ArticlePrediction,
    Campaign,
    CampaignMatch,
)


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


if __name__ == "__main__":
    create_tables()
