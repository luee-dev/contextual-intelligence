
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="contextual_user",
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=5433,
    database="contextual_intelligence",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


def get_db():
    """Provide a database session and close it after use."""

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
