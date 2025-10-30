"""Database connection."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

DATABASE_URL = "sqlite:///./trading.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Initialize database."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Get database session."""
    return SessionLocal()
