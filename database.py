"""
Database Management Module for AI-Powered USM System.

This module initializes the database connection using SQLAlchemy.
It configures the engine and session factory to manage data transactions.
Includes automated schema migration for SQLite table extensions.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL

# Create the SQLAlchemy engine
# connect_args={"check_same_thread": False} is required for SQLite when accessed by multi-threaded frameworks like Streamlit
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False
)

# SessionLocal is the factory for creating new database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


def get_db():
    """
    Context generator for database sessions.
    Ensures that the database session is always closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initializes the database schema.
    Creates all tables and safely applies column migrations if tables already exist.
    """
    import models  # noqa: F401 - Required for SQLAlchemy metadata registration
    Base.metadata.create_all(bind=engine)

    # Safe lightweight schema migration for SQLite
    with engine.connect() as conn:
        try:
            res = conn.exec_driver_sql("PRAGMA table_info(tickets);").fetchall()
            existing_cols = {row[1] for row in res}

            new_columns = [
                ("ticket_type", "VARCHAR(30) DEFAULT 'Incident'"),
                ("response_due_date", "DATETIME"),
                ("sla_due_date", "DATETIME"),
                ("is_sla_breached", "BOOLEAN DEFAULT 0"),
                ("resolution_notes", "TEXT")
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(f"ALTER TABLE tickets ADD COLUMN {col_name} {col_type};")
            conn.commit()
        except Exception:
            pass
