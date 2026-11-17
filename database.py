from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Lightweight migration for existing SQLite databases
    with engine.connect() as conn:
        try:
            res = conn.exec_driver_sql("PRAGMA table_info(tickets);").fetchall()
            existing_cols = {row[1] for row in res}

            new_cols = [
                ("ticket_type", "VARCHAR(30) DEFAULT 'Incident'"),
                ("response_due_date", "DATETIME"),
                ("sla_due_date", "DATETIME"),
                ("is_sla_breached", "BOOLEAN DEFAULT 0"),
                ("resolution_notes", "TEXT")
            ]
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(f"ALTER TABLE tickets ADD COLUMN {col_name} {col_type};")
            conn.commit()
        except Exception:
            pass
