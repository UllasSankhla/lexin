import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

event.listen(engine, "connect", _set_sqlite_pragma)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import call_record, gathered_parameter, call_analytics  # noqa
    Base.metadata.create_all(bind=engine)
    _run_migrations()
    logger.info("Database tables created/verified")


def _run_migrations():
    """Apply additive schema changes that create_all won't handle on existing tables."""
    with engine.connect() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(call_record)"))
        }
        if "end_reason" not in existing:
            conn.execute(text("ALTER TABLE call_record ADD COLUMN end_reason TEXT"))
            conn.commit()
            logger.info("Migration applied: added call_record.end_reason")
