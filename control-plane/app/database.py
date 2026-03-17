import logging
from sqlalchemy import create_engine, event, inspect, text
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


def _run_migrations(engine):
    """Add owner_id column to tables that don't have it yet (safe to run on existing DBs)."""
    tables_needing_owner_id = [
        "assistant_config", "collection_parameter", "faq",
        "webhook_endpoint", "context_file", "spell_rule", "calendly_config",
    ]
    with engine.connect() as conn:
        insp = inspect(engine)
        for table in tables_needing_owner_id:
            if not insp.has_table(table):
                continue
            cols = [c["name"] for c in insp.get_columns(table)]
            if "owner_id" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN owner_id VARCHAR(64) NOT NULL DEFAULT ''"))
        conn.commit()


def init_db():
    from app.models import assistant, parameter, faq, context_file, spell_rule, webhook, calendly_config, calendly_event_type, customer_key  # noqa
    _run_migrations(engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
