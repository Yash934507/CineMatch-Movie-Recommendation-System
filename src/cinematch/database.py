"""Database configuration and session helpers for CineMatch."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import settings
from .models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "alembic.ini"
DEFAULT_DATABASE_URL = settings.database_url


def _create_alembic_config() -> Config:
    config_path_value = os.getenv("CINEMATCH_ALEMBIC_CONFIG")
    config_path = Path(config_path_value) if config_path_value else ALEMBIC_CONFIG_PATH
    if not config_path.is_absolute():
        raise ValueError("CINEMATCH_ALEMBIC_CONFIG must be an absolute path")
    if not config_path.is_file():
        raise FileNotFoundError(f"Alembic configuration not found: {config_path}")

    migrations_path = config_path.parent / "migrations"
    if not migrations_path.is_dir():
        raise FileNotFoundError(f"Alembic migrations directory not found: {migrations_path}")

    config = Config(str(config_path))
    config.set_main_option("script_location", str(migrations_path))
    return config


def get_database_url() -> str:
    return settings.database_url


def create_engine_for_url(database_url: str | None = None):
    url = (database_url or get_database_url()).strip()
    if not url:
        raise ValueError("CINEMATCH_DATABASE_URL cannot be empty")

    connect_args = {}
    engine_kwargs: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        engine_kwargs["connect_args"] = connect_args
        engine_kwargs["future"] = True
        engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(url, **engine_kwargs)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):  # pragma: no cover - SQLite only
        if url.startswith("sqlite"):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()

    return engine


ENGINE = create_engine_for_url()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def initialize_database(database_url: str | None = None) -> None:
    url = database_url or get_database_url()
    cfg = _create_alembic_config()
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


__all__ = [
    "Base",
    "ENGINE",
    "SessionLocal",
    "DEFAULT_DATABASE_URL",
    "ALEMBIC_CONFIG_PATH",
    "create_engine_for_url",
    "get_database_url",
    "get_db",
    "initialize_database",
]
