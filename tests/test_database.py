from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from cinematch.database import ALEMBIC_CONFIG_PATH, _create_alembic_config, initialize_database


def test_alembic_config_has_absolute_script_location(monkeypatch):
    monkeypatch.delenv("CINEMATCH_ALEMBIC_CONFIG", raising=False)

    config = _create_alembic_config()

    assert Path(config.config_file_name).resolve() == ALEMBIC_CONFIG_PATH.resolve()
    assert Path(config.get_main_option("script_location")).is_absolute()
    assert Path(config.get_main_option("script_location")).is_dir()


def test_initialize_database_from_different_working_directory(tmp_path, monkeypatch):
    database_path = tmp_path / "fresh.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"
    deployed_root = tmp_path / "deployed"
    deployed_root.mkdir()
    shutil.copy2(ALEMBIC_CONFIG_PATH, deployed_root / "alembic.ini")
    shutil.copytree(ALEMBIC_CONFIG_PATH.parent / "migrations", deployed_root / "migrations")
    working_directory = tmp_path / "working-directory"
    working_directory.mkdir()
    monkeypatch.setenv("CINEMATCH_ALEMBIC_CONFIG", str(deployed_root / "alembic.ini"))
    monkeypatch.delenv("CINEMATCH_DATABASE_URL", raising=False)
    monkeypatch.chdir(working_directory)

    initialize_database(database_url)

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "alembic_version" in inspector.get_table_names()
        assert {"app_users", "user_sessions", "user_ratings"} <= set(inspector.get_table_names())
        with engine.connect() as connection:
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
        assert revision == "20261107_auth_schema"
    finally:
        engine.dispose()


def test_alembic_config_rejects_relative_override(monkeypatch):
    monkeypatch.setenv("CINEMATCH_ALEMBIC_CONFIG", "alembic.ini")

    with pytest.raises(ValueError, match="absolute path"):
        _create_alembic_config()
