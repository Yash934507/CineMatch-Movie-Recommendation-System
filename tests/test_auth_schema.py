from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from cinematch.models import AppUser, UserSession, utc_now


ROOT = Path(__file__).resolve().parents[1]
INITIAL_REVISION = "20261105_initial_persistent_users"
AUTH_REVISION = "20261107_auth_schema"


def _migration_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_auth_models_define_nullable_legacy_credentials_and_utc_session_defaults():
    assert AppUser.__table__.c.email.nullable
    assert AppUser.__table__.c.password_hash.nullable
    assert not AppUser.__table__.c.is_active.nullable
    assert "uq_app_users_email" in {
        constraint.name for constraint in AppUser.__table__.constraints
    }
    assert not UserSession.__table__.c.token_hash.nullable
    assert "uq_user_sessions_token_hash" in {
        constraint.name for constraint in UserSession.__table__.constraints
    }
    assert not UserSession.__table__.c.expires_at.nullable
    assert UserSession.__table__.c.revoked_at.nullable

    created_at = utc_now()
    assert created_at.tzinfo == timezone.utc
    assert isinstance(created_at, datetime)


def test_auth_migration_preserves_legacy_users_and_interactions(tmp_path, monkeypatch):
    database_path = tmp_path / "auth-migration.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("CINEMATCH_DATABASE_URL", database_url)

    config = _migration_config(database_url)
    command.upgrade(config, INITIAL_REVISION)

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO app_users (id, display_name, movielens_user_id, onboarding_complete) "
                "VALUES (41, 'Legacy demo', 7, 1)"
            )
        )
        connection.execute(
            text("INSERT INTO user_ratings (user_id, movie_id, rating) VALUES (41, 260, 4.5)")
        )
        connection.execute(
            text("INSERT INTO watchlist_items (user_id, movie_id) VALUES (41, 318)")
        )
        connection.execute(
            text("INSERT INTO user_feedback (user_id, movie_id, feedback_type) VALUES (41, 356, 'like')")
        )
        connection.execute(
            text(
                "INSERT INTO recommendation_events (user_id, event_type, movie_id) "
                "VALUES (41, 'recommendation_click', 260)"
            )
        )

    command.upgrade(config, AUTH_REVISION)

    inspector = inspect(engine)
    assert {"email", "password_hash", "is_active"} <= {
        column["name"] for column in inspector.get_columns("app_users")
    }
    assert "user_sessions" in inspector.get_table_names()
    session_indexes = {index["name"] for index in inspector.get_indexes("user_sessions")}
    assert {"ix_user_sessions_user_id", "ix_user_sessions_expires_at"} <= session_indexes
    session_foreign_keys = inspector.get_foreign_keys("user_sessions")
    assert any(
        foreign_key["referred_table"] == "app_users"
        and foreign_key["options"].get("ondelete") == "CASCADE"
        for foreign_key in session_foreign_keys
    )

    with engine.connect() as connection:
        user = connection.execute(
            text(
                "SELECT id, display_name, movielens_user_id, email, password_hash, is_active "
                "FROM app_users WHERE id = 41"
            )
        ).one()
        assert tuple(user) == (41, "Legacy demo", 7, None, None, 1)
        assert connection.execute(
            text("SELECT user_id, movie_id, rating FROM user_ratings")
        ).all() == [(41, 260, 4.5)]
        assert connection.execute(
            text("SELECT user_id, movie_id FROM watchlist_items")
        ).all() == [(41, 318)]
        assert connection.execute(
            text("SELECT user_id, movie_id, feedback_type FROM user_feedback")
        ).all() == [(41, 356, "like")]
        assert connection.execute(
            text("SELECT user_id, event_type, movie_id FROM recommendation_events")
        ).all() == [(41, "recommendation_click", 260)]

    engine.dispose()


def test_auth_migration_enforces_unique_email_and_session_token_hash(tmp_path, monkeypatch):
    database_path = tmp_path / "auth-constraints.sqlite3"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("CINEMATCH_DATABASE_URL", database_url)
    command.upgrade(_migration_config(database_url), "head")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO app_users (display_name, email, password_hash) "
                "VALUES ('Account A', 'a@example.test', 'argon2-hash')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO app_users (display_name, email, password_hash) "
                "VALUES ('Account B', 'b@example.test', 'argon2-hash')"
            )
        )
        account_ids = connection.execute(
            text("SELECT id FROM app_users ORDER BY id")
        ).scalars().all()
        connection.execute(
            text(
                "INSERT INTO user_sessions (user_id, token_hash, expires_at) "
                "VALUES (:user_id, :token_hash, :expires_at)"
            ),
            {
                "user_id": account_ids[0],
                "token_hash": "a" * 64,
                "expires_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO app_users (display_name, email, password_hash) "
                    "VALUES ('Duplicate email', 'a@example.test', 'argon2-hash')"
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO user_sessions (user_id, token_hash, expires_at) "
                    "VALUES (:user_id, :token_hash, :expires_at)"
                ),
                {
                    "user_id": account_ids[1],
                    "token_hash": "a" * 64,
                    "expires_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO user_sessions (user_id, token_hash, expires_at) "
                    "VALUES (:user_id, :token_hash, :expires_at)"
                ),
                {
                    "user_id": account_ids[1],
                    "token_hash": "short",
                    "expires_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    engine.dispose()
