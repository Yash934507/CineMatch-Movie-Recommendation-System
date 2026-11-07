from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from cinematch.models import AppUser, Base, UserSession
from cinematch.services.auth import (
    IneligibleAccountError,
    InvalidPasswordError,
    SESSION_LIFETIME,
    create_session,
    get_session_user,
    hash_password,
    hash_session_token,
    revoke_session,
    session_cookie_options,
    verify_password,
)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def _account(db: Session, *, active: bool = True, credentials: bool = True) -> AppUser:
    user = AppUser(
        display_name="Test account",
        email="test@example.test" if credentials else None,
        password_hash=hash_password("correct horse battery staple") if credentials else None,
        is_active=active,
    )
    db.add(user)
    db.commit()
    return user


def test_argon2id_password_hash_is_not_plaintext_and_verifies():
    password = "correct horse battery staple"
    password_hash = hash_password(password)

    assert password_hash.startswith("$argon2id$")
    assert password not in password_hash
    assert verify_password(password_hash, password)
    assert not verify_password(password_hash, "incorrect password")


def test_password_hashing_rejects_weak_and_oversized_passwords():
    with pytest.raises(InvalidPasswordError):
        hash_password("short")
    with pytest.raises(InvalidPasswordError):
        hash_password("a" * 1025)
    assert not verify_password("not-an-argon-hash", "correct horse battery staple")
    assert not verify_password(None, "correct horse battery staple")


def test_session_uses_256_bit_random_token_and_persists_only_sha256_hash(db):
    user = _account(db)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)

    session, token = create_session(db, user, now=now)

    decoded = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    assert len(decoded) == 32
    assert len(token) == 43
    assert re.fullmatch(r"[a-f0-9]{64}", session.token_hash)
    assert session.token_hash == hash_session_token(token)
    assert token not in session.token_hash
    assert session.created_at.replace(tzinfo=timezone.utc) == now
    assert session.expires_at.replace(tzinfo=timezone.utc) == now + SESSION_LIFETIME

    columns = {column["name"] for column in inspect(db.get_bind()).get_columns("user_sessions")}
    assert "token" not in columns
    assert "token_hash" in columns
    persisted_hash = db.get(UserSession, session.id).token_hash
    assert persisted_hash == hash_session_token(token)


def test_session_lifecycle_rejects_malformed_unknown_expired_and_revoked(db):
    user = _account(db)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    session, token = create_session(db, user, now=now)

    assert get_session_user(db, None, now=now) is None
    assert get_session_user(db, "malformed", now=now) is None
    assert get_session_user(db, "a" * 43, now=now) is None
    assert get_session_user(db, token, now=now) == user
    assert get_session_user(db, token, now=session.expires_at) is None

    assert revoke_session(db, token, now=now + timedelta(minutes=1))
    assert session.revoked_at == now + timedelta(minutes=1)
    assert get_session_user(db, token, now=now + timedelta(minutes=1)) is None
    assert not revoke_session(db, token, now=now + timedelta(minutes=2))


@pytest.mark.parametrize(
    "active,credentials",
    [(False, True), (True, False)],
    ids=["inactive-account", "legacy-account-without-credentials"],
)
def test_ineligible_accounts_cannot_create_or_use_sessions(db, active, credentials):
    user = _account(db, active=active, credentials=credentials)
    with pytest.raises(IneligibleAccountError):
        create_session(db, user)

    assert get_session_user(db, "a" * 43) is None


def test_expired_session_is_rejected_after_persisted_expiration(db):
    user = _account(db)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    session, token = create_session(db, user, lifetime=timedelta(seconds=5), now=now)

    assert get_session_user(db, token, now=session.expires_at - timedelta(microseconds=1)) == user
    assert get_session_user(db, token, now=session.expires_at + timedelta(microseconds=1)) is None


def test_session_lifetime_and_cookie_options_are_safe(db):
    user = _account(db)
    with pytest.raises(ValueError):
        create_session(db, user, lifetime=timedelta(0))

    development = session_cookie_options("development")
    production = session_cookie_options("production")
    assert development.httponly and production.httponly
    assert not development.secure
    assert production.secure
    assert development.samesite == production.samesite == "lax"
    assert production.max_age == int(SESSION_LIFETIME.total_seconds())
    with pytest.raises(ValueError):
        session_cookie_options("unknown")
