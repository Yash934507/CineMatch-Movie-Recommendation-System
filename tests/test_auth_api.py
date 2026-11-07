from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import cinematch.api as api
from cinematch.config import settings as default_settings
from cinematch.models import AppUser, Base, UserSession, utc_now
from cinematch.services.auth import (
    AUTH_RATE_LIMITER,
    hash_password,
    hash_session_token,
)


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_ORIGIN = "http://127.0.0.1:5173"


@pytest.fixture
def auth_client(monkeypatch, request):
    environment = getattr(request, "param", "development")
    monkeypatch.setattr(
        api,
        "settings",
        replace(
            default_settings,
            environment=environment,
            allowed_origins=(TRUSTED_ORIGIN,),
        ),
    )
    AUTH_RATE_LIMITER.reset()

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine, expire_on_commit=False) as db:
            yield db

    app = api.create_app(ROOT)
    app.dependency_overrides[api.get_db] = override_get_db
    client = TestClient(app)
    yield client, engine
    client.close()
    engine.dispose()
    AUTH_RATE_LIMITER.reset()


def _origin_headers(origin: str = TRUSTED_ORIGIN) -> dict[str, str]:
    return {"Origin": origin}


def _register(client: TestClient, email: str = "member@example.test"):
    return client.post(
        "/api/v1/auth/register",
        headers=_origin_headers(),
        json={
            "email": email,
            "password": "a sufficiently long password",
            "display_name": "  Cine Member  ",
        },
    )


def _login(client: TestClient, email: str = "member@example.test", password: str = "a sufficiently long password"):
    return client.post(
        "/api/v1/auth/login",
        headers=_origin_headers(),
        json={"email": email, "password": password},
    )


def test_registration_normalizes_email_and_stores_only_password_hash(auth_client):
    client, engine = auth_client
    response = _register(client, "  Member@Example.Test ")

    assert response.status_code == 201
    payload = response.json()
    assert payload["email"] == "member@example.test"
    assert payload["display_name"] == "Cine Member"
    assert not {"password", "password_hash", "token", "session_token"} & payload.keys()
    assert "set-cookie" not in response.headers

    with Session(engine) as db:
        user = db.execute(select(AppUser)).scalar_one()
        assert user.email == "member@example.test"
        assert user.password_hash is not None
        assert user.password_hash.startswith("$argon2id$")
        assert "a sufficiently long password" not in user.password_hash
        assert user.movielens_user_id is None


def test_duplicate_registration_is_rejected_without_leaking_database_error(auth_client):
    client, _ = auth_client
    assert _register(client).status_code == 201
    response = _register(client, "MEMBER@example.test")

    assert response.status_code == 409
    assert response.json() == {"detail": "An account with this email already exists"}
    assert "IntegrityError" not in response.text


@pytest.mark.parametrize(
    "email,password,display_name",
    [
        ("not-an-email", "a sufficiently long password", "Member"),
        ("member@example.test", "pw", "Member"),
        ("member@example.test", "a sufficiently long password", "  "),
        ("member@example.test", "é" * 600, "Member"),
    ],
)
def test_registration_rejects_invalid_fields(auth_client, email, password, display_name):
    client, _ = auth_client
    response = client.post(
        "/api/v1/auth/register",
        headers=_origin_headers(),
        json={"email": email, "password": password, "display_name": display_name},
    )
    assert response.status_code == 422
    assert password not in response.text


def test_login_sets_secure_session_cookie_and_me_returns_safe_account(auth_client, caplog):
    client, engine = auth_client
    assert _register(client).status_code == 201
    response = _login(client, " MEMBER@EXAMPLE.TEST ")

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "member@example.test"
    assert not {"password", "password_hash", "token", "session_token"} & payload.keys()
    cookie = response.headers["set-cookie"]
    assert "cinematch_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert "Secure" not in cookie
    token = client.cookies.get(api.SESSION_COOKIE_NAME)
    assert isinstance(token, str) and token
    assert token not in response.text

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["id"] == payload["id"]
    assert "password_hash" not in me.json()

    with Session(engine) as db:
        session = db.execute(select(UserSession)).scalar_one()
        assert session.token_hash == hash_session_token(token)
    assert "a sufficiently long password" not in caplog.text
    assert token not in caplog.text


@pytest.mark.parametrize("auth_client", ["production"], indirect=True)
def test_production_session_cookie_has_secure_attribute(auth_client):
    client, _ = auth_client
    assert _register(client).status_code == 201
    response = _login(client)
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie


def test_invalid_missing_and_expired_sessions_return_401(auth_client):
    client, engine = auth_client
    assert client.get("/api/v1/auth/me").status_code == 401
    client.cookies.set(api.SESSION_COOKIE_NAME, "malformed")
    assert client.get("/api/v1/auth/me").status_code == 401

    with Session(engine, expire_on_commit=False) as db:
        user = AppUser(
            email="expired@example.test",
            password_hash=hash_password("another sufficiently long password"),
            display_name="Expired account",
            is_active=True,
        )
        db.add(user)
        db.commit()
        expired_session = UserSession(
            user_id=user.id,
            token_hash=hash_session_token("A" * 43),
            expires_at=utc_now() - timedelta(seconds=1),
        )
        db.add(expired_session)
        db.commit()

    client.cookies.set(api.SESSION_COOKIE_NAME, "A" * 43)
    assert client.get("/api/v1/auth/me").status_code == 401


def test_logout_revokes_session_and_clears_cookie(auth_client):
    client, engine = auth_client
    assert _register(client).status_code == 201
    assert _login(client).status_code == 200
    token = client.cookies.get(api.SESSION_COOKIE_NAME)

    response = client.post("/api/v1/auth/logout", headers=_origin_headers())

    assert response.status_code == 204
    assert api.SESSION_COOKIE_NAME not in client.cookies
    assert client.get("/api/v1/auth/me").status_code == 401
    with Session(engine) as db:
        session = db.execute(select(UserSession)).scalar_one()
        assert session.token_hash == hash_session_token(token)
        assert session.revoked_at is not None

    repeated = client.post(
        "/api/v1/auth/logout",
        headers=_origin_headers(),
    )
    assert repeated.status_code == 204


def test_inactive_accounts_receive_generic_login_failure(auth_client):
    client, engine = auth_client
    assert _register(client).status_code == 201
    with Session(engine) as db:
        user = db.execute(select(AppUser)).scalar_one()
        user.is_active = False
        db.commit()

    response = _login(client)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid email or password"}
    assert "set-cookie" not in response.headers


def test_login_failure_is_generic_for_unknown_email_and_bad_password(auth_client):
    client, _ = auth_client
    unknown = _login(client, "missing@example.test", "a sufficiently long password")
    assert unknown.status_code == 401
    assert unknown.json() == {"detail": "Invalid email or password"}

    assert _register(client).status_code == 201
    wrong_password = _login(client, password="incorrect but long password")
    assert wrong_password.status_code == 401
    assert wrong_password.json() == unknown.json()


def test_auth_state_changes_reject_missing_or_untrusted_origin(auth_client):
    client, _ = auth_client
    missing_origin = client.post(
        "/api/v1/auth/register",
        json={
            "email": "member@example.test",
            "password": "a sufficiently long password",
            "display_name": "Member",
        },
    )
    bad_origin = client.post(
        "/api/v1/auth/login",
        headers=_origin_headers("https://attacker.example"),
        json={"email": "member@example.test", "password": "a sufficiently long password"},
    )
    assert missing_origin.status_code == 403
    assert bad_origin.status_code == 403


def test_cookie_authenticated_writes_require_trusted_origin(auth_client):
    client, _ = auth_client
    assert _register(client).status_code == 201
    assert _login(client).status_code == 200

    response = client.post(
        "/api/v1/users/1/ratings",
        headers=_origin_headers("https://attacker.example"),
        json={"movie_id": 260, "rating": 4.0},
    )
    assert response.status_code == 403


def test_cors_allows_credentials_only_for_configured_origins(auth_client):
    client, _ = auth_client
    response = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": TRUSTED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == TRUSTED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_authentication_rate_limit_returns_429_and_ignores_forwarded_headers(auth_client):
    client, _ = auth_client
    for attempt in range(10):
        response = _login(
            client,
            email=f"unknown{attempt}@example.test",
            password="a sufficiently long password",
        )
        assert response.status_code == 401

    limited = client.post(
        "/api/v1/auth/login",
        headers={**_origin_headers(), "X-Forwarded-For": "203.0.113.99"},
        json={"email": "unknown@example.test", "password": "a sufficiently long password"},
    )
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) > 0
