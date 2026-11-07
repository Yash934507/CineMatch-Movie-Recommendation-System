"""Password and server-side session primitives for account authentication."""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from cinematch.models import AppUser, UserSession, utc_now

SESSION_TOKEN_BYTES = 32
SESSION_LIFETIME = timedelta(days=14)
MAX_PASSWORD_BYTES = 1024
MIN_PASSWORD_LENGTH = 12
SESSION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}\Z"
)

_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)


class InvalidPasswordError(ValueError):
    """Raised when a password is outside the supported input limits."""


class IneligibleAccountError(ValueError):
    """Raised when an account cannot hold authenticated sessions."""


def normalize_email(email: str) -> str:
    if not isinstance(email, str):
        raise ValueError("Invalid email address")
    normalized = email.strip().lower()
    if len(normalized) > 320 or EMAIL_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Invalid email address")
    local_part = normalized.rsplit("@", 1)[0]
    if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
        raise ValueError("Invalid email address")
    return normalized


def _validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise InvalidPasswordError("Password must be text")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise InvalidPasswordError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise InvalidPasswordError("Password exceeds the maximum supported length")


def hash_password(password: str) -> str:
    _validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    if not password_hash or not isinstance(password, str):
        return False
    try:
        _validate_password(password)
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidPasswordError, InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def hash_session_token(token: str) -> str:
    if not isinstance(token, str) or SESSION_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError("Malformed session token")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SessionCookieOptions:
    httponly: bool = True
    secure: bool = False
    samesite: str = "lax"
    max_age: int = int(SESSION_LIFETIME.total_seconds())
    path: str = "/"


def session_cookie_options(environment: str) -> SessionCookieOptions:
    if environment not in {"development", "staging", "production"}:
        raise ValueError("Unsupported application environment")
    return SessionCookieOptions(secure=environment == "production")


def create_session(
    db: Session,
    user: AppUser,
    *,
    lifetime: timedelta = SESSION_LIFETIME,
    now: datetime | None = None,
) -> tuple[UserSession, str]:
    if not user.is_active or not user.email or not user.password_hash:
        raise IneligibleAccountError("Account is not eligible for authentication")
    if lifetime <= timedelta(0):
        raise ValueError("Session lifetime must be positive")

    created_at = _utc_datetime(now or utc_now())
    token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    session = UserSession(
        user_id=user.id,
        token_hash=hash_session_token(token),
        created_at=created_at,
        expires_at=created_at + lifetime,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session, token


def get_session_user(
    db: Session,
    token: str | None,
    *,
    now: datetime | None = None,
) -> AppUser | None:
    if not isinstance(token, str) or SESSION_TOKEN_PATTERN.fullmatch(token) is None:
        return None

    session = db.execute(
        select(UserSession)
        .options(joinedload(UserSession.user))
        .where(UserSession.token_hash == hash_session_token(token))
    ).scalar_one_or_none()
    if session is None or session.revoked_at is not None:
        return None

    current_time = _utc_datetime(now or utc_now())
    if _utc_datetime(session.expires_at) <= current_time:
        return None

    user = session.user
    if not user.is_active or not user.email or not user.password_hash:
        return None
    return user


def revoke_session(
    db: Session,
    token: str | None,
    *,
    now: datetime | None = None,
) -> bool:
    if not isinstance(token, str) or SESSION_TOKEN_PATTERN.fullmatch(token) is None:
        return False

    session = db.execute(
        select(UserSession).where(UserSession.token_hash == hash_session_token(token))
    ).scalar_one_or_none()
    if session is None or session.revoked_at is not None:
        return False

    session.revoked_at = _utc_datetime(now or utc_now())
    db.commit()
    return True


class AuthenticationRateLimiter:
    """Process-local fixed-window limiter; counters are not shared across workers.

    Keys use the direct ASGI client address. Forwarded headers are deliberately
    ignored; multi-worker deployments need shared storage for a global limit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, str], tuple[float, int]] = {}

    def check(
        self,
        action: str,
        client_host: str,
        *,
        limit: int,
        window_seconds: int = 60,
        now: float | None = None,
    ) -> int | None:
        current = time.monotonic() if now is None else now
        key = (action, client_host)
        with self._lock:
            expired = [
                stored_key
                for stored_key, (started, _) in self._windows.items()
                if current - started >= window_seconds
            ]
            for stored_key in expired:
                del self._windows[stored_key]

            existing = self._windows.get(key)
            if existing is None:
                if len(self._windows) >= 10_000:
                    return window_seconds
                self._windows[key] = (current, 1)
                return None

            started, count = existing
            if count >= limit:
                return max(1, int(window_seconds - (current - started)))
            self._windows[key] = (started, count + 1)
            return None

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


AUTH_RATE_LIMITER = AuthenticationRateLimiter()


__all__ = [
    "AUTH_RATE_LIMITER",
    "AuthenticationRateLimiter",
    "IneligibleAccountError",
    "InvalidPasswordError",
    "SESSION_LIFETIME",
    "SESSION_TOKEN_BYTES",
    "SessionCookieOptions",
    "create_session",
    "get_session_user",
    "hash_password",
    "hash_session_token",
    "normalize_email",
    "revoke_session",
    "session_cookie_options",
    "verify_password",
]
