"""Validated application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:4173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:5173",
)


def _read_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _parse_allowed_origins(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_ALLOWED_ORIGINS
    origins = tuple(
        origin.strip().rstrip("/")
        for origin in raw.split(",")
        if origin and origin.strip()
    )
    if not origins:
        raise ValueError("CINEMATCH_ALLOWED_ORIGINS must contain at least one origin")
    for origin in origins:
        if not (origin.startswith("http://") or origin.startswith("https://")):
            raise ValueError(f"CINEMATCH_ALLOWED_ORIGINS contains an invalid origin: {origin!r}")
        split_origin = urlsplit(origin)
        if not split_origin.scheme or not split_origin.netloc or "*" in origin:
            raise ValueError(f"CINEMATCH_ALLOWED_ORIGINS contains an invalid origin: {origin!r}")
    return origins


def redact_url_credentials(value: str) -> str:
    if not value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    if parts.scheme.startswith("sqlite"):
        return value
    username = parts.username
    password = parts.password
    if not username and not password:
        return value
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = hostname + port
    if username:
        redacted_user = username
        if password is not None:
            redacted_user = f"{username}:***"
        netloc = f"{redacted_user}@{netloc}"
    elif password is not None:
        netloc = f"***@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    database_url: str = "sqlite:///./data/cinematch.db"
    read_only_demo: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    model_path: str = "models/svd_model.pkl"
    data_path: str = "data"
    frontend_path: str = "frontend/dist"
    enable_docs: bool = True
    request_timeout_seconds: float = 30.0
    slow_request_ms: int = 500
    app_version: str = "1.1.0"

    @classmethod
    def from_env(cls) -> "Settings":
        environment = (_read_env("CINEMATCH_ENV", "development") or "development").strip().lower()
        if environment not in {"development", "staging", "production"}:
            raise ValueError("CINEMATCH_ENV must be one of: development, staging, production")

        database_url = _read_env("CINEMATCH_DATABASE_URL", "sqlite:///./data/cinematch.db") or "sqlite:///./data/cinematch.db"
        parsed_database_url = urlsplit(database_url)
        if not database_url or not parsed_database_url.scheme:
            raise ValueError("CINEMATCH_DATABASE_URL must be a valid database connection string")
        read_only_raw = (_read_env("CINEMATCH_READ_ONLY_DEMO", "false") or "false").lower()
        if read_only_raw not in {"true", "false"}:
            raise ValueError("CINEMATCH_READ_ONLY_DEMO must be either true or false")
        read_only_demo = read_only_raw == "true"
        if read_only_demo and parsed_database_url.scheme != "sqlite":
            raise ValueError("Read-only demo mode requires an isolated SQLite database")
        if environment == "production" and read_only_demo:
            demo_database_path = PurePosixPath(
                "/" + parsed_database_url.path.lstrip("/")
            )
            if not demo_database_path.is_relative_to("/tmp"):
                raise ValueError("Production read-only demo database must be under /tmp")
        if (
            environment == "production"
            and database_url == "sqlite:///./data/cinematch.db"
            and not read_only_demo
        ):
            raise ValueError("Production deployments must set CINEMATCH_DATABASE_URL to a persistent external database URL")
        if environment != "development" and parsed_database_url.scheme == "sqlite" and not read_only_demo:
            raise ValueError("SQLite is only allowed for local development; use a remote database for non-development environments")

        host = _read_env("CINEMATCH_HOST", "0.0.0.0") or "0.0.0.0"
        port = int(_read_env("CINEMATCH_PORT", "8000") or "8000")
        log_level = (_read_env("CINEMATCH_LOG_LEVEL", "INFO") or "INFO").upper()
        allowed_origins = _parse_allowed_origins(_read_env("CINEMATCH_ALLOWED_ORIGINS"))
        model_path = _read_env("CINEMATCH_MODEL_PATH", "models/svd_model.pkl") or "models/svd_model.pkl"
        data_path = _read_env("CINEMATCH_DATA_PATH", "data") or "data"
        frontend_path = _read_env("CINEMATCH_FRONTEND_PATH", "frontend/dist") or "frontend/dist"
        enable_docs = (_read_env("CINEMATCH_ENABLE_DOCS", "true") or "true").strip().lower() not in {"0", "false", "no"}
        request_timeout = float(_read_env("CINEMATCH_REQUEST_TIMEOUT_SECONDS", "30.0") or "30.0")
        slow_request_ms = int(_read_env("CINEMATCH_SLOW_REQUEST_MS", "500") or "500")

        if not host.strip():
            raise ValueError("CINEMATCH_HOST cannot be empty")
        if not 1 <= port <= 65535:
            raise ValueError(f"CINEMATCH_PORT must be between 1 and 65535; received {port!r}")
        if request_timeout <= 0:
            raise ValueError("CINEMATCH_REQUEST_TIMEOUT_SECONDS must be positive")
        if slow_request_ms <= 0:
            raise ValueError("CINEMATCH_SLOW_REQUEST_MS must be positive")
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"CINEMATCH_LOG_LEVEL must be a valid Python level; received {log_level!r}")

        return cls(
            environment=environment,
            database_url=database_url,
            read_only_demo=read_only_demo,
            host=host,
            port=port,
            log_level=log_level,
            allowed_origins=allowed_origins,
            model_path=model_path,
            data_path=data_path,
            frontend_path=frontend_path,
            enable_docs=enable_docs,
            request_timeout_seconds=request_timeout,
            slow_request_ms=slow_request_ms,
        )

    @property
    def database_url_for_logs(self) -> str:
        return redact_url_credentials(self.database_url)

    @property
    def model_file(self) -> Path:
        return (PROJECT_ROOT / self.model_path).resolve()

    @property
    def data_dir(self) -> Path:
        return (PROJECT_ROOT / self.data_path).resolve()

    @property
    def frontend_dir(self) -> Path:
        return (PROJECT_ROOT / self.frontend_path).resolve()


settings = Settings.from_env()

__all__ = [
    "DEFAULT_ALLOWED_ORIGINS",
    "PROJECT_ROOT",
    "Settings",
    "redact_url_credentials",
    "settings",
]
