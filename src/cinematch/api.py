"""FastAPI application for the CineMatch recommendation engine."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path as APIPath, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import settings
from .data import load_movielens
from .database import get_db, initialize_database
from .evaluation import popularity_fallback
from .hybrid import (
    DEFAULT_WEIGHTS,
    build_genre_profile,
    candidate_genre_similarity,
    rank_hybrid_candidates,
)
from .repositories import (
    FeedbackRepository,
    RatingRepository,
    UserRepository,
    WatchlistRepository,
)
from .logging_config import configure_logging, get_logger
from .onboarding import MIN_RATINGS, fold_in, rank_fold_in, score_fold_in
from .schemas import (
    AppUserRead,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthenticatedUserRead,
    RecommendationEventCreate,
    UserFeedbackCreate,
    UserProfileRead,
    UserRatingCreate,
    UserRatingUpdate,
    WatchlistItemRead,
)
from .models import AppUser
from .services.auth import (
    AUTH_RATE_LIMITER,
    IneligibleAccountError,
    create_session,
    get_session_user,
    hash_password,
    revoke_session,
    session_cookie_options,
    verify_password,
)
from .services.interactions import record_event, set_feedback, toggle_watchlist, upsert_rating
from .services.users import build_user_profile
from .svd_model import load_model, recommend_svd

logger = get_logger("cinematch.api")
configure_logging(settings)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_COOKIE_NAME = "cinematch_session"


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> AppUser:
    user = get_session_user(db, request.cookies.get(SESSION_COOKIE_NAME))
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_account_owner(requested_user_id: int, current_user: AppUser) -> None:
    if requested_user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this resource",
        )


class Movie(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int
    title: str
    genres: list[str]
    rating_count: int
    mean_rating: float | None = None


class ScoredMovie(Movie):
    predicted_rating: float


class SimilarMovie(Movie):
    similarity: float


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str
    cold_start: bool
    recommendations: list[ScoredMovie]


class AccountRecommendationResponse(BaseModel):
    strategy: str
    cold_start: bool
    recommendations: list[ScoredMovie]


class AccountPredictionResponse(BaseModel):
    movie_id: int
    predicted_rating: float
    cold_start: bool
    model: str = "surprise-svd-20"


class OnboardingRating(BaseModel):
    movie_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)


class OnboardingRequest(BaseModel):
    ratings: list[OnboardingRating] = Field(min_length=MIN_RATINGS, max_length=50)
    limit: int = Field(default=10, ge=1, le=50)


class OnboardingResponse(BaseModel):
    strategy: str = "svd-fold-in"
    ratings_used: int
    recommendations: list[ScoredMovie]


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float
    cold_start: bool
    model: str = "surprise-svd-20"


class HealthResponse(BaseModel):
    status: str
    model: str
    users: int
    movies: int
    ratings: int
    version: str | None = None


class ModelHealth(BaseModel):
    name: str
    factors: int | None = None
    loaded: bool = False
    checksum: str | None = None


class DatabaseHealth(BaseModel):
    reachable: bool
    migration_current: bool


class ReadyResponse(BaseModel):
    status: str
    version: str
    model: ModelHealth
    database: DatabaseHealth


class UserCreateResponse(AppUserRead):
    pass


def _genres(value: str) -> list[str]:
    if not value or value == "(no genres listed)":
        return []
    return value.split("|")


def _sanitize_request_id(raw_value: str | None) -> str:
    if raw_value is None or not raw_value.strip():
        return uuid.uuid4().hex
    candidate = raw_value.strip()
    if len(candidate) > 128:
        candidate = candidate[:128]
    if all(ch.isalnum() or ch in "-_." for ch in candidate):
        return candidate
    return uuid.uuid4().hex


def _model_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


@dataclass
class RecommenderService:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    model: object
    cold_start: pd.DataFrame
    movie_lookup: pd.DataFrame
    known_users: set[int]
    recommendation_cache: dict[tuple[int, int], list[dict]] = field(default_factory=dict)

    @classmethod
    def load(cls, project_root: Path) -> "RecommenderService":
        ratings, movies, _ = load_movielens(project_root / "data" / "raw")
        model_path = project_root / "models" / "svd_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved SVD model not found: {model_path}")
        model = load_model(model_path)
        stats = ratings.groupby("movieId").agg(
            rating_count=("rating", "size"), mean_rating=("rating", "mean")
        )
        movie_lookup = movies.merge(stats, on="movieId", how="left").fillna(
            {"rating_count": 0, "mean_rating": 0.0}
        )
        fallback_path = project_root / "data" / "processed" / "week4_cold_start_recommendations.csv"
        if fallback_path.exists():
            cold_start = pd.read_csv(fallback_path)
        else:
            cold_start = popularity_fallback(ratings, movies, n=50, prior_weight=50)
        return cls(
            movies=movies,
            ratings=ratings,
            model=model,
            cold_start=cold_start,
            movie_lookup=movie_lookup.set_index("movieId", drop=False),
            known_users=set(int(value) for value in ratings["userId"].unique()),
        )

    def movie_payload(self, movie_id: int) -> dict:
        if movie_id not in self.movie_lookup.index:
            raise KeyError(movie_id)
        row = self.movie_lookup.loc[movie_id]
        return {
            "movie_id": int(row.movieId),
            "title": str(row.title),
            "genres": _genres(str(row.genres)),
            "rating_count": int(row.rating_count),
            "mean_rating": round(float(row.mean_rating), 4) if row.rating_count else None,
        }

    def search(self, query: str, genre: str | None, limit: int) -> list[dict]:
        frame = self.movie_lookup.reset_index(drop=True)
        if query:
            frame = frame.loc[frame["title"].str.contains(query, case=False, regex=False, na=False)]
        if genre:
            frame = frame.loc[frame["genres"].str.contains(genre, case=False, regex=False, na=False)]
        frame = frame.sort_values(["rating_count", "mean_rating", "movieId"], ascending=[False, False, True])
        return [self.movie_payload(int(movie_id)) for movie_id in frame.head(limit)["movieId"]]

    def recommendations(self, user_id: int, limit: int, strategy: str = "svd") -> RecommendationResponse:
        if user_id not in self.known_users:
            rows = []
            for row in self.cold_start.head(limit).itertuples(index=False):
                payload = self.movie_payload(int(row.movieId))
                payload["predicted_rating"] = round(float(row.weighted_score), 4)
                rows.append(payload)
            return RecommendationResponse(
                user_id=user_id,
                strategy="bayesian-popularity",
                cold_start=True,
                recommendations=rows,
            )

        cache_key = (user_id, limit, strategy.lower())
        if cache_key not in self.recommendation_cache:
            if strategy.lower() == "hybrid":
                known = set(int(value) for value in self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])
                candidate_ids = [
                    int(movie_id)
                    for movie_id in self.movies["movieId"].tolist()
                    if int(movie_id) not in known
                ]
                collaborative_scores = {
                    int(movie_id): float(self.model.predict(int(user_id), int(movie_id)).est)
                    for movie_id in candidate_ids
                }
                profile = build_genre_profile(self.ratings, self.movies, int(user_id))
                genre_scores = {
                    int(movie_id): float(candidate_genre_similarity(profile, self.movies.loc[self.movies["movieId"] == int(movie_id), "genres"].iloc[0]))
                    for movie_id in candidate_ids
                }
                popularity_scores = {
                    int(row.movieId): float(row.weighted_score)
                    for row in self.cold_start.itertuples(index=False)
                    if int(row.movieId) in candidate_ids
                }
                ranked = rank_hybrid_candidates(
                    collaborative_scores,
                    genre_scores,
                    popularity_scores=popularity_scores,
                    weights=DEFAULT_WEIGHTS,
                )
                rows = []
                for movie_id, score in ranked[:limit]:
                    payload = self.movie_payload(int(movie_id))
                    payload["predicted_rating"] = round(float(score), 4)
                    rows.append(payload)
                self.recommendation_cache[cache_key] = rows
            else:
                frame = recommend_svd(self.model, user_id, self.ratings, self.movies, n=limit)
                rows = []
                for row in frame.itertuples(index=False):
                    payload = self.movie_payload(int(row.movieId))
                    payload["predicted_rating"] = round(float(row.predicted_rating), 4)
                    rows.append(payload)
                self.recommendation_cache[cache_key] = rows
        return RecommendationResponse(
            user_id=user_id,
            strategy="hybrid" if strategy.lower() == "hybrid" else "surprise-svd-20",
            cold_start=False,
            recommendations=self.recommendation_cache[cache_key],
        )

    def similar_movies(self, movie_id: int, limit: int) -> list[dict]:
        try:
            target_inner = self.model.trainset.to_inner_iid(movie_id)
        except ValueError as error:
            if movie_id not in self.movie_lookup.index:
                raise KeyError(movie_id) from error
            return []
        factors = self.model.qi
        target = factors[target_inner]
        norms = np.linalg.norm(factors, axis=1)
        denominator = norms * np.linalg.norm(target)
        scores = np.divide(
            factors @ target,
            denominator,
            out=np.zeros(len(factors), dtype=np.float64),
            where=denominator > 0,
        )
        scores[target_inner] = -np.inf
        order = np.argsort(scores)[::-1]
        results = []
        for inner_id in order:
            raw_id = int(self.model.trainset.to_raw_iid(int(inner_id)))
            if raw_id not in self.movie_lookup.index:
                continue
            payload = self.movie_payload(raw_id)
            payload["similarity"] = round(float(scores[inner_id]), 6)
            results.append(payload)
            if len(results) == limit:
                break
        return results

    def onboarding_recommendations(
        self, ratings: list[OnboardingRating], limit: int
    ) -> OnboardingResponse:
        by_movie = {entry.movie_id: entry.rating for entry in ratings}
        if len(by_movie) != len(ratings):
            raise ValueError("Each onboarding movie can only be rated once")

        for movie_id in by_movie:
            if movie_id not in self.movie_lookup.index:
                raise KeyError(movie_id)
        profile = fold_in(self.model, list(by_movie.items()))
        candidates = [
            int(self.model.trainset.to_raw_iid(inner_id))
            for inner_id in range(len(self.model.qi))
            if int(self.model.trainset.to_raw_iid(inner_id)) in self.movie_lookup.index
        ]
        scored = rank_fold_in(self.model, profile, candidates)
        rows = []
        for movie_id, estimate in scored[:limit]:
            payload = self.movie_payload(movie_id)
            payload["predicted_rating"] = round(estimate, 4)
            rows.append(payload)
        return OnboardingResponse(ratings_used=len(ratings), recommendations=rows)


def create_app(
    project_root: str | Path = PROJECT_ROOT,
    frontend_dir: str | Path | None = None,
) -> FastAPI:
    root = Path(project_root).resolve()
    web_root = Path(frontend_dir).resolve() if frontend_dir else root / "frontend" / "dist"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.environment != "production":
            initialize_database()
        app.state.settings = settings
        app.state.version = settings.app_version
        model_file = settings.model_file if settings.model_file.is_absolute() else root / settings.model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Saved SVD model not found: {model_file}")
        if not (root / settings.data_path / "raw").exists():
            raise FileNotFoundError(f"MovieLens data directory not found: {root / settings.data_path / 'raw'}")
        app.state.model_checksum = _model_checksum(model_file)
        try:
            from alembic.config import Config
            from alembic.script import ScriptDirectory

            alembic_config = Config(str(root / "alembic.ini"))
            alembic_config.set_main_option("sqlalchemy.url", settings.database_url)
            script_dir = ScriptDirectory.from_config(alembic_config)
            current_head = script_dir.get_current_head()
            expected_heads = script_dir.get_heads()
            app.state.migration_current = bool(current_head) and bool(expected_heads) and current_head == expected_heads[0]
        except Exception as error:
            app.state.migration_current = False
            logger.warning("migration_check_failed", extra={"error_type": type(error).__name__})
        if not app.state.migration_current:
            raise RuntimeError("Database migrations are not current; run the release migration step before startup")
        app.state.recommender = RecommenderService.load(root)
        app.state.model_loaded = True
        logger.info(
            "application_startup",
            extra={
                "environment": settings.environment,
                "application_version": settings.app_version,
                "model_name": "surprise-svd",
                "model_artifact": model_file.name,
                "model_checksum": app.state.model_checksum,
                "database_url": settings.database_url_for_logs,
            },
        )
        yield
        app.state.recommender.recommendation_cache.clear()
        app.state.model_loaded = False
        logger.info("application_shutdown", extra={"environment": settings.environment, "application_version": settings.app_version})

    app = FastAPI(
        title="CineMatch API",
        version=settings.app_version,
        description="Movie recommendations powered by the selected Surprise SVD model.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )
    app.state.read_only_demo = settings.read_only_demo

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        errors = [
            {
                "loc": item["loc"],
                "msg": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": errors}),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = _sanitize_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started = time.perf_counter()
        unsafe_method = request.method not in {"GET", "HEAD", "OPTIONS"}
        if (
            request.app.state.read_only_demo
            and unsafe_method
            and request.url.path != "/api/v1/onboarding/recommendations"
        ):
            response = JSONResponse(
                status_code=403,
                content={"detail": "This public demo is read-only"},
            )
            response.headers["X-Request-ID"] = request_id
            return response
        auth_endpoint = request.url.path in {
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/auth/logout",
        }
        cookie_authenticated_write = unsafe_method and SESSION_COOKIE_NAME in request.cookies
        if unsafe_method and (auth_endpoint or cookie_authenticated_write):
            origin = request.headers.get("origin", "").rstrip("/")
            if not origin or origin not in settings.allowed_origins:
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Request origin is not allowed"},
                )
                response.headers["X-Request-ID"] = request_id
                return response
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' http://localhost:* http://127.0.0.1:*; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route_path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "slow_request": duration_ms >= settings.slow_request_ms,
            },
        )
        return response

    def service(request: Request) -> RecommenderService:
        return request.app.state.recommender

    def enforce_auth_rate_limit(action: str, request: Request, limit: int) -> None:
        client_host = request.client.host if request.client is not None else "unknown"
        retry_after = AUTH_RATE_LIMITER.check(
            action,
            client_host,
            limit=limit,
            window_seconds=60,
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many authentication attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )

    def authenticated_user_payload(user: AppUser) -> dict[str, int | str]:
        if user.email is None:
            raise RuntimeError("Authenticated account is missing its email")
        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat(),
        }

    @app.post(
        "/api/v1/auth/register",
        response_model=AuthenticatedUserRead,
        status_code=201,
        tags=["authentication"],
    )
    def register(
        payload: AuthRegisterRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> dict[str, int | str]:
        enforce_auth_rate_limit("register", request, limit=5)
        user = AppUser(
            email=payload.email,
            password_hash=hash_password(payload.password),
            display_name=payload.display_name,
            is_active=True,
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError as error:
            db.rollback()
            logger.info("account_registration_conflict")
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists",
            ) from error
        db.refresh(user)
        return authenticated_user_payload(user)

    @app.post(
        "/api/v1/auth/login",
        response_model=AuthenticatedUserRead,
        tags=["authentication"],
    )
    def login(
        payload: AuthLoginRequest,
        request: Request,
        response: Response,
        db: Session = Depends(get_db),
    ) -> dict[str, int | str]:
        enforce_auth_rate_limit("login", request, limit=10)
        user = db.execute(
            select(AppUser).where(AppUser.email == payload.email)
        ).scalar_one_or_none()
        password_matches = verify_password(
            user.password_hash if user is not None else None,
            payload.password,
        )
        if (
            user is None
            or not password_matches
            or not user.is_active
            or not user.email
        ):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        try:
            _, session_token = create_session(db, user)
        except IneligibleAccountError as error:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password",
            ) from error
        cookie = session_cookie_options(settings.environment)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_token,
            max_age=cookie.max_age,
            path=cookie.path,
            secure=cookie.secure,
            httponly=cookie.httponly,
            samesite=cookie.samesite,
        )
        return authenticated_user_payload(user)

    @app.post(
        "/api/v1/auth/logout",
        status_code=204,
        tags=["authentication"],
    )
    def logout(
        request: Request,
        response: Response,
        db: Session = Depends(get_db),
    ) -> Response:
        revoke_session(db, request.cookies.get(SESSION_COOKIE_NAME))
        cookie = session_cookie_options(settings.environment)
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            path=cookie.path,
            secure=cookie.secure,
            httponly=cookie.httponly,
            samesite=cookie.samesite,
        )
        response.status_code = 204
        return response

    @app.get(
        "/api/v1/auth/me",
        response_model=AuthenticatedUserRead,
        tags=["authentication"],
    )
    def auth_me(user: AppUser = Depends(get_current_user)) -> dict[str, int | str]:
        return authenticated_user_payload(user)

    @app.get("/api/v1", tags=["meta"])
    def root_endpoint() -> dict[str, str]:
        return {"name": "CineMatch API", "docs": "/docs", "health": "/api/v1/health"}

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
    def health(recommender: RecommenderService = Depends(service)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model="surprise-svd-20",
            users=len(recommender.known_users),
            movies=len(recommender.movies),
            ratings=len(recommender.ratings),
            version=settings.app_version,
        )

    @app.get("/api/v1/ready", response_model=ReadyResponse, tags=["meta"])
    def ready() -> ReadyResponse:
        database_reachable = False
        migration_current = getattr(app.state, "migration_current", True)
        try:
            with next(get_db()) as session:
                session.execute(text("SELECT 1"))
                database_reachable = True
        except Exception as error:
            logger.warning("readiness_check_failed", extra={"error_type": type(error).__name__})
            return ReadyResponse(
                status="not_ready",
                version=settings.app_version,
                model=ModelHealth(name="surprise-svd", factors=None, loaded=False, checksum=None),
                database=DatabaseHealth(reachable=False, migration_current=migration_current),
            )

        app_state = app.state
        model_loaded = bool(getattr(app_state, "model_loaded", False))
        model_state = getattr(app_state, "recommender", None)
        factors = None
        if model_state is not None and hasattr(model_state.model, "trainset"):
            factors = getattr(model_state.model.trainset, "n_factors", None)
        ready_model = ModelHealth(
            name="surprise-svd",
            factors=factors,
            loaded=model_loaded,
            checksum=getattr(app_state, "model_checksum", None),
        )
        response = ReadyResponse(
            status="ready" if model_loaded and database_reachable else "not_ready",
            version=settings.app_version,
            model=ready_model,
            database=DatabaseHealth(reachable=database_reachable, migration_current=migration_current),
        )
        if response.status != "ready":
            raise HTTPException(status_code=503, detail="Application is not ready")
        return response

    @app.post("/api/v1/users", status_code=410, tags=["users"], deprecated=True)
    def create_user() -> None:
        raise HTTPException(
            status_code=410,
            detail="Use /api/v1/auth/register to create an application account",
        )

    @app.get("/api/v1/users/{user_id}", response_model=AppUserRead, tags=["users"])
    def read_user(
        user_id: int,
        db: Session = Depends(get_db),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        user = UserRepository(db).get_by_id(current_user.id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": user.id,
            "display_name": user.display_name,
            "movielens_user_id": user.movielens_user_id,
            "onboarding_complete": user.onboarding_complete,
            "preferred_strategy": user.preferred_strategy,
            "preferred_reranking": user.preferred_reranking,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        }

    @app.get("/api/v1/users/{user_id}/profile", response_model=UserProfileRead, tags=["users"])
    def read_profile(
        user_id: int,
        db: Session = Depends(get_db),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        try:
            return build_user_profile(db, current_user.id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="User profile not found") from error

    @app.post("/api/v1/users/{user_id}/ratings", response_model=dict, tags=["users"])
    def create_rating(
        user_id: int,
        payload: UserRatingCreate,
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        try:
            return upsert_rating(db, current_user.id, payload.movie_id, payload.rating, movies=recommender.movies)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/v1/users/{user_id}/ratings/{movie_id}", response_model=dict, tags=["users"])
    def update_rating(
        user_id: int,
        movie_id: int,
        payload: UserRatingUpdate,
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        try:
            return upsert_rating(db, current_user.id, movie_id, payload.rating, movies=recommender.movies)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/v1/users/{user_id}/ratings/{movie_id}", tags=["users"])
    def delete_rating(
        user_id: int,
        movie_id: int,
        db: Session = Depends(get_db),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        if not RatingRepository(db).delete(current_user.id, movie_id):
            raise HTTPException(status_code=404, detail="Rating not found")
        return {"deleted": True, "user_id": current_user.id, "movie_id": movie_id}

    @app.post("/api/v1/users/{user_id}/watchlist", response_model=dict, tags=["users"])
    def add_watchlist(
        user_id: int,
        payload: dict,
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        try:
            movie_id = int(payload.get("movie_id"))
            return toggle_watchlist(db, current_user.id, movie_id, movies=recommender.movies, add=True)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail="movie_id must be a positive integer") from error

    @app.delete("/api/v1/users/{user_id}/watchlist/{movie_id}", tags=["users"])
    def remove_watchlist(
        user_id: int,
        movie_id: int,
        db: Session = Depends(get_db),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        removed = WatchlistRepository(db).remove(current_user.id, movie_id)
        return {"removed": removed, "user_id": current_user.id, "movie_id": movie_id}

    @app.post("/api/v1/users/{user_id}/feedback", response_model=dict, tags=["users"])
    def submit_feedback(
        user_id: int,
        payload: UserFeedbackCreate,
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        try:
            return set_feedback(db, current_user.id, payload.movie_id, payload.feedback_type, movies=recommender.movies)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/v1/users/{user_id}/feedback/{movie_id}", tags=["users"])
    def delete_feedback(
        user_id: int,
        movie_id: int,
        db: Session = Depends(get_db),
        current_user: AppUser = Depends(get_current_user),
    ) -> dict:
        require_account_owner(user_id, current_user)
        removed = FeedbackRepository(db).remove(current_user.id, movie_id)
        return {"removed": removed, "user_id": current_user.id, "movie_id": movie_id}

    @app.get(
        "/api/v1/me/recommendations",
        response_model=AccountRecommendationResponse,
        tags=["recommendations"],
    )
    def account_recommendations(
        limit: int = Query(default=10, ge=1, le=50),
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> AccountRecommendationResponse:
        ratings = RatingRepository(db).list_user_ratings(current_user.id)
        rated_movie_ids = {int(rating.movie_id) for rating in ratings}

        if len(ratings) < MIN_RATINGS:
            candidates = recommender.cold_start.loc[
                ~recommender.cold_start["movieId"].isin(rated_movie_ids)
            ]
            recommendations = []
            for row in candidates.head(limit).itertuples(index=False):
                payload = recommender.movie_payload(int(row.movieId))
                payload["predicted_rating"] = round(float(row.weighted_score), 4)
                recommendations.append(payload)
            return AccountRecommendationResponse(
                strategy="bayesian-popularity",
                cold_start=True,
                recommendations=recommendations,
            )

        profile = fold_in(
            recommender.model,
            [(int(rating.movie_id), float(rating.rating)) for rating in ratings],
        )
        candidates = [
            int(movie_id)
            for movie_id in recommender.movies["movieId"].tolist()
            if int(movie_id) not in rated_movie_ids
        ]
        ranked = rank_fold_in(recommender.model, profile, candidates)
        recommendations = []
        for movie_id, estimate in ranked[:limit]:
            payload = recommender.movie_payload(movie_id)
            payload["predicted_rating"] = round(float(estimate), 4)
            recommendations.append(payload)
        return AccountRecommendationResponse(
            strategy="svd-fold-in",
            cold_start=False,
            recommendations=recommendations,
        )

    @app.get(
        "/api/v1/me/predictions/{movie_id}",
        response_model=AccountPredictionResponse,
        tags=["recommendations"],
    )
    def account_prediction(
        movie_id: int = APIPath(gt=0),
        db: Session = Depends(get_db),
        recommender: RecommenderService = Depends(service),
        current_user: AppUser = Depends(get_current_user),
    ) -> AccountPredictionResponse:
        if movie_id not in recommender.movie_lookup.index:
            raise HTTPException(status_code=404, detail="Movie not found")
        ratings = RatingRepository(db).list_user_ratings(current_user.id)
        if len(ratings) < MIN_RATINGS:
            raise HTTPException(
                status_code=409,
                detail=f"At least {MIN_RATINGS} ratings are required for a personalized prediction",
            )
        profile = fold_in(
            recommender.model,
            [(int(rating.movie_id), float(rating.rating)) for rating in ratings],
        )
        estimate = float(
            score_fold_in(recommender.model, profile, [movie_id])[0]
        )
        return AccountPredictionResponse(
            movie_id=movie_id,
            predicted_rating=round(estimate, 4),
            cold_start=False,
        )

    @app.post("/api/v1/events", response_model=dict, tags=["users"])
    def log_event(payload: RecommendationEventCreate, db: Session = Depends(get_db), recommender: RecommenderService = Depends(service)) -> dict:
        try:
            if payload.movie_id is not None:
                recommender.movie_payload(payload.movie_id)
            return record_event(
                db,
                user_id=None,
                session_id=payload.session_id,
                event_type=payload.event_type,
                movie_id=payload.movie_id,
                recommendation_strategy=payload.recommendation_strategy,
                reranking_strategy=payload.reranking_strategy,
                position=payload.position,
                request_id=payload.request_id,
                metadata_json=payload.metadata_json,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Movie {error.args[0]} not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/v1/movies", response_model=list[Movie], tags=["movies"])
    def search_movies(
        q: str = Query(default="", max_length=100),
        genre: str | None = Query(default=None, max_length=50),
        limit: int = Query(default=20, ge=1, le=100),
        recommender: RecommenderService = Depends(service),
    ) -> list[dict]:
        return recommender.search(q.strip(), genre.strip() if genre else None, limit)

    @app.get("/api/v1/movies/{movie_id}", response_model=Movie, tags=["movies"])
    def movie_details(
        movie_id: int,
        recommender: RecommenderService = Depends(service),
    ) -> dict:
        try:
            return recommender.movie_payload(movie_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Movie not found") from error

    @app.get(
        "/api/v1/movies/{movie_id}/similar",
        response_model=list[SimilarMovie],
        tags=["movies"],
    )
    def similar_movies(
        movie_id: int,
        limit: int = Query(default=10, ge=1, le=50),
        recommender: RecommenderService = Depends(service),
    ) -> list[dict]:
        try:
            return recommender.similar_movies(movie_id, limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Movie not found") from error

    @app.get(
        "/api/v1/demo/recommendations/{user_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        summary="Read-only MovieLens demo recommendations",
        description="The path ID is a MovieLens dataset user ID, not an application account ID. This endpoint reads only the public training dataset and does not access application account records.",
    )
    @app.get(
        "/api/v1/recommendations/{user_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        deprecated=True,
        summary="Legacy read-only MovieLens demo recommendations",
        description="The path ID is a MovieLens dataset user ID, not an application account ID. Use /api/v1/demo/recommendations/{user_id}. This endpoint reads only the public training dataset and does not access application account records.",
    )
    def recommendations(
        user_id: int = APIPath(gt=0),
        limit: int = Query(default=10, ge=1, le=50),
        strategy: str = Query(default="svd", pattern="^(svd|hybrid)$"),
        recommender: RecommenderService = Depends(service),
    ) -> RecommendationResponse:
        return recommender.recommendations(user_id, limit, strategy)

    @app.post(
        "/api/v1/onboarding/recommendations",
        response_model=OnboardingResponse,
        tags=["recommendations"],
    )
    def onboarding_recommendations(
        request: OnboardingRequest,
        recommender: RecommenderService = Depends(service),
    ) -> OnboardingResponse:
        try:
            return recommender.onboarding_recommendations(request.ratings, request.limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Movie {error.args[0]} not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get(
        "/api/v1/demo/predictions/{user_id}/{movie_id}",
        response_model=PredictionResponse,
        tags=["recommendations"],
        summary="Read-only MovieLens demo prediction",
        description="The path ID is a MovieLens dataset user ID, not an application account ID. This endpoint reads only the public training dataset and does not access application account records.",
    )
    @app.get(
        "/api/v1/predictions/{user_id}/{movie_id}",
        response_model=PredictionResponse,
        tags=["recommendations"],
        deprecated=True,
        summary="Legacy read-only MovieLens demo prediction",
        description="The path ID is a MovieLens dataset user ID, not an application account ID. Use /api/v1/demo/predictions/{user_id}/{movie_id}. This endpoint reads only the public training dataset and does not access application account records.",
    )
    def prediction(
        user_id: int = APIPath(gt=0),
        movie_id: int = APIPath(gt=0),
        recommender: RecommenderService = Depends(service),
    ) -> PredictionResponse:
        if movie_id not in recommender.movie_lookup.index:
            raise HTTPException(status_code=404, detail="Movie not found")
        estimate = recommender.model.predict(user_id, movie_id)
        return PredictionResponse(
            user_id=user_id,
            movie_id=movie_id,
            predicted_rating=round(float(estimate.est), 4),
            cold_start=user_id not in recommender.known_users,
        )

    # In production the Vite build is served by the same process as the API.
    # API routes are registered first so the SPA fallback cannot shadow them.
    index_file = web_root / "index.html"
    assets_dir = web_root / "assets"
    if index_file.is_file():
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend_fallback(full_path: str) -> Response:
            if full_path == "api" or full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            candidate = (web_root / full_path).resolve()
            if candidate.is_relative_to(web_root) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_file)
    else:
        @app.get("/", tags=["meta"])
        def development_root() -> dict[str, str]:
            return {"name": "CineMatch API", "docs": "/docs", "health": "/api/v1/health"}

    return app


app = create_app()
