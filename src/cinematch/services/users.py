"""Service helpers for user creation and profile construction."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from cinematch.models import AppUser
from cinematch.repositories import FeedbackRepository, RatingRepository, UserRepository, WatchlistRepository


VALID_FEEDBACK = {"like", "dislike"}


def validate_movie_id(movie_id: int, movies: pd.DataFrame | None = None) -> int:
    if movie_id <= 0:
        raise ValueError("Movie ID must be positive")
    if movies is not None and movie_id not in set(int(value) for value in movies["movieId"].tolist()):
        raise ValueError(f"Movie {movie_id} not found in the catalog")
    return int(movie_id)


def create_app_user(db: Session, *, display_name: str, movielens_user_id: int | None = None, onboarding_complete: bool = False, preferred_strategy: str | None = None, preferred_reranking: str | None = None) -> AppUser:
    if not display_name or not display_name.strip():
        raise ValueError("Display name is required")
    if len(display_name.strip()) > 80:
        raise ValueError("Display name must be at most 80 characters")
    repo = UserRepository(db)
    if movielens_user_id is not None:
        existing = repo.get_by_movielens_id(movielens_user_id)
        if existing is not None:
            raise ValueError(f"MovieLens user {movielens_user_id} already mapped to app user {existing.id}")
    return repo.create(
        display_name.strip(),
        movielens_user_id=movielens_user_id,
        onboarding_complete=onboarding_complete,
        preferred_strategy=preferred_strategy,
        preferred_reranking=preferred_reranking,
    )


def build_user_profile(db: Session, user_id: int, *, movies: pd.DataFrame | None = None) -> dict:
    repo = UserRepository(db)
    user = repo.get_by_id(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    ratings = RatingRepository(db).list_user_ratings(user_id)
    watchlist = WatchlistRepository(db).list_for_user(user_id)
    feedback = FeedbackRepository(db).list_for_user(user_id)
    profile = {
        "user": user,
        "ratings": [
            {
                "id": item.id,
                "user_id": item.user_id,
                "movie_id": item.movie_id,
                "rating": item.rating,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in ratings
        ],
        "watchlist": [
            {
                "id": item.id,
                "user_id": item.user_id,
                "movie_id": item.movie_id,
                "created_at": item.created_at.isoformat(),
            }
            for item in watchlist
        ],
        "feedback": [
            {
                "id": item.id,
                "user_id": item.user_id,
                "movie_id": item.movie_id,
                "feedback_type": item.feedback_type,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in feedback
        ],
    }

    if movies is not None:
        movie_ids = {int(value) for value in movies["movieId"].tolist()}
        for item in profile["ratings"]:
            if item["movie_id"] not in movie_ids:
                raise ValueError(f"Rating references unknown movie {item['movie_id']}")
        for item in profile["watchlist"]:
            if item["movie_id"] not in movie_ids:
                raise ValueError(f"Watchlist item references unknown movie {item['movie_id']}")
        for item in profile["feedback"]:
            if item["movie_id"] not in movie_ids:
                raise ValueError(f"Feedback references unknown movie {item['movie_id']}")

    return profile


__all__ = ["VALID_FEEDBACK", "build_user_profile", "create_app_user", "validate_movie_id"]
