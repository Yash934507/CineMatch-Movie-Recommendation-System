"""Interaction service methods for ratings, watchlists, feedback, and events."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from cinematch.repositories import EventRepository, FeedbackRepository, RatingRepository, UserRepository, WatchlistRepository
from cinematch.services.users import VALID_FEEDBACK, validate_movie_id


def upsert_rating(db: Session, user_id: int, movie_id: int, rating: float, *, movies: pd.DataFrame | None = None) -> dict:
    user = UserRepository(db).get_by_id(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    validate_movie_id(movie_id, movies)
    if not (0.5 <= float(rating) <= 5.0):
        raise ValueError("Rating must be between 0.5 and 5.0")

    entry = RatingRepository(db).upsert(user_id, movie_id, rating)
    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "movie_id": entry.movie_id,
        "rating": entry.rating,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def toggle_watchlist(db: Session, user_id: int, movie_id: int, *, movies: pd.DataFrame | None = None, add: bool | None = None) -> dict:
    user = UserRepository(db).get_by_id(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    validate_movie_id(movie_id, movies)

    repo = WatchlistRepository(db)
    exists = repo.has_item(user_id, movie_id)
    if add is None:
        add = not exists
    if add:
        if exists:
            return {"added": False, "already_present": True}
        entry = repo.add(user_id, movie_id)
        return {
            "added": True,
            "item": {
                "id": entry.id,
                "user_id": entry.user_id,
                "movie_id": entry.movie_id,
                "created_at": entry.created_at.isoformat(),
            },
        }
    removed = repo.remove(user_id, movie_id)
    return {"added": False, "removed": removed, "already_present": False}


def set_feedback(db: Session, user_id: int, movie_id: int, feedback_type: str, *, movies: pd.DataFrame | None = None) -> dict:
    user = UserRepository(db).get_by_id(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    validate_movie_id(movie_id, movies)
    normalized = str(feedback_type).strip().lower()
    if normalized not in VALID_FEEDBACK:
        raise ValueError("Feedback type must be 'like' or 'dislike'")

    entry = FeedbackRepository(db).upsert(user_id, movie_id, normalized)
    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "movie_id": entry.movie_id,
        "feedback_type": entry.feedback_type,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def record_event(db: Session, *, user_id: int | None = None, session_id: str | None = None, event_type: str, movie_id: int | None = None, recommendation_strategy: str | None = None, reranking_strategy: str | None = None, position: int | None = None, request_id: str | None = None, metadata_json: dict[str, Any] | None = None) -> dict:
    valid_types = {
        "recommendation_impression",
        "recommendation_click",
        "movie_detail_open",
        "watchlist_add",
        "watchlist_remove",
        "rating_create",
        "rating_update",
        "rating_delete",
        "feedback_like",
        "feedback_dislike",
        "feedback_remove",
    }
    if event_type not in valid_types:
        raise ValueError(f"Unsupported event type: {event_type}")
    if user_id is not None:
        user = UserRepository(db).get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
    event = EventRepository(db).record(
        user_id=user_id,
        session_id=session_id,
        event_type=event_type,
        movie_id=movie_id,
        recommendation_strategy=recommendation_strategy,
        reranking_strategy=reranking_strategy,
        position=position,
        request_id=request_id,
        metadata_json=metadata_json,
    )
    return {
        "id": event.id,
        "user_id": event.user_id,
        "session_id": event.session_id,
        "event_type": event.event_type,
        "movie_id": event.movie_id,
        "recommendation_strategy": event.recommendation_strategy,
        "reranking_strategy": event.reranking_strategy,
        "position": event.position,
        "request_id": event.request_id,
        "metadata_json": event.metadata_json,
        "created_at": event.created_at.isoformat(),
    }


__all__ = ["record_event", "set_feedback", "toggle_watchlist", "upsert_rating"]
