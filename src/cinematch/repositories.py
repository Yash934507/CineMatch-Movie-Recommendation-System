"""Database access routines for persistent user records."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AppUser, RecommendationEvent, UserFeedback, UserRating, WatchlistItem


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> AppUser | None:
        return self.db.get(AppUser, user_id)

    def get_by_movielens_id(self, movielens_user_id: int) -> AppUser | None:
        return self.db.execute(
            select(AppUser).where(AppUser.movielens_user_id == movielens_user_id)
        ).scalar_one_or_none()

    def list_users(self) -> list[AppUser]:
        return self.db.execute(select(AppUser).order_by(AppUser.id)).scalars().all()

    def create(self, display_name: str, *, movielens_user_id: int | None = None, onboarding_complete: bool = False, preferred_strategy: str | None = None, preferred_reranking: str | None = None) -> AppUser:
        user = AppUser(
            display_name=display_name,
            movielens_user_id=movielens_user_id,
            onboarding_complete=onboarding_complete,
            preferred_strategy=preferred_strategy,
            preferred_reranking=preferred_reranking,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, user: AppUser, **changes: object) -> AppUser:
        for key, value in changes.items():
            if value is not None:
                setattr(user, key, value)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: AppUser) -> None:
        self.db.delete(user)
        self.db.commit()


class RatingRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_user_rating(self, user_id: int, movie_id: int) -> UserRating | None:
        return self.db.execute(
            select(UserRating).where(UserRating.user_id == user_id, UserRating.movie_id == movie_id)
        ).scalar_one_or_none()

    def list_user_ratings(self, user_id: int) -> list[UserRating]:
        return self.db.execute(
            select(UserRating).where(UserRating.user_id == user_id).order_by(UserRating.updated_at.desc(), UserRating.movie_id)
        ).scalars().all()

    def upsert(self, user_id: int, movie_id: int, rating: float) -> UserRating:
        existing = self.get_user_rating(user_id, movie_id)
        if existing is not None:
            existing.rating = float(rating)
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        entry = UserRating(user_id=user_id, movie_id=movie_id, rating=float(rating))
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def delete(self, user_id: int, movie_id: int) -> bool:
        entry = self.get_user_rating(user_id, movie_id)
        if entry is None:
            return False
        self.db.delete(entry)
        self.db.commit()
        return True


class WatchlistRepository:
    def __init__(self, db: Session):
        self.db = db

    def has_item(self, user_id: int, movie_id: int) -> bool:
        return self.db.execute(
            select(WatchlistItem.id).where(WatchlistItem.user_id == user_id, WatchlistItem.movie_id == movie_id)
        ).scalar_one_or_none() is not None

    def list_for_user(self, user_id: int) -> list[WatchlistItem]:
        return self.db.execute(
            select(WatchlistItem).where(WatchlistItem.user_id == user_id).order_by(WatchlistItem.created_at.desc(), WatchlistItem.movie_id)
        ).scalars().all()

    def add(self, user_id: int, movie_id: int) -> WatchlistItem:
        if self.has_item(user_id, movie_id):
            return self.db.execute(
                select(WatchlistItem).where(WatchlistItem.user_id == user_id, WatchlistItem.movie_id == movie_id)
            ).scalar_one()
        entry = WatchlistItem(user_id=user_id, movie_id=movie_id)
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def remove(self, user_id: int, movie_id: int) -> bool:
        entry = self.db.execute(
            select(WatchlistItem).where(WatchlistItem.user_id == user_id, WatchlistItem.movie_id == movie_id)
        ).scalar_one_or_none()
        if entry is None:
            return False
        self.db.delete(entry)
        self.db.commit()
        return True


class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_user(self, user_id: int) -> list[UserFeedback]:
        return self.db.execute(
            select(UserFeedback).where(UserFeedback.user_id == user_id).order_by(UserFeedback.updated_at.desc(), UserFeedback.movie_id)
        ).scalars().all()

    def get_for_movie(self, user_id: int, movie_id: int) -> UserFeedback | None:
        return self.db.execute(
            select(UserFeedback).where(UserFeedback.user_id == user_id, UserFeedback.movie_id == movie_id)
        ).scalar_one_or_none()

    def upsert(self, user_id: int, movie_id: int, feedback_type: str) -> UserFeedback:
        existing = self.get_for_movie(user_id, movie_id)
        if existing is not None:
            existing.feedback_type = feedback_type
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        entry = UserFeedback(user_id=user_id, movie_id=movie_id, feedback_type=feedback_type)
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def remove(self, user_id: int, movie_id: int) -> bool:
        entry = self.get_for_movie(user_id, movie_id)
        if entry is None:
            return False
        self.db.delete(entry)
        self.db.commit()
        return True


class EventRepository:
    def __init__(self, db: Session):
        self.db = db

    def record(self, *, user_id: int | None, session_id: str | None, event_type: str, movie_id: int | None = None, recommendation_strategy: str | None = None, reranking_strategy: str | None = None, position: int | None = None, request_id: str | None = None, metadata_json: dict | None = None) -> RecommendationEvent:
        event = RecommendationEvent(
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
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def list_for_user(self, user_id: int, limit: int = 50) -> list[RecommendationEvent]:
        return self.db.execute(
            select(RecommendationEvent)
            .where(RecommendationEvent.user_id == user_id)
            .order_by(RecommendationEvent.created_at.desc())
            .limit(limit)
        ).scalars().all()


__all__ = [
    "EventRepository",
    "FeedbackRepository",
    "RatingRepository",
    "UserRepository",
    "WatchlistRepository",
]
