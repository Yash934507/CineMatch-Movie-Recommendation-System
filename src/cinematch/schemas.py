"""Pydantic schemas for persistent users and interaction APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .services.auth import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH, normalize_email

FeedbackType = Literal["like", "dislike"]
EventType = Literal[
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
]


class AppUserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(..., min_length=1, max_length=80)
    movielens_user_id: int | None = Field(default=None, ge=1)
    onboarding_complete: bool = False
    preferred_strategy: str | None = Field(default=None, max_length=50)
    preferred_reranking: str | None = Field(default=None, max_length=50)


class AuthRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_BYTES)
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Display name is required")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError("Password exceeds the maximum supported length")
        return value


class AuthLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_BYTES)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError("Password exceeds the maximum supported length")
        return value


class AuthenticatedUserRead(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: datetime


class AppUserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str
    movielens_user_id: int | None = None
    onboarding_complete: bool
    preferred_strategy: str | None = None
    preferred_reranking: str | None = None
    created_at: datetime
    updated_at: datetime


class UserRatingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)


class UserRatingUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: float = Field(ge=0.5, le=5.0)


class UserRatingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    rating: float
    created_at: datetime
    updated_at: datetime


class WatchlistItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    created_at: datetime


class UserFeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int = Field(gt=0)
    feedback_type: FeedbackType


class UserFeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    movie_id: int
    feedback_type: str
    created_at: datetime
    updated_at: datetime


class RecommendationEventCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, max_length=64)
    event_type: EventType
    movie_id: int | None = Field(default=None, gt=0)
    recommendation_strategy: str | None = Field(default=None, max_length=50)
    reranking_strategy: str | None = Field(default=None, max_length=50)
    position: int | None = Field(default=None, gt=0)
    request_id: str | None = Field(default=None, max_length=64)
    metadata_json: dict[str, Any] | None = None


class RecommendationEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None = None
    session_id: str | None = None
    event_type: str
    movie_id: int | None = None
    recommendation_strategy: str | None = None
    reranking_strategy: str | None = None
    position: int | None = None
    request_id: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime


class UserProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user: AppUserRead
    ratings: list[UserRatingRead]
    watchlist: list[WatchlistItemRead]
    feedback: list[UserFeedbackRead]


__all__ = [
    "AppUserCreate",
    "AppUserRead",
    "AuthLoginRequest",
    "AuthRegisterRequest",
    "AuthenticatedUserRead",
    "EventType",
    "FeedbackType",
    "RecommendationEventCreate",
    "RecommendationEventRead",
    "UserFeedbackCreate",
    "UserFeedbackRead",
    "UserProfileRead",
    "UserRatingCreate",
    "UserRatingRead",
    "UserRatingUpdate",
    "WatchlistItemRead",
]
