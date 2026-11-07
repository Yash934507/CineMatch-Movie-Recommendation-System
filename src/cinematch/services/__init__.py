"""Service layer for persistent user and interaction logic."""

from .interactions import record_event, set_feedback, toggle_watchlist, upsert_rating
from .users import build_user_profile, create_app_user, validate_movie_id

__all__ = [
    "build_user_profile",
    "create_app_user",
    "record_event",
    "set_feedback",
    "toggle_watchlist",
    "upsert_rating",
    "validate_movie_id",
]
