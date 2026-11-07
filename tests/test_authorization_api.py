from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from sqlalchemy import create_engine

import cinematch.api as api
from cinematch.config import settings as default_settings
from cinematch.database import get_db
from cinematch.models import AppUser, Base, UserFeedback, UserRating, UserSession, WatchlistItem
from cinematch.services.auth import create_session, hash_password
from cinematch.api import RecommenderService


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_ORIGIN = "http://127.0.0.1:5173"


@lru_cache(maxsize=1)
def _recommender() -> RecommenderService:
    return RecommenderService.load(ROOT)


@pytest.fixture
def authorization_client(monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        replace(
            default_settings,
            environment="development",
            allowed_origins=(TRUSTED_ORIGIN,),
        ),
    )
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
    app.dependency_overrides[get_db] = override_get_db
    app.state.recommender = _recommender()
    client = TestClient(app)
    with Session(engine, expire_on_commit=False) as db:
        users = []
        tokens = []
        for index in range(2):
            user = AppUser(
                display_name=f"Account {index + 1}",
                email=f"account{index + 1}@example.test",
                password_hash=hash_password("a sufficiently long password"),
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            _, token = create_session(db, user)
            users.append(user.id)
            tokens.append(token)

    yield client, engine, users, tokens
    client.close()
    engine.dispose()


def _origin(origin: str = TRUSTED_ORIGIN) -> dict[str, str]:
    return {"Origin": origin}


def _authenticate(client: TestClient, token: str) -> None:
    client.cookies.set(api.SESSION_COOKIE_NAME, token)


def test_anonymous_requests_to_all_private_account_routes_return_401(authorization_client):
    client, _, users, _ = authorization_client
    user_id = users[0]
    routes = [
        ("GET", f"/api/v1/users/{user_id}"),
        ("GET", f"/api/v1/users/{user_id}/profile"),
        ("POST", f"/api/v1/users/{user_id}/ratings", {"movie_id": 260, "rating": 4}),
        ("PUT", f"/api/v1/users/{user_id}/ratings/260", {"rating": 4}),
        ("DELETE", f"/api/v1/users/{user_id}/ratings/260"),
        ("POST", f"/api/v1/users/{user_id}/watchlist", {"movie_id": 260}),
        ("DELETE", f"/api/v1/users/{user_id}/watchlist/260"),
        ("POST", f"/api/v1/users/{user_id}/feedback", {"movie_id": 260, "feedback_type": "like"}),
        ("DELETE", f"/api/v1/users/{user_id}/feedback/260"),
        ("GET", "/api/v1/me/recommendations"),
        ("GET", "/api/v1/me/predictions/260"),
    ]
    for method, path, *body in routes:
        response = client.request(
            method,
            path,
            json=body[0] if body else None,
            headers=_origin() if method not in {"GET", "DELETE"} else None,
        )
        assert response.status_code == 401, (method, path, response.text)


def test_user_creation_endpoint_cannot_bypass_registration(authorization_client):
    client, engine, _, _ = authorization_client
    response = client.post(
        "/api/v1/users",
        json={
            "display_name": "Impersonator",
            "movielens_user_id": 1,
            "onboarding_complete": True,
        },
    )
    assert response.status_code == 410
    assert "auth/register" in response.json()["detail"]
    with Session(engine) as db:
        assert db.execute(select(AppUser)).scalars().all().__len__() == 2


def test_owner_can_read_own_profile_but_cannot_read_other_account(authorization_client):
    client, _, users, tokens = authorization_client
    own_id, other_id = users
    _authenticate(client, tokens[0])

    own = client.get(f"/api/v1/users/{own_id}")
    own_profile = client.get(f"/api/v1/users/{own_id}/profile")
    other = client.get(f"/api/v1/users/{other_id}")
    other_profile = client.get(f"/api/v1/users/{other_id}/profile")

    assert own.status_code == 200
    assert own_profile.status_code == 200
    assert own.json()["id"] == own_id
    assert own_profile.json()["user"]["id"] == own_id
    assert other.status_code == other_profile.status_code == 403
    for response in (other, other_profile):
        assert response.json() == {"detail": "Not authorized to access this resource"}
        assert f"account{other_id}@example.test" not in response.text
        assert f"Account {other_id}" not in response.text


def test_owner_can_create_and_update_rating_but_cannot_access_other_rating(authorization_client):
    client, engine, users, tokens = authorization_client
    own_id, other_id = users
    with Session(engine) as db:
        db.add(UserRating(user_id=other_id, movie_id=260, rating=3.0))
        db.commit()
    _authenticate(client, tokens[0])

    own = client.post(
        f"/api/v1/users/{own_id}/ratings",
        headers=_origin(),
        json={"movie_id": 260, "rating": 4.0},
    )
    updated = client.put(
        f"/api/v1/users/{own_id}/ratings/260",
        headers=_origin(),
        json={"rating": 4.5},
    )
    cross_create = client.post(
        f"/api/v1/users/{other_id}/ratings",
        headers=_origin(),
        json={"movie_id": 318, "rating": 5.0},
    )
    cross_update = client.put(
        f"/api/v1/users/{other_id}/ratings/260",
        headers=_origin(),
        json={"rating": 1.0},
    )
    cross_delete = client.delete(
        f"/api/v1/users/{other_id}/ratings/260",
        headers=_origin(),
    )

    assert own.status_code == updated.status_code == 200
    assert own.json()["user_id"] == own_id
    assert updated.json()["rating"] == 4.5
    assert cross_create.status_code == cross_update.status_code == cross_delete.status_code == 403
    with Session(engine) as db:
        ratings = db.execute(select(UserRating)).scalars().all()
        assert {
            (rating.user_id, rating.movie_id, rating.rating) for rating in ratings
        } == {(own_id, 260, 4.5), (other_id, 260, 3.0)}


def test_owner_can_modify_watchlist_but_cannot_modify_another_watchlist(authorization_client):
    client, engine, users, tokens = authorization_client
    own_id, other_id = users
    with Session(engine) as db:
        db.add(WatchlistItem(user_id=other_id, movie_id=260))
        db.commit()
    _authenticate(client, tokens[0])

    own_add = client.post(
        f"/api/v1/users/{own_id}/watchlist",
        headers=_origin(),
        json={"movie_id": 260},
    )
    cross_add = client.post(
        f"/api/v1/users/{other_id}/watchlist",
        headers=_origin(),
        json={"movie_id": 318},
    )
    cross_remove = client.delete(
        f"/api/v1/users/{other_id}/watchlist/260",
        headers=_origin(),
    )
    own_remove = client.delete(
        f"/api/v1/users/{own_id}/watchlist/260",
        headers=_origin(),
    )

    assert own_add.status_code == own_remove.status_code == 200
    assert cross_add.status_code == cross_remove.status_code == 403
    with Session(engine) as db:
        items = db.execute(select(WatchlistItem)).scalars().all()
        assert [(item.user_id, item.movie_id) for item in items] == [(other_id, 260)]


def test_owner_can_modify_feedback_but_cannot_modify_another_users_feedback(authorization_client):
    client, engine, users, tokens = authorization_client
    own_id, other_id = users
    with Session(engine) as db:
        db.add(UserFeedback(user_id=other_id, movie_id=260, feedback_type="dislike"))
        db.commit()
    _authenticate(client, tokens[0])

    own = client.post(
        f"/api/v1/users/{own_id}/feedback",
        headers=_origin(),
        json={"movie_id": 260, "feedback_type": "like"},
    )
    cross_add = client.post(
        f"/api/v1/users/{other_id}/feedback",
        headers=_origin(),
        json={"movie_id": 318, "feedback_type": "dislike"},
    )
    cross_delete = client.delete(
        f"/api/v1/users/{other_id}/feedback/260",
        headers=_origin(),
    )

    assert own.status_code == 200
    assert cross_add.status_code == cross_delete.status_code == 403
    with Session(engine) as db:
        feedback = db.execute(select(UserFeedback)).scalars().all()
        assert {
            (item.user_id, item.movie_id, item.feedback_type) for item in feedback
        } == {(own_id, 260, "like"), (other_id, 260, "dislike")}


def test_me_recommendations_and_predictions_use_only_authenticated_account_ratings(authorization_client):
    client, engine, users, tokens = authorization_client
    own_id, other_id = users
    recommender = _recommender()
    rated_for_owner = [1, 260, 296]
    rated_for_other = [318, 356, 593]
    with Session(engine) as db:
        for movie_id, rating in zip(rated_for_owner, [5.0, 4.0, 4.5]):
            db.add(UserRating(user_id=own_id, movie_id=movie_id, rating=rating))
        for movie_id, rating in zip(rated_for_other, [1.0, 1.5, 2.0]):
            db.add(UserRating(user_id=other_id, movie_id=movie_id, rating=rating))
        db.commit()

    _authenticate(client, tokens[0])
    recommendations = client.get("/api/v1/me/recommendations", params={"limit": 5})
    prediction = client.get("/api/v1/me/predictions/318")
    assert recommendations.status_code == prediction.status_code == 200
    payload = recommendations.json()
    assert payload["cold_start"] is False
    assert payload["strategy"] == "svd-fold-in"
    assert not set(rated_for_owner) & {
        row["movie_id"] for row in payload["recommendations"]
    }
    assert prediction.json()["movie_id"] == 318
    assert 0.5 <= prediction.json()["predicted_rating"] <= 5.0
    assert prediction.json()["cold_start"] is False
    assert "user_id" not in payload and "user_id" not in prediction.json()

    _authenticate(client, tokens[1])
    other_recommendations = client.get("/api/v1/me/recommendations", params={"limit": 5})
    assert other_recommendations.status_code == 200
    assert not set(rated_for_other) & {
        row["movie_id"] for row in other_recommendations.json()["recommendations"]
    }
    assert payload["recommendations"] != other_recommendations.json()["recommendations"]


def test_new_account_gets_cold_start_recommendations_but_requires_history_for_prediction(authorization_client):
    client, _, users, tokens = authorization_client
    _authenticate(client, tokens[0])

    recommendations = client.get("/api/v1/me/recommendations", params={"limit": 3})
    prediction = client.get("/api/v1/me/predictions/260")

    assert recommendations.status_code == 200
    assert recommendations.json()["cold_start"] is True
    assert recommendations.json()["strategy"] == "bayesian-popularity"
    assert len(recommendations.json()["recommendations"]) == 3
    assert prediction.status_code == 409
    assert str(users[0]) not in prediction.text


@pytest.mark.parametrize("session_state", ["invalid", "expired", "revoked"])
def test_invalid_expired_and_revoked_sessions_cannot_access_private_data(
    authorization_client,
    session_state,
):
    client, engine, users, tokens = authorization_client
    if session_state == "invalid":
        token = "invalid"
    else:
        token = tokens[0]
        with Session(engine) as db:
            session = db.execute(
                select(UserSession).where(UserSession.user_id == users[0])
            ).scalar_one()
            if session_state == "expired":
                from datetime import datetime, timezone

                session.expires_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
            else:
                from datetime import datetime, timezone

                session.revoked_at = datetime.now(timezone.utc)
            db.commit()
    _authenticate(client, token)
    assert client.get(f"/api/v1/users/{users[0]}").status_code == 401


@pytest.mark.parametrize(
    "headers,expected_status",
    [
        ({}, 403),
        (_origin("https://attacker.example"), 403),
        (_origin(), 200),
    ],
)
def test_csrf_origin_policy_for_owner_write(authorization_client, headers, expected_status):
    client, _, users, tokens = authorization_client
    _authenticate(client, tokens[0])
    response = client.post(
        f"/api/v1/users/{users[0]}/ratings",
        headers=headers,
        json={"movie_id": 260, "rating": 4.0},
    )
    assert response.status_code == expected_status


def test_invalid_session_with_trusted_origin_still_returns_401(authorization_client):
    client, _, users, _ = authorization_client
    _authenticate(client, "invalid")
    response = client.post(
        f"/api/v1/users/{users[0]}/ratings",
        headers=_origin(),
        json={"movie_id": 260, "rating": 4.0},
    )
    assert response.status_code == 401


def test_public_movie_browsing_and_read_only_movielens_demo_remain_available(authorization_client):
    client, _, _, _ = authorization_client
    assert client.get("/api/v1/movies", params={"q": "A New Hope"}).status_code == 200
    assert client.get("/api/v1/movies/260").status_code == 200
    demo_recommendations = client.get("/api/v1/demo/recommendations/1", params={"limit": 3})
    demo_prediction = client.get("/api/v1/demo/predictions/1/318")
    assert demo_recommendations.status_code == 200
    assert demo_prediction.status_code == 200
    assert demo_recommendations.json()["user_id"] == 1
    assert demo_prediction.json()["user_id"] == 1


def test_public_transient_onboarding_remains_available(authorization_client):
    client, _, _, _ = authorization_client
    response = client.post(
        "/api/v1/onboarding/recommendations",
        json={
            "ratings": [
                {"movie_id": movie_id, "rating": rating}
                for movie_id, rating in [(1, 4), (260, 5), (296, 3)]
            ],
            "limit": 3,
        },
    )
    assert response.status_code == 200
    assert response.json()["ratings_used"] == 3
