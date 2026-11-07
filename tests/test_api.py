import os
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cinematch.api import create_app
from cinematch.config import Settings, redact_url_credentials, settings as default_settings
from cinematch.database import initialize_database


ROOT = Path(__file__).resolve().parents[1]


class CineMatchAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database_url = f"sqlite:///{(Path(cls.temp_directory.name) / 'cinematch-test.db').as_posix()}"
        cls.environment_patch = patch.dict(
            os.environ,
            {"CINEMATCH_DATABASE_URL": cls.database_url},
        )
        cls.environment_patch.start()
        cls.settings_patch = patch(
            "cinematch.api.settings",
            replace(default_settings, database_url=cls.database_url),
        )
        cls.settings_patch.start()
        cls.initialize_patch = patch(
            "cinematch.api.initialize_database",
            lambda: initialize_database(cls.database_url),
        )
        cls.initialize_patch.start()
        cls.context = TestClient(create_app(ROOT))
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        cls.initialize_patch.stop()
        cls.settings_patch.stop()
        cls.environment_patch.stop()
        cls.temp_directory.cleanup()

    def test_health_reports_loaded_artifacts(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "surprise-svd-20")
        self.assertEqual(payload["users"], 610)
        self.assertEqual(payload["ratings"], 100836)

    def test_production_frontend_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('id="root"', response.text)

    def test_spa_route_falls_back_to_frontend(self):
        response = self.client.get("/discover")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="root"', response.text)

    def test_movie_search_finds_star_wars(self):
        response = self.client.get("/api/v1/movies", params={"q": "A New Hope", "limit": 5})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(movie["movie_id"] == 260 for movie in response.json()))

    def test_known_user_gets_personalized_unseen_movies(self):
        response = self.client.get("/api/v1/recommendations/1", params={"limit": 5})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["cold_start"])
        self.assertEqual(payload["strategy"], "surprise-svd-20")
        self.assertEqual(len(payload["recommendations"]), 5)
        known = set(
            self.client.app.state.recommender.ratings.loc[
                self.client.app.state.recommender.ratings["userId"] == 1, "movieId"
            ]
        )
        self.assertFalse(known & {movie["movie_id"] for movie in payload["recommendations"]})

    def test_unknown_user_gets_cold_start_fallback(self):
        response = self.client.get("/api/v1/recommendations/999999", params={"limit": 3})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["cold_start"])
        self.assertEqual(payload["strategy"], "bayesian-popularity")
        self.assertEqual(len(payload["recommendations"]), 3)

    def test_known_user_can_request_hybrid_strategy(self):
        response = self.client.get("/api/v1/recommendations/1", params={"limit": 5, "strategy": "hybrid"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["cold_start"])
        self.assertEqual(payload["strategy"], "hybrid")
        self.assertEqual(len(payload["recommendations"]), 5)
        known = set(
            self.client.app.state.recommender.ratings.loc[
                self.client.app.state.recommender.ratings["userId"] == 1, "movieId"
            ]
        )
        self.assertFalse(known & {movie["movie_id"] for movie in payload["recommendations"]})

    def test_onboarding_fold_in_returns_unseen_recommendations(self):
        submitted = [1, 260, 296, 318, 356]
        response = self.client.post(
            "/api/v1/onboarding/recommendations",
            json={
                "ratings": [
                    {"movie_id": movie_id, "rating": rating}
                    for movie_id, rating in zip(submitted, [3.0, 5.0, 4.5, 5.0, 4.0])
                ],
                "limit": 6,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["strategy"], "svd-fold-in")
        self.assertEqual(payload["ratings_used"], 5)
        self.assertEqual(len(payload["recommendations"]), 6)
        self.assertFalse(set(submitted) & {movie["movie_id"] for movie in payload["recommendations"]})
        for movie in payload["recommendations"]:
            self.assertGreaterEqual(movie["predicted_rating"], 0.5)
            self.assertLessEqual(movie["predicted_rating"], 5.0)

    def test_onboarding_requires_three_ratings(self):
        response = self.client.post(
            "/api/v1/onboarding/recommendations",
            json={"ratings": [{"movie_id": 1, "rating": 5}, {"movie_id": 260, "rating": 4}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_onboarding_three_ratings_preserves_response_contract(self):
        response = self.client.post("/api/v1/onboarding/recommendations", json={
            "ratings": [{"movie_id": movie, "rating": score}
                        for movie, score in [(1, 4), (260, 5), (296, 3)]], "limit": 3,
        })
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"strategy", "ratings_used", "recommendations"})
        self.assertEqual(payload["strategy"], "svd-fold-in")
        self.assertEqual(payload["ratings_used"], 3)
        self.assertEqual(len(payload["recommendations"]), 3)
        self.assertEqual(set(payload["recommendations"][0]), {
            "movie_id", "title", "genres", "rating_count", "mean_rating", "predicted_rating",
        })

    def test_onboarding_duplicate_movies_are_rejected(self):
        response = self.client.post("/api/v1/onboarding/recommendations", json={
            "ratings": [{"movie_id": movie, "rating": 4} for movie in [1, 1, 260]],
        })
        self.assertEqual(response.status_code, 422)

    def test_onboarding_unknown_movie_is_404(self):
        response = self.client.post("/api/v1/onboarding/recommendations", json={
            "ratings": [{"movie_id": movie, "rating": 4} for movie in [1, 260, 99999999]],
        })
        self.assertEqual(response.status_code, 404)

    def test_similar_movies_exclude_source(self):
        response = self.client.get("/api/v1/movies/260/similar", params={"limit": 5})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 5)
        self.assertNotIn(260, {movie["movie_id"] for movie in payload})

    def test_prediction_is_bounded(self):
        response = self.client.get("/api/v1/predictions/1/318")
        self.assertEqual(response.status_code, 200)
        score = response.json()["predicted_rating"]
        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 5.0)

    def test_missing_movie_returns_404(self):
        response = self.client.get("/api/v1/movies/999999999")
        self.assertEqual(response.status_code, 404)

    def test_limit_validation_returns_422(self):
        response = self.client.get("/api/v1/recommendations/1", params={"limit": 51})
        self.assertEqual(response.status_code, 422)

    def test_vite_origin_is_allowed_by_cors(self):
        response = self.client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5173")

    def test_vite_origin_can_post_onboarding_ratings(self):
        response = self.client.options(
            "/api/v1/onboarding/recommendations",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_database_urls_are_redacted_before_logging(self):
        self.assertEqual(
            redact_url_credentials("postgresql://appuser:topsecret@db.example.com:5432/cinematch"),
            "postgresql://appuser:***@db.example.com:5432/cinematch",
        )
        self.assertEqual(
            Settings(database_url="postgresql://appuser:topsecret@db.example.com:5432/cinematch").database_url_for_logs,
            "postgresql://appuser:***@db.example.com:5432/cinematch",
        )

    def test_production_requires_non_sqlite_database_url(self):
        env = {
            "CINEMATCH_ENV": "production",
            "CINEMATCH_DATABASE_URL": "sqlite:///./data/cinematch.db",
            "CINEMATCH_ALLOWED_ORIGINS": "http://localhost:5173",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_invalid_allowed_origin_is_rejected(self):
        env = {
            "CINEMATCH_ENV": "development",
            "CINEMATCH_ALLOWED_ORIGINS": "*",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_production_read_only_demo_allows_only_isolated_sqlite(self):
        env = {
            "CINEMATCH_ENV": "production",
            "CINEMATCH_DATABASE_URL": "sqlite:////tmp/cinematch-demo.db",
            "CINEMATCH_READ_ONLY_DEMO": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            configured = Settings.from_env()
        self.assertTrue(configured.read_only_demo)
        self.assertEqual(configured.database_url, env["CINEMATCH_DATABASE_URL"])

    def test_read_only_demo_rejects_non_sqlite_database(self):
        env = {
            "CINEMATCH_ENV": "production",
            "CINEMATCH_DATABASE_URL": "postgresql://db.example.invalid/cinematch",
            "CINEMATCH_READ_ONLY_DEMO": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "requires an isolated SQLite database"):
                Settings.from_env()

    def test_production_read_only_demo_rejects_non_ephemeral_sqlite_path(self):
        env = {
            "CINEMATCH_ENV": "production",
            "CINEMATCH_DATABASE_URL": "sqlite:////app/data/cinematch.db",
            "CINEMATCH_READ_ONLY_DEMO": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "must be under /tmp"):
                Settings.from_env()

    def test_read_only_demo_rejects_invalid_boolean(self):
        env = {
            "CINEMATCH_ENV": "development",
            "CINEMATCH_READ_ONLY_DEMO": "sometimes",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "must be either true or false"):
                Settings.from_env()

    def test_read_only_demo_blocks_writes_but_preserves_public_reads_and_onboarding(self):
        app = self.client.app
        original_read_only_demo = app.state.read_only_demo
        app.state.read_only_demo = True
        try:
            register = self.client.post(
                "/api/v1/auth/register",
                json={
                    "display_name": "Demo Viewer",
                    "email": "demo-viewer@example.test",
                    "password": "valid synthetic test password",
                },
            )
            rating = self.client.post(
                "/api/v1/users/1/ratings",
                json={"movie_id": 260, "rating": 5.0},
            )
            auth_me = self.client.get("/api/v1/auth/me")
            onboarding = self.client.post(
                "/api/v1/onboarding/recommendations",
                json={
                    "ratings": [
                        {"movie_id": 1, "rating": 3.0},
                        {"movie_id": 260, "rating": 5.0},
                        {"movie_id": 296, "rating": 4.5},
                    ],
                    "limit": 3,
                },
            )
            movies = self.client.get("/api/v1/movies", params={"q": "Star Wars"})
        finally:
            app.state.read_only_demo = original_read_only_demo

        self.assertEqual(register.status_code, 403)
        self.assertEqual(register.json(), {"detail": "This public demo is read-only"})
        self.assertEqual(rating.status_code, 403)
        self.assertEqual(rating.json(), {"detail": "This public demo is read-only"})
        self.assertEqual(auth_me.status_code, 401)
        self.assertEqual(onboarding.status_code, 200)
        self.assertEqual(movies.status_code, 200)


if __name__ == "__main__":
    unittest.main()
