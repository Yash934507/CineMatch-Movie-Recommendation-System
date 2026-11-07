import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cinematch.svd_model import (
    fit_svd,
    load_model,
    predict_frame,
    recommend_svd,
    regression_metrics,
    save_model,
)


class SVDModelTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 10, 5.0), (1, 20, 4.0), (1, 30, 1.0),
                (2, 10, 4.5), (2, 20, 4.0), (2, 40, 5.0),
                (3, 10, 1.0), (3, 30, 5.0), (3, 40, 1.5),
                (4, 20, 2.0), (4, 30, 4.5), (4, 40, 2.0),
            ],
            columns=["userId", "movieId", "rating"],
        )
        self.movies = pd.DataFrame(
            {
                "movieId": [10, 20, 30, 40, 50],
                "title": ["A", "B", "C", "D", "E"],
                "genres": ["Drama"] * 5,
            }
        )
        self.model = fit_svd(self.ratings, n_factors=2, n_epochs=5, random_state=7)

    def test_predictions_are_on_rating_scale(self):
        predictions = predict_frame(self.model, self.ratings.iloc[:3])
        self.assertTrue(predictions["predicted_rating"].between(0.5, 5.0).all())
        self.assertEqual(regression_metrics(predictions)["evaluated_ratings"], 3)

    def test_recommendations_exclude_seen_movies(self):
        result = recommend_svd(self.model, 1, self.ratings, self.movies, n=2)
        self.assertFalse(set(result["movieId"]) & {10, 20, 30})

    def test_fixed_seed_is_reproducible(self):
        second = fit_svd(self.ratings, n_factors=2, n_epochs=5, random_state=7)
        first_score = self.model.predict(1, 40).est
        second_score = second.predict(1, 40).est
        self.assertAlmostEqual(first_score, second_score, places=12)

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save_model(self.model, path)
            loaded = load_model(path)
            self.assertAlmostEqual(self.model.predict(1, 40).est, loaded.predict(1, 40).est)


if __name__ == "__main__":
    unittest.main()
