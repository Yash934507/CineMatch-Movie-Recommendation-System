import unittest

import numpy as np
import pandas as pd

from cinematch.memory_cf import MemoryBasedCF, cosine_similarity, deterministic_user_split


class CosineSimilarityTests(unittest.TestCase):
    def test_parallel_vectors_equal_one(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1, 2]), np.array([2, 4])), 1.0)

    def test_orthogonal_vectors_equal_zero(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(cosine_similarity(np.array([0, 0]), np.array([2, 4])), 0.0)


class MemoryBasedCFTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 10, 5.0), (1, 20, 4.0), (1, 30, 1.0),
                (2, 10, 5.0), (2, 20, 4.0), (2, 30, 1.0), (2, 40, 5.0),
                (3, 10, 1.0), (3, 20, 2.0), (3, 30, 5.0), (3, 40, 1.0),
            ],
            columns=["userId", "movieId", "rating"],
        )
        self.model = MemoryBasedCF(k_neighbors=2, min_common=2).fit(self.ratings)

    def test_similar_user_is_found(self):
        result = self.model.similar_users(1, n=1)
        self.assertEqual(int(result.iloc[0]["userId"]), 2)

    def test_user_prediction_uses_neighbor(self):
        prediction = self.model.predict_user_based(1, 40)
        self.assertFalse(prediction.used_fallback)
        self.assertGreater(prediction.value, 4.0)

    def test_item_prediction_is_bounded(self):
        prediction = self.model.predict_item_based(1, 40)
        self.assertGreaterEqual(prediction.value, 0.5)
        self.assertLessEqual(prediction.value, 5.0)

    def test_recommendations_exclude_seen_movies(self):
        movies = pd.DataFrame(
            {
                "movieId": [10, 20, 30, 40],
                "title": ["A", "B", "C", "D"],
                "genres": ["Drama"] * 4,
            }
        )
        result = self.model.recommend(1, movies, method="user", n=1, candidate_pool=4)
        self.assertEqual(result["movieId"].tolist(), [40])

    def test_unknown_recommendation_method_is_rejected(self):
        movies = pd.DataFrame(columns=["movieId", "title", "genres"])
        with self.assertRaises(ValueError):
            self.model.recommend(1, movies, method="hybrid")

    def test_split_is_reproducible_and_disjoint(self):
        timestamped = self.ratings.assign(timestamp=np.arange(len(self.ratings)))
        train_a, test_a = deterministic_user_split(timestamped, seed=7)
        train_b, test_b = deterministic_user_split(timestamped, seed=7)
        self.assertEqual(train_a.index.tolist(), train_b.index.tolist())
        self.assertEqual(test_a.index.tolist(), test_b.index.tolist())
        self.assertFalse(set(train_a.index) & set(test_a.index))


if __name__ == "__main__":
    unittest.main()
