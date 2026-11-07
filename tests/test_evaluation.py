import unittest

import pandas as pd

from cinematch.evaluation import popularity_fallback, ranking_metrics


class RankingMetricTests(unittest.TestCase):
    def test_metrics_match_known_example(self):
        result = ranking_metrics([10, 20, 30, 40], {10, 30}, k=4)
        self.assertAlmostEqual(result["precision_at_k"], 0.5)
        self.assertAlmostEqual(result["recall_at_k"], 1.0)
        self.assertAlmostEqual(result["hit_rate_at_k"], 1.0)
        self.assertGreater(result["ndcg_at_k"], 0.0)
        self.assertLessEqual(result["ndcg_at_k"], 1.0)

    def test_no_hits_have_zero_metrics(self):
        result = ranking_metrics([10, 20], {30}, k=2)
        self.assertEqual(result["precision_at_k"], 0.0)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["hit_rate_at_k"], 0.0)
        self.assertEqual(result["ndcg_at_k"], 0.0)

    def test_relevance_set_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            ranking_metrics([10], set(), k=1)


class ColdStartTests(unittest.TestCase):
    def test_popularity_fallback_rewards_evidence(self):
        ratings = pd.DataFrame(
            [(1, 10, 5.0), (2, 10, 5.0), (3, 20, 5.0), (4, 30, 1.0)],
            columns=["userId", "movieId", "rating"],
        )
        movies = pd.DataFrame(
            {"movieId": [10, 20, 30], "title": ["A", "B", "C"], "genres": ["Drama"] * 3}
        )
        result = popularity_fallback(ratings, movies, n=3, prior_weight=2)
        self.assertEqual(int(result.iloc[0]["movieId"]), 10)
        self.assertTrue(result["weighted_score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
