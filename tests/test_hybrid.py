import unittest

import pandas as pd

from cinematch.hybrid import (
    build_genre_profile,
    combine_hybrid_scores,
    genre_coverage,
    intra_list_diversity,
    normalize_component_scores,
    rank_hybrid_candidates,
    validate_hybrid_weights,
)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 10, 5.0),
                (1, 20, 4.0),
                (1, 30, 3.0),
                (2, 10, 4.0),
                (2, 40, 5.0),
            ],
            columns=["userId", "movieId", "rating"],
        )
        self.movies = pd.DataFrame(
            {
                "movieId": [10, 20, 30, 40],
                "title": ["A", "B", "C", "D"],
                "genres": ["Action|Drama", "Drama|Comedy", "Comedy", "Action|Thriller"],
            }
        )

    def test_genre_profile_weights_ratings_by_genre(self):
        profile = build_genre_profile(self.ratings, self.movies, 1)
        self.assertEqual(profile["Action"], 5.0)
        self.assertEqual(profile["Drama"], 9.0)
        self.assertEqual(profile["Comedy"], 7.0)

    def test_normalization_handles_constant_scores(self):
        normalized = normalize_component_scores({1: 2.0, 2: 2.0, 3: 2.0})
        self.assertEqual(normalized, {1: 0.0, 2: 0.0, 3: 0.0})

    def test_validate_hybrid_weights_rejects_invalid_totals(self):
        with self.assertRaises(ValueError):
            validate_hybrid_weights({"collaborative": 0.6, "genre": 0.4, "popularity": 0.1})

    def test_hybrid_candidates_are_ranked_by_combined_score(self):
        collaborative = {10: 5.0, 20: 3.0, 30: 1.0}
        genre = {10: 1.0, 20: 0.8, 30: 0.0}
        ranking = rank_hybrid_candidates(collaborative, genre, weights={"collaborative": 0.8, "genre": 0.2, "popularity": 0.0})
        self.assertEqual(ranking[0][0], 10)
        self.assertEqual(ranking[1][0], 20)
        self.assertEqual(ranking[2][0], 30)

    def test_diversity_and_genre_coverage_are_finite(self):
        recommended = [10, 20, 30]
        self.assertGreaterEqual(intra_list_diversity(recommended, self.movies), 0.0)
        self.assertLessEqual(intra_list_diversity(recommended, self.movies), 1.0)
        self.assertGreaterEqual(genre_coverage(recommended, self.movies), 0.0)
        self.assertLessEqual(genre_coverage(recommended, self.movies), 1.0)

    def test_combined_scores_match_weighted_total(self):
        scores = combine_hybrid_scores(
            {1: 10.0, 2: 5.0},
            {1: 1.0, 2: 0.0},
            weights={"collaborative": 0.75, "genre": 0.25, "popularity": 0.0},
        )
        self.assertAlmostEqual(scores[1], 0.75 * 1.0 + 0.25 * 1.0)
        self.assertAlmostEqual(scores[2], 0.75 * 0.0 + 0.25 * 0.0)


if __name__ == "__main__":
    unittest.main()
