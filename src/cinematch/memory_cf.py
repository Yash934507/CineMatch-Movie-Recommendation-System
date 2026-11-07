"""Memory-based collaborative filtering implemented with NumPy.

The module deliberately avoids sklearn's cosine_similarity so the underlying
vector mathematics remains visible for the internship deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cosine_similarity_manual(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute dot(a, b) / (magnitude(a) * magnitude(b)) using NumPy.

    Inputs must be finite, one-dimensional vectors of equal length. Empty
    vectors and vectors with zero magnitude return 0.0 by convention.
    Missing ratings (NaN) must be filtered by the caller before comparison.
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Cosine-similarity inputs must be one-dimensional")
    if a.shape != b.shape:
        raise ValueError("Cosine-similarity vectors must have the same shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cosine-similarity vectors must contain only finite values")
    if a.size == 0:
        return 0.0
    # Positive scaling preserves direction and avoids overflow/underflow.
    scale_a = np.max(np.abs(a))
    scale_b = np.max(np.abs(b))
    if scale_a == 0.0 or scale_b == 0.0:
        return 0.0
    a = a / scale_a
    b = b / scale_b
    dot_product = np.dot(a, b)
    magnitude_a = np.sqrt(np.dot(a, a))
    magnitude_b = np.sqrt(np.dot(b, b))
    return float(np.clip(dot_product / (magnitude_a * magnitude_b), -1.0, 1.0))


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Backward-compatible entry point for the manual implementation."""
    return cosine_similarity_manual(vector_a, vector_b)


def deterministic_user_split(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a deterministic random fraction of every user's ratings."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    for _, group in ratings.groupby("userId", sort=True):
        count = max(1, int(round(len(group) * test_fraction)))
        count = min(count, len(group) - 1)
        test_indices.extend(rng.choice(group.index.to_numpy(), size=count, replace=False).tolist())
    test_mask = ratings.index.isin(test_indices)
    return ratings.loc[~test_mask].copy(), ratings.loc[test_mask].copy()


@dataclass(frozen=True)
class Prediction:
    value: float
    neighbors_used: int
    used_fallback: bool


class MemoryBasedCF:
    """User-User and Item-Item collaborative-filtering model."""

    def __init__(self, k_neighbors: int = 30, min_common: int = 3):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        if min_common < 1:
            raise ValueError("min_common must be positive")
        self.k_neighbors = k_neighbors
        self.min_common = min_common

    def fit(self, ratings: pd.DataFrame) -> "MemoryBasedCF":
        required = {"userId", "movieId", "rating"}
        if missing := required - set(ratings.columns):
            raise ValueError(f"Missing rating columns: {sorted(missing)}")
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_to_index = {int(value): index for index, value in enumerate(self.user_ids)}
        self.movie_to_index = {int(value): index for index, value in enumerate(self.movie_ids)}

        matrix = np.full((len(self.user_ids), len(self.movie_ids)), np.nan, dtype=np.float32)
        user_positions = ratings["userId"].map(self.user_to_index).to_numpy()
        movie_positions = ratings["movieId"].map(self.movie_to_index).to_numpy()
        matrix[user_positions, movie_positions] = ratings["rating"].to_numpy(dtype=np.float32)
        self.ratings = matrix
        self.observed = ~np.isnan(matrix)
        self.global_mean = float(np.nanmean(matrix))
        self.user_means = np.nanmean(matrix, axis=1)
        self.item_means = np.nanmean(matrix, axis=0)

        user_centered = np.where(self.observed, matrix - self.user_means[:, None], 0.0)
        user_norms = np.linalg.norm(user_centered, axis=1)
        denominator = np.outer(user_norms, user_norms)
        self.user_similarity = np.divide(
            user_centered @ user_centered.T,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float32),
            where=denominator > 0,
        )
        common = self.observed.astype(np.int32) @ self.observed.astype(np.int32).T
        self.user_similarity[common < self.min_common] = 0.0
        np.fill_diagonal(self.user_similarity, 0.0)

        self.item_centered = np.where(self.observed, matrix - self.item_means[None, :], 0.0)
        self.item_norms = np.linalg.norm(self.item_centered, axis=0)
        return self

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.5, 5.0))

    def _known_user(self, user_id: int) -> int | None:
        return self.user_to_index.get(int(user_id))

    def _known_movie(self, movie_id: int) -> int | None:
        return self.movie_to_index.get(int(movie_id))

    def predict_user_based(self, user_id: int, movie_id: int) -> Prediction:
        user_index = self._known_user(user_id)
        movie_index = self._known_movie(movie_id)
        if user_index is None:
            return Prediction(self.global_mean, 0, True)
        fallback = float(self.user_means[user_index])
        if movie_index is None:
            return Prediction(self._clip(fallback), 0, True)

        raters = np.flatnonzero(self.observed[:, movie_index])
        similarities = self.user_similarity[user_index, raters]
        eligible = similarities > 0
        raters, similarities = raters[eligible], similarities[eligible]
        if not len(raters):
            return Prediction(self._clip(fallback), 0, True)
        order = np.argsort(similarities)[-self.k_neighbors :]
        raters, similarities = raters[order], similarities[order]
        deviations = self.ratings[raters, movie_index] - self.user_means[raters]
        prediction = fallback + float(np.dot(similarities, deviations) / np.abs(similarities).sum())
        return Prediction(self._clip(prediction), len(raters), False)

    def _item_similarities(self, target_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        target = self.item_centered[:, target_index]
        numerator = target @ self.item_centered[:, candidate_indices]
        denominator = self.item_norms[target_index] * self.item_norms[candidate_indices]
        similarities = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        common = self.observed[:, target_index].astype(np.int16) @ self.observed[:, candidate_indices].astype(np.int16)
        similarities[common < self.min_common] = 0.0
        return similarities

    def predict_item_based(self, user_id: int, movie_id: int) -> Prediction:
        user_index = self._known_user(user_id)
        movie_index = self._known_movie(movie_id)
        if user_index is None:
            return Prediction(self.global_mean, 0, True)
        fallback = float(self.user_means[user_index])
        if movie_index is None:
            return Prediction(self._clip(fallback), 0, True)

        rated_items = np.flatnonzero(self.observed[user_index])
        similarities = self._item_similarities(movie_index, rated_items)
        eligible = similarities > 0
        rated_items, similarities = rated_items[eligible], similarities[eligible]
        if not len(rated_items):
            return Prediction(self._clip(fallback), 0, True)
        order = np.argsort(similarities)[-self.k_neighbors :]
        rated_items, similarities = rated_items[order], similarities[order]
        residuals = self.ratings[user_index, rated_items] - self.item_means[rated_items]
        prediction = self.item_means[movie_index] + float(
            np.dot(similarities, residuals) / np.abs(similarities).sum()
        )
        return Prediction(self._clip(prediction), len(rated_items), False)

    def similar_users(self, user_id: int, n: int = 10) -> pd.DataFrame:
        user_index = self._known_user(user_id)
        if user_index is None:
            raise KeyError(f"Unknown userId: {user_id}")
        scores = self.user_similarity[user_index]
        order = np.argsort(scores)[::-1]
        rows = [
            {"userId": int(self.user_ids[index]), "similarity": float(scores[index])}
            for index in order
            if scores[index] > 0
        ][:n]
        return pd.DataFrame(rows)

    def similar_items(self, movie_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        movie_index = self._known_movie(movie_id)
        if movie_index is None:
            raise KeyError(f"Unknown movieId: {movie_id}")
        candidates = np.arange(len(self.movie_ids))
        scores = self._item_similarities(movie_index, candidates)
        scores[movie_index] = 0.0
        order = np.argsort(scores)[::-1]
        rows = [
            {"movieId": int(self.movie_ids[index]), "similarity": float(scores[index])}
            for index in order
            if scores[index] > 0
        ][:n]
        return pd.DataFrame(rows).merge(movies[["movieId", "title"]], on="movieId", how="left")

    def recommend(
        self,
        user_id: int,
        movies: pd.DataFrame,
        method: str = "user",
        n: int = 5,
        candidate_pool: int = 500,
    ) -> pd.DataFrame:
        """Recommend from the most-rated unseen training items for stable coverage."""
        user_index = self._known_user(user_id)
        if user_index is None:
            raise KeyError(f"Unknown userId: {user_id}")
        if method not in {"user", "item"}:
            raise ValueError("method must be 'user' or 'item'")
        popularity = self.observed.sum(axis=0)
        unseen = np.flatnonzero(~self.observed[user_index])
        candidates = unseen[np.argsort(popularity[unseen])[::-1][:candidate_pool]]
        predictor = self.predict_user_based if method == "user" else self.predict_item_based
        rows = []
        for movie_index in candidates:
            movie_id = int(self.movie_ids[movie_index])
            prediction = predictor(user_id, movie_id)
            if not prediction.used_fallback:
                rows.append(
                    {
                        "movieId": movie_id,
                        "predicted_rating": prediction.value,
                        "neighbors_used": prediction.neighbors_used,
                    }
                )
        result = pd.DataFrame(rows).sort_values(
            ["predicted_rating", "neighbors_used"], ascending=False
        ).head(n)
        return result.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")


def evaluate_predictions(actual: np.ndarray, predictions: list[Prediction]) -> dict[str, float | int]:
    values = np.array([prediction.value for prediction in predictions], dtype=np.float64)
    fallbacks = np.array([prediction.used_fallback for prediction in predictions], dtype=bool)
    errors = values - np.asarray(actual, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "evaluated_ratings": int(len(values)),
        "neighbor_prediction_coverage": float((~fallbacks).mean()),
        "fallback_predictions": int(fallbacks.sum()),
    }
