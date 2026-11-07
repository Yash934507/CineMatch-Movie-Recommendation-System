"""Hybrid recommendation helpers for CineMatch.

The hybrid system combines a collaborative score with a user-genre preference
signal. The normalization step uses min-max scaling within the user's candidate
set so each component occupies the same numeric range while remaining robust to
constant-score candidate pools and zero-variance inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "collaborative": 0.8,
    "genre": 0.2,
    "popularity": 0.0,
}


def normalize_genres(value: object) -> list[str]:
    """Return a canonical genre list from a MovieLens genres string or iterable."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "(no genres listed)":
            return []
        return [genre.strip() for genre in text.split("|") if genre and genre.strip()]
    if isinstance(value, (list, tuple, set)):
        genres = []
        for item in value:
            if item is None:
                continue
            cleaned = str(item).strip()
            if cleaned and cleaned != "(no genres listed)":
                genres.append(cleaned)
        return genres
    raise TypeError(f"Unsupported genre format: {type(value)!r}")


def build_genre_profile(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    *,
    min_rating: float | None = None,
) -> dict[str, float]:
    """Build a weighted genre-preference profile from a user's observed ratings.

    Ratings contribute in proportion to their value so highly liked movies exert
    more influence than neutral or low scores. Missing ratings are never treated
    as dislikes; only observed ratings appear in the profile.
    """
    user_ratings = ratings.loc[ratings["userId"] == int(user_id)].copy()
    if user_ratings.empty:
        return {}

    movie_lookup = movies.set_index("movieId", drop=False)
    profile: dict[str, float] = {}
    for row in user_ratings.itertuples(index=False):
        score = float(row.rating)
        if min_rating is not None and score < min_rating:
            continue
        movie_id = int(row.movieId)
        if movie_id not in movie_lookup.index:
            continue
        for genre in normalize_genres(movie_lookup.loc[movie_id, "genres"]):
            profile[genre] = profile.get(genre, 0.0) + score
    return dict(sorted(profile.items()))


def candidate_genre_similarity(profile: Mapping[str, float], movie_genres: object) -> float:
    """Compute cosine similarity between a user genre profile and a candidate movie."""
    if not profile:
        return 0.0
    candidate = {genre: 1.0 for genre in normalize_genres(movie_genres)}
    if not candidate:
        return 0.0
    genres = sorted(set(profile) | set(candidate))
    user_vector = np.asarray([float(profile.get(genre, 0.0)) for genre in genres], dtype=np.float64)
    movie_vector = np.asarray([float(candidate.get(genre, 0.0)) for genre in genres], dtype=np.float64)
    user_norm = float(np.linalg.norm(user_vector))
    movie_norm = float(np.linalg.norm(movie_vector))
    if user_norm == 0.0 or movie_norm == 0.0:
        return 0.0
    similarity = float(np.dot(user_vector, movie_vector) / (user_norm * movie_norm))
    return float(np.clip(similarity, 0.0, 1.0))


def normalize_component_scores(
    scores: Mapping[int, float] | Iterable[float],
    *,
    method: str = "min_max",
) -> dict[int, float] | list[float]:
    """Normalize component scores within a candidate set.

    Min-max normalization is used because it keeps the result within [0, 1], is
    deterministic for the same candidate set, and avoids NaN/inf for constant
    candidate sets through a safe zero-variance branch.
    """
    if method != "min_max":
        raise ValueError("Only 'min_max' normalization is supported")

    if isinstance(scores, Mapping):
        items = list(scores.items())
        values = np.asarray([float(value) for _, value in items], dtype=np.float64)
        if values.size == 0:
            return {}
        minimum = float(values.min())
        maximum = float(values.max())
        if np.isclose(maximum, minimum):
            return {movie_id: 0.0 for movie_id, _ in items}
        normalized = (values - minimum) / (maximum - minimum)
        return {movie_id: float(score) for (movie_id, _), score in zip(items, normalized)}

    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return []
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(maximum, minimum):
        return [0.0 for _ in values]
    return [(float(value) - minimum) / (maximum - minimum) for value in values]


def validate_hybrid_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate hybrid weights and return a trusted copy."""
    if set(weights) != {"collaborative", "genre", "popularity"}:
        missing = {"collaborative", "genre", "popularity"} - set(weights)
        if missing:
            raise ValueError(f"Missing weights: {sorted(missing)}")
        extra = set(weights) - {"collaborative", "genre", "popularity"}
        if extra:
            raise ValueError(f"Unexpected weights: {sorted(extra)}")
    validated = {name: float(value) for name, value in weights.items()}
    if any(value < 0.0 for value in validated.values()):
        raise ValueError("Hybrid weights must be non-negative")
    total = sum(validated.values())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Hybrid weights must sum to 1.0; received {total!r}")
    return validated


def combine_hybrid_scores(
    collaborative_scores: Mapping[int, float],
    genre_scores: Mapping[int, float],
    *,
    popularity_scores: Mapping[int, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[int, float]:
    """Blend normalized component scores into a single hybrid score per candidate."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    validated = validate_hybrid_weights(weights)
    candidate_ids = sorted(set(collaborative_scores) | set(genre_scores) | set(popularity_scores or {}))
    if not candidate_ids:
        return {}

    collaborative_norm = normalize_component_scores(collaborative_scores)
    genre_norm = normalize_component_scores(genre_scores)
    popularity_norm = (
        normalize_component_scores(popularity_scores)
        if popularity_scores is not None
        else {movie_id: 0.0 for movie_id in candidate_ids}
    )

    combined: dict[int, float] = {}
    for movie_id in candidate_ids:
        total = (
            validated["collaborative"] * collaborative_norm.get(movie_id, 0.0)
            + validated["genre"] * genre_norm.get(movie_id, 0.0)
            + validated["popularity"] * popularity_norm.get(movie_id, 0.0)
        )
        combined[int(movie_id)] = float(total)
    return combined


def rank_hybrid_candidates(
    collaborative_scores: Mapping[int, float],
    genre_scores: Mapping[int, float],
    *,
    popularity_scores: Mapping[int, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Return ranking pairs sorted by hybrid score descending and movie-id ascending."""
    combined = combine_hybrid_scores(
        collaborative_scores,
        genre_scores,
        popularity_scores=popularity_scores,
        weights=weights,
    )
    return sorted(combined.items(), key=lambda item: (-float(item[1]), int(item[0])))


def intra_list_diversity(
    recommended_movie_ids: Iterable[int],
    movies: pd.DataFrame,
) -> float:
    """Return average pairwise genre-set distance over the returned Top-K list."""
    recommendations = [int(movie_id) for movie_id in recommended_movie_ids]
    if len(recommendations) < 2:
        return 0.0
    movie_lookup = movies.set_index("movieId", drop=False)
    genre_sets = []
    for movie_id in recommendations:
        if movie_id not in movie_lookup.index:
            genre_sets.append(set())
            continue
        genre_sets.append(set(normalize_genres(movie_lookup.loc[movie_id, "genres"])))

    total_distance = 0.0
    pair_count = 0
    for left_index in range(len(genre_sets)):
        for right_index in range(left_index + 1, len(genre_sets)):
            left = genre_sets[left_index]
            right = genre_sets[right_index]
            if not left and not right:
                distance = 0.0
            elif not left or not right:
                distance = 1.0
            else:
                union_size = len(left | right)
                if union_size == 0:
                    distance = 0.0
                else:
                    distance = 1.0 - len(left & right) / union_size
            total_distance += distance
            pair_count += 1
    return float(total_distance / pair_count) if pair_count else 0.0


def genre_coverage(
    recommended_movie_ids: Iterable[int],
    movies: pd.DataFrame,
    *,
    catalog_genres: Iterable[str] | None = None,
) -> float:
    """Return the fraction of distinct genres covered by the current recommendation list."""
    recommended = [int(movie_id) for movie_id in recommended_movie_ids]
    if not recommended:
        return 0.0
    movie_lookup = movies.set_index("movieId", drop=False)
    recommended_genres = set()
    for movie_id in recommended:
        if movie_id in movie_lookup.index:
            recommended_genres.update(normalize_genres(movie_lookup.loc[movie_id, "genres"]))
    if catalog_genres is None:
        catalog_genres = set()
        for genre_list in movies["genres"].dropna():
            catalog_genres.update(normalize_genres(genre_list))
    if not recommended_genres:
        return 0.0
    catalog_set = set(catalog_genres)
    if not catalog_set:
        return 1.0
    return float(len(recommended_genres & catalog_set) / len(catalog_set))


def movie_genre_similarity(left: object, right: object) -> float:
    """Compute Jaccard similarity for two movie genre sets.

    Missing or empty genre data is treated as no usable signal rather than as a
    perfect match. This makes the reranking penalty conservative and explainable.
    """
    left_set = set(normalize_genres(left))
    right_set = set(normalize_genres(right))
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set) / len(union))


def compute_item_novelty(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    smooth: float = 1.0,
) -> dict[int, float]:
    """Return training-period self-information novelty for each movie.

    Larger novelty means the movie is less popular in the training history. The
    formula uses training interaction counts only and applies a small smoothing
    constant to avoid infinite novelty for zero-count movies in a finite catalog.
    """
    counts = ratings["movieId"].value_counts().astype(float)
    total = float(counts.sum())
    movie_count = float(len(movies))
    novelty_by_id: dict[int, float] = {}
    for movie_id in movies["movieId"].tolist():
        count = float(counts.get(int(movie_id), 0.0))
        if total <= 0.0:
            novelty_by_id[int(movie_id)] = 0.0
            continue
        probability = (count + smooth) / (total + smooth * movie_count)
        novelty_by_id[int(movie_id)] = float(-np.log2(max(probability, 1e-12)))
    return novelty_by_id


def mmr_rerank(
    relevance_scores: Mapping[int, float],
    candidate_ids: Iterable[int],
    movies: pd.DataFrame,
    *,
    lambda_: float = 1.0,
    k: int = 10,
    exclude: Iterable[int] | None = None,
) -> list[int]:
    """Greedily rerank candidates with maximal marginal relevance."""
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda must be between 0 and 1 inclusive")
    if k < 1:
        raise ValueError("k must be positive")
    if exclude is None:
        exclude = set()
    excluded = set(int(value) for value in exclude)
    remaining = [int(movie_id) for movie_id in candidate_ids if int(movie_id) not in excluded]
    if not remaining:
        return []

    relevance_norm = normalize_component_scores(relevance_scores)
    selected: list[int] = []
    movie_lookup = movies.set_index("movieId", drop=False)
    while len(selected) < min(k, len(remaining)):
        best_movie = None
        best_score = -np.inf
        for movie_id in remaining:
            if movie_id in selected:
                continue
            relevance_term = float(relevance_norm.get(movie_id, 0.0))
            if selected:
                maximum_similarity = 0.0
                for selected_id in selected:
                    current_genres = movie_lookup.loc[movie_id, "genres"] if movie_id in movie_lookup.index else None
                    existing_genres = movie_lookup.loc[selected_id, "genres"] if selected_id in movie_lookup.index else None
                    maximum_similarity = max(
                        maximum_similarity,
                        movie_genre_similarity(current_genres, existing_genres),
                    )
            else:
                maximum_similarity = 0.0
            score = lambda_ * relevance_term - (1.0 - lambda_) * maximum_similarity
            if score > best_score or (np.isclose(score, best_score) and (best_movie is None or movie_id < best_movie)):
                best_score = score
                best_movie = movie_id
        if best_movie is None:
            break
        selected.append(best_movie)
        remaining.remove(best_movie)
    return selected


def novelty_rerank(
    relevance_scores: Mapping[int, float],
    candidate_ids: Iterable[int],
    novelty_scores: Mapping[int, float],
    *,
    novelty_weight: float = 0.5,
    k: int = 10,
) -> list[int]:
    """Rank candidates by a simple relevance-plus-novelty score."""
    if not 0.0 <= float(novelty_weight) <= 1.0:
        raise ValueError("novelty_weight must be between 0 and 1 inclusive")
    relevance_norm = normalize_component_scores(relevance_scores)
    novelty_norm = normalize_component_scores(novelty_scores)
    ranked = sorted(
        [int(movie_id) for movie_id in candidate_ids],
        key=lambda movie_id: (
            -(
                relevance_norm.get(movie_id, 0.0)
                + novelty_weight * novelty_norm.get(movie_id, 0.0)
            ),
            int(movie_id),
        ),
    )
    return ranked[:k]


def mmr_with_novelty_rerank(
    relevance_scores: Mapping[int, float],
    candidate_ids: Iterable[int],
    movies: pd.DataFrame,
    novelty_scores: Mapping[int, float],
    *,
    lambda_: float = 0.7,
    novelty_weight: float = 0.25,
    k: int = 10,
    exclude: Iterable[int] | None = None,
) -> list[int]:
    """Greedy rerank that combines MMR diversity and novelty bonuses."""
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda must be between 0 and 1 inclusive")
    if exclude is None:
        exclude = set()
    excluded = set(int(value) for value in exclude)
    remaining = [int(movie_id) for movie_id in candidate_ids if int(movie_id) not in excluded]
    if not remaining:
        return []
    relevance_norm = normalize_component_scores(relevance_scores)
    novelty_norm = normalize_component_scores(novelty_scores)
    selected: list[int] = []
    movie_lookup = movies.set_index("movieId", drop=False)
    while len(selected) < min(k, len(remaining)):
        best_movie = None
        best_score = -np.inf
        for movie_id in remaining:
            if movie_id in selected:
                continue
            relevance_term = float(relevance_norm.get(movie_id, 0.0))
            novelty_term = float(novelty_norm.get(movie_id, 0.0))
            if selected:
                maximum_similarity = 0.0
                for selected_id in selected:
                    current_genres = movie_lookup.loc[movie_id, "genres"] if movie_id in movie_lookup.index else None
                    existing_genres = movie_lookup.loc[selected_id, "genres"] if selected_id in movie_lookup.index else None
                    maximum_similarity = max(
                        maximum_similarity,
                        movie_genre_similarity(current_genres, existing_genres),
                    )
            else:
                maximum_similarity = 0.0
            score = (
                lambda_ * relevance_term
                - (1.0 - lambda_) * maximum_similarity
                + novelty_weight * novelty_term
            )
            if score > best_score or (np.isclose(score, best_score) and (best_movie is None or movie_id < best_movie)):
                best_score = score
                best_movie = movie_id
        if best_movie is None:
            break
        selected.append(best_movie)
        remaining.remove(best_movie)
    return selected


def mean_topk_novelty(recommended_movie_ids: Iterable[int], novelty_scores: Mapping[int, float]) -> float:
    """Return the mean novelty across the Top-K recommendation list."""
    recommendations = [int(movie_id) for movie_id in recommended_movie_ids]
    if not recommendations:
        return 0.0
    return float(np.mean([float(novelty_scores.get(movie_id, 0.0)) for movie_id in recommendations]))


def unique_genre_count(recommended_movie_ids: Iterable[int], movies: pd.DataFrame) -> int:
    """Return the count of unique genres across a recommendation list."""
    recommendations = [int(movie_id) for movie_id in recommended_movie_ids]
    if not recommendations:
        return 0
    movie_lookup = movies.set_index("movieId", drop=False)
    genres: set[str] = set()
    for movie_id in recommendations:
        if movie_id in movie_lookup.index:
            genres.update(normalize_genres(movie_lookup.loc[movie_id, "genres"]))
    return len(genres)


def catalog_coverage(
    recommended_movie_ids: Iterable[int],
    eligible_movie_ids: Iterable[int],
) -> float:
    """Return the fraction of eligible catalog movies recommended at least once."""
    recommended = {int(movie_id) for movie_id in recommended_movie_ids}
    eligible = {int(movie_id) for movie_id in eligible_movie_ids}
    if not eligible:
        return 0.0
    return float(len(recommended & eligible) / len(eligible))
