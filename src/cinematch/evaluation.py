"""Evaluation and ranking helpers for CineMatch Week 4."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd


Predictor = Callable[[int, int], float]


def ranking_metrics(
    ranked_movie_ids: list[int],
    relevant_movie_ids: set[int],
    k: int = 10,
) -> dict[str, float]:
    """Calculate Precision@K, Recall@K, HitRate@K and NDCG@K."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_movie_ids:
        raise ValueError("At least one relevant item is required")
    top_k = ranked_movie_ids[:k]
    hits = [1 if movie_id in relevant_movie_ids else 0 for movie_id in top_k]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant_movie_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant_movie_ids),
        "hit_rate_at_k": float(hit_count > 0),
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def build_ranking_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movie_ids: np.ndarray,
    *,
    user_limit: int = 100,
    negative_samples: int = 100,
    relevance_threshold: float = 4.0,
    seed: int = 42,
) -> dict[int, dict[str, object]]:
    """Create identical sampled candidate sets for every evaluated model."""
    if user_limit < 1 or negative_samples < 1:
        raise ValueError("user_limit and negative_samples must be positive")
    relevant = test.loc[test["rating"] >= relevance_threshold].groupby("userId")["movieId"].apply(set)
    eligible_users = np.array(sorted(relevant.index.astype(int)))
    rng = np.random.default_rng(seed)
    if len(eligible_users) > user_limit:
        selected_users = np.sort(rng.choice(eligible_users, size=user_limit, replace=False))
    else:
        selected_users = eligible_users

    train_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
    test_by_user = test.groupby("userId")["movieId"].apply(set).to_dict()
    all_movies = np.asarray(sorted(set(int(value) for value in movie_ids)), dtype=np.int64)
    output: dict[int, dict[str, object]] = {}
    for user_id in selected_users:
        known_train = train_by_user.get(int(user_id), set())
        held_out = test_by_user.get(int(user_id), set())
        excluded = known_train | held_out
        negative_pool = all_movies[~np.isin(all_movies, list(excluded))]
        sample_size = min(negative_samples, len(negative_pool))
        negatives = rng.choice(negative_pool, size=sample_size, replace=False)
        candidates = sorted(held_out | set(int(value) for value in negatives))
        output[int(user_id)] = {
            "candidates": candidates,
            "relevant": set(int(value) for value in relevant.loc[user_id]),
        }
    return output


def evaluate_ranking(
    predictors: dict[str, Predictor],
    candidates_by_user: dict[int, dict[str, object]],
    *,
    k: int = 10,
    catalog_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on identical user/candidate sets."""
    aggregate_rows = []
    detail_rows = []
    for model_name, predictor in predictors.items():
        recommended_catalog: set[int] = set()
        model_rows = []
        for user_id, payload in candidates_by_user.items():
            candidates = payload["candidates"]
            relevant = payload["relevant"]
            scored = [(int(movie_id), float(predictor(user_id, int(movie_id)))) for movie_id in candidates]
            ranked = [movie_id for movie_id, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))]
            top_k = ranked[:k]
            recommended_catalog.update(top_k)
            metrics = ranking_metrics(ranked, relevant, k=k)
            model_rows.append(metrics)
            detail_rows.append(
                {
                    "model": model_name,
                    "userId": user_id,
                    "relevant_items": len(relevant),
                    "candidate_items": len(candidates),
                    "recommended_movie_ids": "|".join(str(value) for value in top_k),
                    **metrics,
                }
            )
        frame = pd.DataFrame(model_rows)
        aggregate_rows.append(
            {
                "model": model_name,
                "evaluated_users": len(frame),
                "k": k,
                "precision_at_k": float(frame["precision_at_k"].mean()),
                "recall_at_k": float(frame["recall_at_k"].mean()),
                "hit_rate_at_k": float(frame["hit_rate_at_k"].mean()),
                "ndcg_at_k": float(frame["ndcg_at_k"].mean()),
                "catalog_coverage_at_k": len(recommended_catalog) / catalog_size,
                "unique_recommended_items": len(recommended_catalog),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(detail_rows)


def popularity_fallback(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    n: int = 10,
    prior_weight: int = 50,
) -> pd.DataFrame:
    """Return Bayesian-weighted popular movies for users with no history."""
    if n < 1 or prior_weight < 1:
        raise ValueError("n and prior_weight must be positive")
    global_mean = float(ratings["rating"].mean())
    stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "size"), mean_rating=("rating", "mean")
    )
    stats["weighted_score"] = (
        stats["rating_count"] / (stats["rating_count"] + prior_weight) * stats["mean_rating"]
        + prior_weight / (stats["rating_count"] + prior_weight) * global_mean
    )
    return (
        stats.reset_index()
        .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
        .sort_values(["weighted_score", "rating_count", "movieId"], ascending=[False, False, True])
        .head(n)
        .reset_index(drop=True)
    )
