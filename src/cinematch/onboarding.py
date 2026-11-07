"""Shared SVD fold-in and leakage-safe chronological onboarding evaluation.

The production model is never loaded here. Evaluation fits a fresh base model
before a fixed time cutoff and uses disjoint validation and test users.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .evaluation import popularity_fallback, ranking_metrics
from .svd_model import fit_svd


MIN_RATINGS = 3
DEFAULT_CUTOFF = 1104537600  # 2005-01-01 00:00:00 UTC; fixed before evaluation.
METRICS = ("precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k")


@dataclass(frozen=True)
class FoldInProfile:
    bias: float
    factors: np.ndarray
    rated_ids: frozenset[int]


def fold_in(model, ratings: list[tuple[int, float]]) -> FoldInProfile:
    """Fit a temporary user with the original API's ridge penalties.

    Solve (X.T X + D) c = X.T (r - global_mean - item_bias), where
    X = [1, item_factors], D = diag(0.02, 0.1, ..., 0.1).
    The supplied model and its factors are never modified.
    """
    ids = [movie_id for movie_id, _ in ratings]
    if len(set(ids)) != len(ids):
        raise ValueError("Each onboarding movie can only be rated once")
    if len(ratings) < MIN_RATINGS:
        raise ValueError(f"At least {MIN_RATINGS} unique ratings are required")
    values = np.asarray([rating for _, rating in ratings], dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.5) & (values <= 5)).all():
        raise ValueError("Ratings must be finite and between 0.5 and 5.0")
    inner_ids = []
    for movie_id in ids:
        try:
            inner_ids.append(int(model.trainset.to_inner_iid(movie_id)))
        except ValueError as error:
            raise ValueError(f"Movie {movie_id} has no learned SVD factors") from error
    design = np.column_stack([np.ones(len(ids)), model.qi[np.asarray(inner_ids)]])
    targets = values - float(model.trainset.global_mean) - model.bi[inner_ids]
    penalty = np.eye(design.shape[1]) * 0.1
    penalty[0, 0] = 0.02
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return FoldInProfile(float(coefficients[0]), coefficients[1:], frozenset(ids))


def score_fold_in(model, profile: FoldInProfile, movie_ids: list[int]) -> np.ndarray:
    """Score supplied candidates, including unseen items with zero bias/factors.

    Unknown items get global mean + temporary user bias. Evaluation retains
    them rather than silently dropping future relevant movies.
    """
    scores = np.full(len(movie_ids), float(model.trainset.global_mean) + profile.bias)
    for position, movie_id in enumerate(movie_ids):
        try:
            inner_id = model.trainset.to_inner_iid(movie_id)
        except ValueError:
            continue
        scores[position] += float(model.bi[inner_id]) + float(np.dot(profile.factors, model.qi[inner_id]))
    return np.clip(scores, 0.5, 5.0)


def rank_fold_in(model, profile: FoldInProfile, movie_ids: list[int]) -> list[tuple[int, float]]:
    """Exclude onboarding movies and break score ties by ascending movie ID."""
    candidates = sorted(set(movie_ids) - profile.rated_ids)
    scores = score_fold_in(model, profile, candidates)
    return sorted(zip(candidates, scores.tolist()), key=lambda pair: (-pair[1], pair[0]))


def validate_history(ratings: pd.DataFrame) -> None:
    required = {"userId", "movieId", "rating", "timestamp"}
    if required - set(ratings):
        raise ValueError(f"Missing columns: {sorted(required - set(ratings))}")
    if ratings.empty or not np.isfinite(ratings[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("History must be nonempty and finite")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user/movie rating")
    if not ratings.rating.between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0")


def partition_history(ratings: pd.DataFrame, cutoff: int, seed: int):
    """Train strictly before cutoff; split genuinely new users 50/50 by seed."""
    validate_history(ratings)
    train = ratings.loc[ratings.timestamp < cutoff].sort_values(
        ["userId", "timestamp", "movieId"]
    ).reset_index(drop=True)
    if train.empty:
        raise ValueError("No ratings before cutoff")
    first = ratings.groupby("userId").timestamp.min()
    new_users = np.asarray(sorted(first[first >= cutoff].index), dtype=np.int64)
    if len(new_users) < 2:
        raise ValueError("Need at least two new users after cutoff")
    shuffled = np.random.default_rng(seed).permutation(new_users)
    split = len(shuffled) // 2
    cohorts = {"validation": sorted(shuffled[:split].tolist()), "test": sorted(shuffled[split:].tolist())}
    return train, cohorts


def chronological_profile(history: pd.DataFrame, size: int, known_items: set[int]):
    """Return earliest N ratings and strictly later ratings, or an exclusion reason.

    Equal timestamps are ordered by movieId. Unselected ratings tied with the
    Nth timestamp are neither onboarding nor test observations.
    """
    if size < MIN_RATINGS:
        raise ValueError(f"Onboarding size must be at least {MIN_RATINGS}")
    validate_history(history)
    if history.userId.nunique() != 1:
        raise ValueError("Expected one user's history")
    ordered = history.sort_values(["timestamp", "movieId"]).reset_index(drop=True)
    if len(ordered) <= size:
        return None, "insufficient_history"
    early = ordered.iloc[:size].copy()
    if not set(early.movieId).issubset(known_items):
        return None, "unknown_onboarding_item"
    later = ordered.loc[ordered.timestamp > early.timestamp.max()].copy()
    if not (later.rating >= 4.0).any():
        return None, "no_later_relevant_movie"
    return (early, later), None


def generate_candidates(history: pd.DataFrame, later: pd.DataFrame, catalog: set[int], *,
                        negative_samples: int = 100, seed: int = 42, user_id: int = 0):
    """All later movies plus uniform unseen training-catalog samples, no replacement.

    Exclude the user's entire observed history from the negative pool, including
    tied timestamps. User-specific seeds preserve samples across onboarding sizes.
    """
    if negative_samples < 0:
        raise ValueError("negative_samples cannot be negative")
    pool = np.asarray(sorted(catalog - set(history.movieId)), dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, user_id]))
    negatives = rng.choice(pool, size=min(negative_samples, len(pool)), replace=False)
    return sorted(set(later.movieId) | set(negatives.tolist())), len(negatives)


def aggregate_details(details: pd.DataFrame, catalog_size: int) -> pd.DataFrame:
    rows = []
    for (cohort, size, strategy), group in details.groupby(["cohort", "onboarding_size", "strategy"], sort=True):
        recommended = set()
        for serialized in group.recommended_movie_ids:
            recommended.update(json.loads(serialized))
        rows.append({
            "cohort": cohort, "onboarding_size": int(size), "strategy": strategy,
            "eligible_users": len(group), "k": 10,
            "average_relevant_movies": float(group.relevant_items.mean()),
            "average_candidate_size": float(group.candidate_items.mean()),
            "average_unknown_test_movies": float(group.unknown_test_items.mean()),
            **{metric: float(group[metric].mean()) for metric in METRICS},
            "catalog_coverage": len(recommended) / catalog_size,
            "unique_recommended_items": len(recommended), "catalog_size": catalog_size,
        })
    return pd.DataFrame(rows)


def select_product_policy(validation: pd.DataFrame) -> dict:
    """Validation only: common users, NDCG, 0.01 absolute tolerance for friction.

    Keep the smallest size within 0.01 of best fold-in NDCG. Retain fold-in only
    if its paired mean NDCG exceeds popularity at that size. No test selection.
    """
    if not (validation.cohort == "validation").all():
        raise ValueError("Product selection accepts validation users only")
    sizes = sorted(validation.onboarding_size.unique())
    common = set.intersection(*[
        set(validation.loc[validation.onboarding_size == size, "userId"]) for size in sizes
    ])
    if not common or sizes != [3, 5, 10]:
        raise ValueError("Selection requires common validation users at 3, 5, and 10")
    paired = validation.loc[validation.userId.isin(common)]
    means = paired.groupby(["onboarding_size", "strategy"]).ndcg_at_k.mean().unstack()
    best = float(means["svd-fold-in"].max())
    selected = int(means.index[means["svd-fold-in"] >= best - 0.01].min())
    delta = float(means.loc[selected, "svd-fold-in"] - means.loc[selected, "bayesian-popularity"])
    # A small or uncertain difference warrants retaining current behavior rather
    # than declaring superiority. The paired bootstrap is fixed and reproducible.
    chosen = paired.loc[paired.onboarding_size == selected].pivot(
        index="userId", columns="strategy", values="ndcg_at_k"
    )
    differences = (chosen["svd-fold-in"] - chosen["bayesian-popularity"]).to_numpy()
    rng = np.random.default_rng(42)
    samples = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "selection_source": "common validation users only", "common_validation_users": len(common),
        "best_validation_size": int(means["svd-fold-in"].idxmax()),
        "selected_minimum": selected if low > 0 else 3,
        "provisional_preferred_size": selected,
        "continue_fold_in": high >= 0,
        "popularity_remains_fallback": True,
        "paired_ndcg_gain": delta, "paired_bootstrap_95_interval": [low, high],
        "decision": ("Evidence favors fold-in; use the smallest near-best minimum." if low > 0 else
                     "Evidence favors popularity; fold-in should remain experimental." if high < 0 else
                     "Mixed evidence: retain three-rating fold-in provisionally; do not claim superiority."),
        "selection_rule": "Paired validation NDCG; smallest size within 0.01 of best; change minimum only with positive 95% paired bootstrap gain over popularity.",
        "validation_ndcg": {str(int(size)): {key: float(value) for key, value in row.items()}
                            for size, row in means.to_dict(orient="index").items()},
    }


def evaluate_onboarding(ratings: pd.DataFrame, movies: pd.DataFrame, *,
                        cutoff: int = DEFAULT_CUTOFF, seed: int = 42,
                        negative_samples: int = 100):
    """Fit once on permitted data; lock validation policy before scoring test users."""
    train, cohorts = partition_history(ratings, cutoff, seed)
    model = fit_svd(train, n_factors=20, n_epochs=20, random_state=seed)
    catalog = set(int(x) for x in train.movieId.unique())
    full_catalog = set(int(x) for x in movies.movieId)
    if not set(ratings.movieId).issubset(full_catalog):
        raise ValueError("Ratings reference a movie outside the catalog")
    popularity = popularity_fallback(train, movies, n=len(catalog), prior_weight=50)
    popular_scores = popularity.set_index("movieId").weighted_score.to_dict()
    counts = popularity.set_index("movieId").rating_count.to_dict()
    global_mean = float(train.rating.mean())
    histories = {int(uid): group for uid, group in ratings.groupby("userId")}
    rows, exclusions = [], []
    policy = None
    for cohort, users in cohorts.items():
        for size in (3, 5, 10):
            for user_id in users:
                history = histories[user_id]
                result, reason = chronological_profile(history, size, catalog)
                if reason:
                    exclusions.append({"cohort": cohort, "onboarding_size": size,
                                       "userId": user_id, "reason": reason})
                    continue
                early, later = result
                candidates, negative_count = generate_candidates(
                    history, later, catalog, negative_samples=negative_samples, seed=seed, user_id=user_id
                )
                relevant = set(int(x) for x in later.loc[later.rating >= 4, "movieId"])
                profile = fold_in(model, list(early[["movieId", "rating"]].itertuples(index=False, name=None)))
                ranked_fold = [movie for movie, _ in rank_fold_in(model, profile, candidates)]
                ranked_pop = sorted(candidates, key=lambda movie: (
                    -popular_scores.get(movie, global_mean), -counts.get(movie, 0), movie
                ))
                if set(ranked_fold) != set(ranked_pop) or set(candidates) & profile.rated_ids:
                    raise AssertionError("Strategies must rank the same unseen candidates")
                candidate_json = json.dumps(candidates)
                for strategy, ranked in (("svd-fold-in", ranked_fold), ("bayesian-popularity", ranked_pop)):
                    rows.append({
                        "cohort": cohort, "onboarding_size": size, "strategy": strategy, "userId": user_id,
                        "onboarding_movie_ids": json.dumps(early.movieId.astype(int).tolist()),
                        "onboarding_end_timestamp": int(early.timestamp.max()),
                        "test_start_timestamp": int(later.timestamp.min()),
                        "test_items": len(later), "relevant_items": len(relevant),
                        "unknown_test_items": len(set(later.movieId) - catalog),
                        "tied_ratings_excluded": int(((history.timestamp == early.timestamp.max()) &
                                                     ~history.movieId.isin(early.movieId)).sum()),
                        "candidate_items": len(candidates), "negative_samples": negative_count,
                        "candidate_sha256": hashlib.sha256(candidate_json.encode()).hexdigest(),
                        "candidate_movie_ids": candidate_json,
                        "relevant_movie_ids": json.dumps(sorted(relevant)),
                        "recommended_movie_ids": json.dumps(ranked[:10]),
                        **ranking_metrics(ranked, relevant, k=10),
                    })
        if cohort == "validation":
            policy = select_product_policy(pd.DataFrame(rows))
    details = pd.DataFrame(rows)
    metrics = aggregate_details(details, len(full_catalog))
    summary = {
        "seed": seed, "cutoff_timestamp": cutoff,
        "cutoff_utc": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "training_ratings": len(train), "training_users": int(train.userId.nunique()),
        "training_items": len(catalog), "training_max_timestamp": int(train.timestamp.max()),
        "new_user_cohorts": cohorts, "excluded_user_sizes": exclusions,
        "n_factors": 20, "n_epochs": 20, "factor_penalty": 0.1, "bias_penalty": 0.02,
        "popularity_prior_weight": 50, "negative_samples_requested": negative_samples,
        "k": 10, "relevance_threshold": 4.0, "catalog_size": len(full_catalog),
        "product_policy": policy, "metrics": metrics.to_dict(orient="records"),
    }
    return metrics, details, summary
